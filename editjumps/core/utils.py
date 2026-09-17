"""Utils for all pipelines."""

# Annotations reference names imported only under TYPE_CHECKING below.
from __future__ import annotations

import json
import logging
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    import mlflow

DEFAULT_MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"


def get_mlflow_tracking_uri() -> str:
    """Resolve the MLflow tracking store URI, in precedence order. 1. ``MLFLOW_TRACKING_URI`` env var —."""
    env = os.environ.get("MLFLOW_TRACKING_URI")
    if env:
        return env
    try:
        import yaml

        params = yaml.safe_load(Path("params.yaml").read_text())
        uri = ((params or {}).get("mlflow") or {}).get("tracking_uri")
        if uri:
            return str(uri)
    except Exception:  # noqa: BLE001 - params.yaml is optional; any read failure -> local default
        pass
    return DEFAULT_MLFLOW_TRACKING_URI


def _gcp_identity_token() -> str | None:
    """Best-effort GCP identity token for the private Cloud Run MLflow server."""
    import subprocess

    try:
        out = subprocess.run(
            ["gcloud", "auth", "print-identity-token"],
            capture_output=True, text=True, timeout=15, check=False,
        )
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001 - gcloud missing/slow/unauth -> no token
        return None


# A GCP identity token expires after one hour, and the sky jobs mint exactly one before python starts.
METADATA_IDENTITY_URL = (
    "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/identity"
)
TOKEN_REFRESH_SECONDS = 2400  # 40 min, against a 60 min expiry - two chances before it matters
_REFRESHER_NAME = "mlflow-token-refresh"


def _metadata_identity_token(audience: str, timeout: float = 5.0) -> str | None:
    """Mint an identity token from the GCE metadata server for ``audience``."""
    import urllib.request

    request = urllib.request.Request(
        f"{METADATA_IDENTITY_URL}?audience={audience}", headers={"Metadata-Flavor": "Google"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - fixed link-local URL
            return response.read().decode().strip() or None
    except Exception:  # noqa: BLE001 - not on GCE, or no identity for this SA -> caller falls back
        return None


def refresh_identity_token(audience: str) -> str | None:
    """Mint a fresh identity token, metadata server first and ``gcloud`` second."""
    return _metadata_identity_token(audience) or _gcp_identity_token()


def _start_token_refresher(audience: str) -> None:
    """Keep ``MLFLOW_TRACKING_TOKEN`` fresh in ``os.environ`` for the life of the process."""
    import threading
    import time

    if any(t.name == _REFRESHER_NAME for t in threading.enumerate()):
        return

    def loop() -> None:
        while True:
            time.sleep(TOKEN_REFRESH_SECONDS)
            refresh_token_once(audience)

    threading.Thread(target=loop, name=_REFRESHER_NAME, daemon=True).start()
    get_logger(__file__).info(
        f"MLflow identity token will refresh every {TOKEN_REFRESH_SECONDS}s (expiry is 3600s)"
    )


def refresh_token_once(audience: str) -> bool:
    """Replace ``MLFLOW_TRACKING_TOKEN`` with a freshly minted one."""
    logger = get_logger(__file__)
    token = refresh_identity_token(audience)
    if not token:
        logger.warning("could not refresh the MLflow identity token; keeping the current one")
        return False
    os.environ["MLFLOW_TRACKING_TOKEN"] = token
    logger.info(f"refreshed the MLflow identity token ({len(token)} bytes)")
    return True


def make_tracking_nonfatal() -> None:
    """Wrap mlflow's logging calls so a tracking-server failure cannot kill the run."""
    import mlflow

    logger = get_logger(__file__)
    failures = {"n": 0}

    def guard(fn: "Callable") -> "Callable":
        label = getattr(fn, "__name__", "log call")

        def wrapped(*args: object, **kwargs: object) -> object:
            try:
                return fn(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001 - tracking must never abort real work
                failures["n"] += 1
                if failures["n"] == 1:
                    logger.warning(
                        f"MLflow {label} failed ({type(exc).__name__}: {exc}); continuing without "
                        f"tracking - the stage's metrics JSON is the durable record"
                    )
                elif failures["n"] % 50 == 0:
                    logger.warning(f"MLflow logging still failing ({failures['n']} calls dropped)")
                return None

        wrapped._editjumps_guarded = True  # ty: ignore[unresolved-attribute]
        wrapped.__name__ = label
        return wrapped

    def guard_attribute(module: object, name: str) -> None:
        """Wrap ``module.name`` unless it is already wrapped (idempotent per attribute)."""
        fn = getattr(module, name)
        if not getattr(fn, "_editjumps_guarded", False):
            setattr(module, name, guard(fn))

    # Log_artifacts (plural) matters as much as the singular and was missing: it is the call that uploads a.
    for name in ("log_metric", "log_metrics", "log_params", "log_param", "set_tag",
                 "log_artifact", "log_artifacts", "log_text", "log_dict", "log_figure"):
        guard_attribute(mlflow, name)

    # Leaving the `with start_mlflow_run(...)` block is a REST call too, and it resolves the module global.
    from mlflow.tracking import fluent

    guard_attribute(fluent, "end_run")
    mlflow.end_run = fluent.end_run


# The modeling "design space" axes, recorded per run in ArmConfig / params.yaml: the categorical coordinates that.
DESIGN_AXES: tuple[str, ...] = (
    "pretrain_data",
    "split_method",
    "chain_handling",   # combine (VH.VL joined) | separate (H/L independent): clustering + training
    "target_property",  # always None here: this reproduction is unconditional generation
    "pair_source",      # always `homolog` here -- natural OAS pairs. The upstream project also
                        # emitted `repair` pairs (a labelled molecule damaged by one substitution);
                        # the axis is kept so a run from either records which it was.
    "backbone",
    "weight_init",
    "objective",
)


def design_space_tags(**axes: object) -> dict[str, str]:
    """Build a validated MLflow tag dict for the modeling design-space axes."""
    unknown = [k for k in axes if k not in DESIGN_AXES]
    if unknown:
        get_logger(__file__).warning(
            f"design_space_tags: non-standard axis keys {unknown}; standard axes are {list(DESIGN_AXES)}"
        )
    return {k: str(v) for k, v in axes.items()}


#: Environment variables a job sets to say where this run's artefacts will land in GCS.
GCS_DESTINATION_ENV: dict[str, str] = {
    "MODEL_GCS_URI": "gcs_model_uri",
    "CHECKPOINT_GCS_URI": "gcs_checkpoint_uri",
    "METRICS_GCS_URI": "gcs_metrics_uri",
}


def gcs_destination_tags(**extra: "str | None") -> dict[str, str]:
    """Collect the GCS destinations of this run's artefacts, as MLflow tags."""
    tags = {tag: os.environ[env] for env, tag in GCS_DESTINATION_ENV.items() if os.environ.get(env)}
    tags.update({name: value for name, value in extra.items() if value})
    return tags


def start_mlflow_run(
    experiment_name: str, run_name: str | None = None, tags: dict[str, str] | None = None
) -> "mlflow.ActiveRun":
    """Set the tracking URI/experiment and start an MLflow run."""
    import mlflow

    # Stamped here rather than per stage: a stage that forgets is a run whose weights cannot be
    # found, and the failure is silent. Caller tags win on a key collision.
    destinations = gcs_destination_tags()
    if destinations:
        tags = {**destinations, **(tags or {})}

    uri = get_mlflow_tracking_uri()
    # Private Cloud Run server needs a bearer token; mint one if the caller (e.g. a sky job or CI) hasn't.
    if uri.startswith("https://"):
        if not os.environ.get("MLFLOW_TRACKING_TOKEN"):
            token = _gcp_identity_token()
            if token:
                os.environ["MLFLOW_TRACKING_TOKEN"] = token
        _start_token_refresher(uri)
    make_tracking_nonfatal()
    # Fail fast instead of hanging if the server is unreachable/unauthorized.
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "10")
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")
    try:
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(experiment_name)
        return mlflow.start_run(run_name=run_name, tags=tags)
    except Exception as exc:  # noqa: BLE001 - never fail a run on a tracking-server issue
        if uri.startswith("sqlite"):
            raise
        get_logger(__file__).warning(
            f"MLflow server {uri} unavailable ({type(exc).__name__}); logging to {DEFAULT_MLFLOW_TRACKING_URI}"
        )
        mlflow.set_tracking_uri(DEFAULT_MLFLOW_TRACKING_URI)
        mlflow.set_experiment(experiment_name)
        return mlflow.start_run(run_name=run_name, tags=tags)


def load_params(path: Path = Path("params.yaml")) -> dict:
    """Parse ``params.yaml`` if it exists, else ``{}``."""
    if not path.exists():
        return {}
    import yaml

    return yaml.safe_load(path.read_text()) or {}


def sanitize_mlflow_name(key: object) -> str:
    """Make a string safe to use as an MLflow metric, param or tag name."""
    return re.sub(r"[^0-9A-Za-z_\-.: ]", "_", str(key))


def numbers_within(value: object) -> list[float]:
    """Collect every finite number anywhere inside a value, recursing through lists and dicts."""
    if isinstance(value, bool):
        return [float(value)]
    if isinstance(value, (int, float)):
        return [] if math.isnan(float(value)) else [float(value)]
    if isinstance(value, (list, tuple)):
        return [number for item in value for number in numbers_within(item)]
    if isinstance(value, dict):
        return [number for item in value.values() for number in numbers_within(item)]
    return []


def flatten_metrics(tree: dict, prefix: str = "", *, summarise_lists: bool = False) -> dict[str, float]:
    """Flatten a nested result dict into ``{path: value}`` for every finite number in it."""
    out: dict[str, float] = {}
    for key, value in tree.items():
        safe = sanitize_mlflow_name(key)
        path = f"{prefix}/{safe}" if prefix else safe
        if isinstance(value, dict):
            out.update(flatten_metrics(value, path, summarise_lists=summarise_lists))
        elif isinstance(value, bool):
            out[path] = float(value)
        elif isinstance(value, (int, float)) and not math.isnan(value):
            out[path] = float(value)
        elif summarise_lists and isinstance(value, (list, tuple)):
            # Length ALWAYS, so a list is visible in the run even when it holds no numbers at all (`caveats`.
            out[f"{path}/n"] = float(len(value))
            numbers = numbers_within(value)
            if numbers:
                out[f"{path}/mean"] = sum(numbers) / len(numbers)
    return out


def get_logger(name: str = __name__, level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger with a standard format."""
    logger = logging.getLogger(Path(name).stem)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)s [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False

    return logger


def load_tokenizer(source: "str | Path") -> object:
    """Load a tokenizer and narrow away the ``None`` its loader is typed to allow.."""
    from transformers import AutoTokenizer  # ty: ignore[unresolved-import]

    tokenizer = AutoTokenizer.from_pretrained(str(source))
    if tokenizer is None:
        raise RuntimeError(f"AutoTokenizer.from_pretrained({source!r}) returned None")
    return tokenizer


def decode_one(tokenizer: object, token_ids: object, *, skip_special_tokens: bool = True) -> str:
    """Decode one sequence of ids to a string, narrowing the ``str | list[str]`` return. ``decode`` is."""
    from editjumps.core.sequences import AA, PAIR_SEP

    decoded = tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)  # ty: ignore[unresolved-attribute]
    if not isinstance(decoded, str):
        raise TypeError(f"decode_one expected one sequence, got {type(decoded).__name__}")
    # Strip <...> first so a multi-character token cannot survive as its individual letters.
    stripped = re.sub(r"<[^<>]*>", "", decoded).replace(" ", "")
    return "".join(c for c in stripped if c in AA or c == PAIR_SEP)


def encode_one(tokenizer: object, text: str) -> list[int]:
    """Tokenise one sequence to a flat list of ids. ``tokenizer(text)`` is typed through the same."""
    return list(tokenizer(text)["input_ids"])  # ty: ignore[call-non-callable]


def tokens_of(tokenizer: object, token_ids: object) -> list[str]:
    """Convert ids to their token strings, narrowing the ``str | list[str]`` return.."""
    tokens = tokenizer.convert_ids_to_tokens(token_ids)  # ty: ignore[unresolved-attribute]
    return [tokens] if isinstance(tokens, str) else list(tokens)


def tok_attr(tokenizer: object, name: str) -> "int | None":
    """Read a special-token id off a tokenizer, past the possibly-``None`` union. ``mask_token_id``."""
    return getattr(tokenizer, name, None)


def token_id(tokenizer: object, token: str) -> "int | None":
    """Map one token string to its id."""
    return tokenizer.convert_tokens_to_ids(token)  # ty: ignore[unresolved-attribute]


def write_generated_fasta(
    metrics_path: "Path", sets: dict[str, list[str]], header: dict[str, object] | None = None
) -> "Path | None":
    """Save the sequences a run generated, beside its metrics JSON."""
    if not any(sets.values()):
        return None
    path = metrics_path.with_suffix(".fasta")
    lines = [f"; {key}: {value}" for key, value in (header or {}).items()]
    for name, sequences in sets.items():
        for index, sequence in enumerate(sequences):
            lines.append(f">{name}_{index}")
            lines.append(sequence)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path


#: How long to wait on ``git`` before recording nothing.
_GIT_TIMEOUT_S = 5.0


def code_provenance() -> dict[str, object]:
    """Describe the code that produced an artefact: the commit, whether the tree was clean, the command."""
    record: dict[str, object] = {"command": " ".join(sys.argv)}
    root = Path(__file__).resolve().parents[2]

    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True,
                              timeout=_GIT_TIMEOUT_S, check=True).stdout.strip()

    try:
        record["git_commit"] = git("rev-parse", "HEAD")
        # ``--untracked-files=no`` on purpose, twice over: an untracked scratch file does not change
        # the code that ran, and the untracked scan walks ``data/`` on this repo, which is slow.
        record["git_dirty"] = bool(git("status", "--porcelain", "--untracked-files=no"))
    except (OSError, subprocess.SubprocessError):
        pass
    return record


def write_metrics(metrics_path: "Path", report: dict, **extra: object) -> "Path":
    """Write a stage's metrics JSON with a ``provenance`` block naming the code behind it."""
    payload = {**report, "provenance": {**code_provenance(), **extra}}
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(payload, indent=2, default=str))
    return metrics_path
