"""Best-effort GCS checkpoint sync for long GPU training runs (torch-free)."""

import os
import re
import subprocess
from pathlib import Path

from editjumps.core.utils import get_logger

# Runtime override for the checkpoint URI, mirroring the repo's MLFLOW_TRACKING_URI idiom, so a SkyPilot job.
CHECKPOINT_URI_ENV = "EDITJUMPS_CHECKPOINT_URI"

logger = get_logger(__file__)

# gcloud can hang on a wedged network; bound every call so a stuck upload cannot stall a training
# step indefinitely (best-effort -> a timeout just logs + skips).
_GCLOUD_TIMEOUT_S = 600

# `gcloud storage cp` treats these as glob metacharacters and REFUSES a destination containing them.
GLOB_METACHARACTERS = "[]*?"


def gcs_safe_component(name: str) -> str:
    """Make one path component safe to use in a ``gs://`` URI."""
    return re.sub(rf"[{re.escape(GLOB_METACHARACTERS)}]+", "-", name).strip("-")


def assert_gcs_safe_uri(uri: str) -> None:
    """Raise if ``uri`` contains a character ``gcloud storage cp`` will reject as a wildcard."""
    found = sorted(set(uri) & set(GLOB_METACHARACTERS))
    if found:
        raise ValueError(
            f"GCS URI {uri!r} contains {found}, which `gcloud storage cp` rejects as a wildcard - "
            f"every write to it would fail. Pass each component through gcs_safe_component()."
        )


def gcs_upload(local: Path, gcs_uri: str) -> bool:
    """Upload a local file or directory to a ``gs://…`` destination (best-effort)."""
    return _run_cp(str(local), gcs_uri, what=f"upload {local} -> {gcs_uri}")


def gcs_download(gcs_uri: str, local: Path) -> bool:
    """Download a ``gs://…`` file or directory to a local path (best-effort)."""
    Path(local).parent.mkdir(parents=True, exist_ok=True)
    return _run_cp(gcs_uri, str(local), what=f"download {gcs_uri} -> {local}")


def gcs_copy(src_uri: str, dst_uri: str) -> bool:
    """Copy one ``gs://…`` object to another ``gs://…`` name, server-side (best-effort)."""
    return _run_cp(src_uri, dst_uri, what=f"copy {src_uri} -> {dst_uri}")


def gcs_delete(gcs_uri: str) -> bool:
    """Delete a ``gs://…`` object (best-effort)."""
    try:
        out = subprocess.run(
            ["gcloud", "storage", "rm", gcs_uri],
            capture_output=True, text=True, timeout=_GCLOUD_TIMEOUT_S, check=False,
        )
    except Exception as exc:  # noqa: BLE001 - gcloud missing/slow -> best-effort, never crash training
        logger.warning(f"gcs delete {gcs_uri} failed: {type(exc).__name__}: {exc}")
        return False
    if out.returncode != 0:
        logger.warning(f"gcs delete {gcs_uri} failed (exit {out.returncode}): {out.stderr.strip()}")
        return False
    return True


def gcs_list(gcs_uri: str) -> list[str]:
    """List the immediate entries under a ``gs://…`` prefix (best-effort)."""
    try:
        out = subprocess.run(
            ["gcloud", "storage", "ls", gcs_uri],
            capture_output=True, text=True, timeout=_GCLOUD_TIMEOUT_S, check=False,
        )
    except Exception as exc:  # noqa: BLE001 - gcloud missing/slow -> best-effort empty listing
        logger.warning(f"gcs_list {gcs_uri} failed: {type(exc).__name__}: {exc}")
        return []
    if out.returncode != 0:
        logger.warning(f"gcs_list {gcs_uri} failed (exit {out.returncode}): {out.stderr.strip()}")
        return []
    names = []
    for line in out.stdout.splitlines():
        line = line.strip().rstrip("/")
        if line and line != gcs_uri.rstrip("/"):
            names.append(line.rsplit("/", 1)[-1])
    return names


def resolve_checkpoint_uri(cli_value: str | None, params: dict | None, section: str) -> str | None:
    """Resolve the effective checkpoint URI: CLI flag > env override > params.yaml; ``''`` -> off."""
    params_value = ((params or {}).get(section) or {}).get("checkpoint_uri")
    for candidate in (cli_value, os.environ.get(CHECKPOINT_URI_ENV), params_value):
        if candidate and str(candidate).strip():
            return str(candidate).strip()
    return None


def latest_checkpoint(names: list[str]) -> str | None:
    """Pick the ``checkpoint-<N>`` entry with the largest ``N`` (pure, no I/O)."""
    best_name: str | None = None
    best_n = -1
    for name in names:
        prefix, _, suffix = name.partition("-")
        if prefix == "checkpoint" and suffix.isdigit():
            n = int(suffix)
            if n > best_n:
                best_n, best_name = n, name
    return best_name


def _run_cp(src: str, dst: str, what: str) -> bool:
    """Run ``gcloud storage cp -r src dst`` best-effort; log + return False on failure."""
    try:
        out = subprocess.run(
            ["gcloud", "storage", "cp", "-r", src, dst],
            capture_output=True, text=True, timeout=_GCLOUD_TIMEOUT_S, check=False,
        )
    except Exception as exc:  # noqa: BLE001 - gcloud missing/slow/hung -> best-effort, never crash training
        logger.warning(f"gcs {what} failed: {type(exc).__name__}: {exc}")
        return False
    if out.returncode != 0:
        logger.warning(f"gcs {what} failed (exit {out.returncode}): {out.stderr.strip()}")
        return False
    return True
