"""Shared run plumbing: the design-space axes, MLflow tracking, its token refresh and metrics."""

import json
import re
from pathlib import Path

import pytest


def test_design_space_tags_stringifies_and_covers_readme_axes() -> None:
    """The design-space tag helper stringifies values and its standard keys are the README's axes."""
    from editjumps.core.utils import DESIGN_AXES, design_space_tags

    assert set(DESIGN_AXES) == {
        "pretrain_data", "split_method", "chain_handling", "target_property",
        "pair_source",  # always `homolog` here; see the axis list for why the value is recorded
        "backbone", "weight_init", "objective",
    }
    tags = design_space_tags(objective="mlm", backbone="esm2_t12_35M_UR50D", weight_init="pretrained")
    assert tags == {"objective": "mlm", "backbone": "esm2_t12_35M_UR50D", "weight_init": "pretrained"}
    assert all(isinstance(v, str) for v in design_space_tags(objective=1, backbone=None).values())


def test_get_mlflow_tracking_uri_precedence(monkeypatch: "pytest.MonkeyPatch", tmp_path: Path) -> None:
    """Tracking URI precedence: env override > params.yaml mlflow.tracking_uri > local sqlite."""
    from editjumps.core import utils

    monkeypatch.setenv("MLFLOW_TRACKING_URI", "sqlite:///custom.db")
    assert utils.get_mlflow_tracking_uri() == "sqlite:///custom.db"  # env wins

    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.chdir(tmp_path)
    Path("params.yaml").write_text("mlflow:\n  tracking_uri: https://server.example\n")
    assert utils.get_mlflow_tracking_uri() == "https://server.example"  # params default

    Path("params.yaml").write_text("other: 1\n")  # no mlflow block
    assert utils.get_mlflow_tracking_uri() == utils.DEFAULT_MLFLOW_TRACKING_URI  # sqlite fallback


def test_mlflow_rereads_token_from_env_on_every_request() -> None:
    """MLflow must read MLFLOW_TRACKING_TOKEN per request, which is what token refresh relies on."""
    import os
    from functools import partial

    from mlflow.store.tracking.rest_store import RestStore
    from mlflow.utils.credentials import get_default_host_creds

    before = os.environ.get("MLFLOW_TRACKING_TOKEN")
    try:
        store = RestStore(partial(get_default_host_creds, "https://server.example"))
        os.environ["MLFLOW_TRACKING_TOKEN"] = "first-token"
        assert store.get_host_creds().token == "first-token"
        os.environ["MLFLOW_TRACKING_TOKEN"] = "second-token"  # the refresher's move
        assert store.get_host_creds().token == "second-token", (
            "mlflow cached the token; editjumps.core.utils._start_token_refresher no longer works"
        )
    finally:
        if before is None:
            os.environ.pop("MLFLOW_TRACKING_TOKEN", None)
        else:
            os.environ["MLFLOW_TRACKING_TOKEN"] = before


def test_token_refresh_replaces_env_token_and_survives_a_dry_mint(
    monkeypatch: "pytest.MonkeyPatch",
) -> None:
    """One refresh swaps the env token; a failed mint keeps the old one rather than clearing it."""
    import os

    from editjumps.core import utils

    monkeypatch.setenv("MLFLOW_TRACKING_TOKEN", "stale")
    monkeypatch.setattr(utils, "refresh_identity_token", lambda audience: "fresh")
    assert utils.refresh_token_once("https://server.example") is True
    assert os.environ["MLFLOW_TRACKING_TOKEN"] == "fresh"

    # A mint that comes back empty must not wipe a token that at least might still be valid.
    monkeypatch.setattr(utils, "refresh_identity_token", lambda audience: None)
    assert utils.refresh_token_once("https://server.example") is False
    assert os.environ["MLFLOW_TRACKING_TOKEN"] == "fresh"


def test_token_refresher_starts_one_daemon_thread(monkeypatch: "pytest.MonkeyPatch") -> None:
    """The refresher is a single daemon thread and starting it twice does not stack a second."""
    import threading

    from editjumps.core import utils

    monkeypatch.setattr(utils, "refresh_identity_token", lambda audience: "fresh")
    utils._start_token_refresher("https://server.example")
    utils._start_token_refresher("https://server.example")  # idempotent
    threads = [t for t in threading.enumerate() if t.name == utils._REFRESHER_NAME]
    assert len(threads) == 1
    assert threads[0].daemon  # must not hold the process open at exit


def test_tracking_failures_do_not_abort_the_run() -> None:
    """A raising mlflow.log_metric must not propagate: losing metrics beats losing hours of GPU."""
    import mlflow

    from editjumps.core.utils import make_tracking_nonfatal

    original = mlflow.log_metric
    try:
        def explode(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("API request ... failed with error code 401 != 200")

        mlflow.log_metric = explode  # ty: ignore[invalid-assignment]
        make_tracking_nonfatal()
        assert mlflow.log_metric("loss", 0.5) is None  # swallowed, run continues
        guarded = mlflow.log_metric
        make_tracking_nonfatal()  # idempotent - double-wrapping would double-count failures
        assert mlflow.log_metric is guarded
    finally:
        mlflow.log_metric = original


def test_run_exit_is_guarded_where_mlflow_actually_looks_it_up() -> None:
    """`with start_mlflow_run(...)` must not raise on exit: ActiveRun.__exit__ calls fluent.end_run."""
    from mlflow.tracking import fluent

    from editjumps.core.utils import make_tracking_nonfatal

    original = fluent.end_run
    try:
        def explode(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("401 != 200")

        fluent.end_run = explode  # ty: ignore[invalid-assignment]
        make_tracking_nonfatal()
        assert fluent.end_run("FINISHED") is None
    finally:
        fluent.end_run = original


def test_metadata_identity_token_requests_the_service_url_as_audience(
    monkeypatch: "pytest.MonkeyPatch",
) -> None:
    """The audience must be the Cloud Run URL - a token minted for anything else authenticates as nobody."""
    import urllib.request

    from editjumps.core import utils

    seen: dict[str, object] = {}

    class FakeResponse:
        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *_exc: object) -> None:
            return None

        def read(self) -> bytes:
            return b"  tok  "

    def fake_urlopen(request: object, timeout: float = 0.0) -> FakeResponse:
        seen["url"] = request.full_url  # ty: ignore[unresolved-attribute]
        seen["header"] = request.get_header("Metadata-flavor")  # ty: ignore[unresolved-attribute]
        return FakeResponse()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    assert utils._metadata_identity_token("https://server.example") == "tok"
    assert seen["url"] == f"{utils.METADATA_IDENTITY_URL}?audience=https://server.example"
    assert seen["header"] == "Google"  # metadata server rejects the request without it


def test_flatten_metrics_logs_every_number_including_n_and_the_ci() -> None:
    """MLflow must receive the qualifying numbers, not just the flattering ones."""
    from editjumps.core.utils import flatten_metrics

    flat = flatten_metrics(
        {
            "comparison": {
                "a": {"score": 0.5, "n": 100, "gain_ci95_lo": -0.05, "gain_ci95_hi": 0.15},
                "b": {"score": 0.4, "n": 100},
            },
            "by_group": {"group/holdout": {"a": 0.25, "n": 40, "scored": 1.0}},
            "verdict": {"tied": True},
        }
    )
    assert flat["comparison/a/gain_ci95_lo"] == pytest.approx(-0.05)
    assert flat["comparison/a/gain_ci95_hi"] == pytest.approx(0.15)
    assert flat["comparison/a/n"] == pytest.approx(100.0)
    # Group names carry a slash, which would otherwise fake a nesting level in the metric path.
    assert flat["by_group/group_holdout/n"] == pytest.approx(40.0)
    assert flat["verdict/tied"] == pytest.approx(1.0), "bools must log as 0/1"

    # NaN is skipped (MLflow stores it unhelpfully) but strings must not crash the walk.
    assert flatten_metrics({"a": float("nan"), "b": "text", "c": 1}) == {"c": 1.0}


def test_flatten_metrics_summarises_list_values_instead_of_dropping_them() -> None:
    """A list leaf must neither crash the walk nor vanish from the run."""
    from editjumps.core.utils import flatten_metrics

    report = {
        "defined_by_the_paper": {
            "levenshtein_to_template": 4.0,
            "levenshtein_to_template_per_template": [3.0, 5.0],
            "levenshtein_to_template_per_sequence": [[2.0, 4.0], [4.0, 6.0]],
        },
        "caveats": ["one", "two", "three"],
    }
    assert flatten_metrics(report) == {"defined_by_the_paper/levenshtein_to_template": 4.0}, (
        "the default must stay list-free, or every existing caller's metric set changes"
    )

    flat = flatten_metrics(report, summarise_lists=True)
    assert flat["defined_by_the_paper/levenshtein_to_template_per_template/n"] == 2.0
    assert flat["defined_by_the_paper/levenshtein_to_template_per_template/mean"] == pytest.approx(4.0)
    # A list of lists: `n` counts the groups, `mean` pools across them.
    assert flat["defined_by_the_paper/levenshtein_to_template_per_sequence/n"] == 2.0
    assert flat["defined_by_the_paper/levenshtein_to_template_per_sequence/mean"] == pytest.approx(4.0)
    # A list holding no numbers still records its length; inventing a mean of nothing would not.
    assert flat["caveats/n"] == 3.0
    assert "caveats/mean" not in flat


def test_every_mlflow_run_records_where_its_artefacts_land(monkeypatch: pytest.MonkeyPatch) -> None:
    """A run must carry its GCS destinations, because MLflow does not hold the weights themselves."""
    from editjumps.core.utils import GCS_DESTINATION_ENV, gcs_destination_tags

    for env in GCS_DESTINATION_ENV:
        monkeypatch.delenv(env, raising=False)
    assert gcs_destination_tags() == {}, "a local run uploads nothing and should be tagged with nothing"

    monkeypatch.setenv("MODEL_GCS_URI", "gs://bucket/models/edit_flows/demo")
    monkeypatch.setenv("CHECKPOINT_GCS_URI", "gs://bucket/checkpoints/experiments/demo")
    tags = gcs_destination_tags()
    assert tags["gcs_model_uri"] == "gs://bucket/models/edit_flows/demo"
    assert tags["gcs_checkpoint_uri"] == "gs://bucket/checkpoints/experiments/demo"

    # A stage that knows a URI directly can pass it, and None never becomes a tag.
    assert "gcs_extra_uri" not in gcs_destination_tags(gcs_extra_uri=None)
    assert gcs_destination_tags(gcs_extra_uri="gs://b/extra/x")["gcs_extra_uri"] == "gs://b/extra/x"

    # The job must export every name the helper reads, or the tag is silently always absent.
    job = (Path(__file__).parents[2] / "deploy" / "gcp" / "train.sky.yaml").read_text()
    for env in ("MODEL_GCS_URI", "CHECKPOINT_GCS_URI", "METRICS_GCS_URI"):
        assert f"export {env}=" in job, f"{env} is read by gcs_destination_tags and never exported"


def test_write_metrics_records_the_commit_without_touching_the_report(tmp_path: Path) -> None:
    """A stage's metrics land with a `provenance` block, and the caller's dict is left alone."""
    from editjumps.core.utils import flatten_metrics, write_metrics

    report = {"levenshtein_to_template": 4.58, "nested": {"n": 400}}
    path = write_metrics(tmp_path / "sub" / "cell.json", report, checkpoint_uri="gs://bucket/x", step=20000)

    written = json.loads(path.read_text())
    assert "provenance" not in report, "write_metrics mutated the caller's report"
    assert flatten_metrics(report) == {"levenshtein_to_template": 4.58, "nested/n": 400.0}

    provenance = written["provenance"]
    assert re.fullmatch(r"[0-9a-f]{40}", str(provenance["git_commit"])), provenance
    assert isinstance(provenance["git_dirty"], bool)
    assert provenance["command"]
    # Stage-specific provenance rides along under whatever key the caller passes.
    assert provenance["checkpoint_uri"] == "gs://bucket/x"
    assert provenance["step"] == 20000
    # No wall clock and no hostname: several artefacts here claim bit-for-bit re-derivation, and a
    # timestamp would break every one of those comparisons for nothing the commit does not say.
    assert not {"timestamp", "created_at", "written_at", "hostname"} & set(provenance)


def test_code_provenance_omits_git_fields_rather_than_guessing(monkeypatch: "pytest.MonkeyPatch") -> None:
    """Outside a checkout -- an installed wheel, a GPU VM that never cloned -- the fields are absent."""
    import subprocess

    from editjumps.core import utils

    def refuse(*args: object, **kwargs: object) -> None:
        raise FileNotFoundError("git")

    monkeypatch.setattr(subprocess, "run", refuse)
    record = utils.code_provenance()
    assert set(record) == {"command"}, record


def _esm2_tokenizer() -> object:
    """Load the real ESM-2 tokenizer, or skip -- the two facts below need the real vocabulary."""
    pytest.importorskip("transformers", reason="`transformers` is in the `train` group")
    from transformers import AutoTokenizer  # ty: ignore[unresolved-import]

    try:
        return AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")
    except OSError as exc:  # no network and no cached copy
        pytest.skip(f"facebook/esm2_t12_35M_UR50D is neither cached nor reachable: {exc}")


def test_decode_one_drops_the_null_1_filler_but_keeps_the_vh_vl_separator() -> None:
    """ESM-2's `<null_1>` is samplable but is not a registered special token. `skip_special_tokens=True`."""
    from editjumps.core.utils import decode_one

    tokenizer = _esm2_tokenizer()
    vocab = tokenizer.get_vocab()  # ty: ignore[unresolved-attribute]
    assert vocab["<null_1>"] not in tokenizer.all_special_ids, (  # ty: ignore[unresolved-attribute]
        "premise of this test: <null_1> is not a registered special token"
    )

    with_filler = [vocab["<cls>"], vocab["Q"], vocab["V"], vocab["<null_1>"], vocab["Q"],
                   vocab["<eos>"]]
    assert decode_one(tokenizer, with_filler) == "QVQ"

    with_separator = [vocab["<cls>"], vocab["Q"], vocab["V"], vocab["."], vocab["Q"],
                      vocab["<eos>"]]
    assert decode_one(tokenizer, with_separator) == "QV.Q"


def test_decode_one_drops_every_non_standard_vocabulary_token() -> None:
    """The token head can sample any vocab id, including seven that are not standard residues. `-` is a."""
    from editjumps.core.sequences import AA, PAIR_SEP
    from editjumps.core.utils import decode_one

    tokenizer = _esm2_tokenizer()
    vocab = tokenizer.get_vocab()  # ty: ignore[unresolved-attribute]

    emittable = {t for t in vocab if not t.startswith("<")}
    non_standard = sorted(emittable - AA - {PAIR_SEP})
    assert non_standard == ["-", "B", "O", "U", "X", "Z"], (
        f"premise changed: ESM-2's non-standard emittable tokens are now {non_standard}"
    )

    ids = [vocab["<cls>"], vocab["Q"], *[vocab[t] for t in non_standard], vocab["V"],
           vocab["<eos>"]]
    assert decode_one(tokenizer, ids) == "QV"
