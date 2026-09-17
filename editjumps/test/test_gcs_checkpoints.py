"""Periodic checkpoints to GCS: safe URIs, picking the latest, and the default-off contract."""

import pytest


def test_latest_checkpoint_picks_max_and_ignores_non_matching() -> None:
    """Latest_checkpoint returns checkpoint-<max N>, ignores junk, and is None on empty (pure)."""
    from editjumps.core.gcs_checkpoints import latest_checkpoint

    assert latest_checkpoint(["checkpoint-3", "checkpoint-100", "checkpoint-20"]) == "checkpoint-100"
    # non-matching entries (other files, malformed suffixes) are ignored
    assert latest_checkpoint(["config.json", "checkpoint-5", "checkpoint-", "checkpoint-x"]) == "checkpoint-5"
    assert latest_checkpoint([]) is None
    assert latest_checkpoint(["notacheckpoint", "runs", "checkpoint-abc"]) is None


def test_resolve_checkpoint_uri_is_off_by_default(monkeypatch: "pytest.MonkeyPatch") -> None:
    """Default-off contract: no flag, no env, empty params -> None (guarantees zero GCS calls)."""
    from editjumps.core.gcs_checkpoints import CHECKPOINT_URI_ENV, resolve_checkpoint_uri

    monkeypatch.delenv(CHECKPOINT_URI_ENV, raising=False)
    assert resolve_checkpoint_uri(None, None, "pretrain") is None
    assert resolve_checkpoint_uri("", {"pretrain": {"checkpoint_uri": ""}}, "pretrain") is None
    assert resolve_checkpoint_uri("  ", {}, "edit_flows") is None  # whitespace-only counts as off

    # precedence: explicit flag > env override > params.yaml
    assert resolve_checkpoint_uri("gs://flag", {"pretrain": {"checkpoint_uri": "gs://p"}}, "pretrain") == "gs://flag"
    assert resolve_checkpoint_uri(None, {"edit_flows": {"checkpoint_uri": "gs://p"}}, "edit_flows") == "gs://p"
    monkeypatch.setenv(CHECKPOINT_URI_ENV, "gs://env")
    assert resolve_checkpoint_uri(None, {"pretrain": {"checkpoint_uri": "gs://p"}}, "pretrain") == "gs://env"
