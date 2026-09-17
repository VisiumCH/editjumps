"""The ESM pretraining stage's defaults: step-based checkpointing and an unclobberable metrics path."""

import inspect

import pytest
import yaml


def test_pretrain_esm_default_checkpointing_is_step_based() -> None:
    """Checkpointing must be step-based (not epoch-only) with resume on by default."""
    # pretrain_esm imports the optional `train` group, absent from the lean CI env.
    pretrain_esm = pytest.importorskip("editjumps.pipeline.train.pretrain_esm")
    sig = inspect.signature(pretrain_esm.pretrain)
    assert sig.parameters["save_steps"].default > 0
    assert sig.parameters["save_total_limit"].default > 0
    assert sig.parameters["resume"].default is True

    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    assert params["pretrain"]["save_steps"] > 0
    assert params["pretrain"]["save_total_limit"] > 0


def test_pretrain_esm_metrics_path_defaults_to_none() -> None:
    """Metrics_path must default to None so an ad-hoc run can't clobber the tracked metrics file."""
    pretrain_esm = pytest.importorskip("editjumps.pipeline.train.pretrain_esm")
    assert inspect.signature(pretrain_esm.pretrain).parameters["metrics_path"].default is None
    assert inspect.signature(pretrain_esm.main).parameters["metrics_path"].default is None
