"""One choice per DAG stage: every name validates, and every pure stage dispatches."""

from pathlib import Path

import pytest
import yaml


def test_edit_stages_config_and_pure_stages() -> None:
    """EditFlowConfig validates stage names; the pure ② path/schedule stages dispatch correctly."""
    import random

    from editjumps.core.edit_flows.path import strip_epsilon
    from editjumps.core.edit_flows.stages import PATH_BUILDERS, SCHEDULES, EditFlowConfig

    # defaults, params.yaml block, CLI-style override, and fail-fast on a bad name
    assert EditFlowConfig().path == "needleman_wunsch"
    assert EditFlowConfig.from_params({"edit_flows": {"path": "random_noise"}}).path == "random_noise"
    assert EditFlowConfig.from_params({}, sampler="euler").sampler == "euler"
    cfg = EditFlowConfig.from_params({"edit_flows": {"sampler": "gillespie"}})
    assert cfg.sampler == "gillespie"  # ⑤ inference stepper
    with pytest.raises(ValueError):
        EditFlowConfig(loss="bogus")

    assert SCHEDULES["linear"](0.3) == (0.3, 1.0)  # ② schedule

    rng = random.Random(0)
    nw0, nw1 = PATH_BUILDERS["needleman_wunsch"]([1, 2, 3], [1, 2, 4], 30, rng)
    assert len(nw0) == len(nw1)  # ② aligned to equal length
    rn0, rn1 = PATH_BUILDERS["random_noise"]([9], [1, 2, 3], 30, rng)
    assert strip_epsilon(rn1) == [1, 2, 3]  # z1 recovers the data (ids1)
    assert rn0[0] == 1 and rn1[0] == 1  # BOS held fixed at the front


def test_edit_flow_config_reads_every_field_from_params_and_logs_every_field() -> None:
    """A hand-listed key set silently drops new knobs; both sides are now derived from the fields.."""
    import yaml

    from editjumps.core.edit_flows.stages import EditFlowConfig

    fields = set(EditFlowConfig.__dataclass_fields__)
    block = yaml.safe_load(Path("params.yaml").read_text())["edit_flows"]
    # The block also holds training hyperparameters reaching the CLI via ${} interpolation, so not
    # every key is a field. What matters: every field params.yaml sets is read, not defaulted.
    from_params = EditFlowConfig.from_params({"edit_flows": block})
    for field in fields & set(block):
        assert getattr(from_params, field) == block[field], (
            f"edit_flows.{field} is set in params.yaml but from_params did not read it")

    # A non-default value round-trips rather than being replaced by the default.
    tweaked = {"edit_flows": {**block, "rate_head": "mlp", "q_head": "esm_lm_head"}}
    cfg = EditFlowConfig.from_params(tweaked)
    assert (cfg.rate_head, cfg.q_head) == ("mlp", "esm_lm_head")

    # And every field is logged, so two runs with different heads are distinguishable in MLflow.
    assert set(cfg.as_dict()) == fields

def test_gillespie_is_a_selectable_sampler_stage() -> None:
    """`edit_flows.sampler: gillespie` validates and params.yaml still names a registered sampler."""
    from editjumps.core.edit_flows.stages import SAMPLERS, EditFlowConfig

    assert "gillespie" in SAMPLERS and "euler" in SAMPLERS
    assert EditFlowConfig(sampler="gillespie").sampler == "gillespie"
    with pytest.raises(ValueError):
        EditFlowConfig(sampler="tau_leaping")
    params = yaml.safe_load(Path("params.yaml").read_text())
    assert params["edit_flows"]["sampler"] in SAMPLERS

def test_blosum62_path_is_selectable_and_needs_the_vocab_but_is_not_the_default() -> None:
    """The faithful scoring must be reachable from params.yaml — and must not become the default."""
    import random

    import yaml

    from editjumps.core.edit_flows.stages import PATH_BUILDERS, EditFlowConfig

    assert "needleman_wunsch_blosum62" in PATH_BUILDERS
    assert EditFlowConfig().path == "needleman_wunsch"
    assert yaml.safe_load(Path("params.yaml").read_text())["edit_flows"]["path"] == "needleman_wunsch"

    config = EditFlowConfig(path="needleman_wunsch_blosum62")
    with pytest.raises(ValueError, match="tokenizer vocab"):
        config.path_builder()([1], [1], 33, random.Random(0))
    with pytest.raises(ValueError, match="tokenizer vocab"):
        config.path_builder(tokens={"W": 22}, vocab_size=0)([1], [1], 33, random.Random(0))

    build = config.path_builder(tokens={"W": 22, "F": 18, "P": 14}, vocab_size=33)
    z_0, z_1 = build([0, 22, 22, 2], [0, 22, 18, 22, 2], 33, random.Random(0))
    assert len(z_0) == len(z_1) == 5
    # and the selected path is symmetric, unlike the default one
    reverse_0, reverse_1 = build([0, 22, 18, 22, 2], [0, 22, 22, 2], 33, random.Random(0))
    assert (z_0, z_1) == (reverse_1, reverse_0)
