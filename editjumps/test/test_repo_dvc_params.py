"""DVC and params.yaml must agree: every knob is tracked, wired, bounded and derivable."""

from pathlib import Path

import yaml


def test_head_parameterisation_knobs_are_tracked_by_dvc() -> None:
    """An untracked knob cannot trigger a rerun, so flipping a head would reuse a stale model."""
    import yaml

    from editjumps.core.edit_flows.stages import EditFlowConfig

    dvc = yaml.safe_load(Path("dvc.yaml").read_text())
    tracked = {p.split(".", 1)[1] for p in dvc["stages"]["train_edit_flows"]["params"]
               if p.startswith("edit_flows.")}
    for knob in ("rate_head", "q_head"):
        assert knob in tracked, f"edit_flows.{knob} is not a dvc param of train_edit_flows"
        assert knob in EditFlowConfig.__dataclass_fields__


def test_evotune_params_are_all_tracked_by_dvc() -> None:
    """Every evotune knob in params.yaml is a tracked param of a stage that reads it."""
    params = set(yaml.safe_load(Path("params.yaml").read_text())["evotune"])
    stages = yaml.safe_load(Path("dvc.yaml").read_text())["stages"]
    tracked = {
        key.split(".", 1)[1]
        for name, stage in stages.items() if name.startswith("evotune")
        for key in stage.get("params", []) if key.startswith("evotune.")
    }
    assert not params - tracked, f"params.yaml evotune keys no dvc stage tracks: {sorted(params - tracked)}"
    assert not tracked - params, f"dvc tracks evotune keys that do not exist: {sorted(tracked - params)}"

    # The §4.2 partition params must be tracked by BOTH the tuning stage and the baselines.
    shared = {"evotune.n_templates", "evotune.holdout_size", "evotune.seed"}
    for name in ("evotune_esm", "evotune_baseline", "evotune_baseline_forced"):
        assert shared <= set(stages[name]["params"]), f"{name} must track the partition params"


def test_evodiff_params_are_all_tracked_by_dvc() -> None:
    """Every evodiff knob in params.yaml is a tracked param of the stage that reads it."""
    params = set(yaml.safe_load(Path("params.yaml").read_text())["evodiff"])
    stage = yaml.safe_load(Path("dvc.yaml").read_text())["stages"]["evodiff_msa_baseline"]
    tracked = {key.split(".", 1)[1] for key in stage["params"] if key.startswith("evodiff.")}
    assert not params - tracked, f"params.yaml evodiff keys the stage does not track: {sorted(params - tracked)}"
    assert not tracked - params, f"the stage tracks evodiff keys that do not exist: {sorted(tracked - params)}"

    # The §4.2 partition and matched count are deliberately NOT duplicated here: reading evotune.*
    # keeps this baseline, the evotuned ones and the editor on one holdout at one n.
    assert not {key for key in params if key in {"n_templates", "holdout_size", "seed", "mutations"}}
    shared = {"evotune.n_templates", "evotune.holdout_size", "evotune.seed", "evotune.mutations"}
    assert shared <= set(stage["params"]), "the baseline must track the shared §4.2 partition"


def test_no_training_stage_leaves_its_step_budget_unbounded() -> None:
    """Training params must name an explicit `max_steps`, never -1 with an epoch count. `evotune.epochs."""
    import yaml as pyyaml

    params = pyyaml.safe_load((Path(__file__).parents[2] / "params.yaml").read_text())
    for section, config in params.items():
        if not isinstance(config, dict) or "max_steps" not in config:
            continue
        steps, epochs = config["max_steps"], config.get("epochs")
        # One pass with no step cap is a defined budget -- `pretrain` runs epochs: 1.0 over the corpus.
        if steps == -1 and epochs is not None and epochs > 1:
            raise AssertionError(
                f"{section}: max_steps -1 with epochs={epochs} multiplies a corpus size nobody "
                f"measured. Set max_steps explicitly, or drop to a single pass."
            )
    # The specific one this test exists for.
    assert params["evotune"]["max_steps"] > 0, "evotune.max_steps must be explicit"


