"""The run config: it round-trips through YAML and reproduces the equivalent DVC flags."""

from pathlib import Path

import yaml


def _dvc_stage_flags(stage: str) -> dict[str, str]:
    """Extract the params-driven flags of one ``dvc.yaml`` stage, resolved against ``params.yaml``."""
    import re

    root = Path(__file__).parents[2]
    params = yaml.safe_load((root / "params.yaml").read_text())
    command = yaml.safe_load((root / "dvc.yaml").read_text())["stages"][stage]["cmd"]

    def resolve(text: str) -> str:
        """Substitute every ``${a.b}`` in ``text`` with its ``params.yaml`` value."""
        def one(match: "re.Match[str]") -> str:
            node = params
            for key in match.group(1).split("."):
                node = node[key]
            return str(node)

        return re.sub(r"\$\{([^}]+)\}", one, text)

    tokens = command.split()
    flags: dict[str, str] = {}
    for index, token in enumerate(tokens):
        if not token.startswith("--"):
            continue
        value = tokens[index + 1] if index + 1 < len(tokens) else ""
        if value.startswith("--") or "${" not in value:
            continue                                    # boolean switch, or a literal: not a setting
        flags[token] = resolve(value).strip('"')
    return flags


def test_comparability_group_covers_every_cross_run_axis() -> None:
    """The split earns its third file only if that file holds EVERY axis a comparison depends on."""
    from dataclasses import fields, replace

    from editjumps.core.run_config import (
        COMPARABILITY_AXES,
        ArmConfig,
        ComparabilityConfig,
        TargetConfig,
    )

    declared = {f.name for f in fields(ComparabilityConfig)}
    covered = {name for names in COMPARABILITY_AXES.values() for name in names}

    # 1 + 3: the concept -> field map is total in both directions.
    assert covered <= declared, f"axes name fields that do not exist: {sorted(covered - declared)}"
    assert declared <= covered, (
        f"comparability fields with no declared axis: {sorted(declared - covered)}. Either say which "
        "cross-run comparison the field breaks, or move it to target.yaml / arm.yaml"
    )

    # 2: the three named in the brief, by concept and not merely by field name.
    axis_text = " | ".join(COMPARABILITY_AXES)
    assert "holdout_size" in covered and "reference size" in axis_text
    assert "alignment_template" in covered and "alignment template" in axis_text
    assert {"n_templates", "n_variants"} <= covered and "MMD sample size" in axis_text
    assert COMPARABILITY_AXES["MMD sample size (generated side)"] == ("n_templates", "n_variants"), (
        "the generated-set size IS n_templates * n_variants; if that stops being true the MMD "
        "sample-size axis is no longer pinned by these two fields"
    )
    assert ComparabilityConfig().n_generated == 20 * 20

    # 4: the fingerprint is only worth recording if it responds to every field.
    base = ComparabilityConfig()
    bumped = {
        "n_templates": 21, "n_variants": 19, "holdout_size": 201, "ceiling_n": 301,
        "alignment_template": "first_sampled", "split_seed": 1, "spectrum_k": 4, "mutations": 5,
        "match_budget_to_editor": False, "budget_mode": "masked", "max_rounds": 9,
        "pll_model": "facebook/esm2_t12_35M_UR50D", "pll_positions": 25,
    }
    assert set(bumped) == declared, "a new comparability field needs a case here"
    for name, value in bumped.items():
        assert replace(base, **{name: value}).fingerprint() != base.fingerprint(), (
            f"{name} does not move the comparability fingerprint, so two runs differing only in it "
            "would advertise themselves as comparable"
        )

    # 5: no field lives in two groups.
    for other in (TargetConfig, ArmConfig):
        overlap = declared & {f.name for f in fields(other)}
        assert not overlap, f"{other.__name__} duplicates comparability field(s) {sorted(overlap)}"


def test_run_config_round_trips_and_matches_the_dvc_flags() -> None:
    """A config written from `params.yaml` survives YAML and reproduces the pipeline's own flags."""
    import tempfile

    from editjumps.core.run_config import RunConfig

    config = RunConfig.from_params()

    with tempfile.TemporaryDirectory() as directory:
        config.write(Path(directory))
        assert RunConfig.load(Path(directory)) == config, (
            "a RunConfig does not survive a write/load round trip, so `editjumps evaluate --init` "
            "can emit a file that means something other than what it was built from"
        )

    # Calling load() with no arguments defaults directly to params.yaml
    assert RunConfig.load() == config

    # And the flags themselves, resolved from the stage's own `cmd` string, so a renamed flag or a re-pointed
    # knob is caught.
    shared = {"--path": config.arm.path, "--schedule": config.arm.schedule,
              "--loss": config.arm.loss}
    flags = _dvc_stage_flags("train_edit_flows")
    for flag, value in shared.items():
        assert flag in flags, f"{flag} is no longer a params-driven flag of train_edit_flows"
        assert flags[flag] == str(value), (
            f"{flag}: dvc.yaml resolves to {flags[flag]!r} but params.yaml says {value!r}"
        )
