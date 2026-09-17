"""The `editjumps evaluate` front door: one config in, mutually comparable methods out."""

import json
from pathlib import Path

import pytest

from editjumps.test.fakes import _fake_shim


def test_the_front_door_hands_every_method_the_same_comparability_settings() -> None:
    """One config in, four mutually comparable methods out - checked flag by flag."""
    from dataclasses import replace

    from editjumps.core.run_config import RunConfig
    from editjumps.pipeline.evaluate.standard import DEFAULT_STEPS, build_plan

    config = RunConfig.from_params()
    plan = build_plan(config, Path("metrics/standard"))
    assert [step.name for step in plan.steps] == list(DEFAULT_STEPS)
    assert plan.steps[0].name == "editor", "the editor must run first: the baselines match its budget"

    shared = ("--n-templates", "--n-variants", "--holdout-size", "--ceiling-n", "--seed", "--k")
    for flag in shared:
        seen = set()
        for step in plan.steps:
            argv = list(step.command)
            assert flag in argv, f"{step.name} does not receive {flag}"
            seen.add(argv[argv.index(flag) + 1])
        assert len(seen) == 1, f"{flag} differs across methods: {seen} - the numbers are not comparable"

    # §4.2's matched mutation count comes from the editor's measured mean edit distance, so it cannot
    # drift from what the editor actually did. The editor step is the one that must NOT carry it.
    editor, baselines = plan.steps[0], plan.steps[1:]
    assert "--mutations-from" not in editor.command
    for step in baselines:
        argv = list(step.command)
        assert "--mutations-from" in argv
        assert argv[argv.index("--mutations-from") + 1] == str(editor.metrics_path)
        assert step.needs_editor_metrics

    # The fingerprint is recorded with the plan, so a report always carries the settings it was made
    # under instead of relying on someone having kept the command line.
    assert plan.as_dict()["comparability_fingerprint"] == config.comparability.fingerprint()

    # Turning the matching off is allowed, and then the budget is an explicit shared number.
    unmatched = config.replace(
        comparability=replace(config.comparability, match_budget_to_editor=False, mutations=7)
    )
    for step in build_plan(unmatched, Path("metrics/standard")).steps[1:]:
        argv = list(step.command)
        assert "--mutations-from" not in argv
        assert argv[argv.index("--mutations") + 1] == "7"

    with pytest.raises(ValueError, match="unknown step"):
        build_plan(config, Path("metrics/standard"), ("editor", "typo"))


def test_the_front_door_runs_a_method_end_to_end_with_the_configured_ceiling(tmp_path: Path) -> None:
    """Config -> plan -> a real stage main -> a report carrying the configured sample sizes."""
    from dataclasses import replace

    from editjumps.core.run_config import ArmConfig, ComparabilityConfig, RunConfig, TargetConfig
    from editjumps.pipeline.evaluate.standard import build_plan, run_step, summarise

    shim, _log = _fake_shim(tmp_path / "shim")
    letters = "ACDEFGHIKLMNPQRSTVY"
    members = [letters[i // 19] + letters[i % 19] + letters * 4 for i in range(40)]
    fasta = tmp_path / "families" / "fam.fasta"
    fasta.parent.mkdir(parents=True)
    fasta.write_text("".join(f">m{i}\n{seq}\n" for i, seq in enumerate(members)))

    config = RunConfig(
        target=TargetConfig(seed_family="fam.fasta", families_dir=fasta.parent),
        comparability=ComparabilityConfig(
            n_templates=4, n_variants=3, holdout_size=12, ceiling_n=5, mutations=2,
            match_budget_to_editor=False,
        ),
        arm=ArmConfig(evodiff_msa_size=16, evodiff_n_sequences=8),
    )
    plan = build_plan(config, tmp_path / "out", ("evodiff",))
    (step,) = plan.steps
    # The shim is a test seam on the stage, not a config field, so it is injected here rather than
    # given a home in arm.yaml where a real run could pick it up by accident.
    run_step(replace(step, kwargs={**step.kwargs, "shim": shim, "workdir": tmp_path / "msa"}))

    report = json.loads(step.metrics_path.read_text())
    assert report["n_generated"] == 12 == config.comparability.n_generated
    assert report["n_natural_reference"] == config.comparability.holdout_size
    assert report["mutation_budget"]["target"] == config.comparability.mutations
    # The pool is 40 - 4 templates - 12 holdout = 24 members, so 5 is a value only the config can
    # have produced: neither the old hardcoded 300 nor the pool size.
    assert report["defined_by_the_paper"]["ceiling_n"] == 5.0, "comparability.ceiling_n is decorative"

    # And the summary puts the sizes next to the metrics, which is the comparison a reader must make.
    (row,) = summarise(plan)
    assert row["method"] == "evodiff"
    assert row["n_generated"] == 12 and row["n_reference"] == 12
    assert row["ceiling_n"] == 5.0 and "L=" in str(row["alignment"])


