"""`editjumps/repo/audit_stage_coverage.py`: what the DVC column of the coverage table may claim."""

from editjumps.repo.audit_stage_coverage import ROOT, cli_commands, dvc_modules, stages


def test_dvc_coverage_reads_commands_not_raw_yaml_text() -> None:
    """A module mentioned anywhere in dvc.yaml is not thereby a stage: only the commands count."""
    modules = dvc_modules()
    raw = (ROOT / "dvc.yaml").read_text()

    # generation_eval.py is named three times in dvc.yaml, every one of them a `deps:` entry. The
    # old substring test read that as coverage; nothing actually runs it as a stage.
    assert "editjumps/pipeline/evaluate/generation_eval.py" in raw
    assert "editjumps.pipeline.evaluate.generation_eval" not in modules


def test_a_stage_name_does_not_vouch_for_a_module_it_merely_contains() -> None:
    """`evotune_baseline_forced` is a stage name, not a module; it must not supply the coverage."""
    modules = dvc_modules()
    assert not any(m.endswith("evotune_baseline_forced") for m in modules)
    # The real module is present because a command invokes it, not because a name contains it.
    assert "editjumps.pipeline.evaluate.evotune_baseline" in modules


def test_stages_reached_through_the_console_script_still_resolve() -> None:
    """`uv run editjumps <command>` names a CLI command, which resolves back through main.py."""
    assert "build-deterministic-pairs" in cli_commands()
    assert "editjumps.pipeline.preprocess.pretrain.deterministic_pairs" in dvc_modules()


def test_every_resolved_module_is_a_file_that_exists() -> None:
    """The dotted paths are matched against real stage modules, so they have to name real files."""
    known = {p.with_suffix("").as_posix().replace("/", ".") for p in stages()}
    for module in dvc_modules():
        assert module in known, f"{module} is invoked by dvc.yaml but is not a stage module"
