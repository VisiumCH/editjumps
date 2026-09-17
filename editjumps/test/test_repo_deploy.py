"""The launch surface: sky yamls, isolated environments and what the workdir sync carries."""

from pathlib import Path


def test_every_isolated_env_an_installer_creates_is_skyignored() -> None:
    """Any `.*_env` an install script builds must be excluded from the SkyPilot workdir sync. bucket on."""
    import re

    root = Path(__file__).parents[2]
    ignored = {
        line.strip().rstrip("/")
        for line in (root / ".skyignore").read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    }
    declared: set[str] = set()
    for script in (root / "editjumps").rglob("install_*.sh"):
        declared |= set(re.findall(r"\.[a-z0-9]+_env", script.read_text()))

    assert declared, "no installer declares an isolated env — has the naming convention changed?"
    missing = sorted(env for env in declared if env not in ignored)
    assert not missing, (
        f"install scripts create {missing} but .skyignore does not exclude them; each would be "
        "uploaded to the SkyPilot filemount bucket on every launch"
    )

    # Agent worktrees are the other way this bucket fills up: each is a full repo copy, and one
    # carrying a 910 MB .evodiff_env took the tree to 6.4 GB, hanging two launches for ~50 min.
    assert ".claude" in ignored, (
        ".skyignore must exclude .claude/ — agent worktrees live there and are full repo copies"
    )


def test_gpu_jobs_pin_the_environment_before_any_uv_run() -> None:
    """No sky yaml may let `uv run` re-resolve the environment before it trains."""
    deploy = Path(__file__).parents[2] / "deploy" / "gcp"
    for yaml_path in sorted(deploy.glob("*.sky.yaml")):
        text = yaml_path.read_text()
        if "dvc repro" not in text and "train-edit-flows" not in text:
            continue                                  # not a training job
        run_block = text[text.index("\nrun: |"):] if "\nrun: |" in text else ""
        assert run_block, f"{yaml_path.name}: no run block found"
        # What reverts torch is a `uv run` that re-resolves WITH the train group, which is where torch lives.
        def first_code_offset(needle: str) -> int | None:
            at = 0
            for text_line in run_block.splitlines(keepends=True):
                if not text_line.strip().startswith("#") and needle in text_line:
                    return at + text_line.index(needle)
                at += len(text_line)
            return None

        exported_at = first_code_offset("UV_NO_SYNC")
        exported_at = -1 if exported_at is None else exported_at
        repro_at = first_code_offset("dvc repro")
        if repro_at is not None:
            assert exported_at != -1 and exported_at < repro_at, (
                f"{yaml_path.name}: `dvc repro` runs each stage's own `uv run --group train` from "
                f"dvc.yaml, which this file cannot flag. Export UV_NO_SYNC=1 before it, or the "
                f"stages re-resolve torch to the cu130 wheel this VM's driver cannot run and "
                f"training falls to CPU. Sky job 58 lost 23 hours of A100 to exactly this."
            )
        offset = 0
        for line in run_block.splitlines(keepends=True):
            body = line.strip()
            # Comment lines are skipped on purpose: the block explaining why UV_NO_SYNC exists necessarily.
            if not body.startswith("#") and "uv run" in body and "--group train" in body:
                covered = "--no-sync" in body or (exported_at != -1 and exported_at < offset)
                assert covered, (
                    f"{yaml_path.name}: this `uv run --group train` re-resolves the environment "
                    f"and reverts setup's CUDA-matched torch to the cu130 wheel this VM's driver "
                    f"cannot run, so training falls to CPU. Add --no-sync, or export UV_NO_SYNC=1 "
                    f"earlier in the run block.\n    {body[:110]}"
                )
            offset += len(line)

        # And a run-environment CUDA check, because setup's cannot catch a later revert.
        assert "run-env CUDA" in run_block, (
            f"{yaml_path.name}: run block must re-check torch.cuda.is_available() with the "
            f"invocation the stages use; setup's check cannot see a revert that happens after it"
        )


def test_every_isolated_env_installs_a_driver_matched_torch() -> None:
    """An installer that puts torch in its own env must let uv match the wheel to the driver."""
    root = Path(__file__).parents[2]
    checked = 0
    for script in sorted((root / "editjumps").rglob("install_*.sh")):
        text = script.read_text()
        installs = [
            line for line in text.splitlines()
            if not line.strip().startswith("#") and "uv pip install" in line
        ]
        # The torch install is the one that matters; an installer without one is not this test's business.
        torch_lines = [line for line in installs if "torch" in line]
        if not torch_lines:
            continue
        checked += 1
        for line in torch_lines:
            assert "--torch-backend=auto" in line, (
                f"{script.name}: this installs torch into an isolated env without "
                f"--torch-backend=auto, so it takes the default wheel for the newest CUDA and the "
                f"GPU VMs' driver cannot initialise it -- torch.cuda.is_available() goes False and "
                f"the stage silently runs on CPU. Sky job 71 lost an hour of L4 to exactly this.\n"
                f"    {line.strip()[:110]}"
            )
    assert checked, "no installer installs torch — has the isolated-env pattern changed?"

    # And the job must assert the result before it starts paying for GPU time: the installer's own
    # smoke test uses --device cpu, so it passes either way.
    repro = (root / "deploy" / "gcp" / "repro.sky.yaml").read_text()
    assert "evodiff-env CUDA" in repro, (
        "repro.sky.yaml must fail fast in setup when .evodiff_env's torch cannot see the GPU; the "
        "run-env check covers .venv only, and .evodiff_env has its own torch"
    )


def test_no_package_module_imports_a_foreign_interpreter_runner() -> None:
    """The three foreign-interpreter runners must never be imported by package code. torch."""
    import ast

    runners = {"evodiff_msa_runner"}
    root = Path(__file__).parents[2]
    offenders = []
    for path in sorted((root / "editjumps").rglob("*.py")):
        if path.stem in runners:
            continue                       # a runner importing a sibling runner is not the concern
        for node in ast.walk(ast.parse(path.read_text())):
            # Narrowed to the two import nodes, which is also what gives `lineno` a type.
            if isinstance(node, ast.Import):
                named = {alias.name.rsplit(".", 1)[-1] for alias in node.names}
            elif isinstance(node, ast.ImportFrom):
                named = {alias.name for alias in node.names}
                if node.module:
                    named.add(node.module.rsplit(".", 1)[-1])
            else:
                continue
            hit = named & runners
            if hit:
                offenders.append(f"{path.relative_to(root)}:{node.lineno} imports {sorted(hit)}")
    assert not offenders, (
        "these modules import a runner that does not run under this interpreter -- call the shim "
        + "\n".join(f"    {line}" for line in offenders)
    )


def test_source_stages_resolve_to_no_pullable_inputs() -> None:
    """`fetch_base_checkpoint` and `download_oas` fetch from outside DVC, so an empty list is right.

    The resolver is correct to return nothing here; it is the caller that must not read "nothing"
    as "everything".
    """
    from editjumps.repo.stage_data_deps import data_deps

    assert data_deps(["fetch_base_checkpoint"]) == []
    assert data_deps(["download_oas"]) == []
    # A stage with cached inputs still resolves them, so the empty case above is not vacuous.
    assert data_deps(["split_corpus"])


def test_the_repro_job_never_hands_dvc_pull_an_empty_target_list() -> None:
    """`dvc pull` with no targets fetches the ENTIRE remote, which is never what a job wants.

    A blanket pull also has to check out every declared output, including those of stages that have
    never been recorded in dvc.lock, so it fails on work the job was not asking for.
    """
    from pathlib import Path

    yaml_text = (Path(__file__).parents[2] / "deploy" / "gcp" / "repro.sky.yaml").read_text()

    # Real invocations only: other lines mention "dvc pull" inside comments and error messages.
    pull_lines = [line.strip() for line in yaml_text.splitlines()
                  if "uv run dvc pull" in line and not line.strip().startswith("#")]
    assert pull_lines, "repro.sky.yaml no longer pulls anything"
    for line in pull_lines:
        assert "$DEPS" in line, f"pull without the resolved target list: {line}"
    # And that pull is reached only when the list is non-empty.
    assert 'if [ -n "$DEPS" ]; then' in yaml_text, (
        "the pull is not guarded on a non-empty $DEPS; an empty list pulls the whole remote"
    )
