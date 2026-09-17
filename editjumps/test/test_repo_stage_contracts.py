"""Contracts every stage must honour, checked by scanning the stages rather than one by one."""

import inspect
from pathlib import Path


def test_every_evaluation_path_passes_the_clock_to_the_sampler() -> None:
    """No evaluation path may reach the ⑤ sampler without a clock argument."""
    import ast

    threaded = {"sample_edits", "sample_edits_gillespie", "sampler"}
    paths = [
        "editjumps/pipeline/evaluate/generation_eval.py",    # Appendix-B generation eval
        "editjumps/pipeline/evaluate/deterministic_benchmark.py",  # §4.1 clock sweep
    ]
    for path in paths:
        tree = ast.parse(Path(path).read_text())
        # A module may call the sampler through an alias bound from resolve_sampler(); discover
        # those names rather than hard-coding them, or the next refactor exempts a path.
        aliases = {
            target.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and ast.unparse(node.value.func).endswith("resolve_sampler")
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        recognised = threaded | aliases
        calls = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id in recognised
        ]
        assert calls, f"{path} no longer calls any of {sorted(recognised)} - is this test stale?"
        for call in calls:
            called = ast.unparse(call.func)
            assert any(kw.arg == "clock" for kw in call.keywords), (
                f"{path}:{call.lineno} calls {called}() without clock=, so it samples at "
                f"unscaled rates whatever params.yaml says"
            )


def test_every_evaluator_takes_the_disjoint_filter_and_passes_it_on() -> None:
    """All three evaluators accept ``disjoint_from_pairs``, and each CLI actually forwards it."""
    import inspect

    from editjumps.pipeline.evaluate import (
        evodiff_msa_baseline,
        evotune_baseline,
        generation_eval,
    )

    for module in (generation_eval, evotune_baseline, evodiff_msa_baseline):
        name = module.__name__.rsplit(".", 1)[-1]
        inner = inspect.signature(module.evaluate).parameters
        assert "disjoint_from_pairs" in inner, f"{name}.evaluate lacks the parameter"
        assert inner["disjoint_from_pairs"].kind is inspect.Parameter.KEYWORD_ONLY, (
            f"{name}: must be keyword-only, or a positional call binds the wrong argument to it"
        )
        assert "disjoint_from_pairs" in inspect.signature(module.main).parameters, (
            f"{name}.main does not expose the flag"
        )
        # The CLI must FORWARD it, not merely accept it.
        source = inspect.getsource(module.main)
        assert "disjoint_from_pairs=disjoint_from_pairs" in source, (
            f"{name}.main accepts the flag and never passes it on"
        )


def test_mlflow_tracking_failures_cannot_abort_a_finished_run() -> None:
    """Every MLflow call this repo makes is guarded, and the metrics JSON is written first."""
    import inspect
    import re

    from editjumps.core import utils
    from editjumps.pipeline.train import pretrain_esm

    source = inspect.getsource(utils)
    block = source[source.index('for name in ("log_metric"'):]
    guarded = set(re.findall(r'"([a-z_]+)"', block[:block.index(")")]))
    for call in ("log_metric", "log_metrics", "log_params", "log_param", "set_tag",
                 "log_artifact", "log_artifacts", "log_text"):
        assert call in guarded, f"mlflow.{call} is not guarded; a tracking failure would abort real work"

    # Every mlflow call site in the pipeline must be a guarded name, or tracking can still kill a stage.
    root = Path(__file__).parents[1]
    setup_only = {"start_run", "set_tracking_uri", "set_experiment", "end_run", "active_run",
                  "get_tracking_uri", "search_runs", "set_registry_uri"}
    allowed = guarded | setup_only
    unguarded = []
    for path in root.rglob("*.py"):
        if "/test" in str(path):
            continue
        for match in re.finditer(r"\bmlflow\.([a-z_]+)\s*\(", path.read_text()):
            if match.group(1) not in allowed:
                unguarded.append(f"{path.relative_to(root)}: mlflow.{match.group(1)}")
    assert not unguarded, f"unguarded MLflow calls: {unguarded}"

    # Ordering: the metrics JSON must be written before the artifact upload.
    body = inspect.getsource(pretrain_esm)
    json_at = body.index("wrote DVC metrics file to")
    upload_at = body.index("mlflow.log_artifacts(str(model_dir)")
    assert json_at < upload_at, (
        "the artifact upload precedes the metrics JSON write; a 413 there loses the durable "
        "record of a run whose training already succeeded, which is what happened to sky job 66"
    )
    # And the upload is gated on file size, so the common case does not upload-then-fail.
    assert "MLFLOW_ARTIFACT_FILE_LIMIT" in body, "the artifact upload must be size-gated"
    assert pretrain_esm.MLFLOW_ARTIFACT_FILE_LIMIT < 32_000_000, (
        "the limit must sit below Cloud Run's 32 MB request cap"
    )


def test_every_training_stage_opens_an_mlflow_run() -> None:
    """A stage that writes weights must record, somewhere durable, that it ran and where they went.."""
    root = Path(__file__).parents[1] / "pipeline" / "train"
    # where that whole half moved; the rule about it is unchanged.
    fitters = [root / name for name in ("pretrain_esm.py", "evotune.py", "evoflows.py")]
    for path in fitters:
        name = path.name
        source = path.read_text()
        assert "start_mlflow_run(" in source, (
            f"{name} fits a model and opens no MLflow run; a run whose weights nobody can find is "
            "how the evotuned trunk was lost"
        )

    # And the wrapper delegates rather than nests: `pretrain` joins an active run instead of trying
    # to start a second one, which is what lets `evotune`'s run hold the family AND the losses.
    assert "mlflow.active_run()" in (root / "pretrain_esm.py").read_text(), (
        "pretrain must join a caller's run; starting a nested one raises, and splitting one "
        "training run across two runs is what hid the family in the first place"
    )


def test_every_family_evaluator_aligns_to_the_same_template() -> None:
    """Editor and baselines must project onto the same template."""
    root = Path(__file__).parents[1] / "pipeline" / "evaluate"
    for name in ("generation_eval.py", "evotune_baseline.py", "evodiff_msa_baseline.py"):
        text = (root / name).read_text()
        assignments = [line.strip() for line in text.splitlines()
                       if ("alignment_template = " in line or "reference_template = " in line)
                       and not line.strip().startswith("#")]
        assert assignments, f"{name}: no alignment template assignment found"
        for line in assignments:
            assert "split.templates[0]" in line, (
                f"{name} aligns to something other than split.templates[0], so its per-position "
                f"metrics cannot be compared with the other evaluators':\n    {line}"
            )


def test_every_trainer_and_evaluator_takes_its_pairs_file_and_family_as_arguments() -> None:
    """No stage reads a corpus path out of the air, so an arbitrary pairs table can be benchmarked."""
    import ast

    from editjumps.pipeline.evaluate import (
        evodiff_msa_baseline,
        evotune_baseline,
        generation_eval,
        standard,
    )
    from editjumps.pipeline.train import evotune, train_edit_flows

    required = {
        train_edit_flows: {"pairs"},
        evotune: {"family_fasta", "disjoint_from_pairs"},
        generation_eval: {"pairs", "family_fasta"},
        evotune_baseline: {"family_fasta", "train_corpus", "disjoint_from_pairs"},
        evodiff_msa_baseline: {"family_fasta", "disjoint_from_pairs"},
    }
    for module, names in required.items():
        parameters = set(inspect.signature(module.main).parameters)
        missing = names - parameters
        assert not missing, f"{module.__name__}.main cannot be pointed at another table: {sorted(missing)}"

    # Every data/metrics path literal must sit in a parameter default, never in a function body.
    for module in (*required, standard):
        tree = ast.parse(inspect.getsource(module))
        allowed: set[int] = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                defaults = [*node.args.defaults, *(d for d in node.args.kw_defaults if d is not None)]
                for default in defaults:
                    allowed.update(id(sub) for sub in ast.walk(default))
        buried = [
            node.value for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
            and (node.value.startswith("data/") or node.value.startswith("metrics/"))
            and node.value.endswith((".gz", ".fasta", ".json", ".tsv"))
            and id(node) not in allowed
        ]
        assert not buried, (
            f"{module.__name__} hardcodes {buried} outside a parameter default, so a run on another "
            "pairs file or family would silently read the old one"
        )
