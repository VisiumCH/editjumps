"""``editjumps evaluate`` — the front door: one config, the whole §4.2/§4.3 comparison, one command."""

import json
import shutil
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated

import typer

from editjumps.core.length_capability import REFUSED, LengthCapability
from editjumps.core.run_config import CONFIG_FILENAMES, DEFAULT_CONFIG_DIR, ArmConfig, RunConfig
from editjumps.core.utils import get_logger

logger = get_logger(__file__)

#: Step name -> the stage module whose ``main`` runs it. The step names are what ``--steps`` takes.
RUNNERS: dict[str, str] = {
    "generation_eval": "editjumps.pipeline.evaluate.generation_eval",
    "evotune_baseline": "editjumps.pipeline.evaluate.evotune_baseline",
    "evodiff_msa_baseline": "editjumps.pipeline.evaluate.evodiff_msa_baseline",
}

#: The default sequence: the editor first, because the baselines' mutation budget is matched to its measured.
DEFAULT_STEPS: tuple[str, ...] = ("editor", "evotuned", "evotuned_forced", "evodiff")


@dataclass(frozen=True)
class PlannedStep:
    """One method of the standard evaluation, fully resolved but not yet run."""

    name: str
    runner: str
    kwargs: dict
    command: tuple[str, ...]
    metrics_path: Path
    needs_editor_metrics: bool = False
    inapplicable: str = ""


@dataclass(frozen=True)
class Plan:
    """A resolved standard evaluation: the configuration and the steps it expands to."""

    config: RunConfig
    steps: list[PlannedStep] = field(default_factory=list)
    out_dir: Path = Path("metrics/standard")

    def as_dict(self) -> dict[str, object]:
        """Render the plan as JSON-able data, configuration and fingerprint included."""
        return {
            **self.config.as_dict(),
            "out_dir": str(self.out_dir),
            "steps": [
                {"name": s.name, "command": " ".join(s.command), "metrics_path": str(s.metrics_path),
                 **({"inapplicable": s.inapplicable} if s.inapplicable else {})}
                for s in self.steps
            ],
        }


def _flag_list(flags: Mapping[str, str | None]) -> list[str]:
    """Flatten a flag mapping into an argv fragment."""
    argv: list[str] = []
    for flag, value in flags.items():
        argv.append(flag)
        if value is not None:
            argv.append(value if value != "" else '""')
    return argv


def step_capability(name: str, arm: ArmConfig) -> LengthCapability:
    """Look up what one step's method can do to a sequence's LENGTH."""
    import importlib

    if name == "editor":
        return importlib.import_module(RUNNERS["generation_eval"]).LENGTH_CAPABILITY
    if name in ("evotuned", "evotuned_forced"):
        module = importlib.import_module(RUNNERS["evotune_baseline"])
        return module.length_capability(name == "evotuned_forced")
    if name == "evodiff":
        module = importlib.import_module(RUNNERS["evodiff_msa_baseline"])
        return module.length_capability(arm.evodiff_mode)
    raise ValueError(f"unknown step {name!r}; options: {DEFAULT_STEPS}")


def build_plan(
    config: RunConfig,
    out_dir: Path,
    steps: tuple[str, ...] = DEFAULT_STEPS,
    target_length: int | None = None,
    allow_train_overlap: bool = False,
) -> Plan:
    """Expand a configuration into the steps of the standard evaluation."""
    unknown = [s for s in steps if s not in DEFAULT_STEPS]
    if unknown:
        raise ValueError(f"unknown step(s) {unknown}; options: {DEFAULT_STEPS}")

    comparability, arm = config.comparability, config.arm
    editor_metrics = out_dir / "generation_eval.json"
    planned: list[PlannedStep] = []

    def growth(name: str) -> tuple[dict[str, str | None], dict[str, object], str]:
        """Resolve one step's growth target into flags, kwargs and an inapplicability reason."""
        if target_length is None:
            return {}, {}, ""
        capability = step_capability(name, arm)
        if capability.target_length == REFUSED:
            return {}, {}, capability.mechanism
        return {"--target-length": str(target_length)}, {"target_length": target_length}, ""

    if "editor" in steps:
        growth_flags, growth_kwargs, growth_reason = growth("editor")
        flags = {**config.generation_eval_flags(), **growth_flags, "--metrics-path": str(editor_metrics)}
        if allow_train_overlap:
            flags["--allow-train-overlap"] = None
        planned.append(PlannedStep(
            name="editor",
            runner="generation_eval",
            inapplicable=growth_reason,
            kwargs={
                **growth_kwargs,
                "model_folder": arm.model_folder, "pairs": config.target.pairs,
                "n_templates": comparability.n_templates, "n_variants": comparability.n_variants,
                "n_steps": arm.n_steps, "seed": comparability.split_seed, "k": comparability.spectrum_k,
                "clock": config.target.clock, "pll_model": comparability.pll_model,
                "pll_positions": comparability.pll_positions, "family_fasta": config.target.family_fasta(),
                "holdout_size": comparability.holdout_size, "ceiling_n": comparability.ceiling_n,
                "rate_head": arm.rate_head, "q_head": arm.q_head, "metrics_path": editor_metrics,
                "allow_train_overlap": allow_train_overlap,
            },
            command=("editjumps", "generation-eval", *_flag_list(flags)),
            metrics_path=editor_metrics,
        ))

    baseline_steps = [s for s in steps if s != "editor"]
    if baseline_steps and config.target.seed_family is None:
        raise ValueError(
            "the §4.2 baselines need target.seed_family: each one partitions ONE family into "
            "templates / holdout / pool, and without a family there is no partition to share with "
            "the editor. Set a seed family, or run --steps editor alone and read its numbers as "
            "self-comparable only."
        )
    # §4.2 compares methods "matching the expected number of mutations per sequence".
    matched = comparability.match_budget_to_editor
    if matched and baseline_steps and "editor" not in steps and not editor_metrics.exists():
        raise ValueError(
            f"comparability.match_budget_to_editor is on, but {editor_metrics} does not exist and the "
            "editor step was not requested. §4.2's matched mutation budget is read off the editor's "
            "measured mean edit distance, so either include the `editor` step or turn the matching "
            "off and set comparability.mutations explicitly."
        )
    budget_flags = ({"--mutations-from": str(editor_metrics)} if matched
                    else {"--mutations": str(comparability.mutations)})

    for name, forced in (("evotuned", False), ("evotuned_forced", True)):
        if name not in steps:
            continue
        metrics = out_dir / f"{name}.json"
        growth_flags, growth_kwargs, growth_reason = growth(name)
        flags: dict[str, str | None] = {**config.evotune_baseline_flags(), **budget_flags,
                                        "--forced" if forced else "--no-forced": None,
                                        **growth_flags,
                                        "--metrics-path": str(metrics)}
        planned.append(PlannedStep(
            name=name,
            runner="evotune_baseline",
            inapplicable=growth_reason,
            kwargs={
                **growth_kwargs,
                "model_folder": arm.evotuned_model, "family_fasta": config.target.family_fasta(),
                "train_corpus": arm.evotune_train_corpus, "forced": forced,
                "n_templates": comparability.n_templates, "n_variants": comparability.n_variants,
                "mutations": comparability.mutations,
                "mutations_from": editor_metrics if matched else None,
                "temperature": arm.evotune_temperature, "holdout_size": comparability.holdout_size,
                "profile_size": arm.evotune_profile_size, "budget_mode": comparability.budget_mode,
                "max_rounds": comparability.max_rounds, "seed": comparability.split_seed,
                "k": comparability.spectrum_k, "ceiling_n": comparability.ceiling_n,
                "pll_model": comparability.pll_model, "pll_positions": comparability.pll_positions,
                "metrics_path": metrics,
            },
            command=("editjumps", "evotune-baseline", *_flag_list(flags)),
            metrics_path=metrics,
            needs_editor_metrics=matched,
        ))

    if "evodiff" in steps:
        metrics = out_dir / "evodiff_msa_baseline.json"
        growth_flags, growth_kwargs, growth_reason = growth("evodiff")
        flags = {**config.evodiff_baseline_flags(), **budget_flags, **growth_flags,
                 "--metrics-path": str(metrics)}
        planned.append(PlannedStep(
            name="evodiff",
            runner="evodiff_msa_baseline",
            inapplicable=growth_reason,
            kwargs={
                **growth_kwargs,
                "family_fasta": config.target.family_fasta(), "mode": arm.evodiff_mode,
                "model": arm.evodiff_model, "n_templates": comparability.n_templates,
                "n_variants": comparability.n_variants, "mutations": comparability.mutations,
                "mutations_from": editor_metrics if matched else None,
                "holdout_size": comparability.holdout_size, "msa_size": arm.evodiff_msa_size,
                "n_sequences": arm.evodiff_n_sequences, "selection_type": arm.evodiff_selection_type,
                "temperature": arm.evodiff_temperature, "device": arm.evodiff_device,
                "budget_mode": comparability.budget_mode, "max_rounds": comparability.max_rounds,
                "penalty_value": arm.evodiff_penalty_value, "seed": comparability.split_seed,
                "k": comparability.spectrum_k, "ceiling_n": comparability.ceiling_n,
                "pll_model": comparability.pll_model, "pll_positions": comparability.pll_positions,
                "metrics_path": metrics,
            },
            command=("editjumps", "evodiff-msa-baseline", *_flag_list(flags)),
            metrics_path=metrics,
            needs_editor_metrics=matched,
        ))

    return Plan(config=config, steps=planned, out_dir=out_dir)


def run_step(step: PlannedStep) -> None:
    """Execute one planned step by calling its stage module's ``main``."""
    import importlib

    module = importlib.import_module(RUNNERS[step.runner])
    module.main(**step.kwargs)


def summarise(plan: Plan) -> list[dict[str, object]]:
    """Read back each step's report and pull out the numbers and the sizes they were measured at."""
    rows: list[dict[str, object]] = []
    for step in plan.steps:
        # An inapplicable row carries NO metric keys, deliberately: a null spectrum_mmd would still sort, plot.
        if step.inapplicable:
            rows.append({"method": step.name, "inapplicable": True, "reason": step.inapplicable})
            continue
        if not step.metrics_path.exists():
            continue
        report = json.loads(step.metrics_path.read_text())
        defined = report.get("defined_by_the_paper", {})
        diagnostic = report.get("mmd_diagnostic", {})
        rows.append({
            "method": step.name,
            "spectrum_mmd": defined.get("spectrum_mmd"),
            "covariance_agreement": defined.get("covariance_agreement"),
            "mip_agreement": defined.get("mip_agreement"),
            "n_generated": diagnostic.get("n_generated", report.get("n_generated")),
            "n_reference": diagnostic.get("n_reference", report.get("n_natural_reference")),
            "agreement_reference_n": defined.get("agreement_reference_n"),
            "ceiling_n": defined.get("ceiling_n"),
            "alignment": report.get("alignment"),
        })
    return rows


def main(
    config_dir: Annotated[
        Path | None,
        typer.Option("--config-dir", help="Directory holding target/comparability/arm files; defaults to params.yaml"),
    ] = DEFAULT_CONFIG_DIR,
    target: Annotated[Path | None, typer.Option(help="Override the target group's file")] = None,
    comparability: Annotated[Path | None, typer.Option(help="Override the comparability group's file")] = None,
    arm: Annotated[Path | None, typer.Option(help="Override the arm group's file")] = None,
    out_dir: Annotated[Path, typer.Option(help="Where the reports and run_config.json land")] = Path(
        "metrics/standard"
    ),
    steps: Annotated[str, typer.Option(help="Comma-separated: editor,evotuned,evotuned_forced,evodiff")] = ",".join(
        DEFAULT_STEPS
    ),
    target_length: Annotated[int | None, typer.Option(
        help="Run the growth benchmark at this length; methods that cannot target one are reported "
             "as inapplicable instead of run")] = None,
    dry_run: Annotated[bool, typer.Option(help="Resolve, record the plan and the fingerprint, run nothing")] = False,
    init: Annotated[
        bool,
        typer.Option(help="Write three YAML files from params.yaml into --config-dir, then exit"),
    ] = False,
    allow_train_overlap: Annotated[
        bool,
        typer.Option(help="Allow training pairs to overlap reference sequences (for minimal smoke runs)"),
    ] = False,
) -> None:
    """Run the standard EvoFlows §4.2/§4.3 evaluation from one configuration."""
    if init:
        target_dir = config_dir or Path("configs")
        existing = [target_dir / name for name in CONFIG_FILENAMES.values() if (target_dir / name).exists()]
        if existing:
            logger.error(f"refusing to overwrite {', '.join(str(p) for p in existing)}; "
                         "pass --config-dir <somewhere-else> to write a fresh set")
            raise typer.Exit(code=1)
        written = RunConfig.from_params().write(target_dir)
        for name, path in written.items():
            logger.info(f"wrote {path}  ({name})")
        logger.info("edit them, then: editjumps evaluate --config-dir %s", target_dir)
        return

    config = RunConfig.load(config_dir, target=target, comparability=comparability, arm=arm)
    chosen = tuple(s.strip() for s in steps.split(",") if s.strip())
    plan = build_plan(config, out_dir, chosen, target_length=target_length, allow_train_overlap=allow_train_overlap)

    out_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = config.comparability.fingerprint()
    (out_dir / "run_config.json").write_text(json.dumps(plan.as_dict(), indent=2))
    logger.info(f"comparability fingerprint {fingerprint} "
                f"(holdout {config.comparability.holdout_size}, "
                f"generated {config.comparability.n_generated}, "
                f"frame {config.comparability.alignment_template}, "
                f"ceiling {config.comparability.ceiling_n})")
    comp_source = f"{config_dir / 'comparability.yaml'}" if config_dir is not None else "params.yaml"
    logger.info(f"a run whose fingerprint differs is NOT comparable with this one; "
                f"diff against {comp_source} to see which field moved")
    logger.info(f"wrote {out_dir / 'run_config.json'}")

    if dry_run:
        for step in plan.steps:
            logger.info(f"[dry-run] {step.name}: {' '.join(step.command)}")
        return

    for step in plan.steps:
        # A method with no construction for the requested length is skipped and reported as inapplicable.
        if step.inapplicable:
            logger.warning(f"skipping {step.name}: inapplicable at --target-length {target_length}. "
                           f"{step.inapplicable}")
            continue
        # The EvoDiff baseline runs under its own interpreter (evodiff pins numpy<2), so a missing shim is a.
        if step.runner == "evodiff_msa_baseline" and shutil.which("evodiff-msa") is None:
            logger.warning(f"skipping {step.name}: `evodiff-msa` not on PATH "
                           "(run `bash editjumps/core/evodiff_msa/install_evodiff.sh`)")
            continue
        logger.info(f"--- {step.name}: {' '.join(step.command)}")
        run_step(step)

    for row in summarise(plan):
        if row.get("inapplicable"):
            logger.info(f"  {str(row['method']):<16} INAPPLICABLE - {row['reason']}")
            continue
        logger.info(
            f"  {str(row['method']):<16} mmd={row['spectrum_mmd']}  cov={row['covariance_agreement']}  "
            f"mip={row['mip_agreement']}  n_gen={row['n_generated']}  n_ref={row['n_reference']}  "
            f"{row['alignment']}"
        )
    logger.info(f"reports in {out_dir} (fingerprint {fingerprint})")


if __name__ == "__main__":
    typer.run(main)
