"""Unified ``editjumps`` command-line interface. **Start here.** Three commands, for the three things."""

import typer

# Each stage module exposes a Typer-style ``main``; they are grouped into
# help panels at registration below (imports are kept import-sorted).
from editjumps.core.provenance import main as provenance_cmd
from editjumps.editing import main as edit_cmd
from editjumps.pipeline.evaluate.agreement_interval import main as agreement_interval_cmd
from editjumps.pipeline.evaluate.alignment_scoring import main as alignment_scoring_cmd
from editjumps.pipeline.evaluate.consolidate import main as consolidate_cmd
from editjumps.pipeline.evaluate.deterministic_benchmark import main as deterministic_benchmark_cmd
from editjumps.pipeline.evaluate.evodiff_msa_baseline import main as evodiff_msa_baseline_cmd
from editjumps.pipeline.evaluate.evotune_baseline import main as evotune_baseline_cmd
from editjumps.pipeline.evaluate.figure_extract import main as figure_extract_cmd
from editjumps.pipeline.evaluate.generation_eval import main as generation_eval_cmd
from editjumps.pipeline.evaluate.inference_trace import main as inference_trace_cmd
from editjumps.pipeline.evaluate.rescore_positional import main as rescore_positional_cmd
from editjumps.pipeline.evaluate.sampler_step_size import main as sampler_step_size_cmd
from editjumps.pipeline.evaluate.standard import main as evaluate_cmd
from editjumps.pipeline.preprocess.pretrain.deterministic_pairs import main as deterministic_pairs_cmd
from editjumps.pipeline.preprocess.pretrain.download_oas import main as download_oas_cmd
from editjumps.pipeline.preprocess.pretrain.homolog_pairs import main as build_homolog_pairs_cmd
from editjumps.pipeline.preprocess.pretrain.seed_homologs import main as seed_homologs_cmd
from editjumps.pipeline.preprocess.pretrain.split_corpus import main as split_corpus_cmd
from editjumps.pipeline.train.calibrate_throughput import main as calibrate_throughput_cmd
from editjumps.pipeline.train.convert_esm_checkpoint import main as fetch_base_checkpoint_cmd
from editjumps.pipeline.train.evotune import main as evotune_cmd
from editjumps.pipeline.train.pretrain_esm import main as pretrain_cmd
from editjumps.pipeline.train.restore_editor import main as restore_editor_cmd
from editjumps.pipeline.train.train_edit_flows import main as train_edit_flows_cmd
from editjumps.ranking import main as rank_cmd
from editjumps.repo.audit_stage_coverage import main as stage_coverage_cmd

app = typer.Typer(
    name="editjumps",
    add_completion=False,
    no_args_is_help=True,
    rich_markup_mode="rich",
    help="Fine-grained protein sequence editing with learned generative jump edits: "
    "an EvoFlows reproduction, with the corpus build and ESM-2 pretraining behind it.\n\n"
    "[bold]Start with[/bold] [green]editjumps edit[/green] - one sequence in, edited variants out, "
    "or [green]editjumps rank[/green] - a shortlist in, an order out - "
    "or [green]editjumps evaluate[/green] to reproduce the paper's comparison from one config file. "
    "Everything else is a pipeline stage that `dvc repro` calls.",
)

# The front door gets its own panel, registered first so `--help` opens on it.
ENTRY_PANEL = "Start here"
PIPELINE_PANEL = "Stages: whole pipeline"
OAS_PANEL = "Stages: OAS pretraining corpus"
TRAIN_PANEL = "Stages: ESM-2 pretraining, the editor, and the §4.2 baselines"
HELPERS_PANEL = "Stages: helpers / manual (not in the DAG)"


app.command(
    "edit", rich_help_panel=ENTRY_PANEL,
    help="[bold]Sequence in, edited variants out.[/bold] The product surface: one sequence -> N variants "
         "with their realised edit distances. Needs a checkpoint (see --help). No quality score - it cannot.",
)(edit_cmd)
app.command(
    "rank", rich_help_panel=ENTRY_PANEL,
    help="[bold]Candidates in, an order out.[/bold] Ranks a shortlist by ESM-2 naturalness "
         "(Appendix B.2's estimator). Ranks how protein-like each is - NOT your property.",
)(rank_cmd)
app.command(
    "evaluate", rich_help_panel=ENTRY_PANEL,
    help="[bold]The front door.[/bold] Run the standard §4.2/§4.3 evaluation from params.yaml "
         "(--init writes standalone YAMLs; --dry-run shows the plan).",
)(evaluate_cmd)


@app.command(
    "run-pipeline",
    rich_help_panel=PIPELINE_PANEL,
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    help="Run the whole DAG via `dvc repro` (all args pass straight through).",
)
def run_pipeline(ctx: typer.Context) -> None:
    """Run the full pipeline by delegating to ``dvc repro``."""
    import shutil
    import subprocess

    if shutil.which("dvc") is None:
        typer.secho("dvc not found - install the dev group first: `uv sync --dev`", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    raise typer.Exit(code=subprocess.call(["dvc", "repro", *ctx.args]))


# Register each stage's existing ``main`` as a subcommand.

app.command("download-oas", rich_help_panel=OAS_PANEL, help="Crawl OAS and build the pretraining corpus.")(
    download_oas_cmd
)
app.command(
    "seed-homologs", rich_help_panel=OAS_PANEL,
    help="Build EvoFlows-style seed families and report them in the paper's Table 2 columns.",
)(seed_homologs_cmd)
app.command("split-corpus", rich_help_panel=OAS_PANEL, help="Cluster (MMseqs2) and split the corpus leakage-aware.")(
    split_corpus_cmd
)
app.command(
    "build-homolog-pairs", rich_help_panel=OAS_PANEL,
    help="Cluster the OAS corpus into homolog families and emit intra-family training pairs.",
)(build_homolog_pairs_cmd)



app.command(
    "fetch-base-checkpoint", rich_help_panel=TRAIN_PANEL, help="Download+convert a base ESM-2 checkpoint to HF format."
)(fetch_base_checkpoint_cmd)
app.command("pretrain", rich_help_panel=TRAIN_PANEL, help="Continue-pretrain (MLM) ESM-2 on the OAS corpus.")(
    pretrain_cmd
)
app.command(
    "train-edit-flows", rich_help_panel=TRAIN_PANEL, help="Train the EvoFlows edit-flow model on homolog pairs."
)(train_edit_flows_cmd)
app.command(
    "calibrate-throughput", rich_help_panel=TRAIN_PANEL,
    help="Measure s/step, MFU and peak memory across batch sizes (decides if a bigger GPU helps).",
)(calibrate_throughput_cmd)
app.command(
    "generation-eval", rich_help_panel=TRAIN_PANEL,
    help="Score a trained editor's generations against held-out homologs (EvoFlows Appendix B).",
)(generation_eval_cmd)
app.command(
    "restore-editor", rich_help_panel=TRAIN_PANEL,
    help="Rebuild a loadable editor folder from a training-state checkpoint.pt (local or gs://).",
)(restore_editor_cmd)
app.command(
    "deterministic-benchmark", rich_help_panel=TRAIN_PANEL,
    help="EvoFlows §4.1 ground-truth benchmark: per-class precision/recall + clock sweep.",
)(deterministic_benchmark_cmd)
app.command(
    "alignment-scoring", rich_help_panel=TRAIN_PANEL,
    help="Measure what the ② alignment scoring does to the edit-op label distribution (CPU only).",
)(alignment_scoring_cmd)
app.command(
    "evotune", rich_help_panel=TRAIN_PANEL,
    help="EvoFlows §2.2 Evotuning: MLM-adapt ESM-2 to one homolog family (the baselines' generator).",
)(evotune_cmd)
app.command(
    "evotune-baseline", rich_help_panel=TRAIN_PANEL,
    help="EvoFlows' evotuned-PLM baseline: entropy profile picks positions, the MLM infills (--forced).",
)(evotune_baseline_cmd)
app.command(
    "evodiff-msa-baseline", rich_help_panel=TRAIN_PANEL,
    help="EvoFlows' EvoDiff-MSA baseline (§4.2): their released MSA model infills a matched mutation budget.",
)(evodiff_msa_baseline_cmd)
app.command(
    "build-deterministic-pairs", rich_help_panel=OAS_PANEL,
    help="Build EvoFlows §4.1's synthetic (z0, z1) training pairs from natural sequences.",
)(deterministic_pairs_cmd)

# Standalone helpers - not part of the DAG (not in dvc.yaml). They support the reported numbers
# without producing any: diagnostics, digitisers and readers over artefacts already committed.
app.command(
    "sampler-step-size", rich_help_panel=HELPERS_PANEL,
    help="Measure how the Euler grid resolution changes the realised edit count (CPU, model-free).",
)(sampler_step_size_cmd)
app.command(
    "figure-extract", rich_help_panel=HELPERS_PANEL,
    help="Digitise a figure's plotted values out of a paper PDF.",
)(figure_extract_cmd)
app.command(
    "inference-trace", rich_help_panel=HELPERS_PANEL,
    help="Print the sampled inference path a trained model walks from a real template.",
)(inference_trace_cmd)
app.command(
    "consolidate", rich_help_panel=HELPERS_PANEL,
    help="Collect every §4.2 result into metrics/all_results.{json,csv}, with its comparability frame.",
)(consolidate_cmd)
app.command(
    "rescore-positional", rich_help_panel=HELPERS_PANEL,
    help="Fill a per-position metric into a committed cell from the sequences it saved (no model).",
)(rescore_positional_cmd)
app.command(
    "agreement-interval", rich_help_panel=HELPERS_PANEL,
    help="Give a committed cell's covariance/MIP agreement a template-resampled interval, and "
         "compare two methods PAIRED on the same templates (no model).",
)(agreement_interval_cmd)
app.command(
    "provenance", rich_help_panel=HELPERS_PANEL,
    help="Which stages reproduce EvoFlows and which support it (--write syncs the README).",
)(provenance_cmd)
# The two repo-shape reports.
app.command(
    "stage-coverage", rich_help_panel=HELPERS_PANEL,
    help="Which pipeline stages are tracked by DVC, MLflow and the CLI, and which are not.",
)(stage_coverage_cmd)


if __name__ == "__main__":
    app()
