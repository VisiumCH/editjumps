"""CLI for EvoFlows edit-flow training (Phase 2b)."""

from pathlib import Path
from typing import Annotated

import typer


def main(
    pairs: Annotated[Path, typer.Option(help="Homolog pairs from build-homolog-pairs")] = Path(
        "data/pretrain/oas_homolog_pairs.tsv.gz"
    ),
    model_name: Annotated[str, typer.Option(help="ESM-2 trunk to init from (local path or HF id)")] = (
        "data/pretrain/base_checkpoints/esm2_t12_35M_UR50D"
    ),
    output_folder: Annotated[Path, typer.Option()] = Path("data/pretrain/edit_flows"),
    max_steps: Annotated[int, typer.Option()] = 200,
    batch_size: Annotated[int, typer.Option(help="Homolog pairs summed per optimizer step")] = 2,
    lr: Annotated[float, typer.Option()] = 1e-4,
    save_steps: Annotated[int, typer.Option(help="Checkpoint the encoder every N steps (0 = never)")] = 100,
    seed: Annotated[int, typer.Option()] = 0,
    path: Annotated[str | None, typer.Option(
        help="② path: needleman_wunsch | needleman_wunsch_blosum62 | random_noise")] = None,
    schedule: Annotated[str | None, typer.Option(help="② schedule: linear | cubic")] = None,
    rate_head: Annotated[
        str | None, typer.Option(help="linear | mlp (Appendix A's shallow MLPs); overrides params.yaml")
    ] = None,
    q_head: Annotated[
        str | None, typer.Option(help="fresh | esm_lm_head (Appendix A eq 16); overrides params.yaml")
    ] = None,
    loss: Annotated[str | None, typer.Option(help="④ loss: edit_flow | edit_flow_figure13")] = None,
    checkpoint_uri: Annotated[
        str | None,
        typer.Option(help="gs://… prefix to save/resume the full training state (default: off, no GCS calls)"),
    ] = None,
    val_frac: Annotated[
        float, typer.Option(help="Fraction of pairs held out for val_loss, sequence-disjoint (0 = off)")
    ] = 0.05,
    eval_steps: Annotated[int, typer.Option(help="Log val_loss every N steps")] = 500,
    val_pairs: Annotated[int, typer.Option(help="Held-out pairs the val loss averages over")] = 256,
    patience: Annotated[
        int, typer.Option(help="Stop after N validations with no val_loss improvement (0 = off)")
    ] = 0,
    min_delta: Annotated[float, typer.Option(help="Minimum val_loss decrease that counts as improvement")] = 0.0,
    run_tag: Annotated[
        str | None, typer.Option(help="Label this run in MLflow (required to compare hyperparameters)")
    ] = None,
    chain: Annotated[
        str,
        typer.Option(help="heavy | light | joined - which chain of a joined VH.VL pair line to train on"),
    ] = "joined",
) -> None:
    """Train the EvoFlows edit-flow model on homolog pairs (logs to MLflow)."""
    import gzip

    import yaml

    from editjumps.core.edit_flows.stages import EditFlowConfig
    from editjumps.core.gcs_checkpoints import resolve_checkpoint_uri
    from editjumps.core.utils import get_logger
    from editjumps.pipeline.preprocess.pretrain.seed_homologs import CHAIN_CHOICES
    from editjumps.pipeline.train.evoflows import select_chain_pairs, train_edit_flows

    if chain not in CHAIN_CHOICES:
        raise typer.BadParameter(f"chain={chain!r}; options: {CHAIN_CHOICES}", param_hint="--chain")

    params = yaml.safe_load(Path("params.yaml").read_text()) if Path("params.yaml").exists() else {}
    config = EditFlowConfig.from_params(
        params, path=path, schedule=schedule, loss=loss, rate_head=rate_head, q_head=q_head
    )

    source = "homolog"
    sample: list[tuple[str, str]] = []
    with gzip.open(pairs, "rt") as fh:
        for i, line in enumerate(fh):
            if i >= 2000:
                break
            x0, tab, x1 = line.rstrip("\n").partition("\t")
            if tab:
                sample.append((x0, x1))
    sample, sample_skipped = select_chain_pairs(sample, chain)
    get_logger(__file__).info(
        f"pairs: {len(sample)} sampled, unoriented (chain={chain}, "
        f"skipped={sum(sample_skipped.values())} of the sample, pair_source={source})"
    )

    train_edit_flows(
        pairs,
        model_name=model_name,
        output_folder=output_folder,
        max_steps=max_steps,
        batch_size=batch_size,
        lr=lr,
        save_steps=save_steps,
        seed=seed,
        config=config,
        checkpoint_uri=resolve_checkpoint_uri(checkpoint_uri, params, "edit_flows"),
        val_frac=val_frac,
        eval_steps=eval_steps,
        val_pairs=val_pairs,
        patience=patience,
        min_delta=min_delta,
        run_tag=run_tag,
        chain=chain,
    )


if __name__ == "__main__":
    typer.run(main)
