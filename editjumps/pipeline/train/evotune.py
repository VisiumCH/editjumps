"""Evotuning: the same MLM objective, family-scoped instead of corpus-scoped (EvoFlows §2.2)."""

import gzip
import random
from pathlib import Path
from typing import Annotated

import typer

from editjumps.core.family_split import disjoint_members, split_family, usable_members
from editjumps.core.sequences import AA
from editjumps.core.utils import get_logger

logger = get_logger(__file__)

#: Its own experiment, not ``pretrain_esm``'s: same objective, but a family-scoped trunk is a different.
EXPERIMENT_NAME = "editjumps-esm2-evotune"


def training_members(pool: list[str]) -> list[str]:
    """De-duplicate and alphabet-filter a family's train part into MLM training sequences."""
    seen: set[str] = set()
    members: list[str] = []
    for raw in pool:
        sequence = raw.strip().upper()
        if not sequence or sequence in seen or not set(sequence) <= AA:
            continue
        seen.add(sequence)
        members.append(sequence)
    return members


def family_corpus(
    family_fasta: Path,
    train_path: Path,
    val_path: Path,
    n_templates: int = 20,
    holdout_size: int = 200,
    split_seed: int = 0,
    val_fraction: float = 0.05,
    seed: int = 0,
    *,
    disjoint_from_pairs: Path | None = None,
) -> tuple[int, int]:
    """Write one family's train part as the gzipped one-sequence-per-line corpus ``pretrain`` reads."""
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError(f"val_fraction={val_fraction}; must be in [0, 1)")
    from editjumps.pipeline.preprocess.pretrain.seed_homologs import read_fasta

    members = usable_members(read_fasta(family_fasta).values())
    if disjoint_from_pairs is not None:
        # The trunk must be adapted on the SAME population its baseline will be scored against, or its.
        kept = disjoint_members(members, disjoint_from_pairs)
        logger.info(f"disjoint draw: {len(kept)}/{len(members)} family members are not in "
                    f"{disjoint_from_pairs}")
        members = kept
    split = split_family(members, n_templates, holdout_size, split_seed)
    train_members = training_members(split.pool)
    if not train_members:
        raise ValueError(
            f"{family_fasta}: {len(members)} usable members leave nothing to train on after "
            f"{n_templates} templates and a holdout of {holdout_size} — lower --holdout-size, "
            "or build a larger family (see `seed_homologs.min_identity`)"
        )

    random.Random(seed).shuffle(train_members)
    n_val = int(len(train_members) * val_fraction)
    # Never all of it: an empty val file is fine, an empty train file fails inside the Trainer.
    n_val = min(n_val, max(0, len(train_members) - 1))
    val, train = train_members[:n_val], train_members[n_val:]

    for path, group in ((train_path, train), (val_path, val)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt") as out:
            for sequence in group:
                out.write(f"{sequence}\n")
    logger.info(
        f"{family_fasta.name}: {len(members)} usable members -> {len(split.pool)} in the train part "
        f"-> {len(train_members)} on-alphabet -> {len(train)} train / {len(val)} val"
    )
    return len(train), len(val)


def main(
    disjoint_from_pairs: Annotated[Path | None, typer.Option(
        help="Adapt only on members absent from this pairs file; match the evaluation")] = None,
    family_fasta: Annotated[Path, typer.Option(help="One seed family FASTA from `seed-homologs`")] = Path(
        "data/interim/seed_families/Anti-SARS-CoV-2_VHH_Ty1.fasta"
    ),
    output_folder: Annotated[Path, typer.Option()] = Path("data/pretrain"),
    corpus_stem: Annotated[str, typer.Option(help="Basename stem for the two corpus files")] = "evotune_family",
    model_dir_name: Annotated[str, typer.Option(help="Saved-model subfolder under output_folder")] = "esm2_evotuned",
    model_name: Annotated[str, typer.Option(help="Base ESM-2 checkpoint to evotune from")] = (
        "facebook/esm2_t12_35M_UR50D"
    ),
    epochs: Annotated[float, typer.Option()] = 3.0,
    max_steps: Annotated[int, typer.Option(help="Cap steps; -1 = full epochs")] = -1,
    batch_size: Annotated[int, typer.Option()] = 16,
    max_length: Annotated[int, typer.Option(help="Truncation length; a single V domain needs ~160")] = 160,
    mlm_probability: Annotated[float, typer.Option()] = 0.15,
    n_templates: Annotated[int, typer.Option(help="Inference part excluded from training (match the eval)")] = 20,
    holdout_size: Annotated[int, typer.Option(help="Scoring holdout excluded from training (match the eval)")] = 200,
    split_seed: Annotated[int, typer.Option(help="Seed of the §4.2 family partition (match the eval)")] = 0,
    val_fraction: Annotated[float, typer.Option(help="Train-part fraction held out for eval loss")] = 0.05,
    save_steps: Annotated[int, typer.Option()] = 500,
    save_total_limit: Annotated[int, typer.Option()] = 2,
    precision: Annotated[str, typer.Option(help="auto | bf16 | fp32")] = "auto",
    dataloader_workers: Annotated[int, typer.Option()] = 0,
    seed: Annotated[int, typer.Option(help="Seed for the train/val shuffle inside the train part")] = 0,
    metrics_path: Annotated[
        Path | None, typer.Option(help="DVC metrics JSON (default: don't write one, as in `pretrain`)")
    ] = None,
) -> None:
    """Evotune a pre-trained ESM-2 on one homolog family (§2.2), for the baselines to generate with."""
    # Imported here: `pretrain_esm` pulls mlflow at import, and `--help` must work in the lean env.
    import mlflow

    from editjumps.core.utils import design_space_tags, start_mlflow_run
    from editjumps.pipeline.train.pretrain_esm import pretrain

    train_path = output_folder / f"{corpus_stem}.train.txt.gz"
    val_path = output_folder / f"{corpus_stem}.val.txt.gz"
    # This run's coordinate in the modeling design space (see ArmConfig / params.yaml), for filtering..
    tags = design_space_tags(
        objective="mlm",
        backbone=Path(model_name).name,
        weight_init="pretrained",
        pretrain_data=family_fasta.name,
        chain_handling="separate",
    )
    # `start_mlflow_run` merges the job's GCS destinations (MODEL_GCS_URI and friends) in itself, so
    # the run says where the trunk was uploaded to as well as where it was written locally.
    run_name = f"evotune-{Path(model_name).name}-{family_fasta.stem}"
    # The corpus build is inside the run on purpose: an empty train part raises in `family_corpus`,
    # and a run marked FAILED with the family and the split sizes on it says why.
    with start_mlflow_run(EXPERIMENT_NAME, run_name=run_name, tags=tags):
        # Keys `pretrain` does not log, so the two param sets compose into one description of the
        # run: what family, which §4.2 partition, and where the trunk lands.
        mlflow.log_params(
            {
                "family": family_fasta.name,
                "family_fasta": str(family_fasta),
                "output_folder": str(output_folder),
                "corpus_stem": corpus_stem,
                "model_dir_name": model_dir_name,
                "model_dir": str(output_folder / model_dir_name),
                "n_templates": n_templates,
                "holdout_size": holdout_size,
                "split_seed": split_seed,
                "val_fraction": val_fraction,
                "shuffle_seed": seed,
                "precision": precision,
                "dataloader_workers": dataloader_workers,
                "metrics_path": str(metrics_path) if metrics_path else None,
            }
        )
        n_train, n_val = family_corpus(
            family_fasta, train_path, val_path, n_templates=n_templates, holdout_size=holdout_size,
            split_seed=split_seed, val_fraction=val_fraction, seed=seed,
            disjoint_from_pairs=disjoint_from_pairs,
        )
        # Metrics, not params: they are measured off the family FASTA (how many members survived
        # de-duplication, the alphabet filter and the holdout), not chosen by the caller.
        mlflow.log_metrics({"n_train_sequences": n_train, "n_val_sequences": n_val})
        logger.info(f"evotuning {model_name} on {family_fasta.name}: {n_train} train / {n_val} val sequences")
        model_dir = pretrain(
            train_path,
            output_folder,
            model_name,
            epochs,
            max_steps,
            batch_size,
            max_length=max_length,
            mlm_probability=mlm_probability,
            val_corpus=val_path if n_val else None,
            save_steps=save_steps,
            save_total_limit=save_total_limit,
            metrics_path=metrics_path,
            precision=precision,
            dataloader_workers=dataloader_workers,
            model_dir_name=model_dir_name,
        )
        # The one fact whose absence cost the hour: a run that finished and a path to look in.
        mlflow.set_tag("evotuned_model_path", str(model_dir))
        logger.info(f"evotuned model at {model_dir}")


if __name__ == "__main__":
    typer.run(main)
