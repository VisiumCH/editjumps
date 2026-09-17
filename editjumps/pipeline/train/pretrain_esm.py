"""Continue-pretrain an ESM-2 model (MLM) on the OAS antibody corpus."""

import math
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import Annotated

import mlflow
import typer

from editjumps.core.utils import design_space_tags, get_logger, start_mlflow_run, write_metrics

logger = get_logger(__file__)

# Largest single file MLflow will be asked to upload.
MLFLOW_ARTIFACT_FILE_LIMIT = 24_000_000

EXPERIMENT_NAME = "editjumps-esm2-pretrain"


def pretrain(
    corpus: Path,
    output_folder: Path,
    model_name: str = "facebook/esm2_t12_35M_UR50D",
    epochs: float = 1.0,
    max_steps: int = -1,
    batch_size: int = 16,
    max_length: int = 280,
    max_seqs: int = 0,
    mlm_probability: float = 0.15,
    val_corpus: Path | None = None,
    save_steps: int = 500,
    save_total_limit: int = 3,
    resume: bool = True,
    metrics_path: Path | None = None,
    checkpoint_uri: str | None = None,
    precision: str = "auto",
    dataloader_workers: int = 0,
    model_dir_name: str = "esm2_oas",
) -> Path:
    """Continue-pretrain an ESM-2 model with MLM on the OAS corpus."""
    # Lazy, so the `editjumps` CLI works without the optional `train` deps that only training needs.
    import torch
    from datasets import load_dataset
    from transformers import (
        AutoModelForMaskedLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainerCallback,
        TrainingArguments,
    )
    from transformers.integrations import MLflowCallback
    from transformers.trainer_utils import get_last_checkpoint

    from editjumps.core.gcs_checkpoints import gcs_download, gcs_list, gcs_upload, latest_checkpoint

    logger.info(f"loading tokenizer + model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    assert tokenizer is not None  # narrow Optional return of from_pretrained
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    n_params = sum(p.numel() for p in model.parameters())

    logger.info(f"loading corpus: {corpus}")
    data_files = {"train": str(corpus)}
    if val_corpus is not None and val_corpus.exists():
        data_files["validation"] = str(val_corpus)
        logger.info(f"loading held-out validation corpus: {val_corpus}")
    else:
        logger.warning("no validation corpus provided/found - training without eval loss")
    ds = load_dataset("text", data_files=data_files)
    if max_seqs:
        ds["train"] = ds["train"].select(range(min(max_seqs, len(ds["train"]))))
        logger.info(f"subset train to {len(ds['train'])} sequences (max_seqs={max_seqs})")

    def tokenize(batch: dict) -> dict:
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    ds = ds.map(tokenize, batched=True, remove_columns=["text"])
    val_note = f", {len(ds['validation'])} val" if "validation" in ds else ""
    logger.info(f"tokenized {len(ds['train'])} train sequences{val_note}")

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=mlm_probability)
    has_val = "validation" in ds
    checkpoints_dir = output_folder / "checkpoints"
    # Resolved here, not passed through, so `auto` logs what it picked: bf16 needs CUDA and Ampere+,
    # and an unlogged fp32 fallback on an A100 is the ~8x cost described above.
    if precision == "auto":
        use_bf16 = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    elif precision in ("bf16", "fp32"):
        use_bf16 = precision == "bf16"
    else:
        raise ValueError(f"precision={precision!r}; options: auto, bf16, fp32")
    get_logger(__file__).info(
        f"precision={precision} -> bf16={use_bf16}; dataloader_workers={dataloader_workers}"
    )

    args = TrainingArguments(
        output_dir=str(checkpoints_dir),
        num_train_epochs=epochs,
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        logging_steps=50,
        eval_strategy="steps" if has_val else "no",
        eval_steps=save_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        dataloader_num_workers=dataloader_workers,
        bf16=use_bf16,
        report_to=[],  # MLflow logging is wired explicitly below via MLflowCallback
    )
    # Best-effort mirror of every Trainer checkpoint; only when checkpoint_uri is set, else no calls.
    class GCSCheckpointCallback(TrainerCallback):
        """Upload the newest local ``checkpoint-<N>`` dir to ``checkpoint_uri`` after each save."""

        def on_save(self, args: object, state: object, control: object, **kwargs: object) -> None:
            """Mirror the just-written checkpoint to GCS (best-effort; never raises)."""
            names = [p.name for p in checkpoints_dir.iterdir() if p.is_dir()] if checkpoints_dir.exists() else []
            latest = latest_checkpoint(names)
            if latest:
                gcs_upload(checkpoints_dir / latest, f"{checkpoint_uri.rstrip('/')}/{latest}")

    callbacks: list[TrainerCallback] = [MLflowCallback()]
    if checkpoint_uri:
        callbacks.append(GCSCheckpointCallback())
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"] if has_val else None,
        data_collator=collator,
        callbacks=callbacks,
    )

    # Pull the latest remote checkpoint in first, so the get_last_checkpoint path below resumes it.
    if checkpoint_uri:
        latest = latest_checkpoint(gcs_list(checkpoint_uri))
        if latest:
            checkpoints_dir.mkdir(parents=True, exist_ok=True)
            if gcs_download(f"{checkpoint_uri.rstrip('/')}/{latest}", checkpoints_dir / latest):
                logger.info(f"pulled remote checkpoint {latest} from {checkpoint_uri}")

    resume_from_checkpoint = None
    if resume and checkpoints_dir.exists():
        resume_from_checkpoint = get_last_checkpoint(str(checkpoints_dir))
        if resume_from_checkpoint:
            logger.info(f"resuming from checkpoint: {resume_from_checkpoint}")
        else:
            logger.info(f"no checkpoint found under {checkpoints_dir}; starting fresh")

    # A caller that already opened a run owns it: `evotune` wraps this function to MLM-adapt one homolog.
    active = mlflow.active_run()
    if active is not None:
        logger.info(f"logging into the caller's MLflow run {active.info.run_id}")
        run_context: "AbstractContextManager[object]" = nullcontext()
    else:
        # This run's coordinate in the modeling design space (see ArmConfig / params.yaml), for filtering.
        tags = design_space_tags(
            objective="mlm",
            backbone=Path(model_name).name,
            weight_init="pretrained",
            pretrain_data=corpus.name,
            chain_handling="combine",  # VH.VL joined; "separate" is a planned design-space option
        )
        run_context = start_mlflow_run(EXPERIMENT_NAME, run_name=f"{Path(model_name).name}-{corpus.stem}", tags=tags)
    with run_context:
        mlflow.log_params(
            {
                "model_name": model_name,
                "n_params": n_params,
                "corpus": str(corpus),
                "val_corpus": str(val_corpus) if val_corpus else None,
                "n_train_seqs": len(ds["train"]),
                "n_val_seqs": len(ds["validation"]) if has_val else 0,
                "epochs": epochs,
                "max_steps": max_steps,
                "batch_size": batch_size,
                "max_length": max_length,
                "max_seqs": max_seqs,
                "mlm_probability": mlm_probability,
                "save_steps": save_steps,
                "save_total_limit": save_total_limit,
                "resumed_from_checkpoint": resume_from_checkpoint,
            }
        )

        logger.info("starting MLM continue-pretraining ...")
        train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        metrics_out = {"final_train_loss": train_result.training_loss}
        mlflow.log_metrics({"final_train_loss": train_result.training_loss})

        if has_val:
            eval_metrics = trainer.evaluate()
            numeric_eval = {f"final_{k}": v for k, v in eval_metrics.items() if isinstance(v, int | float)}
            mlflow.log_metrics(numeric_eval)
            metrics_out.update(numeric_eval)
            logger.info(f"final eval metrics: {eval_metrics}")

        # Perplexity is the MLM readout people compare across corpora and model sizes, and it is exp(loss) —.
        mlflow.log_metrics({
            key.removesuffix("loss") + "perplexity": math.exp(value)
            for key, value in metrics_out.items() if key.endswith("_loss") and value < 20
        })

        model_dir = output_folder / model_dir_name
        trainer.save_model(str(model_dir))
        tokenizer.save_pretrained(str(model_dir))
        logger.info(f"saved continue-pretrained model to {model_dir}")

        # Headline metrics as a small JSON so `dvc metrics show`/`diff` work without the MLflow store.
        if metrics_path is not None:
            metrics_path = Path(metrics_path)
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            write_metrics(metrics_path, metrics_out)
            logger.info(f"wrote DVC metrics file to {metrics_path}")

        # The model itself goes to MLflow only when the store can take it.
        largest = max((f.stat().st_size for f in model_dir.rglob("*") if f.is_file()), default=0)
        if largest > MLFLOW_ARTIFACT_FILE_LIMIT:
            logger.info(
                f"skipping MLflow artifact upload: largest file is {largest / 1e6:.0f} MB, over the "
                f"{MLFLOW_ARTIFACT_FILE_LIMIT / 1e6:.0f} MB limit a remote tracking server accepts; "
                f"the model is at {model_dir} and is copied to GCS by the job"
            )
            mlflow.set_tag("model_artifact_skipped_mb", f"{largest / 1e6:.0f}")
            # Where it went, not only that it did not come here. A run recording a skip and
            # no destination is a run whose weights nobody can find.
            mlflow.set_tag("model_local_path", str(model_dir))
        else:
            mlflow.log_artifacts(str(model_dir), artifact_path=model_dir_name)

    return model_dir


def main(
    corpus: Annotated[Path, typer.Option()] = Path("data/pretrain/oas_corpus.train.txt.gz"),
    val_corpus: Annotated[
        Path | None, typer.Option(help="Held-out corpus for eval loss (from split_corpus)")
    ] = Path("data/pretrain/oas_corpus.val.txt.gz"),
    output_folder: Annotated[Path, typer.Option()] = Path("data/pretrain"),
    model_name: Annotated[str, typer.Option(help="Base ESM-2 checkpoint")] = "facebook/esm2_t12_35M_UR50D",
    epochs: Annotated[float, typer.Option()] = 1.0,
    max_steps: Annotated[int, typer.Option(help="Cap steps for a smoke test; -1 = full epochs")] = -1,
    batch_size: Annotated[int, typer.Option()] = 16,
    max_length: Annotated[int, typer.Option(help="Truncation length; ~280 for joined VH.VL pairs")] = 280,
    max_seqs: Annotated[int, typer.Option(help="Train on only the first N sequences (0 = all)")] = 0,
    mlm_probability: Annotated[float, typer.Option(help="Fraction of tokens masked for MLM")] = 0.15,
    save_steps: Annotated[
        int, typer.Option(help="Checkpoint every N steps (long runs need this, not just epoch-end)")
    ] = 500,
    save_total_limit: Annotated[int, typer.Option(help="Keep only the N most recent step checkpoints")] = 3,
    resume: Annotated[
        bool, typer.Option(help="Resume from the last checkpoint under output_folder/checkpoints if one exists")
    ] = True,
    metrics_path: Annotated[
        Path | None,
        typer.Option(
            help="Where to write the DVC metrics JSON (default: don't write one)."
            " Pass an explicit path (e.g. dvc.yaml's metrics/pretrain_esm.json) so ad hoc"
            " runs can never clobber the real pipeline's tracked metrics file by defaulting to it"
        ),
    ] = None,
    checkpoint_uri: Annotated[
        str | None,
        typer.Option(help="gs://… prefix to mirror checkpoints to + resume from (default: off, no GCS calls)"),
    ] = None,
    precision: Annotated[
        str, typer.Option(help="auto (bf16 where supported) | bf16 | fp32; fp32 on A100 is ~8x slower")
    ] = "auto",
    dataloader_workers: Annotated[int, typer.Option(help="Dataloader worker processes (0 = in-process)")] = 0,
) -> None:
    """Run ESM-2 continue-pretraining on the OAS corpus."""
    import yaml

    from editjumps.core.gcs_checkpoints import resolve_checkpoint_uri

    params = yaml.safe_load(Path("params.yaml").read_text()) if Path("params.yaml").exists() else {}
    pretrain(
        corpus,
        output_folder,
        model_name,
        epochs,
        max_steps,
        batch_size,
        max_length=max_length,
        max_seqs=max_seqs,
        mlm_probability=mlm_probability,
        val_corpus=val_corpus,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        resume=resume,
        metrics_path=metrics_path,
        checkpoint_uri=resolve_checkpoint_uri(checkpoint_uri, params, "pretrain"),
        precision=precision,
        dataloader_workers=dataloader_workers,
    )


if __name__ == "__main__":
    typer.run(main)
