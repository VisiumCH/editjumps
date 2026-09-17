"""Rebuild a loadable editor folder from a training-state ``checkpoint.pt``."""

from pathlib import Path
from typing import Annotated

import typer

from editjumps.core.gcs_checkpoints import gcs_download
from editjumps.core.utils import get_logger, load_params

logger = get_logger(__file__)


def main(
    checkpoint: Annotated[str, typer.Option(help="checkpoint.pt path, or a gs://… URI to download")] = "",
    output_folder: Annotated[Path, typer.Option(help="Destination; gets encoder/ + evoflows_model.pt")] = Path(
        "data/pretrain/edit_flows_restored"
    ),
    model_name: Annotated[str, typer.Option(help="Trunk the run was initialised from")] = (
        "data/pretrain/esm2_oas"
    ),
    rate_head: Annotated[str, typer.Option(help="Override params.yaml; must match the checkpoint")] = "",
    q_head: Annotated[str, typer.Option(help="Override params.yaml; must match the checkpoint")] = "",
    verify: Annotated[bool, typer.Option(help="Reload the result with load_trained before declaring success")] = True,
) -> None:
    """Rebuild ``encoder/`` + ``evoflows_model.pt`` from a training-state checkpoint."""
    from editjumps.pipeline.train.evoflows import EvoFlowsModel, restore_model_folder

    if not checkpoint:
        raise typer.BadParameter("--checkpoint is required (a local checkpoint.pt or a gs://… URI)")

    local = Path(checkpoint)
    if checkpoint.startswith("gs://"):
        local = output_folder.parent / "checkpoint.pt"
        logger.info(f"downloading {checkpoint} -> {local}")
        if not gcs_download(checkpoint, local):
            raise typer.BadParameter(f"could not download {checkpoint} (gcs_download is best-effort; see the warning)")
    if not local.exists():
        raise typer.BadParameter(f"{local} does not exist")

    # Overridable because arms were trained with different heads while params.yaml holds one value:
    # the strict load_state_dict only protects a caller who can say which arm this checkpoint is.
    edit_flows = (load_params() or {}).get("edit_flows") or {}
    rate_head = rate_head or str(edit_flows.get("rate_head", "linear"))
    q_head = q_head or str(edit_flows.get("q_head", "fresh"))
    logger.info(f"restoring with rate_head={rate_head} q_head={q_head} from trunk {model_name}")

    step = restore_model_folder(local, output_folder, model_name, rate_head=rate_head, q_head=q_head)
    logger.info(f"wrote {output_folder}/encoder and {output_folder}/evoflows_model.pt (checkpoint step {step})")

    # This module exists because a run produced an artefact nothing could open; reporting success
    # without opening it would repeat that.
    if verify:
        EvoFlowsModel.load_trained(output_folder, rate_head=rate_head, q_head=q_head)
        logger.info(f"verified: load_trained({output_folder}) succeeds")


if __name__ == "__main__":
    typer.run(main)
