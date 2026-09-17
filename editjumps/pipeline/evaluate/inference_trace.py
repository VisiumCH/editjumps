"""Print the SAMPLED inference path a trained model walks from a real template, step by step."""

import gzip
import random
from pathlib import Path
from typing import Annotated

import typer

from editjumps.core.edit_flows.alignment import levenshtein
from editjumps.core.edit_flows.metrics import edit_ops
from editjumps.core.sequences import PAIR_SEP
from editjumps.core.utils import get_logger

logger = get_logger(__file__)

#: The three seed sequences a ``--dataset`` name resolves to.
DATASETS: dict[str, str] = {
    "ty1": "data/interim/seed_families/Anti-SARS-CoV-2_VHH_Ty1.fasta",
    "her2vh": "data/interim/seed_families/Anti-HER2_scFv_VH_trastuzumab.fasta",
    "corpus": "data/pretrain/oas_homolog_pairs.tsv.gz",
}

#: so the last interval emits the predicted endpoint and everything before it is an interpolant.
FRACTIONS: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99, 1.0)


def read_template(dataset: str, index: int = 0) -> str:
    """Return one starting sequence from one of `DATASETS`."""
    path = Path(DATASETS.get(dataset, dataset))
    if path.name.endswith((".tsv.gz", ".tsv")):
        with gzip.open(path, "rt") if path.name.endswith(".gz") else path.open() as handle:
            for i, line in enumerate(handle):
                if i == index:
                    return line.rstrip("\n").partition("\t")[0].split(PAIR_SEP)[0]
        raise ValueError(f"{path} has no line {index}")
    records, current = [], ""
    with open(path) as handle:
        for line in handle:
            if line.startswith(">"):
                if current:
                    records.append(current)
                    if len(records) > index:
                        return records[index]
                current = ""
            else:
                current += line.strip()
    records.append(current)
    if len(records) <= index:
        raise ValueError(f"{path} has no record {index}")
    return records[index]


def pick_rows(count: int, fractions: "tuple[float, ...]" = FRACTIONS) -> list[int]:
    """Return the snapshot indices to print, one per fraction, de-duplicated and in order."""
    if count <= 0:
        return []
    wanted = sorted({min(count - 1, int(round(f * (count - 1)))) for f in fractions})
    return wanted


def format_rows(
    template: str,
    snapshots: "list[tuple[float, str]]",
    fractions: "tuple[float, ...]" = FRACTIONS,
) -> list[str]:
    """Turn ``(t, sequence)`` snapshots into the printable rows, one per selected snapshot."""
    width = max([len(template)] + [len(s) for _, s in snapshots]) + 2
    rows = [f"{'template':>10}  L={len(template):3d}  {template}"]
    for index in pick_rows(len(snapshots), fractions):
        t, sequence = snapshots[index]
        ops = edit_ops(template, sequence)
        rows.append(
            f"  t={t:<7.4f} L={len(sequence):3d}  {sequence:<{width}}"
            f"d={levenshtein(sequence, template):3d}  "
            f"sub={ops['substitutions']:3d} ins={ops['insertions']:3d} del={ops['deletions']:3d}"
        )
    return rows


def trace_editor(
    model_folder: Path,
    template: str,
    n_steps: int = 50,
    seed: int = 0,
    device: str = "",
    rate_head: str = "mlp",
    q_head: str = "esm_lm_head",
    clock: float | None = 40.0,
    gillespie: bool = False,
) -> "tuple[list[tuple[float, str]], dict]":
    """Sample one editor trajectory from ``template`` and return every rate evaluation's sequence."""
    # Deferred, as everywhere a pipeline module reaches the train extras: the lean `build` CI job
    # type-checks without them and `editjumps --help` must work without them installed.
    from transformers import AutoTokenizer  # ty: ignore[unresolved-import]

    from editjumps.pipeline.train.evoflows import (
        EvoFlowsModel,
        pick_device,
        sample_edits,
        sample_edits_gillespie,
    )

    torch_device = device or pick_device()
    model = EvoFlowsModel.load_trained(model_folder, rate_head=rate_head, q_head=q_head).to(torch_device)
    tokenizer = AutoTokenizer.from_pretrained(str(Path(model_folder) / "encoder"))
    # Same two lines `generation_eval`'s editor path uses, so the trace tokenises the template the way the.
    ids = list(tokenizer(template)["input_ids"])  # ty: ignore[call-non-callable]

    def decode(tokens: "list[int]") -> str:
        """Decode token ids back to residues, dropping the special tokens the sampler protects."""
        text = tokenizer.decode(tokens, skip_special_tokens=True)  # ty: ignore[unresolved-attribute]
        return str(text).replace(" ", "")

    snapshots: list[tuple[float, str]] = []

    def tap(x: "list[int]", t: float) -> None:
        """Record the state a rate evaluation sees."""
        snapshots.append((t, decode(list(x))))

    sampler = sample_edits_gillespie if gillespie else sample_edits
    result = sampler(
        model,
        list(ids),
        n_steps=n_steps,
        rng=random.Random(seed),
        clock=clock,
        observe=tap,
        provenance=True,
    )
    final, origins = result if isinstance(result, tuple) else (result, [])
    snapshots.append((1.0, decode(list(final))))
    provenance = {
        "method": "edit_flows_gillespie" if gillespie else "edit_flows",
        "n_steps": n_steps,
        "clock": clock,
        "rate_head": rate_head,
        "q_head": q_head,
        "device": str(torch_device),
        "seed": seed,
        "rate_evaluations": len(snapshots) - 1,
        "ops_provenance": provenance_ops(list(ids), list(final), list(origins)),
    }
    return snapshots, provenance


def provenance_ops(input_ids: "list[int]", output_ids: "list[int]", origins: "list[int | None]") -> dict[str, int]:
    """Return the editor sampler's OWN op counts, read off its per-token provenance."""
    if not origins:
        return {}
    substitutions = sum(
        1
        for token, origin in zip(output_ids, origins, strict=True)
        if origin is not None and token != input_ids[origin]
    )
    insertions = sum(1 for origin in origins if origin is None)
    survivors = {origin for origin in origins if origin is not None}
    # Position 0 is BOS, which the sampler protects from deletion, so it is not a survivor to count.
    deletions = len([i for i in range(1, len(input_ids)) if i not in survivors])
    ops = {"substitutions": substitutions, "insertions": insertions, "deletions": deletions}
    ops["total"] = sum(ops.values())
    return ops


def main(
    method: Annotated[str, typer.Option(help="editor | editor-gillespie")] = "editor",
    model_folder: Annotated[Path, typer.Option(help="Checkpoint folder for the chosen method")] = Path(
        "data/pretrain/eval_B_stock_appA"
    ),
    dataset: Annotated[str, typer.Option(help=f"{' | '.join(DATASETS)}, or a FASTA/pairs path")] = "ty1",
    index: Annotated[int, typer.Option(help="Which entry of --dataset to start from")] = 0,
    template: Annotated[str, typer.Option(help="Start from this sequence instead of --dataset")] = "",
    seed: Annotated[int, typer.Option()] = 0,
    device: Annotated[str, typer.Option(help="Torch device; empty picks the best available")] = "",
    rate_head: Annotated[str, typer.Option(help="editor: linear | mlp; must match the checkpoint")] = "mlp",
    q_head: Annotated[str, typer.Option(help="editor: fresh | esm_lm_head; must match")] = "esm_lm_head",
    clock: Annotated[float, typer.Option(help="editor: §3.3 clock; 0 runs unclocked")] = 40.0,
    n_steps: Annotated[int, typer.Option(help="Sampler steps; 0 uses the method default")] = 0,
) -> None:
    """Print the sampled inference path a trained model walks from a real template."""
    start = template or read_template(dataset, index)
    if not template:
        logger.info(f"template {index} of {DATASETS.get(dataset, dataset)}, L={len(start)}")
    if method in ("editor", "editor-gillespie"):
        snapshots, provenance = trace_editor(
            model_folder, start, n_steps=n_steps or 50, seed=seed, device=device,
            rate_head=rate_head, q_head=q_head, clock=clock or None,
            gillespie=method == "editor-gillespie",
        )
    else:
        raise typer.BadParameter(f"--method must be editor or editor-gillespie, got {method!r}")
    for row in format_rows(start, snapshots):
        print(row)
    print("  " + "  ".join(f"{key}={value}" for key, value in provenance.items() if key != "ops_provenance"))
    if provenance.get("ops_provenance"):
        exact = provenance["ops_provenance"]
        print(
            f"  sampler's own op counts (exact, not a re-alignment): "
            f"sub={exact['substitutions']} ins={exact['insertions']} del={exact['deletions']}"
        )


if __name__ == "__main__":
    typer.run(main)
