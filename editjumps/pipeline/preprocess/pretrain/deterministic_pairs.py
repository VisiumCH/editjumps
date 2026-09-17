"""Build EvoFlows §4.1's synthetic training set: (z0, z1) pairs under the deterministic rules. §4.1."""

import gzip
from pathlib import Path
from typing import Annotated

import typer

from editjumps.core.edit_flows.deterministic import EDIT_CLASSES, apply_deterministic_edits, class_counts
from editjumps.core.sequences import AA, PAIR_SEP
from editjumps.core.utils import get_logger

logger = get_logger(__file__)

OUTPUT_NAME = "deterministic_pairs.tsv.gz"


def build_pairs(sequences: list[str]) -> list[tuple[str, str]]:
    """Apply §4.1's rules to each source sequence, dropping any that the rules leave unchanged."""
    pairs: list[tuple[str, str]] = []
    skipped_alphabet = skipped_identity = 0
    for z0 in sequences:
        if set(z0) - AA:
            skipped_alphabet += 1
            continue
        z1 = apply_deterministic_edits(z0)
        if z1 == z0:
            skipped_identity += 1
            continue
        pairs.append((z0, z1))
    if skipped_alphabet:
        logger.info(f"skipped {skipped_alphabet} sequence(s) with non-amino-acid characters")
    if skipped_identity:
        logger.info(f"skipped {skipped_identity} sequence(s) the rules leave unchanged")
    return pairs


def main(
    corpus: Annotated[Path, typer.Option(help="Gzipped source corpus, one sequence per line")] = Path(
        "data/pretrain/oas_corpus.train.txt.gz"
    ),
    output_folder: Annotated[Path, typer.Option()] = Path("data/pretrain"),
    n: Annotated[int, typer.Option(help="How many source sequences to draw (0 = all)")] = 20000,
    chain: Annotated[str, typer.Option(help="heavy | light — which side of the separator to use")] = "heavy",
) -> None:
    """Build the §4.1 synthetic pair file."""
    if chain not in ("heavy", "light"):
        raise ValueError(f"chain={chain!r} must be 'heavy' or 'light'")
    index = 0 if chain == "heavy" else 1

    sequences: list[str] = []
    with gzip.open(corpus, "rt") as handle:
        for line in handle:
            parts = line.strip().split(PAIR_SEP)
            if len(parts) > index and parts[index]:
                sequences.append(parts[index])
            if n and len(sequences) >= n:
                break
    logger.info(f"{len(sequences)} source sequences from {corpus} ({chain} chain)")

    pairs = build_pairs(sequences)
    if not pairs:
        raise ValueError(f"{corpus} produced no usable pairs; check the chain and the alphabet")

    counts = class_counts([z0 for z0, _ in pairs])
    edited = sum(counts[c] for c in EDIT_CLASSES if c != "no_op")
    logger.info(f"{len(pairs)} pairs; ground-truth positions: {counts}")
    logger.info(f"  {edited} edited positions vs {counts['no_op']} no-ops "
                f"({100 * edited / max(1, edited + counts['no_op']):.1f}% edited)")

    output_folder.mkdir(parents=True, exist_ok=True)
    output_path = output_folder / OUTPUT_NAME
    with gzip.open(output_path, "wt") as handle:
        for z0, z1 in pairs:
            handle.write(f"{z0}\t{z1}\n")
    logger.info(f"wrote {output_path}")


if __name__ == "__main__":
    typer.run(main)
