"""What the ② alignment scoring does to the training label distribution (EvoFlows §3.2)."""

import gzip
import random
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Annotated

import typer

from editjumps.core.edit_flows.alignment import (
    UNIT_SCORING,
    AffineScoring,
    blosum62_scoring,
    needleman_wunsch,
    needleman_wunsch_affine,
)
from editjumps.core.edit_flows.path import (
    EPS,
)
from editjumps.core.sequences import AA
from editjumps.core.utils import get_logger, write_metrics

logger = get_logger(__file__)

#: Stand-in token ids (see the module docstring): ``ord(residue)`` for residues, with two sentinels standing in.
BOS_ID = 1
EOS_ID = 2
VOCAB_SIZE = 128

#: The vocab handed to `blosum62_scoring`.
RESIDUE_TOKENS: dict[str, int] = {residue: ord(residue) for residue in sorted(AA)}

#: The mutation types Table 1 scores, plus the no-op class it also reports.
OP_CLASSES = ("no_op", "substitution", "insertion", "deletion")

#: One alignment call under one scoring: ``(ids_a, ids_b) -> (z_0, z_1)``.
Aligner = Callable[[list[int], list[int]], tuple[list[int], list[int]]]


def encode(sequence: str) -> list[int]:
    """Tokenize one ``VH.VL`` corpus line into stand-in ids with BOS/EOS."""
    return [BOS_ID, *(ord(char) for char in sequence), EOS_ID]


def column_counts(z_0: list[int], z_1: list[int]) -> dict[str, int]:
    """Count the four column types of an aligned pair, plus gap-run structure."""
    counts = {name: 0 for name in OP_CLASSES}
    gap_runs = 0
    in_run = False
    for token_0, token_1 in zip(z_0, z_1, strict=True):
        if token_0 == EPS:
            counts["insertion"] += 1
        elif token_1 == EPS:
            counts["deletion"] += 1
        elif token_0 == token_1:
            counts["no_op"] += 1
        else:
            counts["substitution"] += 1
        is_gap = token_0 == EPS or token_1 == EPS
        if is_gap and not in_run:
            gap_runs += 1
        in_run = is_gap
    counts["columns"] = len(z_0)
    counts["gap_runs"] = gap_runs
    return counts


def op_mix(alignments: Iterator[tuple[list[int], list[int]]], n_pairs: int) -> dict[str, float]:
    """Aggregate column counts over many alignments into label proportions."""
    totals = {name: 0 for name in (*OP_CLASSES, "columns", "gap_runs")}
    for z_0, z_1 in alignments:
        for name, value in column_counts(z_0, z_1).items():
            totals[name] += value
    columns = totals["columns"] or 1
    gaps = totals["insertion"] + totals["deletion"]
    report = {name: totals[name] / columns for name in OP_CLASSES}
    report["mean_columns"] = totals["columns"] / n_pairs
    report["mean_edits"] = (gaps + totals["substitution"]) / n_pairs
    report["mean_gap_runs"] = totals["gap_runs"] / n_pairs
    report["mean_gap_run_length"] = gaps / totals["gap_runs"] if totals["gap_runs"] else 0.0
    report["n_pairs"] = n_pairs
    return report


def read_pairs(pairs_path: Path, n_pairs: int, seed: int) -> list[tuple[str, str]]:
    """Reservoir-sample homolog pairs from the gzipped pair file."""
    if not pairs_path.exists():
        raise FileNotFoundError(
            f"no homolog pairs at {pairs_path}.\n\n"
            "`data/` is git-ignored and DVC-backed, so a fresh clone has no pair file. Either:\n"
            "  uv run dvc pull data/pretrain/oas_homolog_pairs.tsv.gz   # needs the private remote\n"
            "  uv run dvc repro build_homolog_pairs                     # rebuild from public OAS\n\n"
            "docs/training.md builds a small one locally in about a minute."
        )
    rng = random.Random(seed)
    reservoir: list[tuple[str, str]] = []
    with gzip.open(pairs_path, "rt") as handle:
        for index, line in enumerate(handle):
            if "\t" not in line:
                continue
            x_0, x_1 = line.rstrip("\n").split("\t")[:2]
            if len(reservoir) < n_pairs:
                reservoir.append((x_0, x_1))
            else:
                slot = rng.randint(0, index)
                if slot < n_pairs:
                    reservoir[slot] = (x_0, x_1)
    logger.info(f"sampled {len(reservoir)} of {index + 1} pairs from {pairs_path}")
    return reservoir


def random_pairs(n_pairs: int, alphabet_size: int, min_len: int, max_len: int,
                 seed: int) -> list[tuple[list[int], list[int]]]:
    """Random short id pairs, the setting the PR #60 review measured its asymmetry rate on."""
    rng = random.Random(seed)
    pairs: list[tuple[list[int], list[int]]] = []
    for _ in range(n_pairs):
        a = [rng.randrange(alphabet_size) for _ in range(rng.randint(min_len, max_len))]
        b = [rng.randrange(alphabet_size) for _ in range(rng.randint(min_len, max_len))]
        pairs.append((a, b))
    return pairs


def asymmetry_rate(pairs: list[tuple[list[int], list[int]]], align: "Aligner") -> dict[str, float]:
    """How often aligning a pair the other way round does not give the mirrored alignment. ``align(a, b)."""
    n_asymmetric = 0
    column_delta = 0
    for a, b in pairs:
        forward_0, forward_1 = align(a, b)
        reverse_0, reverse_1 = align(b, a)
        if (forward_0, forward_1) != (reverse_1, reverse_0):
            n_asymmetric += 1
            column_delta += abs(len(forward_0) - len(reverse_0))
    return {
        "rate": n_asymmetric / len(pairs),
        "n_asymmetric": n_asymmetric,
        "n_pairs": len(pairs),
        "mean_abs_column_delta": column_delta / n_asymmetric if n_asymmetric else 0.0,
    }


def scorings(gap_opens: list[float], gap_extend: float) -> dict[str, AffineScoring]:
    """Build the BLOSUM62 scorings to compare, keyed by label."""
    return {
        f"blosum62_open{open_gap:g}_ext{gap_extend:g}": blosum62_scoring(
            RESIDUE_TOKENS, VOCAB_SIZE, gap_open=open_gap, gap_extend=gap_extend
        )
        for open_gap in gap_opens
    }


def main(
    pairs_path: Annotated[Path, typer.Option()] = Path("data/pretrain/oas_homolog_pairs.tsv.gz"),
    n_pairs: Annotated[int, typer.Option(help="Homolog pairs to sample (pure-Python DP: ~50 ms/pair).")] = 5000,
    gap_opens: Annotated[str, typer.Option(help="BLOSUM62 gap-open sweep; the first is the headline.")] = "11,5,1",
    gap_extend: Annotated[float, typer.Option()] = 1.0,
    n_random: Annotated[int, typer.Option(help="Random short pairs for the asymmetry reproduction.")] = 3000,
    random_alphabet: Annotated[int, typer.Option()] = 4,
    random_min_len: Annotated[int, typer.Option()] = 4,
    random_max_len: Annotated[int, typer.Option()] = 10,
    seed: Annotated[int, typer.Option()] = 0,
    metrics_path: Annotated[Path, typer.Option()] = Path("metrics/alignment_scoring.json"),
) -> None:
    """Measure the op mix and the direction-dependence of every ② alignment scoring."""
    opens = [float(value) for value in gap_opens.split(",")]
    blosum = scorings(opens, gap_extend)
    headline = next(iter(blosum))
    try:
        pairs = read_pairs(pairs_path, n_pairs, seed)
    except FileNotFoundError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from None
    encoded = [(encode(x_0), encode(x_1)) for x_0, x_1 in pairs]

    # ① op mix, per scoring. `unit_asdrawn` is the live default: legacy aligner, pair in whatever order the.
    aligners: dict[str, Aligner] = {
        "unit_asdrawn": needleman_wunsch,
        "unit_symmetric": lambda a, b: needleman_wunsch_affine(a, b, UNIT_SCORING, symmetric=True),
        **{
            f"{name}_symmetric": (lambda a, b, scoring=scoring: needleman_wunsch_affine(a, b, scoring))
            for name, scoring in blosum.items()
        },
    }
    mixes: dict[str, dict[str, float]] = {}
    asymmetries: dict[str, dict[str, float]] = {}
    for name, align in aligners.items():
        mix = op_mix((align(a, b) for a, b in encoded), len(encoded))
        mixes[name] = mix
        logger.info(
            f"{name:34s} no-op={mix['no_op']:.4f} sub={mix['substitution']:.4f} "
            f"ins={mix['insertion']:.4f} del={mix['deletion']:.4f} "
            f"edits/pair={mix['mean_edits']:.2f} gap-runs/pair={mix['mean_gap_runs']:.2f} "
            f"run-len={mix['mean_gap_run_length']:.2f}"
        )
    reversed_mix = op_mix((needleman_wunsch(b, a) for a, b in encoded), len(encoded))
    swapped = dict(reversed_mix)
    swapped["insertion"], swapped["deletion"] = reversed_mix["deletion"], reversed_mix["insertion"]
    mixes["unit_reversed_relabelled"] = swapped
    logger.info(
        f"{'unit_reversed_relabelled':34s} no-op={swapped['no_op']:.4f} sub={swapped['substitution']:.4f} "
        f"ins={swapped['insertion']:.4f} del={swapped['deletion']:.4f} "
        f"edits/pair={swapped['mean_edits']:.2f}"
    )

    # ② asymmetry: default and BLOSUM reading, with and without canonical orientation, on real
    # homolog pairs and on the review's random short pairs.
    synthetic = random_pairs(n_random, random_alphabet, random_min_len, random_max_len, seed)
    probes: dict[str, Aligner] = {
        "unit_asdrawn": needleman_wunsch,
        "unit_symmetric": lambda a, b: needleman_wunsch_affine(a, b, UNIT_SCORING, symmetric=True),
        f"{headline}_asdrawn": lambda a, b: needleman_wunsch_affine(a, b, blosum[headline], symmetric=False),
        f"{headline}_symmetric": lambda a, b: needleman_wunsch_affine(a, b, blosum[headline]),
    }
    for name, align in probes.items():
        for population, tested in (("homolog", encoded), ("random", synthetic)):
            stats = asymmetry_rate(tested, align)
            asymmetries[f"{name}_{population}"] = stats
            logger.info(
                f"asymmetry {name:34s} {population:8s} {stats['n_asymmetric']}/{stats['n_pairs']} "
                f"= {stats['rate']:.4f} (mean |Δcolumns| {stats['mean_abs_column_delta']:.2f})"
            )

    report = {
        "n_pairs": len(pairs),
        "n_random_pairs": n_random,
        "gap_extend": gap_extend,
        "gap_opens": opens,
        "seed": seed,
        "random_pairs": {"alphabet": random_alphabet, "min_len": random_min_len, "max_len": random_max_len},
        "op_mix": mixes,
        "asymmetry": asymmetries,
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    write_metrics(metrics_path, report)
    logger.info(f"wrote {metrics_path}")


if __name__ == "__main__":
    typer.run(main)
