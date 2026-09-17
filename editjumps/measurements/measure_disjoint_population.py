"""Whether the family members absent from the pairs file are a different population."""
import argparse
import gzip
import random
import statistics as st
from pathlib import Path

from editjumps.core.family_split import usable_members
from editjumps.core.generation_metrics import mean_pairwise_levenshtein
from editjumps.core.sequences import PAIR_SEP
from editjumps.pipeline.preprocess.pretrain.seed_homologs import read_fasta

#: Default sequences drawn per arm, overridable with --sample-size. A pool smaller than this is
#: reported in full rather than sampled.
SAMPLE_SIZE = 200

#: Where the pairs file and the seed families live, relative to the repository root.
PAIRS_PATH = Path("data/pretrain/oas_homolog_pairs.tsv.gz")
FAMILIES = (
    ("Ty1", Path("data/interim/seed_families/Anti-SARS-CoV-2_VHH_Ty1.fasta")),
    ("HER2-VH", Path("data/interim/seed_families/Anti-HER2_scFv_VH_trastuzumab.fasta")),
)


def sequences_in_pairs(pairs_path: Path) -> set[str]:
    """Read every sequence the pairs file mentions, both whole cells and their chain halves.

    Args:
        pairs_path: The gzipped homolog-pairs TSV.

    Returns:
        Every sequence appearing in it, whole cells and chain halves alike.
    """
    seen: set[str] = set()
    with gzip.open(pairs_path, "rt") as handle:
        for line in handle:
            for cell in line.rstrip("\n").split("\t"):
                if cell:
                    seen.add(cell)
                    seen.update(part for part in cell.split(PAIR_SEP) if part)
    return seen


def summarise_pool(label: str, pool: list[str], rng: random.Random,
                   sample_size: int = SAMPLE_SIZE) -> str:
    """Describe one arm: pooled pairwise distance and length spread over a draw from ``pool``.

    A family need not hold ``sample_size`` members. Asking `random.sample` for more than the pool
    holds raises, and below two members there is no pairwise distance and no length spread to
    report, so both cases are answered rather than computed.

    Args:
        label: Name of the arm, printed at the start of the line.
        pool: The sequences this arm draws from.
        rng: Draw source, so a run is reproducible.
        sample_size: Sequences to draw; the whole pool is used when it holds fewer.

    Returns:
        One formatted line, ready to print.
    """
    if len(pool) < 2:
        return f"   {label:<26} {len(pool)} sequences - too few to compare"
    sample = rng.sample(pool, min(sample_size, len(pool)))
    lengths = [len(sequence) for sequence in sample]
    # Say so when an arm is the whole pool rather than a draw: otherwise a 5-member pool reads as
    # if it had been measured the same way as a 200-member one.
    note = "" if len(sample) == sample_size else f"   (whole pool: {len(sample)})"
    return (f"   {label:<26} pooled pairwise Lev {mean_pairwise_levenshtein(sample):7.2f}   "
            f"len {st.mean(lengths):6.1f} +/- {st.stdev(lengths):4.1f}{note}")


def main() -> None:
    """Print the disjoint-vs-in-pairs comparison for each seed family."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-size", type=int, default=SAMPLE_SIZE,
                        help="Sequences drawn per arm; a smaller pool is reported in full")
    parser.add_argument("--pairs", type=Path, default=PAIRS_PATH,
                        help="The homolog-pairs file that defines which members are 'in pairs'")
    args = parser.parse_args()

    seen = sequences_in_pairs(args.pairs)
    for name, path in FAMILIES:
        members = list(dict.fromkeys(usable_members(read_fasta(path).values())))
        disjoint = [m for m in members if m not in seen]
        in_pairs = [m for m in members if m in seen]
        rng = random.Random(0)
        print(f"\n{name}: {len(disjoint)} disjoint, {len(in_pairs)} in pairs")
        for label, pool in (("disjoint (new reference)", disjoint),
                            ("in pairs (old reference)", in_pairs)):
            print(summarise_pool(label, pool, rng, args.sample_size))


if __name__ == "__main__":
    main()
