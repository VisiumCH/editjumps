"""How often Edit Flows' two readings of the delete branch disagree on real data."""
import gzip
import random
import statistics
from pathlib import Path

from editjumps.core.edit_flows.alignment import needleman_wunsch
from editjumps.core.edit_flows.path import EPS, mixture_path
from editjumps.core.edit_flows.targets import edit_targets

PAIRS = Path("data/pretrain/oas_homolog_pairs.tsv.gz")
N_PAIRS, N_TIMES, SEED = 400, 5, 0


BOS = 0


def encode(seq: str) -> list[int]:
    """Tokenise a sequence as ord() codes with a BOS, the form edit_targets requires."""
    # BOS prepended so an insertion at the very start attaches at index >= 0, which is what
    # edit_targets requires and what the real tokenizer provides.
    return [BOS] + [ord(c) for c in seq.strip()]


def read_pairs(path: Path, n_pairs: int = N_PAIRS) -> list[tuple[str, str]]:
    """Read up to ``n_pairs`` two-column rows out of the pairs file.

    Args:
        path: The gzipped homolog-pairs TSV.
        n_pairs: Stop after this many usable rows.

    Returns:
        ``(source, target)`` pairs; rows that are not two non-empty columns are skipped.
    """
    pairs: list[tuple[str, str]] = []
    with gzip.open(path, "rt") as handle:
        for line in handle:
            cells = line.rstrip("\n").split("\t")
            if len(cells) == 2 and cells[0] and cells[1]:
                pairs.append((cells[0], cells[1]))
            if len(pairs) >= n_pairs:
                break
    return pairs


def main() -> None:
    """Print how much of Fig 13's delete supervision eq 23 does not produce."""
    rng = random.Random(SEED)
    pairs = read_pairs(PAIRS)
    # Every summary below divides by a count from these rows; with none, it is a ZeroDivisionError.
    if not pairs:
        raise SystemExit(f"{PAIRS} yielded no usable two-column rows; nothing to measure")

    real_edits, spurious, eps_eps_cols, total_cols, examples_affected = [], [], 0, 0, 0
    for source, target in pairs:
        z0, z1 = needleman_wunsch(encode(source), encode(target))
        for _ in range(N_TIMES):
            kappa = rng.random()
            z_t = mixture_path(z0, z1, kappa=kappa, rng=rng)
            # eq 23's reading, which is what the code implements.
            ours = edit_targets(z_t, z1)
            # Fig 13's extra deletions: columns where z_t already took the gap AND z_1 is a gap.
            extra = sum(1 for a, b in zip(z_t, z1) if a == EPS and b == EPS)
            real_edits.append(len(ours))
            spurious.append(extra)
            eps_eps_cols += extra
            total_cols += len(z_t)
            examples_affected += 1 if extra else 0

    n = len(real_edits)
    print(f"pairs {len(pairs)}, times per pair {N_TIMES}, training examples {n}")
    print(f"aligned columns inspected      {total_cols}")
    print(f"(EPS, EPS) columns             {eps_eps_cols}  ({100*eps_eps_cols/total_cols:.3f}% of columns)")
    print(f"real edits per example         mean {statistics.mean(real_edits):.2f}")
    print(f"spurious deletes per example   mean {statistics.mean(spurious):.3f}, max {max(spurious)}")
    print(f"examples with >=1 spurious     {examples_affected}/{n} ({100*examples_affected/n:.1f}%)")
    share = sum(spurious) / max(1, sum(real_edits) + sum(spurious))
    print(f"\nshare of Fig-13 supervision that is spurious: {100*share:.2f}%")


if __name__ == "__main__":
    main()
