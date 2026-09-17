"""Analyze CDR and framework mutations."""

import argparse
import math
from collections import Counter
from pathlib import Path

from editjumps.core.cdr import IMGT_CDR_RANGES, imgt_positions

#: IMGT positions of the intradomain disulfide. Both must be cysteine in a well-formed V domain.
CANONICAL_CYS = (23, 104)
#: How the paper's §4.2 reference is cut: dedup, shuffle with this seed, skip the templates.
PAPER_SEED, PAPER_TEMPLATES, PAPER_REFERENCE_N = 0, 20, 200


def read_fasta(path: Path, only_set: str | None = None) -> list[str]:
    """Read sequences from a FASTA, optionally keeping only one named set. ``write_generated_fasta`` puts."""
    records: list[str] = []
    current: list[str] = []
    keep = True
    for line in path.read_text().splitlines():
        if line.startswith(";"):
            continue
        if line.startswith(">"):
            if current and keep:
                records.append("".join(current))
            current = []
            keep = only_set is None or line[1:].startswith(only_set)
        elif line.strip():
            current.append(line.strip())
    if current and keep:
        records.append("".join(current))
    return records


def paper_reference(family_fasta: Path) -> list[str]:
    """Cut the 200-sequence natural reference the paper scores against."""
    import random

    unique = list(dict.fromkeys(read_fasta(family_fasta)))
    random.Random(PAPER_SEED).shuffle(unique)
    return unique[PAPER_TEMPLATES:PAPER_TEMPLATES + PAPER_REFERENCE_N]


def imgt_map(sequence: str, chain: str) -> dict[int, str] | None:
    """Map IMGT position to residue for one sequence."""
    positions = imgt_positions(sequence, chain)
    if positions is None:
        return None
    return dict(zip(positions, sequence, strict=False))


def is_cdr(position: int) -> bool:
    """Whether an IMGT position falls in a CDR."""
    return any(low <= position <= high for low, high in IMGT_CDR_RANGES)


def cysteine_retention(sequences: list[str], chain: str) -> tuple[int, int, int]:
    """Count sequences retaining both canonical cysteines."""
    retained = numbered = unnumberable = 0
    for sequence in sequences:
        mapped = imgt_map(sequence, chain)
        if mapped is None:
            unnumberable += 1
            continue
        numbered += 1
        if all(mapped.get(position) == "C" for position in CANONICAL_CYS):
            retained += 1
    return retained, numbered, unnumberable


def region_composition(sequences: list[str], chain: str) -> tuple[Counter, Counter]:
    """Amino-acid composition inside CDR columns and inside framework columns, separately."""
    cdr: Counter = Counter()
    framework: Counter = Counter()
    for sequence in sequences:
        mapped = imgt_map(sequence, chain)
        if mapped is None:
            continue
        for position, residue in mapped.items():
            (cdr if is_cdr(position) else framework)[residue] += 1
    return cdr, framework


def kl(p: Counter, q: Counter) -> float:
    """``D_KL(p || q)`` over residue frequencies, with q smoothed."""
    total_p = sum(p.values()) or 1
    total_q = sum(q.values()) or 1
    out = 0.0
    for residue in set(p) | set(q):
        pi = p.get(residue, 0) / total_p
        qi = (q.get(residue, 0) + 0.5) / (total_q + 10)
        if pi > 0:
            out += pi * math.log(pi / qi)
    return out


def main() -> None:
    """Report cysteine retention and the CDR/framework split for one run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family-fasta", type=Path, required=True,
                        help="Seed family; the paper's 200-sequence reference is cut from it")
    parser.add_argument("--generated", type=Path, default=None,
                        help="A run's FASTA. Omit to report the natural baseline alone")
    parser.add_argument("--set", dest="set_name", default=None,
                        help="Keep only this set from --generated, e.g. `model`")
    parser.add_argument("--chain", default="heavy", choices=("heavy", "light"))
    args = parser.parse_args()

    natural = paper_reference(args.family_fasta)
    retained, numbered, skipped = cysteine_retention(natural, args.chain)
    print(f"natural reference: {len(natural)} sequences, {numbered} numbered, {skipped} unnumberable")
    print(f"   canonical cysteines (IMGT 23 & 104) retained {retained}/{numbered}"
          f" ({100 * retained / max(1, numbered):.1f}%)")
    nat_cdr, nat_fwk = region_composition(natural, args.chain)
    total = sum(nat_cdr.values()) + sum(nat_fwk.values())
    print(f"   region split: {sum(nat_cdr.values())} CDR / {sum(nat_fwk.values())} framework "
          f"({100 * sum(nat_cdr.values()) / max(1, total):.0f}% CDR)")

    if args.generated is None:
        return

    generated = read_fasta(args.generated, only_set=args.set_name)
    g_retained, g_numbered, g_skipped = cysteine_retention(generated, args.chain)
    print(f"\ngenerated ({args.set_name or 'all sets'}): {len(generated)} sequences, "
          f"{g_numbered} numbered, {g_skipped} unnumberable")
    print(f"   canonical cysteines retained {g_retained}/{g_numbered}"
          f" ({100 * g_retained / max(1, g_numbered):.1f}%)")
    gen_cdr, gen_fwk = region_composition(generated, args.chain)
    print("\ncomposition KL(generated || natural)")
    print(f"   CDR columns        {kl(gen_cdr, nat_cdr):.4f}")
    print(f"   framework columns  {kl(gen_fwk, nat_fwk):.4f}")


if __name__ == "__main__":
    main()
