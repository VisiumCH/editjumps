"""One homolog family, split into templates / reference holdout / pool — EvoFlows §4.2. "Each homolog."""

import random
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FamilySplit:
    """A family partitioned the way §4.2 partitions one."""

    templates: list[str]
    reference: list[str]
    pool: list[str]


def split_family(members: list[str], n_templates: int, holdout_size: int, seed: int) -> FamilySplit:
    """Shuffle a family once and cut it into inference / holdout / pool."""
    unique = list(dict.fromkeys(members))
    if len(unique) < n_templates + 2:
        raise ValueError(
            f"family has {len(unique)} distinct usable members; need > {n_templates + 1} to leave "
            "a holdout"
        )
    shuffled = unique
    random.Random(seed).shuffle(shuffled)
    templates, holdout = shuffled[:n_templates], shuffled[n_templates:]
    return FamilySplit(templates=templates, reference=holdout[:holdout_size], pool=holdout[holdout_size:])


def usable_members(sequences: Iterable[str], min_length: int = 50) -> list[str]:
    """Filter a family to the members an evaluation may use, in file order."""
    return [sequence for sequence in sequences if len(sequence) > min_length]


def sequences_in_pairs(pairs: Path) -> set[str]:
    """Every sequence appearing in a training-pairs TSV, in both joined and per-chain form."""
    import gzip

    from editjumps.core.sequences import PAIR_SEP

    if not pairs.exists():
        return set()
    seen: set[str] = set()
    with gzip.open(pairs, "rt") as handle:
        for line in handle:
            for cell in line.rstrip("\n").split("\t"):
                if not cell:
                    continue
                seen.add(cell)
                seen.update(part for part in cell.split(PAIR_SEP) if part)
    return seen


def disjoint_members(members: list[str], pairs: Path) -> list[str]:
    """Family members that do not appear in the editor's training pairs."""
    seen = sequences_in_pairs(pairs)
    return [member for member in members if member not in seen]
