"""The positional distribution that decides WHERE a substitution goes (EvoFlows arXiv 2603.11703, eq.."""

import math
import random
from collections.abc import Iterable, Sequence

from editjumps.core.sequences import AA

#: Gap character in an alignment, matching :mod:`editjumps.core.generation_metrics`.
GAP = "-"

#: The amino acids eq 12 sums over. Ordered so the profile is reproducible across runs.
ALPHABET: tuple[str, ...] = tuple(sorted(AA))


def column_entropy(aligned: Sequence[str], eps: float = 1e-12) -> list[float]:
    """Return eq. 12's per-column entropy in nats; empty if ``aligned`` is empty."""
    if not aligned:
        return []
    width = len(aligned[0])
    ragged = [len(seq) for seq in aligned if len(seq) != width]
    if ragged:
        raise ValueError(f"alignment is ragged: width {width} but also lengths {sorted(set(ragged))}")

    entropy: list[float] = []
    for column in range(width):
        counts = {letter: 0 for letter in ALPHABET}
        total = 0
        for seq in aligned:
            residue = seq[column]
            if residue in counts:
                counts[residue] += 1
                total += 1
        if total == 0:
            entropy.append(0.0)
            continue
        value = 0.0
        for letter in ALPHABET:
            p = counts[letter] / total
            value -= p * math.log(p + eps)
        entropy.append(value)
    return entropy


def normalise_weights(entropy: Sequence[float]) -> list[float]:
    """Turn per-column entropies into the sampling distribution ("entropy weights are normalized")."""
    clipped = [max(0.0, value) for value in entropy]
    total = sum(clipped)
    if not clipped:
        return []
    if total <= 0.0:
        return [1.0 / len(clipped)] * len(clipped)
    return [value / total for value in clipped]


def ungapped_weights(weights: Sequence[float], aligned_template: str) -> list[float]:
    """Map alignment-column weights onto one sequence's own coordinates ("mapped to ungapped")."""
    if len(weights) != len(aligned_template):
        raise ValueError(
            f"{len(weights)} column weights against a template row of {len(aligned_template)}; "
            "the profile and the template must come from the same alignment"
        )
    kept = [weight for weight, residue in zip(weights, aligned_template, strict=True) if residue != GAP]
    return normalise_weights(kept)


def position_weights(aligned_family: Sequence[str], aligned_template: str, eps: float = 1e-12) -> list[float]:
    """Build the paper's positional distribution: eq. 12, normalised, in template coordinates.."""
    return ungapped_weights(normalise_weights(column_entropy(aligned_family, eps)), aligned_template)


def sample_positions(
    weights: Sequence[float], k: int, rng: random.Random, exclude: Iterable[int] = ()
) -> list[int]:
    """Draw ``k`` distinct positions in draw order, proportional to the profile, skipping ``exclude``."""
    if k < 0:
        raise ValueError(f"k={k}; cannot draw a negative number of positions")
    blocked = set(exclude)
    pool = [(i, max(0.0, weights[i])) for i in range(len(weights)) if i not in blocked]
    drawn: list[int] = []
    while len(drawn) < k and pool:
        total = sum(weight for _, weight in pool)
        if total <= 0.0:
            index = rng.randrange(len(pool))
        else:
            threshold = rng.random() * total
            cumulative = 0.0
            index = len(pool) - 1
            for position, (_, weight) in enumerate(pool):
                cumulative += weight
                if cumulative >= threshold:
                    index = position
                    break
        drawn.append(pool.pop(index)[0])
    return drawn
