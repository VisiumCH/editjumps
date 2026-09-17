"""The infill loop and the matched mutation budget (EvoFlows arXiv 2603.11703, §2.2 and §4.2)."""

import random
from collections.abc import Callable, Sequence

from editjumps.core.evotune.profile import sample_positions

#: What the runner hands in for the network half: given the working residue list (``None`` marks a still-masked.
Proposer = Callable[[list[str | None], int, str], str]


def substitute_by_profile(
    template: str,
    weights: Sequence[float],
    budget: int,
    propose: Proposer,
    rng: random.Random,
    *,
    forced: bool = False,
    top_up: bool = True,
    max_rounds: int = 8,
) -> tuple[str, dict]:
    """Generate one variant: the profile picks the positions, ``propose`` picks the residues."""
    if len(weights) != len(template):
        raise ValueError(
            f"{len(weights)} weights for a template of {len(template)}; the profile must be in "
            "ungapped template coordinates (see ungapped_weights)"
        )
    if budget < 0:
        raise ValueError(f"budget={budget}; the mutation budget cannot be negative")

    residues: list[str | None] = list(template)
    spent: set[int] = set()
    changed: set[int] = set()
    rounds = 0

    while rounds < max_rounds:
        remaining = budget - len(changed)
        if remaining <= 0:
            break
        drawn = sample_positions(weights, remaining, rng, exclude=spent)
        if not drawn:
            break
        rounds += 1
        for position in drawn:
            residues[position] = None
        # Filled one at a time, in a random order, so each choice conditions on the previous ones.
        for position in rng.sample(drawn, len(drawn)):
            blocked = template[position] if forced else ""
            residues[position] = propose(residues, position, blocked)
        for position in drawn:
            spent.add(position)
            if residues[position] != template[position]:
                changed.add(position)
        if not top_up:
            break

    variant = "".join(residue if residue is not None else template[i]
                      for i, residue in enumerate(residues))
    return variant, {
        "budget": budget,
        "n_masked": len(spent),
        "n_changed": len(changed),
        "rounds": rounds,
        "hit_budget": len(changed) >= budget,
    }


def matched_budget(mean_edit_distance: float) -> int:
    """Round the editor's own mean edit distance into §4.2's matched mutation count, at least 1."""
    return max(1, int(round(mean_edit_distance)))
