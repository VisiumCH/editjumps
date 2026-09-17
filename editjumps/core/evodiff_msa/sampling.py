"""Where a mutation goes, and which residue fills it — the two choices that are ours, not theirs."""

import random
from math import exp

from editjumps.core.sequences import AA

#: Residues row 0 may be given.
PROPOSABLE: tuple[str, ...] = tuple(sorted(AA))


def uniform_weights(template: str) -> list[float]:
    """Position weights for a baseline with no positional model: uniform over the template."""
    if not template:
        return []
    return [1.0 / len(template)] * len(template)


def softmax_choice(logits: dict[str, float], blocked: str, temperature: float, rng: random.Random) -> str:
    """Draw one residue from temperature-scaled ``logits``, with ``blocked`` removed from the support.."""
    if temperature <= 0:
        raise ValueError(f"temperature={temperature}; must be > 0")
    candidates = [residue for residue in PROPOSABLE if residue != blocked and residue in logits]
    if not candidates:
        raise ValueError(f"no candidate residue left after blocking {blocked!r}; logits={sorted(logits)}")
    # Shifted by the max before exponentiating: an MSATransformer logit can sit well above 30, and
    # exp() of that overflows a float long before the softmax would.
    top = max(logits[residue] for residue in candidates)
    weights = [exp((logits[residue] - top) / temperature) for residue in candidates]
    return rng.choices(candidates, weights=weights, k=1)[0]
