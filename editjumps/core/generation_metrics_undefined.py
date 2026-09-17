"""Metrics EvoFlows *reports* but never defines — our reading, kept separate on purpose."""

import numpy as np

from editjumps.core.generation_metrics import (
    ALPHABET,
    GAP_INDEX,
    background_frequencies,
    one_hot,
    smoothed_composition,
)


def per_position_entropy(aligned: list[str], eps: float = 1e-12) -> np.ndarray:
    """Shannon entropy at each aligned column, in nats."""
    encoded = one_hot(aligned)
    if encoded.size == 0:
        return np.zeros(0)
    freq = encoded.mean(axis=0)                       # (L, 21)
    return -(freq * np.log(np.maximum(freq, eps))).sum(axis=1)


def entropy_delta(generated: list[str], reference: list[str]) -> float:
    """Mean per-column entropy of the generated set minus that of the reference set (our reading)."""
    gen, ref = per_position_entropy(generated), per_position_entropy(reference)
    if gen.size == 0 or ref.size == 0:
        return 0.0
    return float(gen.mean() - ref.mean())


def js_divergence(generated: list[str], reference: list[str], alpha: float = 1.0) -> float:
    """Jensen–Shannon divergence between amino-acid compositions (our reading)."""
    from editjumps.core.generation_metrics import kl_divergence

    p = smoothed_composition(generated, alpha)
    q = smoothed_composition(reference, alpha)
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


def js_divergence_positional(generated: list[str], reference: list[str], alpha: float = 1.0,
                             eps: float = 1e-12) -> float:
    """Mean per-column Jensen-Shannon divergence between two aligned sets (our reading)."""
    left, right = one_hot(generated), one_hot(reference)
    if left.size == 0 or right.size == 0:
        return 0.0
    prior = background_frequencies()
    p = (left.sum(axis=0) + alpha * prior) / (len(generated) + alpha)      # (L, 21)
    q = (right.sum(axis=0) + alpha * prior) / (len(reference) + alpha)
    m = 0.5 * (p + q)
    half = 0.5 * (p * np.log(np.maximum(p, eps) / np.maximum(m, eps))).sum(axis=1)
    half += 0.5 * (q * np.log(np.maximum(q, eps) / np.maximum(m, eps))).sum(axis=1)
    return float(half.mean())


def profile_log_likelihood(generated: list[str], reference: list[str], alpha: float = 1.0,
                           eps: float = 1e-12) -> float:
    """Mean log-likelihood of generated sequences under a position-specific profile of the reference (our reading)."""
    if not generated or not reference:
        return 0.0
    ref_encoded = one_hot(reference)
    length = ref_encoded.shape[1]
    if any(len(s) != length for s in generated):
        raise ValueError("generated and reference must share one alignment length for a profile score")

    prior = background_frequencies()
    counts = ref_encoded.sum(axis=0)                                        # (L, 21)
    # Same normalisation correction as smoothed_composition: eq 24's printed denominator
    # does not yield a distribution when mu sums to 1.
    profile = (counts + alpha * prior) / (counts.sum(axis=1, keepdims=True) + alpha)

    index = {a: i for i, a in enumerate(ALPHABET)}
    scores = []
    for seq in generated:
        scores.append(sum(float(np.log(max(profile[i, index.get(char, GAP_INDEX)], eps)))
                          for i, char in enumerate(seq)))
    return float(np.mean(scores))


def kl_divergence_positional(generated: list[str], reference: list[str], alpha: float = 1.0,
                             eps: float = 1e-12) -> float:
    """Mean per-column KL divergence of the generated set from the reference set."""
    left, right = one_hot(generated), one_hot(reference)
    if left.size == 0 or right.size == 0:
        return 0.0
    prior = background_frequencies()
    p = (left.sum(axis=0) + alpha * prior) / (len(generated) + alpha)
    q = (right.sum(axis=0) + alpha * prior) / (len(reference) + alpha)
    per_column = (p * np.log(np.maximum(p, eps) / np.maximum(q, eps))).sum(axis=1)
    return float(per_column.mean())
