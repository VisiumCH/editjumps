"""Diversity and novelty, the two axes generative-protein papers use that EvoFlows does not.."""

from editjumps.core.utils import get_logger

logger = get_logger(__file__)


def nearest_neighbour_distance(
    queries: list[str],
    reference: list[str],
    max_reference: int = 20000,
    seed: int = 0,
) -> dict:
    """Mean distance from each query to its nearest neighbour in ``reference`` -- i.e. novelty.

    Args:
        queries: Sequences to score.
        reference: Sequences to score them against.
        max_reference: Cap on the reference, subsampled above it; this can only over-estimate.
        seed: Draw source for that subsample.

    Returns:
        ``mean``, ``median`` and ``min`` of the per-query nearest-neighbour distances, with the two
        counts they were computed over.
    """
    import random

    import numpy as np
    from rapidfuzz.distance import Levenshtein
    from rapidfuzz.process import cdist

    if not queries or not reference:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "n_queries": 0, "n_reference": 0}

    searched = reference
    if len(reference) > max_reference:
        searched = random.Random(seed).sample(reference, max_reference)
        logger.info(f"novelty: subsampled reference {len(reference)} -> {max_reference} "
                    f"(this can only over-estimate novelty)")

    # workers=-1 parallelises across cores; scores are plain edit distances, not similarities.
    matrix = cdist(queries, searched, scorer=Levenshtein.distance, workers=-1)
    nearest = matrix.min(axis=1)
    return {
        "mean": float(np.mean(nearest)),
        "median": float(np.median(nearest)),
        "min": float(np.min(nearest)),
        "n_queries": len(queries),
        "n_reference": len(searched),
    }


def diversity_novelty(
    generated: list[str],
    templates: list[str],
    reference: list[str],
    max_pairs: int = 2000,
    max_reference: int = 20000,
    seed: int = 0,
) -> dict:
    """Report both axes, plus the template baseline that makes novelty interpretable."""
    from editjumps.core.generation_metrics import mean_pairwise_levenshtein

    diversity = mean_pairwise_levenshtein(generated, max_pairs=max_pairs, seed=seed)
    gen_novelty = nearest_neighbour_distance(generated, reference, max_reference, seed)
    tpl_novelty = nearest_neighbour_distance(templates, reference, max_reference, seed)
    return {
        "diversity": diversity,
        "novelty": gen_novelty["mean"],
        "novelty_templates": tpl_novelty["mean"],
        "novelty_delta": gen_novelty["mean"] - tpl_novelty["mean"],
        "novelty_detail": gen_novelty,
        "novelty_templates_detail": tpl_novelty,
    }
