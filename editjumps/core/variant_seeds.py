"""Per-variant RNG seeds: the block each template owns, and the guard that keeps blocks apart."""

#: Variant streams are laid out as ``seed + VARIANT_SEED_STRIDE * template_index + variant``, so
#: every template owns a contiguous block of this many RNG seeds. The layout is load-bearing for
#: reproducibility: changing the stride reseeds every generation on record.
VARIANT_SEED_STRIDE = 1000


def template_seed(seed: int, index: int) -> int:
    """Return the base of the seed block that one template owns.

    Args:
        seed: The run's base seed, as passed on the command line.
        index: Position of the template in the run's template list.

    Returns:
        The first RNG seed of that template's block.
    """
    return seed + VARIANT_SEED_STRIDE * index


def variant_seed(seed: int, index: int, variant: int) -> int:
    """Return the RNG seed for one variant of one template.

    Args:
        seed: The run's base seed, as passed on the command line.
        index: Position of the template in the run's template list.
        variant: Position of the variant within that template, below `VARIANT_SEED_STRIDE`.

    Returns:
        The RNG seed for that one draw.
    """
    return template_seed(seed, index) + variant


def check_variant_budget(n_variants: int) -> None:
    """Reject an ``n_variants`` that would run one template's seeds into the next template's block.

    Past the stride, template ``i``'s variants reuse template ``i+1``'s seeds: the same draws are
    made twice under different templates, which duplicates sequences and skews diversity, novelty
    and MMD. Nothing raises on its own, so the run looks healthy and the numbers are wrong.

    Args:
        n_variants: Variants the caller intends to generate per template.

    Raises:
        ValueError: If ``n_variants`` exceeds `VARIANT_SEED_STRIDE`.
    """
    if n_variants > VARIANT_SEED_STRIDE:
        raise ValueError(
            f"n_variants={n_variants} exceeds the per-template seed stride {VARIANT_SEED_STRIDE}: "
            f"template i's variant seeds would collide with template i+1's, silently duplicating "
            f"sequences and skewing diversity/novelty/MMD. Lower n_variants, or raise "
            f"VARIANT_SEED_STRIDE - which reseeds every stream and breaks comparability with the "
            f"runs already on record."
        )
