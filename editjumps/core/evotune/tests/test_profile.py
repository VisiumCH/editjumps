"""Eq. 12 verbatim, its normalisation, and the position draw it feeds."""

import pytest


def test_evotune_column_entropy_is_equation_12_verbatim() -> None:
    """Eq 12's three edge cases, by hand: conserved, two-way split, and all-gap. ``H(l) = -sum_a p_l(a)."""
    from editjumps.core.evotune.profile import column_entropy

    #            col0 conserved   col1 two-way   col2 all-gap
    aligned = ["AC-", "AD-", "AC-", "AD-"]
    entropy = column_entropy(aligned)
    assert entropy[0] == pytest.approx(0.0, abs=1e-9)
    assert entropy[1] == pytest.approx(0.6931471805599453, abs=1e-9)  # ln 2
    # An all-gap column has no residue distribution: no information, and crucially not a NaN.
    assert entropy[2] == 0.0

    with pytest.raises(ValueError, match="ragged"):
        column_entropy(["AC", "ACD"])

def test_evotune_entropy_excludes_gaps_from_the_denominator() -> None:
    """"Excluding gaps" is the paper's word, and it changes the number, so it is pinned."""
    from editjumps.core.evotune.profile import column_entropy

    assert column_entropy(["A", "A", "-"])[0] == pytest.approx(0.0, abs=1e-9)
    # `X` is neither a gap nor one of the 20 residues; excluded from the denominator the same way.
    assert column_entropy(["A", "A", "X"])[0] == pytest.approx(0.0, abs=1e-9)

def test_evotune_weights_normalise_and_map_to_ungapped_coordinates() -> None:
    """Weights sum to 1 in the template's OWN coordinates, dropping columns it has no residue at."""
    from editjumps.core.evotune.profile import normalise_weights, position_weights, ungapped_weights

    assert normalise_weights([1.0, 3.0]) == pytest.approx([0.25, 0.75])
    # A perfectly conserved family gets a uniform vector, not an undrawable all-zero one.
    assert normalise_weights([0.0, 0.0, 0.0]) == pytest.approx([1 / 3, 1 / 3, 1 / 3])

    # The gap column must not survive, and the two that do must be renormalised, not left at 0.75.
    mapped = ungapped_weights([0.25, 0.25, 0.5], "A-C")
    assert len(mapped) == 2
    assert mapped == pytest.approx([1 / 3, 2 / 3])

    # End to end: the varying column outranks the conserved one, in template coordinates.
    weights = position_weights(["AC", "AD", "AC", "AD"], "AC")
    assert len(weights) == 2
    assert weights[1] > weights[0]
    assert sum(weights) == pytest.approx(1.0)

    with pytest.raises(ValueError, match="same alignment"):
        ungapped_weights([0.5, 0.5], "ABC")

def test_evotune_position_sampling_follows_the_profile_and_never_repeats() -> None:
    """Positions are drawn without replacement, proportional to entropy, zero-weight ones last."""
    import random

    from editjumps.core.evotune.profile import sample_positions

    weights = [0.0, 0.9, 0.1, 0.0]
    counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for trial in range(400):
        counts[sample_positions(weights, 1, random.Random(trial))[0]] += 1
    assert counts[1] > counts[2] > 0
    # Zero-weight positions come last, not never: refusing them makes the budget unreachable.
    assert counts[0] == counts[3] == 0

    drawn = sample_positions(weights, 4, random.Random(1))
    assert sorted(drawn) == [0, 1, 2, 3], "all four must be reachable once the budget demands them"

    # k is clipped, not rejected: a budget larger than the sequence means "edit everything".
    assert len(sample_positions(weights, 10, random.Random(2))) == 4
    assert set(sample_positions(weights, 2, random.Random(3), exclude=[1, 2])) == {0, 3}
    with pytest.raises(ValueError):
        sample_positions(weights, -1, random.Random(0))
