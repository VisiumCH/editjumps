"""The residue draw is ours: seeded, temperature-scaled, and safe at an MSATransformer logit."""

import random

import pytest


def test_evodiff_residue_draw_is_ours_seeded_temperature_scaled_and_overflow_safe() -> None:
    """Sampling happens on our side of the pipe, so one --seed reproduces a whole run."""
    from editjumps.core.evodiff_msa.sampling import softmax_choice

    logits = {"A": 800.0, "W": 799.0, "C": -800.0}
    # exp(800) is inf; the max-shift is what keeps this a distribution rather than nan.
    assert softmax_choice(logits, "", 1.0, random.Random(0)) in {"A", "W"}
    assert softmax_choice(logits, "A", 1.0, random.Random(0)) == "W", "blocking must remove A"

    # Temperature is ours: the paper gives no value for this baseline.
    cold = [softmax_choice(logits, "", 0.05, random.Random(seed)) for seed in range(40)]
    hot = [softmax_choice(logits, "", 5000.0, random.Random(seed)) for seed in range(40)]
    assert set(cold) == {"A"}
    assert len(set(hot)) > 1, "a high temperature must not still be greedy"

    with pytest.raises(ValueError, match="temperature"):
        softmax_choice(logits, "", 0.0, random.Random(0))
    with pytest.raises(ValueError, match="no candidate residue left"):
        softmax_choice({"A": 1.0}, "A", 1.0, random.Random(0))
