"""The Appendix B generation metrics, against the properties their equations force."""

from pathlib import Path

import pytest

ALPHABET_INDEX = {a: i for i, a in enumerate("ACDEFGHIKLMNPQRSTVWY-")}


def test_generation_metrics_match_their_definitions() -> None:
    """The Appendix B metrics, checked against properties their equations force."""
    import numpy as np

    from editjumps.core.generation_metrics import (
        background_frequencies,
        frequencies,
        kl_divergence,
        levenshtein,
        mutual_information_apc,
        one_hot,
        positional_interaction_strength,
        smoothed_composition,
        spectrum_kernel,
        spectrum_mmd,
    )

    # eq 17 shapes: f_i is (L,21), f_ij and C_ij are (L,L,21,21) over 20 AAs PLUS gap.
    aligned = ["ACDE", "ACDE", "AC-E"]
    encoded = one_hot(aligned)
    assert encoded.shape == (3, 4, 21)
    f_i, f_ij, c_ij = frequencies(encoded)
    assert f_i.shape == (4, 21) and c_ij.shape == (4, 4, 21, 21)
    assert np.allclose(f_i.sum(axis=1), 1.0)          # a distribution at every column
    # A column that never varies has zero covariance with anything.
    assert np.allclose(c_ij[0, :, :, :], 0.0, atol=1e-12)

    # eq 18: Frobenius norm over the amino-acid axes is non-negative and symmetric in (i, j).
    strength = positional_interaction_strength(c_ij)
    assert strength.shape == (4, 4)
    assert (strength >= 0).all() and np.allclose(strength, strength.T)

    # eq 19-22: MIp is symmetric.
    mip = mutual_information_apc(f_i, f_ij)
    assert mip.shape == (4, 4) and np.allclose(mip, mip.T, atol=1e-10)

    # eq 24: smoothing gives an unobserved symbol positive probability, or KL becomes infinite.
    composition = smoothed_composition(["AAAA"])
    assert (composition > 0).all() and abs(composition.sum() - 1.0) < 1e-12
    assert np.isfinite(kl_divergence(smoothed_composition(["WWWW"]), composition))
    # The prior is BLOSUM62-weighted, not uniform: leucine (0.099) must outrank tryptophan (0.013).
    prior = background_frequencies()
    assert prior[ALPHABET_INDEX["L"]] > prior[ALPHABET_INDEX["W"]]

    # eq 23: KL is asymmetric, as the paper states explicitly.
    p, q = smoothed_composition(["AAAC"]), smoothed_composition(["ACCC"])
    assert abs(kl_divergence(p, q) - kl_divergence(q, p)) > 1e-6
    assert kl_divergence(p, p) < 1e-12

    # eq 25-27: MMD is 0 for identical sets and never negative — why the biased estimator is used.
    same = ["ACDEFGHIK", "ACDEFGHIK"]
    assert spectrum_mmd(same, same) == 0.0
    assert spectrum_mmd(["AAAAAAAAA"], ["WWWWWWWWW"]) > 0.0
    assert spectrum_mmd(["ACDEFGHIK"], ["ACDEFGHIL"]) >= 0.0
    # The kernel is an inner product of k-mer counts, so it is symmetric.
    from editjumps.core.generation_metrics import spectrum_features
    a, b = spectrum_features("ACDEFG"), spectrum_features("ACDEFH")
    assert spectrum_kernel(a, b) == spectrum_kernel(b, a)

    assert levenshtein("ACDE", "ACDE") == 0
    assert levenshtein("ACDE", "ACD") == 1


def test_matrix_agreement_separates_what_the_scalar_reductions_cannot() -> None:
    """`matrix_agreement` ranks coupling matrices; the reductions it replaces cannot."""
    import numpy as np

    from editjumps.core.generation_metrics import (
        frequencies,
        matrix_agreement,
        mutual_information_apc,
        one_hot,
    )

    rng = np.random.default_rng(0)
    natural = ["".join(c) for c in rng.choice(list("ACDEFGHIKLMNPQRSTVWY"), size=(40, 24))]
    # Same set with a real coupling injected: position 0 forces position 12.
    coupled = [("A" + s[1:12] + "W" + s[13:]) if i % 2 else ("C" + s[1:12] + "Y" + s[13:])
               for i, s in enumerate(natural)]

    def mip_of(aligned: list[str]) -> np.ndarray:
        f_i, f_ij, _ = frequencies(one_hot(aligned))
        return mutual_information_apc(f_i, f_ij)

    mip_a, mip_b = mip_of(coupled), mip_of(natural)
    # The reduction we used to report is ~0 for BOTH, so it cannot tell them apart.
    assert abs(mip_a.mean()) < 1e-9 and abs(mip_b.mean()) < 1e-9, "MIp mean is not ~0 by construction"

    # A matrix agrees perfectly with itself, and much less with an unrelated one.
    assert abs(matrix_agreement(mip_a, mip_a) - 1.0) < 1e-9
    assert matrix_agreement(mip_a, mip_b) < 0.9

    # Norm-blindness: permuting rows and columns preserves the Frobenius norm exactly while
    # destroying the pattern, so the norm ratio says "identical" and the correlation does not.
    order = rng.permutation(mip_a.shape[0])
    shuffled = mip_a[np.ix_(order, order)]
    assert abs(np.linalg.norm(shuffled) - np.linalg.norm(mip_a)) < 1e-9
    assert abs(matrix_agreement(mip_a, shuffled)) < 0.6

    # The diagonal band is excluded: a matrix that agrees ONLY there must not score high.
    length = mip_a.shape[0]
    band = np.zeros((length, length))
    for offset in range(4):
        np.fill_diagonal(band[:, offset:], 1.0)
    assert np.isnan(matrix_agreement(band, band)), "near-diagonal-only pairs were not excluded"

    with pytest.raises(ValueError, match="shapes differ"):
        matrix_agreement(mip_a, mip_a[:-1, :-1])


def test_pooled_pairwise_is_not_the_within_template_mean() -> None:
    """Pooling across templates is a different, larger quantity -- and non-zero at one variant each."""
    import numpy as np

    from editjumps.core.generation_metrics import mean_pairwise_levenshtein

    # One variant per template: the within-template mean is 0 by definition.
    assert mean_pairwise_levenshtein(["ACDE"]) == 0.0
    # Pooled over three single-variant templates, it is not.
    pooled = mean_pairwise_levenshtein(["ACDE", "ACDW", "WWWW"])
    assert pooled > 0.0

    # Pooling across dissimilar templates exceeds the within-template average, because it also
    # carries the between-template distance.
    group_a, group_b = ["ACDE", "ACDW"], ["WWWW", "WWWY"]
    within = float(np.mean([mean_pairwise_levenshtein(group_a),
                            mean_pairwise_levenshtein(group_b)]))
    assert mean_pairwise_levenshtein(group_a + group_b) > within


def test_agreement_scores_are_biased_by_a_small_reference() -> None:
    """A 20-sequence reference drags both agreements toward zero, so the size must be recorded."""
    import numpy as np

    from editjumps.core.generation_metrics import (
        frequencies,
        matrix_agreement,
        mutual_information_apc,
        one_hot,
        positional_interaction_strength,
    )

    rng = np.random.default_rng(1)
    # One family: a fixed backbone with a few variable columns, so the couplings are real.
    backbone = list("ACDEFGHIKLMNPQRSTVWYACDEFGHIKL")
    variable = [3, 7, 11, 19, 23]

    def draw(n: int) -> list[str]:
        out = []
        for _ in range(n):
            seq = backbone.copy()
            flip = rng.random() < 0.5
            for position in variable:                    # the coupling: all move together
                seq[position] = "W" if flip else "Y"
            seq[rng.integers(len(seq))] = str(rng.choice(list("ACDEFG")))
            out.append("".join(seq))
        return out

    def couplings(aligned: list[str]) -> tuple[np.ndarray, np.ndarray]:
        f_i, f_ij, c_ij = frequencies(one_hot(aligned))
        return positional_interaction_strength(c_ij), mutual_information_apc(f_i, f_ij)

    cov_g, mip_g = couplings(draw(400))
    scores = {}
    for n in (20, 200):
        cov_r, mip_r = couplings(draw(n))
        scores[n] = (matrix_agreement(cov_g, cov_r), matrix_agreement(mip_g, mip_r))

    # Both agreements improve with the reference size, on identically-distributed data.
    assert scores[200][0] > scores[20][0], f"covariance: {scores}"
    assert scores[200][1] > scores[20][1], f"MIP: {scores}"
    # The generation_eval and baseline reports must both carry the size, or a reader cannot tell a
    # model difference from this.
    for module, name in (("editjumps.pipeline.evaluate.generation_eval", "generation_eval"),
                         ("editjumps.pipeline.evaluate.evotune_baseline", "evotune_baseline")):
        source = Path(__file__).parents[1].joinpath(*module.split(".")[1:]).with_suffix(".py")
        text = source.read_text()
        assert "agreement_reference_n" in text, f"{name} must record the reference size"
        assert '"covariance_agreement": matrix_agreement(strength, strength_full' in text, (
            f"{name} must score the agreements against the FULL reference, not the per-template "
            f"partners -- see this test's docstring"
        )
