"""`editjumps/measurements/homolog_cap_bias.py`: the estimator behind the cap-vs-uncapped column in."""

import json
import random
from pathlib import Path
from types import ModuleType

import pytest


def test_homolog_cap_bias_uncapped_estimate_is_unbiased_and_the_cap_is_not() -> None:
    """The cap-vs-uncapped comparison must not be an artefact of its own estimator."""
    import importlib.util
    import statistics
    from itertools import combinations

    script = Path(__file__).parents[2] / "editjumps" / "measurements" / "homolog_cap_bias.py"
    spec = importlib.util.spec_from_file_location("homolog_cap_bias", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Differing pair distances AND sizes — the only way a per-family cap can bias anything.
    small = ["AAAAAAAA", "AAAAAAAC"]                     # 1 pair, distance 1
    large = ["C" * 8, "CD" * 4, "CDD" + "C" * 5, "D" * 8, "DC" * 4, "DDC" + "D" * 5]
    families = [small, large]
    exact = statistics.fmean([module.score_pair(a, b)[0]
                              for members in families
                              for a, b in combinations(members, 2)])

    report = module.measure(families, (1, 15), 15, 0)
    assert report["totals"]["pairs_uncapped"] == 16
    assert report["uncapped_weighted"]["mean"] == pytest.approx(exact)
    assert report["uncapped_weighted"]["n_pairs"] == pytest.approx(16.0)
    # cap 1 gives the two families equal weight; uncapped gives the big one 15/16 of it.
    assert report["arms"]["cap1"]["mean"] != pytest.approx(exact)
    # Wasserstein-1 to the uncapped distribution: zero for the arm that IS it, positive otherwise.
    assert report["wasserstein1_to_uncapped"]["cap15"] == pytest.approx(0.0)
    assert report["wasserstein1_to_uncapped"]["cap1"] > 0.0
    # The per-size breakdown is the mechanism claim, so both bins must be present and add up.
    shares = report["per_family_size"]
    assert sum(b["share_of_uncapped_pairs"] for b in shares.values()) == pytest.approx(1.0)
    assert sum(b["share_of_capped_pairs"] for b in shares.values()) == pytest.approx(1.0)
    json.dumps(report)


def test_homolog_cap_bias_draw_is_uniform_in_every_prefix() -> None:
    """Reporting several caps from one draw is only valid if any prefix is itself uniform. `measure`."""
    import importlib.util
    from collections import Counter
    from itertools import combinations

    script = Path(__file__).parents[2] / "editjumps" / "measurements" / "homolog_cap_bias.py"
    spec = importlib.util.spec_from_file_location("homolog_cap_bias", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    rng = random.Random(0)
    # Below the cap: exhaustive, so every pair appears exactly once.
    drawn = module.draw_pairs(6, 20, rng)
    assert sorted(drawn) == sorted(combinations(range(6), 2))
    # Above the cap: exactly `cap` DISTINCT pairs, and the first position is not stuck on member 0.
    first: Counter[int] = Counter()
    for _ in range(200):
        sample = module.draw_pairs(20, 5, rng)
        assert len(sample) == len(set(sample)) == 5
        first[sample[0][0]] += 1
    assert len(first) > 5, f"first-slot members barely vary ({sorted(first)}), draw is not shuffled"


def _load_module() -> "ModuleType":
    """Import the measurement script by path, as the other tests in this file do."""
    import importlib.util

    script = Path(__file__).parents[2] / "editjumps" / "measurements" / "homolog_cap_bias.py"
    spec = importlib.util.spec_from_file_location("homolog_cap_bias", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_cap_above_cap_max_is_refused_rather_than_silently_truncated() -> None:
    """Every arm slices the same `cap_max` draws, so a larger cap would publish a label it never reached."""
    module = _load_module()
    families = [["ACDE", "ACDF", "ACDG"], ["WYWY", "WYWF", "WFWY"]]

    with pytest.raises(ValueError, match="cap_max"):
        module.measure(families, (10, 2000), 1000, 0)

    # The boundary itself is legitimate: cap == cap_max measures exactly what it claims.
    report = module.measure(families, (1, 3), 3, 0)
    assert "cap3" in report["arms"]


def test_degenerate_cap_arguments_are_refused() -> None:
    """An empty or non-positive cap yields an arm with no pairs in it, which is not a measurement."""
    module = _load_module()
    families = [["ACDE", "ACDF", "ACDG"]]

    with pytest.raises(ValueError, match="empty"):
        module.measure(families, (), 10, 0)
    with pytest.raises(ValueError, match="non-positive"):
        module.measure(families, (0, 5), 10, 0)
