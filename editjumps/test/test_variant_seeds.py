"""Per-variant RNG seeds: the block layout, and the guard that stops one template reaching the next."""

from pathlib import Path

import pytest

from editjumps.core.variant_seeds import (
    VARIANT_SEED_STRIDE,
    check_variant_budget,
    template_seed,
    variant_seed,
)


def test_seed_layout_is_unchanged_from_the_arithmetic_it_replaced() -> None:
    """The helper must reproduce ``seed + 1000 * index + variant`` exactly: runs on record depend on it."""
    for seed, index, variant in ((0, 0, 0), (7, 3, 19), (123, 11, 999)):
        assert variant_seed(seed, index, variant) == seed + 1000 * index + variant
    assert template_seed(5, 4) == 5 + 1000 * 4


def test_every_variant_of_every_template_gets_its_own_stream_within_the_stride() -> None:
    """Inside the budget the map (index, variant) -> seed is injective, so no two draws collide."""
    n_templates, n_variants = 12, VARIANT_SEED_STRIDE
    seeds = [variant_seed(0, i, v) for i in range(n_templates) for v in range(n_variants)]
    assert len(set(seeds)) == len(seeds)


def test_overrunning_the_stride_is_exactly_the_collision_the_guard_describes() -> None:
    """One past the stride, template i's variant reuses template i+1's first seed."""
    assert variant_seed(0, 0, VARIANT_SEED_STRIDE) == variant_seed(0, 1, 0)


def test_check_variant_budget_accepts_the_budget_and_rejects_one_past_it() -> None:
    """The guard fires only above the stride, and names the parameter the caller has to change."""
    check_variant_budget(1)
    check_variant_budget(VARIANT_SEED_STRIDE)  # the last collision-free value
    with pytest.raises(ValueError, match="n_variants"):
        check_variant_budget(VARIANT_SEED_STRIDE + 1)


def test_the_three_generators_refuse_an_over_budget_call(tmp_path: Path) -> None:
    """Each generator raises on an over-budget `n_variants`, before it loads a model or writes anything.

    Called for real rather than inspected for the guard's name: a source check passes whenever the
    right words appear, including after the loop that would have used the colliding seeds.
    """
    pytest.importorskip("torch")
    from editjumps.pipeline.evaluate.evodiff_msa_baseline import generate_variants as evodiff
    from editjumps.pipeline.evaluate.evotune_baseline import generate_variants as evotune
    from editjumps.pipeline.evaluate.generation_eval import evaluate

    over = VARIANT_SEED_STRIDE + 1

    def never_called(masked: list[str | None], position: int, template: str, /) -> str:
        """Stand-in proposer: reaching it means the guard did not fire first."""
        raise AssertionError("the guard must refuse the budget before any proposing happens")

    # Paths that do not exist, and a proposer that refuses to be called: the guard has to fire
    # before any of it is touched, so reaching them fails differently from the assertion below.
    with pytest.raises(ValueError, match="n_variants"):
        evaluate(tmp_path / "no-model", tmp_path / "no-pairs", 1, over, 1, 0, 1)

    with pytest.raises(ValueError, match="n_variants"):
        evotune(templates=["ACDE"], train_members=["ACDEF"], budget=1, propose=never_called,
                n_variants=over, seed=0, forced=False)

    with pytest.raises(ValueError, match="n_variants"):
        evodiff(templates=["ACDE"], train_members=["ACDEF"], budget=1, n_variants=over, seed=0,
                workdir=tmp_path / "unused")
