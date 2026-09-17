"""`editjumps/measurements/`: the shape every script there has to have, and the sampling it does."""

import importlib
import random
from pathlib import Path

import pytest

MEASUREMENTS = Path(__file__).parents[1] / "measurements"


def measurement_modules() -> list[str]:
    """Every measurement script, as an importable dotted name."""
    return [f"editjumps.measurements.{path.stem}"
            for path in sorted(MEASUREMENTS.glob("*.py")) if path.name != "__init__.py"]


@pytest.mark.parametrize("module", measurement_modules())
def test_a_measurement_script_defines_its_work_instead_of_doing_it(module: str) -> None:
    """Importing a measurement script must not run it.

    A script that works at module level cannot be imported, reused or tested, and it reads
    DVC-tracked corpora that a fresh clone does not have - so importing it raises there. That shape
    is why the `rng.sample(pool, 200)` crash in measure_disjoint_population.py had no regression
    test until its logic moved into a function.
    """
    importlib.import_module(module)


@pytest.mark.parametrize("module", measurement_modules())
def test_a_measurement_script_is_runnable_on_its_own(module: str) -> None:
    """Each script keeps a `main` behind a `__main__` guard, so it stays runnable by hand."""
    imported = importlib.import_module(module)
    assert callable(getattr(imported, "main", None)), f"{module} has no main() to run"
    assert '__name__ == "__main__"' in Path(imported.__file__ or "").read_text(), (
        f"{module} defines main() but never calls it"
    )


def test_a_pool_smaller_than_the_sample_size_is_reported_in_full() -> None:
    """`random.sample` raises above the pool size, which used to kill the whole comparison."""
    from editjumps.measurements.measure_disjoint_population import SAMPLE_SIZE, summarise_pool

    pool: list[str] = [f"ACDEFG{'H' * (i % 4)}" for i in range(5)]
    line = summarise_pool("small family", pool, random.Random(0), sample_size=SAMPLE_SIZE)

    assert "pooled pairwise Lev" in line
    # The arm says it is the whole pool, so it cannot be read as a 200-sequence draw.
    assert "(whole pool: 5)" in line


def test_a_pool_at_or_above_the_sample_size_is_drawn_from_and_says_nothing_extra() -> None:
    """A full-size arm is the normal case and carries no note."""
    from editjumps.measurements.measure_disjoint_population import summarise_pool

    pool: list[str] = [f"ACDEFG{'H' * (i % 7)}" for i in range(40)]
    line = summarise_pool("big family", pool, random.Random(0), sample_size=10)

    assert "pooled pairwise Lev" in line
    assert "whole pool" not in line


@pytest.mark.parametrize("size", [0, 1])
def test_a_pool_too_small_to_compare_is_said_so_rather_than_computed(size: int) -> None:
    """Below two members there is no pairwise distance and no length spread; `stdev` would raise."""
    from editjumps.measurements.measure_disjoint_population import summarise_pool

    line = summarise_pool("tiny", ["ACDEF"][:size], random.Random(0))

    assert "too few to compare" in line
    assert f"{size} sequences" in line
