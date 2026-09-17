"""The local capacity guard: fail fast on a job this machine cannot hold, and stay out of the way."""

import pytest


def test_local_capacity_guard_blocks_only_big_jobs_on_tight_machines(
    monkeypatch: "pytest.MonkeyPatch",
) -> None:
    """The capacity guard fails fast on a big local job, and never gets in the way otherwise."""
    import pytest as _pytest

    from editjumps.core import capacity

    tight = lambda: (1.0, 0.95)  # noqa: E731 - 1GB free, swap 95% full
    roomy = lambda: (64.0, 0.05)  # noqa: E731
    big, small = capacity.LARGE_WORKLOAD_ITEMS + 1, capacity.LARGE_WORKLOAD_ITEMS - 1
    monkeypatch.delenv(capacity.ALLOW_ENV, raising=False)

    # Fires only for a big job on a tight machine, and names the escape hatch + the make target.
    monkeypatch.setattr(capacity, "local_memory_state", tight)
    with _pytest.raises(RuntimeError) as err:
        capacity.require_local_capacity(big, "clustering", stage="split_corpus")
    assert "make jobs-repro STAGES=split_corpus" in str(err.value)
    assert capacity.ALLOW_ENV in str(err.value)

    # Must NOT fire: small job; roomy machine; platform where memory is unreadable.
    capacity.require_local_capacity(small, "clustering", stage="split_corpus")
    monkeypatch.setattr(capacity, "local_memory_state", roomy)
    capacity.require_local_capacity(big, "clustering", stage="split_corpus")
    monkeypatch.setattr(capacity, "local_memory_state", lambda: (None, None))
    capacity.require_local_capacity(big, "clustering", stage="split_corpus")

    # An explicit override wins even on a tight machine.
    monkeypatch.setattr(capacity, "local_memory_state", tight)
    monkeypatch.setenv(capacity.ALLOW_ENV, "1")
    capacity.require_local_capacity(big, "clustering", stage="split_corpus")


def test_local_memory_state_reads_this_machine() -> None:
    """`local_memory_state` returns plausible numbers here, or None (never a crash/garbage)."""
    from editjumps.core.capacity import local_memory_state

    free, swap = local_memory_state()
    assert free is None or 0.0 <= free < 4096.0
    assert swap is None or 0.0 <= swap <= 1.0
