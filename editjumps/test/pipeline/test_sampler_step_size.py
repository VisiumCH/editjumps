"""The sampler step-size study: Euler's error against the clock and the step count."""


def test_euler_error_vanishes_by_the_pipeline_step_count_and_flips_sign_with_the_clock() -> None:
    """Euler's step-size error, and the two things about it that are easy to get backwards."""
    from editjumps.pipeline.evaluate.sampler_step_size import compare

    # The runner's own default length, because the claim being pinned is about that configuration; 250.
    report = compare(120, insert=1.0, delete=0.5, substitute=1.0, step_counts=[2, 50],
                     clocks=[40.0, None], n_traj=250, seed=777)
    clocked = {row["n_steps"]: row for row in report["regimes"]["clock_40"]["euler"]}
    unclocked = {row["n_steps"]: row for row in report["regimes"]["clock_none"]["euler"]}

    # Asserted FIRST because it is what makes the next assertion mean anything: the comparison does detect a.
    assert clocked[2]["edit_distance_resolved"], "n_steps=2 must be distinguishable from exact"
    assert clocked[2]["edit_distance_delta"] > 3.0, "the coarse-h error must be several edits wide"
    for regime, rows in (("clock_40", clocked), ("clock_none", unclocked)):
        assert not rows[50]["edit_distance_resolved"], (
            f"{regime}: at the pipeline's n_steps=50 Euler must be indistinguishable from exact. "
            "If this fires, every Euler run in the paper needs a discretisation caveat it does not "
            "currently carry -- check it against a larger n_traj before believing it"
        )
    # Claim 2: the sign, in both regimes. Same rate field, opposite errors.
    assert clocked[2]["edit_distance_delta"] > 0, "clock-normalised: rates fall, so Euler over-fires"
    assert unclocked[2]["edit_distance_delta"] < 0, "unnormalised: rates rise, so Euler under-fires"
