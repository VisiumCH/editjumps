"""The two ⑤ steppers, driven by SYNTHETIC rate fields. torch is optional, so the samplers' numeric."""

import random

import pytest

from editjumps.core.edit_flows.inference import RateField, RateFn


def _constant_rate_fn(insert: float, delete: float, substitute: float, vocab: int = 4) -> RateFn:
    """Build a state-independent rate field: same rates everywhere, Q uniform over ``vocab``."""

    def rate_fn(x: list[int], t: float) -> RateField:
        n = len(x)
        row = [1.0 / vocab] * vocab
        return [insert] * n, [row] * n, [delete] * n, [substitute] * n, [row] * n

    return rate_fn

def _count_insertions(provenance: list[int | None] | None) -> int:
    """Count the tokens the sampler inserted, read straight off the provenance."""
    assert provenance is not None
    return sum(1 for origin in provenance if origin is None)

def test_next_event_time_is_the_exponential_inverse_cdf() -> None:
    """The waiting-time draw is exactly -ln(1-u)/R, and degenerates safely."""
    import math

    from editjumps.core.edit_flows.inference import next_event_time

    assert next_event_time(2.0, 0.0) == 0.0
    assert next_event_time(2.0, 0.5) == pytest.approx(math.log(2.0) / 2.0)
    # doubling the total rate halves the waiting time for the same uniform draw
    assert next_event_time(4.0, 0.75) == pytest.approx(next_event_time(2.0, 0.75) / 2.0)
    # nothing can fire -> infinite wait, so the caller stops rather than looping
    assert next_event_time(0.0, 0.5) == math.inf
    assert next_event_time(-1.0, 0.5) == math.inf
    assert next_event_time(1.0, 1.0) == math.inf
    # and the draws really are Exp(R): mean 1/R
    rng = random.Random(0)
    draws = [next_event_time(4.0, rng.random()) for _ in range(20000)]
    assert sum(draws) / len(draws) == pytest.approx(0.25, rel=0.05)

def test_choose_event_is_proportional_to_rate() -> None:
    """The event pick is inverse-CDF over unnormalised rates, boundaries included."""
    from editjumps.core.edit_flows.inference import choose_event

    weights = [1.0, 3.0, 0.0, 4.0]  # total 8: cuts at 0.125, 0.5, 0.5, 1.0
    assert choose_event(weights, 0.0) == 0
    assert choose_event(weights, 0.124) == 0
    assert choose_event(weights, 0.126) == 1
    assert choose_event(weights, 0.499) == 1
    assert choose_event(weights, 0.6) == 3
    assert choose_event(weights, 1.0) == 3
    # a zero-rate event is never chosen
    rng = random.Random(0)
    assert all(choose_event(weights, rng.random()) != 2 for _ in range(2000))
    with pytest.raises(ValueError):
        choose_event([0.0, 0.0], 0.5)

def test_apply_event_moves_provenance_correctly_for_each_edit_kind() -> None:
    """Substitute keeps the origin, delete removes it, insert adds a None — the lockstep contract."""
    from editjumps.core.edit_flows.inference import EditEvent, apply_event

    x: list[int] = [9, 1, 2, 3]
    prov: list[int | None] = [0, 1, 2, 3]
    mask = [False, True, True, True]

    # substitute at 2: same slot, same origin (this is what makes a substitution scoreable)
    sx, sm, sp = apply_event(x, EditEvent("substitute", 2), 7, mask=mask, provenance=prov)
    assert (sx, sp, sm) == ([9, 1, 7, 3], [0, 1, 2, 3], [False, True, True, True])

    # delete at 1: origin 1 simply vanishes, so deletions are a set difference
    dx, dm, dp = apply_event(x, EditEvent("delete", 1), None, mask=mask, provenance=prov)
    assert (dx, dp, dm) == ([9, 2, 3], [0, 2, 3], [False, True, True])

    # insert after 1: a new origin-less token, and the mask inherits the neighbour's region
    ix, im, ip = apply_event(x, EditEvent("insert", 1), 5, mask=mask, provenance=prov)
    assert (ix, ip, im) == ([9, 1, 5, 2, 3], [0, 1, None, 2, 3], [False, True, True, True, True])

    # the inputs are never mutated (both samplers rebind, so aliasing would be silent)
    assert x == [9, 1, 2, 3] and prov == [0, 1, 2, 3] and mask == [False, True, True, True]
    with pytest.raises(ValueError):
        apply_event(x, EditEvent("substitute", 1), None)
    with pytest.raises(ValueError):
        apply_event(x, EditEvent("transpose", 1), 3)

def test_euler_trace_reproduces_the_pre_refactor_loop_exactly() -> None:
    """The Euler stepper was LIFTED out of `sample_edits`, so it must be bit-identical to it."""
    from editjumps.core.edit_flows.inference import euler_trace

    def old_sample(x0: list[int], rate_fn: RateFn, rng: random.Random, n_steps: int,
                   max_len: int, mask: list[bool] | None, clock: float | None) -> list[int]:
        """Run the pre-refactor body of sample_edits, verbatim except for the model call."""
        def old_categorical(probs: list[float]) -> int:
            threshold = rng.random()
            cumulative = 0.0
            for i, p in enumerate(probs):
                cumulative += p
                if threshold <= cumulative:
                    return i
            return len(probs) - 1

        x, m, h = list(x0), (list(mask) if mask is not None else None), 1.0 / n_steps
        for step in range(n_steps):
            t = step * h
            il, iq, dl, sl, sq = rate_fn(x, t)
            if clock is not None and len(x) > 0:
                scale = clock / len(x)
                il = [v * scale for v in il]
                dl = [v * scale for v in dl]
                sl = [v * scale for v in sl]
            new_x: list[int] = []
            new_m: list[bool] | None = [] if m is not None else None
            for pos in range(len(x)):
                editable = pos != 0 and (m is None or m[pos])
                region = True if m is None else m[pos]
                keep = True
                if editable:
                    r = rng.random()
                    if r < h * dl[pos]:
                        keep = False
                    elif r < h * (dl[pos] + sl[pos]):
                        new_x.append(old_categorical(sq[pos]))
                        keep = False
                        if new_m is not None:
                            new_m.append(region)
                if keep:
                    new_x.append(x[pos])
                    if new_m is not None:
                        new_m.append(region)
                if (m is None or m[pos]) and len(new_x) < max_len and rng.random() < h * il[pos]:
                    new_x.append(old_categorical(iq[pos]))
                    if new_m is not None:
                        new_m.append(region)
            x, m = new_x, new_m
            if len(x) >= max_len:
                break
        return x

    rate_fn = _constant_rate_fn(insert=0.6, delete=0.8, substitute=1.4)
    x0 = [0, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2]
    for seed in range(8):
        for clock, mask in [(None, None), (20.0, None), (20.0, [True, True, False] * 4)]:
            want = old_sample(x0, rate_fn, random.Random(seed), 12, 60, mask, clock)
            got, prov = euler_trace(
                x0, rate_fn, random.Random(seed), n_steps=12, max_len=60, mask=mask, clock=clock,
            )
            assert got == want, f"draw order changed at seed={seed} clock={clock}"
            assert prov is None  # provenance is off by default, so no caller's return shape moves

def test_gillespie_fires_exactly_one_edit_per_event_and_reconditions() -> None:
    """One edit per event, and the NEXT event sees the state the previous one left."""
    from editjumps.core.edit_flows.inference import gillespie_trace

    def rate_fn(x: list[int], t: float) -> RateField:
        n = len(x)
        row = [0.0, 0.0, 1.0]  # every write emits token 2
        hot = [0.0] * n
        if len(x) > 1 and x[1] == 1:
            hot[1] = 1000.0  # enormous while unfixed, zero once fixed
        return [0.0] * n, [row] * n, [0.0] * n, hot, [row] * n

    out, prov = gillespie_trace([0, 1, 1, 1], rate_fn, random.Random(0), track_provenance=True)
    assert out == [0, 2, 1, 1]  # exactly one substitution landed, then the field went silent
    assert prov == [0, 1, 2, 3]  # a substitution keeps its slot's origin

def test_gillespie_edit_count_matches_the_analytic_rate_integral() -> None:
    """E[edits] = the time-integral of the total rate — the matched-budget guarantee, as arithmetic."""
    from editjumps.core.edit_flows.inference import gillespie_trace

    rate_fn = _constant_rate_fn(insert=1.0, delete=0.0, substitute=0.0)
    x0 = [0] + [1] * 60
    for clock in (20.0, 50.0):
        counts = [
            _count_insertions(gillespie_trace(
                x0, rate_fn, random.Random(seed), clock=clock, max_events=1000, max_len=1000,
                track_provenance=True,
            )[1])
            for seed in range(300)
        ]
        assert sum(counts) / len(counts) == pytest.approx(clock, rel=0.1)

def test_euler_and_gillespie_agree_on_the_edit_budget_when_h_is_small() -> None:
    """The matched-budget claim, measured — and the boundary where it stops holding."""
    from editjumps.core.edit_flows.inference import euler_trace, gillespie_trace

    rate_fn = _constant_rate_fn(insert=1.0, delete=0.0, substitute=0.0)
    x0 = [0] + [1] * 60

    def mean_inserts(kind: str, n_steps: int, clock: float) -> float:
        """Mean insertion count over many seeds, read off the provenance."""
        counts = []
        for seed in range(200):
            if kind == "euler":
                _, prov = euler_trace(x0, rate_fn, random.Random(seed), n_steps=n_steps,
                                      clock=clock, max_len=1000, track_provenance=True)
            else:
                _, prov = gillespie_trace(x0, rate_fn, random.Random(seed), clock=clock,
                                          max_events=1000, max_len=1000, track_provenance=True)
            counts.append(_count_insertions(prov))
        return sum(counts) / len(counts)

    fine_euler = mean_inserts("euler", 50, 40.0)
    fine_gillespie = mean_inserts("gillespie", 50, 40.0)
    assert fine_euler == pytest.approx(40.0, rel=0.1)
    assert fine_euler == pytest.approx(fine_gillespie, rel=0.1)

    # One step, clock 200: h*lambda > 1, so Euler saturates at the sequence length.
    assert mean_inserts("euler", 1, 200.0) == pytest.approx(len(x0), rel=0.02)
    assert mean_inserts("gillespie", 1, 200.0) == pytest.approx(200.0, rel=0.1)

def test_re_conditioning_does_not_change_set_diversity_at_our_step_size() -> None:
    """The diversity hypothesis, pinned as a NEGATIVE result."""
    import statistics

    from editjumps.core.edit_flows.alignment import levenshtein
    from editjumps.core.edit_flows.inference import euler_trace, gillespie_trace

    bad, vocab = set(range(5)), 20

    def self_limiting(x: list[int], t: float) -> RateField:
        """Substitution rate is hot only where the residue is still one the field wants gone."""
        n = len(x)
        row = [0.0] * 5 + [1.0 / (vocab - 5)] * (vocab - 5)  # writes only 'fixed' residues
        return ([0.05] * n, [row] * n, [0.05] * n,
                [1.0 if tok in bad else 0.02 for tok in x], [row] * n)

    def as_str(ids: list[int]) -> str:
        """Render token ids as characters so `levenshtein` (a string metric) can be reused."""
        return "".join(chr(65 + i) for i in ids)

    def diversity(kind: str, clock: float) -> float:
        """Mean pairwise Levenshtein within each template's variant set, averaged over templates."""
        per_template = []
        for tseed in range(3):
            r = random.Random(tseed)
            x0 = [0] + [r.choice(sorted(bad)) if r.random() < 0.3 else r.randrange(5, vocab)
                        for _ in range(60)]
            variants = []
            for v in range(12):
                rng = random.Random(1000 * tseed + v)
                if kind == "euler":
                    out, _ = euler_trace(x0, self_limiting, rng, n_steps=50, clock=clock)
                else:
                    out, _ = gillespie_trace(x0, self_limiting, rng, clock=clock, max_events=500)
                variants.append(as_str(out))
            pairs = [(i, j) for i in range(len(variants)) for j in range(i + 1, len(variants))]
            per_template.append(statistics.fmean(levenshtein(variants[i], variants[j]) for i, j in pairs))
        return statistics.fmean(per_template)

    for clock in (25.0, 40.0):
        euler, gillespie = diversity("euler", clock), diversity("gillespie", clock)
        # within 10%: too close to source a diversity gap, whether the gap is the retracted ~4x
        # or the 1.1-1.3x that survives
        assert euler == pytest.approx(gillespie, rel=0.10), (
            f"clock={clock}: euler {euler:.2f} vs gillespie {gillespie:.2f}"
        )

def test_gillespie_respects_bos_and_the_framework_mask() -> None:
    """A frozen position is never deleted, substituted, or inserted after — same contract as Euler."""
    from editjumps.core.edit_flows.inference import gillespie_trace

    def delete_everything(x: list[int], t: float) -> RateField:
        row = [0.0, 0.0, 0.0, 0.0, 1.0]
        return [0.0] * len(x), [row] * len(x), [50.0] * len(x), [0.0] * len(x), [row] * len(x)

    def insert_everywhere(x: list[int], t: float) -> RateField:
        row = [0.0, 0.0, 0.0, 0.0, 1.0]  # every write emits token 4
        return [5.0] * len(x), [row] * len(x), [0.0] * len(x), [0.0] * len(x), [row] * len(x)

    x = [0, 1, 2, 3]
    mask = [False, True, False, True]  # freeze position 2 on top of BOS's own lock

    out, prov = gillespie_trace(x, delete_everything, random.Random(0), mask=mask,
                                max_events=200, track_provenance=True)
    assert out == [0, 2]      # only BOS and the frozen residue are left; the field then goes silent
    assert prov == [0, 2]     # and their origins are the input positions they came from

    for seed in range(10):
        out, prov = gillespie_trace(x, insert_everywhere, random.Random(seed), mask=mask,
                                    max_events=200, max_len=60, track_provenance=True)
        assert prov is not None
        # Every inserted run must anchor to an EDITABLE position (1 or 3).
        for i, origin in enumerate(prov):
            if origin is not None:
                continue
            j = i - 1
            while j >= 0 and prov[j] is None:
                j -= 1
            assert prov[j] in (1, 3), f"insertion anchored to locked position {prov[j]}"

def test_provenance_is_ordered_and_covers_every_output_token() -> None:
    """Provenance is a per-token index list: same length as the output, strictly increasing origins."""
    from editjumps.core.edit_flows.inference import euler_trace, gillespie_trace

    rate_fn = _constant_rate_fn(insert=0.5, delete=0.5, substitute=0.7)
    x0 = [0] + [1] * 40
    for kind in ("euler", "gillespie"):
        for seed in range(20):
            if kind == "euler":
                out, prov = euler_trace(x0, rate_fn, random.Random(seed), n_steps=20, clock=30.0,
                                        track_provenance=True)
            else:
                out, prov = gillespie_trace(x0, rate_fn, random.Random(seed), clock=30.0,
                                            track_provenance=True)
            assert prov is not None and len(prov) == len(out)
            origins = [p for p in prov if p is not None]
            assert origins == sorted(origins) and len(origins) == len(set(origins))
            assert all(0 <= o < len(x0) for o in origins)
