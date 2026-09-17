"""CTMC samplers for integration over t in [0, 1]."""

import math
import random
from typing import Callable, NamedTuple, Sequence

#: Rate field emitted by rate heads: (ins_lambda, ins_q, del_lambda, sub_lambda, sub_q).
RateField = tuple[list[float], list[list[float]], list[float], list[float], list[list[float]]]

#: Rate evaluation callable: rate_fn(x, t) -> RateField.
RateFn = Callable[[list[int], float], RateField]

#: Per-token input provenance tracking.
Provenance = list[int | None]


class EditEvent(NamedTuple):
    """Candidate edit event in the CTMC."""

    kind: str
    position: int


def sample_categorical(probs: Sequence[float], u: float) -> int:
    """Sample an index from probabilities via inverse CDF."""
    cumulative = 0.0
    for i, p in enumerate(probs):
        cumulative += p
        if u <= cumulative:
            return i
    return len(probs) - 1


def next_event_time(total_rate: float, u: float) -> float:
    """Draw waiting time to next event: dt = -ln(1 - u) / total_rate."""
    if total_rate <= 0.0 or u >= 1.0:
        return math.inf
    return -math.log1p(-u) / total_rate


def choose_event(weights: Sequence[float], u: float) -> int:
    """Inverse-CDF pick of WHICH edit fires, proportional to its rate — the other half of §3.3."""
    total = sum(weights)
    if total <= 0.0:
        raise ValueError("choose_event called with all-zero rates; next_event_time returns inf there")
    threshold = u * total
    cumulative = 0.0
    for i, weight in enumerate(weights):
        cumulative += weight
        if threshold <= cumulative:
            return i
    return len(weights) - 1


def clock_scale(clock: float | None, length: int) -> float:
    """Compute clock normalization factor: clock / length."""
    if clock is None or length <= 0:
        return 1.0
    return clock / length


def enabled_events(
    length: int, insert_lambda: Sequence[float], delete_lambda: Sequence[float],
    substitute_lambda: Sequence[float], *, mask: Sequence[bool] | None = None,
    allow_insert: bool = True, scale: float = 1.0,
) -> tuple[list[EditEvent], list[float]]:
    """Enumerate allowed edit events and their scaled rates."""
    events: list[EditEvent] = []
    rates: list[float] = []
    for pos in range(length):
        in_region = mask is None or mask[pos]
        if pos != 0 and in_region:
            events.append(EditEvent("delete", pos))
            rates.append(delete_lambda[pos] * scale)
            events.append(EditEvent("substitute", pos))
            rates.append(substitute_lambda[pos] * scale)
        if in_region and allow_insert:
            events.append(EditEvent("insert", pos))
            rates.append(insert_lambda[pos] * scale)
    return events, rates


def apply_event(
    x: list[int], event: EditEvent, token: int | None, *,
    mask: list[bool] | None = None, provenance: Provenance | None = None,
) -> tuple[list[int], list[bool] | None, Provenance | None]:
    """Apply an edit event to the sequence and update mask and provenance."""
    new_x = list(x)
    new_mask = None if mask is None else list(mask)
    new_prov = None if provenance is None else list(provenance)
    pos = event.position
    if event.kind == "delete":
        del new_x[pos]
        if new_mask is not None:
            del new_mask[pos]
        if new_prov is not None:
            del new_prov[pos]
        return new_x, new_mask, new_prov
    if token is None:
        raise ValueError(f"event {event.kind!r} needs a token to write")
    if event.kind == "substitute":
        new_x[pos] = token
        return new_x, new_mask, new_prov
    if event.kind == "insert":
        new_x.insert(pos + 1, token)
        if new_mask is not None:
            new_mask.insert(pos + 1, new_mask[pos])
        if new_prov is not None:
            new_prov.insert(pos + 1, None)
        return new_x, new_mask, new_prov
    raise ValueError(f"unknown edit kind {event.kind!r}")


def euler_trace(
    x0: Sequence[int], rate_fn: RateFn, rng: random.Random, *, n_steps: int = 50,
    max_len: int = 400, mask: Sequence[bool] | None = None, clock: float | None = None,
    track_provenance: bool = False,
) -> tuple[list[int], Provenance | None]:
    """Euler tau-leaping simulation over n_steps."""
    x = list(x0)
    m = list(mask) if mask is not None else None
    p: Provenance | None = list(range(len(x))) if track_provenance else None
    h = 1.0 / n_steps
    for step in range(n_steps):
        t = step * h
        insert_lambda, insert_q, delete_lambda, substitute_lambda, substitute_q = rate_fn(x, t)
        scale = clock_scale(clock, len(x))
        il = [v * scale for v in insert_lambda]
        dl = [v * scale for v in delete_lambda]
        sl = [v * scale for v in substitute_lambda]
        iq, sq = insert_q, substitute_q
        new_x: list[int] = []
        new_m: list[bool] | None = [] if m is not None else None
        new_p: Provenance | None = [] if p is not None else None
        for pos in range(len(x)):
            editable = pos != 0 and (m is None or m[pos])  # BOS + frozen positions are locked
            region = True if m is None else m[pos]  # region tag carried onto new tokens
            keep = True
            if editable:
                r = rng.random()
                if r < h * dl[pos]:
                    keep = False  # delete
                elif r < h * (dl[pos] + sl[pos]):
                    new_x.append(sample_categorical(sq[pos], rng.random()))  # substitute
                    keep = False
                    if new_m is not None:
                        new_m.append(region)
                    if new_p is not None and p is not None:
                        new_p.append(p[pos])  # a substitution keeps the slot's origin
            if keep:
                new_x.append(x[pos])
                if new_m is not None:
                    new_m.append(region)
                if new_p is not None and p is not None:
                    new_p.append(p[pos])
            # insert after pos only within an editable region (m is None => always, as before)
            if (m is None or m[pos]) and len(new_x) < max_len and rng.random() < h * il[pos]:
                new_x.append(sample_categorical(iq[pos], rng.random()))  # insert after
                if new_m is not None:
                    new_m.append(region)
                if new_p is not None:
                    new_p.append(None)  # inserted material has no origin
        x = new_x
        m = new_m
        p = new_p
        if len(x) >= max_len:
            break
    return x, p


def gillespie_trace(
    x0: Sequence[int], rate_fn: RateFn, rng: random.Random, *, max_events: int = 500,
    max_len: int = 400, mask: Sequence[bool] | None = None, clock: float | None = None,
    track_provenance: bool = False, t_end: float = 1.0,
) -> tuple[list[int], Provenance | None]:
    """Gillespie / SSA next-event simulation of the edit CTMC."""
    x = list(x0)
    m = list(mask) if mask is not None else None
    p: Provenance | None = list(range(len(x))) if track_provenance else None
    t = 0.0
    for _ in range(max_events):
        insert_lambda, insert_q, delete_lambda, substitute_lambda, substitute_q = rate_fn(x, t)
        events, rates = enabled_events(
            len(x), insert_lambda, delete_lambda, substitute_lambda,
            mask=m, allow_insert=len(x) < max_len, scale=clock_scale(clock, len(x)),
        )
        dt = next_event_time(sum(rates), rng.random())
        if t + dt > t_end:
            break  # the next event would land outside the flow's interval: the trajectory is done
        t += dt
        event = events[choose_event(rates, rng.random())]
        token: int | None = None
        if event.kind == "substitute":
            token = sample_categorical(substitute_q[event.position], rng.random())
        elif event.kind == "insert":
            token = sample_categorical(insert_q[event.position], rng.random())
        x, m, p = apply_event(x, event, token, mask=m, provenance=p)
    return x, p
