"""Euler tau-leaping against exact Gillespie on a fixed synthetic rate field, versus step count."""

import random
import statistics
from pathlib import Path
from typing import Annotated

import typer

from editjumps.core.edit_flows.inference import RateFn, euler_trace, gillespie_trace
from editjumps.core.generation_metrics import levenshtein
from editjumps.core.utils import get_logger, write_metrics

logger = get_logger(__file__)

#: Token ids the synthetic field inserts and substitutes with.
FIRST_TOKEN = 5
N_TOKENS = 20


def constant_rate_fn(insert: float, delete: float, substitute: float) -> RateFn:
    """Build a state-independent ``rate_fn(x, t)`` with uniform token distributions."""
    uniform = [1.0 / N_TOKENS] * N_TOKENS

    def rate_fn(x: list[int], t: float) -> tuple:
        """Return the five rate-field components at state ``x``, ignoring ``t``."""
        n = len(x)
        return ([insert] * n, [uniform] * n, [delete] * n, [substitute] * n, [uniform] * n)

    return rate_fn


def _summarise(values: list[float]) -> dict[str, float]:
    """Mean and standard error of the mean."""
    if len(values) < 2:
        return {"mean": float(values[0]) if values else 0.0, "sem": 0.0}
    return {"mean": statistics.fmean(values),
            "sem": statistics.stdev(values) / len(values) ** 0.5}


def run_trajectories(x0: list[int], rate_fn: RateFn, clock: float | None, n_traj: int, *,
                     n_steps: int | None, seed: int) -> dict[str, dict[str, float]]:
    """Simulate ``n_traj`` trajectories with one sampler and summarise two scalars. ``n_steps=None``."""
    lengths: list[float] = []
    distances: list[float] = []
    start = "".join(chr(c) for c in x0)
    for i in range(n_traj):
        rng = random.Random(seed + i)
        if n_steps is None:
            # max_events well above the expected count: the guard must never truncate a trajectory,
            # or the row measures the guard. Same reason max_len is far above any reachable length.
            out, _ = gillespie_trace(x0, rate_fn, rng, max_events=8000, clock=clock, max_len=8000)
        else:
            out, _ = euler_trace(x0, rate_fn, rng, n_steps=n_steps, clock=clock, max_len=8000)
        lengths.append(float(len(out)))
        distances.append(float(levenshtein(start, "".join(chr(c) for c in out))))
    return {"length": _summarise(lengths), "edit_distance": _summarise(distances)}


def compare(length: int, insert: float, delete: float, substitute: float,
            step_counts: list[int], clocks: list[float | None], n_traj: int,
            seed: int) -> dict:
    """Sweep Euler step counts against Gillespie in each clock regime."""
    x0 = [0] + [FIRST_TOKEN + (i % N_TOKENS) for i in range(length - 1)]
    rate_fn = constant_rate_fn(insert, delete, substitute)
    regimes = {}
    for clock in clocks:
        exact = run_trajectories(x0, rate_fn, clock, n_traj, n_steps=None, seed=seed)
        logger.info(f"clock={clock}: gillespie length {exact['length']['mean']:.3f} "
                    f"edits {exact['edit_distance']['mean']:.3f}")
        rows = []
        for n_steps in step_counts:
            approx = run_trajectories(x0, rate_fn, clock, n_traj, n_steps=n_steps, seed=seed)
            row: dict = {"n_steps": n_steps, **approx}
            for key in ("length", "edit_distance"):
                delta = approx[key]["mean"] - exact[key]["mean"]
                # Combined standard error of the difference of two independent means. "Resolved" means the two.
                sem = (approx[key]["sem"] ** 2 + exact[key]["sem"] ** 2) ** 0.5
                row[f"{key}_delta"] = delta
                row[f"{key}_delta_sem"] = sem
                row[f"{key}_resolved"] = bool(abs(delta) > 2.0 * sem)
            rows.append(row)
            logger.info(f"  euler n_steps={n_steps:<5} d_length {row['length_delta']:+7.3f} "
                        f"d_edits {row['edit_distance_delta']:+7.3f} "
                        f"(resolved: {row['edit_distance_resolved']})")
        regimes["clock_none" if clock is None else f"clock_{clock:g}"] = {
            "clock": clock, "gillespie": exact, "euler": rows,
        }
    return {
        "starting_length": length,
        "rates_per_position": {"insert": insert, "delete": delete, "substitute": substitute},
        "n_trajectories": n_traj,
        "seed": seed,
        "rate_field": "constant per position and in time; uniform token distributions",
        "regimes": regimes,
    }


def main(
    length: Annotated[int, typer.Option(help="Starting sequence length, BOS included")] = 120,
    insert: Annotated[float, typer.Option(help="Per-position insertion rate")] = 1.0,
    delete: Annotated[float, typer.Option(help="Per-position deletion rate")] = 0.5,
    substitute: Annotated[float, typer.Option(help="Per-position substitution rate")] = 1.0,
    n_traj: Annotated[int, typer.Option(help="Trajectories per configuration")] = 2000,
    seed: Annotated[int, typer.Option()] = 777,
    metrics_path: Annotated[Path, typer.Option(help="Where to write the report")] = Path(
        "metrics/sampler_step_size.json"
    ),
) -> None:
    """Measure Euler's step-size dependence against exact Gillespie, in both clock regimes."""
    # 50 is the pipeline's default (`edit_flows.n_steps`), so it must be in the sweep: the reportable
    # claim is about the error at the setting the runs actually used, not at a convenient one.
    payload = compare(length, insert, delete, substitute,
                      step_counts=[2, 5, 10, 25, 50, 100], clocks=[40.0, None],
                      n_traj=n_traj, seed=seed)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    write_metrics(metrics_path, payload)
    logger.info(f"wrote {metrics_path}")


if __name__ == "__main__":
    typer.run(main)
