"""Template-resampled intervals for a committed cell's SET-LEVEL agreement metrics.."""

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import numpy as np
import typer

from editjumps.core.family_split import disjoint_members, split_family, usable_members
from editjumps.core.generation_metrics import (
    frequencies,
    levenshtein,
    matrix_agreement,
    mutual_information_apc,
    one_hot,
    positional_interaction_strength,
)
from editjumps.core.utils import get_logger, write_metrics
from editjumps.pipeline.evaluate.generation_eval import project_to_template
from editjumps.pipeline.evaluate.rescore_positional import (
    fasta_header,
    fasta_sets,
    read_family,
)

logger = get_logger(__file__)

#: The set-level metrics this fills an interval for, and the artefact block each is read from.
AGREEMENTS: tuple[tuple[str, str], ...] = (
    ("covariance_agreement", "defined_by_the_paper"),
    ("mip_agreement", "defined_by_the_paper"),
)

#: Draw counts, smallest first.
DRAWS: tuple[int, ...] = (100, 2000)

#: Bootstrap seed. Fixed so an interval can be checked against the artefact it came from.
SEED = 0

#: ``mutual_information_apc``'s own default.
EPS = 1e-12


@dataclass(frozen=True)
class Blocks:
    """One projected generated set, regrouped so that a template resample is a matrix product."""

    name: str
    joint: np.ndarray
    position: np.ndarray
    starts: np.ndarray
    length: int
    block_size: int
    templates: tuple[int, ...]


@dataclass(frozen=True)
class Cell:
    """A committed cell, reconstructed far enough to be rescored."""

    path: Path
    header: dict[str, str]
    strength_reference: np.ndarray
    mip_reference: np.ndarray
    sets: dict[str, Blocks]


def segment_sum(matrix: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """Contract a ``(K, K)`` flat-slot matrix to ``(L, L)`` by summing each column's slots."""
    return np.add.reduceat(np.add.reduceat(matrix, starts, axis=0), starts, axis=1)


def compact(name: str, aligned: list[str], block_size: int, templates: tuple[int, ...]) -> Blocks:
    """Regroup a projected set into per-template joint-occupancy counts over occupied slots."""
    if len(aligned) % block_size:
        raise ValueError(
            f"{len(aligned)} sequences do not divide into blocks of {block_size}; the template-major "
            "layout the template resample depends on does not hold"
        )
    encoded = one_hot(aligned)                                   # (N, L, 21), the paper's X (eq 17)
    n, length, symbols = encoded.shape
    flat = encoded.reshape(n, length * symbols)
    # Slots no sequence in this set occupies.
    occupied = np.flatnonzero(flat.any(axis=0))
    position = occupied // symbols
    starts = np.flatnonzero(np.r_[True, position[1:] != position[:-1]])
    x = flat[:, occupied].astype(np.float32)
    blocks = x.reshape(n // block_size, block_size, len(occupied))
    joint = np.matmul(blocks.transpose(0, 2, 1), blocks)         # (T, K, K), counts <= block_size
    return Blocks(name=name, joint=joint, position=position, starts=starts, length=length,
                  block_size=block_size, templates=templates)


def coupling_matrices(blocks: Blocks, counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eq-18 interaction strength and eq-19-22 MIp for one resample of a set's template blocks.."""
    total = np.tensordot(counts.astype(np.float32), blocks.joint, axes=(0, 0)).astype(np.float64)
    f_ij = total / (float(counts.sum()) * blocks.block_size)
    f_i = np.diag(f_ij).copy()
    outer = np.outer(f_i, f_i)
    strength = np.sqrt(segment_sum((f_ij - outer) ** 2, blocks.starts))
    ratio = np.where(f_ij > 0, f_ij / np.maximum(outer, EPS), 1.0)
    information = segment_sum(f_ij * np.log(np.maximum(ratio, EPS)), blocks.starts)
    row, col, mean = information.mean(axis=1), information.mean(axis=0), information.mean()
    return strength, information - np.outer(row, col) / max(mean, EPS)


def agreements(cell: Cell, name: str, counts: np.ndarray) -> dict[str, float]:
    """Score one set's resample against the cell's fixed reference."""
    strength, mip = coupling_matrices(cell.sets[name], counts)
    return {
        "covariance_agreement": matrix_agreement(strength, cell.strength_reference),
        "mip_agreement": matrix_agreement(mip, cell.mip_reference),
    }


def check_fast_path(aligned: list[str], blocks: Blocks, tolerance: float = 1e-9) -> None:
    """Assert the regrouped path reproduces the library functions on the identity draw."""
    f_i, f_ij, c_ij = frequencies(one_hot(aligned))
    want_strength = positional_interaction_strength(c_ij)
    want_mip = mutual_information_apc(f_i, f_ij)
    got_strength, got_mip = coupling_matrices(blocks, np.ones(len(blocks.joint)))
    for label, want, got in (("strength", want_strength, got_strength), ("MIp", want_mip, got_mip)):
        gap = float(np.abs(want - got).max())
        if gap > tolerance:
            raise ValueError(
                f"the block-regrouped {label} matrix differs from generation_metrics' by {gap:.3g} "
                "on the identity draw; every interval from it would be wrong"
            )


def assign_blocks(cost: np.ndarray) -> tuple[int, ...]:
    """Solve the block-to-template assignment that minimises total distance. ``scipy``, not a greedy."""
    from scipy.optimize import linear_sum_assignment

    rows, columns = linear_sum_assignment(cost)
    by_row = dict(zip(rows.tolist(), columns.tolist(), strict=True))
    return tuple(int(by_row[block]) for block in range(cost.shape[0]))


def template_of_each_block(sequences: list[str], templates: list[str], block_size: int
                           ) -> tuple[int, ...]:
    """Identify which split template each block of a saved set was generated from."""
    blocks = [sequences[start:start + block_size] for start in range(0, len(sequences), block_size)]
    order = [min(range(len(templates)), key=lambda i: levenshtein(block[0], templates[i]))
             for block in blocks]
    if sorted(order) == list(range(len(templates))):
        return tuple(order)
    refusal = (f"per-block nearest gives {order}, which is not a permutation of the "
               f"{len(templates)} templates")
    if len(blocks) != len(templates):
        raise ValueError(f"{refusal}, and {len(blocks)} blocks against {len(templates)} templates "
                         "admits no bijection either, so these blocks cannot be paired with "
                         "another method's by template")
    if block_size < 2:
        raise ValueError(f"{refusal}, and a block of {block_size} cannot be split in half, so the "
                         "assignment has nothing to check itself against and these blocks cannot "
                         "be paired with another method's by template")

    # (B, V, T).
    distances = np.array([[[float(levenshtein(variant, template)) for template in templates]
                           for variant in block] for block in blocks])
    full = assign_blocks(distances.mean(axis=1))
    halves = (assign_blocks(distances[:, 0::2].mean(axis=1)),
              assign_blocks(distances[:, 1::2].mean(axis=1)))
    if halves[0] != halves[1] or full != halves[0]:
        raise ValueError(
            f"{refusal}, and the minimum-cost assignment does not survive a split-half check: the "
            f"even-indexed variants of each block give {halves[0]} and the odd-indexed ones give "
            f"{halves[1]}. A set whose blocks really came from templates gives the same answer from "
            "either half; this one does not, so these blocks cannot be paired with another method's "
            "by template"
        )
    chosen = distances.mean(axis=1)[range(len(blocks)), full]
    logger.info(
        f"block-to-template recovery fell back to minimum-cost assignment: {full}, at a mean "
        f"distance of {chosen.mean():.1f} against {distances.mean():.1f} for a random pairing, and "
        f"agreeing between both disjoint halves of every block"
    )
    return tuple(full)


def load_cell(path: Path, pairs: Path) -> Cell:
    """Reconstruct a committed cell's alignment frame, reference and projected sets."""
    fasta = path.with_suffix(".fasta")
    if not fasta.exists():
        raise ValueError(f"{path} has no sibling FASTA, so nothing can be resampled")
    header = fasta_header(fasta)
    family = header.get("family")
    if not family or family == "None":
        raise ValueError(f"{fasta}: header records no family, so no holdout can be rebuilt")
    n_templates, n_variants = int(header["n_templates"]), int(header["n_variants"])
    holdout_size = int(header.get("holdout_size", 200))

    members = usable_members(read_family(Path(family)))
    if header.get("disjoint_from_pairs") not in (None, "None", "False"):
        members = disjoint_members(members, pairs)
    split = split_family(members, n_templates, holdout_size, int(header["seed"]))

    # The frame every evaluator in metrics/disjoint/ scores in: split.templates[0] for the
    # alignment, and the family holdout capped exactly as generation_eval caps it for the reference.
    reference_template = split.templates[0]
    cap = max(300, holdout_size)
    aligned_reference = [project_to_template(s, reference_template) for s in split.reference[:cap]]
    f_i, f_ij, c_ij = frequencies(one_hot(aligned_reference))

    sets: dict[str, Blocks] = {}
    for name, sequences in fasta_sets(fasta).items():
        if len(sequences) % n_variants:
            # The random-pairing set is the one allowed to fall short, when the family pool cannot supply as.
            logger.info(f"{path.name} [{name}]: {len(sequences)} sequences do not divide into "
                        f"blocks of {n_variants}, no template unit to resample, skipped")
            continue
        aligned = [project_to_template(s, reference_template) for s in sequences]
        try:
            order = template_of_each_block(sequences, split.templates, n_variants)
        except ValueError as exc:
            # Pairing is unavailable for this set; the unpaired interval still is, because blocks
            # are exchangeable whether or not we can say which template each came from.
            logger.info(f"{path.name} [{name}]: not pairable by template ({exc})")
            order = ()
        blocks = compact(name, aligned, n_variants, order)
        check_fast_path(aligned, blocks)
        sets[name] = blocks
    return Cell(path=path, header=header,
                strength_reference=positional_interaction_strength(c_ij),
                mip_reference=mutual_information_apc(f_i, f_ij), sets=sets)


def percentile_interval(values: np.ndarray) -> tuple[float, float]:
    """Take the 2.5th and 97.5th order statistics of a bootstrap distribution."""
    ordered = np.sort(values)
    draws = len(ordered)
    return float(ordered[int(0.025 * draws)]), float(ordered[int(0.975 * draws)])


def draw_counts(rng: np.random.Generator, n_templates: int, draws: int) -> np.ndarray:
    """Resample template indices with replacement, as multiplicity counts."""
    picks = rng.integers(0, n_templates, size=(draws, n_templates))
    counts = np.zeros((draws, n_templates), dtype=np.int64)
    np.add.at(counts, (np.repeat(np.arange(draws), n_templates), picks.ravel()), 1)
    return counts


def primary_set(cell: Cell) -> str:
    """Name the set that is the METHOD's own output, not a model-free baseline the same run scored."""
    return next(name for name in cell.sets if not name.startswith("random_"))


def bootstrap(cell: Cell, name: str, counts: np.ndarray) -> dict[str, np.ndarray]:
    """Score one set over every draw."""
    out: dict[str, list[float]] = {metric: [] for metric, _ in AGREEMENTS}
    for row in counts:
        scored = agreements(cell, name, row)
        for metric in out:
            out[metric].append(scored[metric])
    return {metric: np.array(values) for metric, values in out.items()}


def blocks_for_templates(cell: Cell, name: str, counts: np.ndarray) -> np.ndarray:
    """Translate template multiplicities into this set's BLOCK multiplicities."""
    order = cell.sets[name].templates
    if not order:
        raise ValueError(f"{cell.path.name} [{name}] has no block-to-template assignment")
    # order[b] is the template block b came from, so block b's multiplicity is that template's.
    return counts[:, np.array(order)]


def compare(left: Cell, right: Cell, seed: int, counts_by_draws: dict[int, np.ndarray]) -> list[dict]:
    """Paired bootstrap of two cells' agreements over one shared set of template resamples."""
    for field in ("family", "seed", "n_templates", "n_variants"):
        if left.header.get(field) != right.header.get(field):
            raise ValueError(
                f"{left.path.name} and {right.path.name} disagree about {field}: "
                f"{left.header.get(field)!r} vs {right.header.get(field)!r}; they are not one frame"
            )
    if left.strength_reference.shape != right.strength_reference.shape:
        raise ValueError(
            f"{left.path.name} and {right.path.name} are in different alignments: "
            f"{left.strength_reference.shape} vs {right.strength_reference.shape}"
        )
    a_name, b_name = primary_set(left), primary_set(right)
    identity = np.ones((1, len(left.sets[a_name].joint)), dtype=np.int64)
    centre = {
        "a": agreements(left, a_name, blocks_for_templates(left, a_name, identity)[0]),
        "b": agreements(right, b_name, blocks_for_templates(right, b_name, identity)[0]),
    }
    records: dict[str, dict] = {}
    for metric, _ in AGREEMENTS:
        records[metric] = {
            "family": left.header.get("family"),
            "metric": metric,
            "a": {"cell": str(left.path), "set": a_name, "value": centre["a"][metric]},
            "b": {"cell": str(right.path), "set": b_name, "value": centre["b"][metric]},
            "margin": centre["a"][metric] - centre["b"][metric],
            "seed": seed,
            "unit": "template",
            "by_draws": {},
        }
    for draws, counts in sorted(counts_by_draws.items()):
        started = time.time()
        a_values = bootstrap(left, a_name, blocks_for_templates(left, a_name, counts))
        b_values = bootstrap(right, b_name, blocks_for_templates(right, b_name, counts))
        # The same difference with the template mapping DROPPED, i.e. paired on raw block position.
        a_unaligned = bootstrap(left, a_name, counts)
        b_unaligned = bootstrap(right, b_name, counts)
        for metric, _ in AGREEMENTS:
            record = records[metric]
            diff = a_values[metric] - b_values[metric]
            unaligned = a_unaligned[metric] - b_unaligned[metric]
            low, high = percentile_interval(diff)
            record["by_draws"][str(draws)] = {
                "paired_diff_ci": [low, high],
                "includes_zero": bool(low <= 0.0 <= high),
                # The bootstrap's own centre for the difference.
                "paired_diff_mean": float(diff.mean()),
                # One-sided share of draws in which the lead survives.
                "share_of_draws_a_ahead": float((diff > 0).mean()),
                "a_ci": list(percentile_interval(a_values[metric])),
                "b_ci": list(percentile_interval(b_values[metric])),
                "a_bootstrap_mean": float(a_values[metric].mean()),
                "b_bootstrap_mean": float(b_values[metric].mean()),
                "if_paired_on_raw_block_index": {
                    "paired_diff_ci": list(percentile_interval(unaligned)),
                    "note": (
                        "WRONG PAIRING, kept as a diagnostic. generation_eval walks a seeded "
                        "permutation of split.templates and the baselines walk it in order, so "
                        "block i is not the same template in the two files; differencing on block "
                        "index cancels nothing and widens the interval by about 3x."
                    ),
                },
            }
        logger.info(f"{left.path.name} vs {right.path.name}: {draws} paired draws in "
                    f"{time.time() - started:.1f}s")
    return list(records.values())


def _reads_as(shift: float, contains_centre: bool) -> str:
    """Describe an interval by what was measured about it, not by a rule assumed to hold."""
    direction = "BELOW" if shift < 0 else "ABOVE"
    measured = (
        f"Measured here, the bootstrap's own centre sits {abs(shift):.4f} {direction} this cell's"
    )
    if not contains_centre:
        placement = (
            ", and the interval does NOT contain the centre -- so it is not a coverage interval on "
            "it, and a reader who takes it for one will conclude the opposite of what it says"
        )
    else:
        placement = (
            ", and the interval does happen to bracket the centre. That does NOT make it a coverage "
            "interval: it is still the spread of a redrawn template set, and which cells bracket is "
            "not predictable from the metric, the size of the shift or the distance to the ceiling "
            "-- across the 24 committed cells the two NARROWEST intervals bracket and so does the "
            "WIDEST, while the 16 that do not are spread across the whole range in between"
        )
    return (
        "SPREAD, not coverage. Resampling 20 templates with replacement leaves 12.8 distinct ones "
        "on average, and this estimator's value moves with the number of sequences behind the "
        f"matrix (see matrix_agreement's docstring). {measured}{placement}. Read it as how far the "
        "value moves when the twenty templates are redrawn. To compare two methods use the PAIRED "
        "difference (editjumps agreement-interval --against), where the shift largely cancels; the "
        "overlap of two of these intervals is not a test."
    )


def own_intervals(cell: Cell, counts_by_draws: dict[int, np.ndarray], seed: int
                  ) -> dict[str, dict[str, dict]]:
    """Unpaired template-resampled intervals for every set of one cell."""
    out: dict[str, dict[str, dict]] = {}
    for name in cell.sets:
        per_draws = {}
        for draws, counts in sorted(counts_by_draws.items()):
            started = time.time()
            mapped = blocks_for_templates(cell, name, counts) if cell.sets[name].templates else counts
            per_draws[draws] = bootstrap(cell, name, mapped)
            logger.info(f"{cell.path.name} [{name}]: {draws} draws in {time.time() - started:.1f}s")
        largest = max(per_draws)
        # This set's own centre, from the identity draw, so `reads_as` can state what was MEASURED about the.
        identity = np.ones((1, len(cell.sets[name].joint)), dtype=np.int64)
        mapped_identity = (blocks_for_templates(cell, name, identity)
                           if cell.sets[name].templates else identity)
        centres = agreements(cell, name, mapped_identity[0])
        out[name] = {}
        for metric, _ in AGREEMENTS:
            low, high = percentile_interval(per_draws[largest][metric])
            bootstrap_mean = float(per_draws[largest][metric].mean())
            shift = bootstrap_mean - centres[metric]
            contains_centre = bool(low <= centres[metric] <= high)
            out[name][metric] = {
                "low": low, "high": high, "draws": largest, "seed": seed, "unit": "template",
                # The bootstrap distribution's own centre, recorded because it is usually NOT the
                # artefact's centre, and the gap is what `reads_as` describes.
                "bootstrap_mean": bootstrap_mean,
                # Both recorded, because the direction of the shift is a MEASUREMENT and not a property of the.
                "bootstrap_shift": shift,
                "contains_centre": contains_centre,
                "reads_as": _reads_as(shift, contains_centre),
                "at_other_draw_counts": {
                    str(draws): list(percentile_interval(values[metric]))
                    for draws, values in per_draws.items() if draws != largest
                },
            }
    return out


def write_intervals(cell: Cell, intervals: dict[str, dict[str, dict]], tolerance: float) -> None:
    """Write each set's intervals into the cell's JSON, beside the centre and stamped as recovered."""
    report = json.loads(cell.path.read_text())
    fasta = cell.path.with_suffix(".fasta")
    for name, per_metric in intervals.items():
        recomputed = agreements(cell, name, np.ones(len(cell.sets[name].joint)))
        if name == primary_set(cell):
            block = report.setdefault("defined_by_the_paper", {})
        elif name in report.get("baselines", {}):
            block = report["baselines"][name]
        else:
            logger.info(f"{cell.path.name}: set {name!r} is in the FASTA but not the JSON, skipped")
            continue
        for metric, interval in per_metric.items():
            existing = block.get(metric)
            if existing is None:
                logger.info(f"{cell.path.name} [{name}] {metric}: no centre on record, skipped")
                continue
            if abs(existing - recomputed[metric]) > tolerance:
                raise ValueError(
                    f"{cell.path}: [{name}] {metric} recomputes to {recomputed[metric]:.10g}, not "
                    f"the {existing:.10g} on record; the reconstruction and the run disagree, so "
                    "the interval would not belong to this centre"
                )
            block[f"{metric}_ci"] = interval
            notes = report.setdefault("reading_notes", {})
            notes.setdefault("recomputed_intervals", {})[f"{name}.{metric}_ci"] = (
                f"percentile bootstrap over the 20 TEMPLATE blocks of {fasta.name}, "
                f"{interval['draws']} draws, seed {interval['seed']}, recovered by "
                "editjumps agreement-interval. The CENTRE is this run's own emitted value (checked to "
                f"{tolerance:g}); the INTERVAL was not produced by the run -- a set-level metric is "
                "computed from all 400 sequences at once, so this evaluator records no per-template "
                "values for it and the interval had to be obtained by recomputing the metric from "
                "the saved sequences. Paired comparisons between two methods must use the paired "
                "difference (editjumps agreement-interval --against), not the overlap of two of these."
            )
            logger.info(f"{cell.path.name} [{name}] {metric}: {existing:.4f} "
                        f"[{interval['low']:.4f}, {interval['high']:.4f}] "
                        f"({interval['draws']} draws)")
    cell.path.write_text(json.dumps(report, indent=2))


def main(
    cells: Annotated[
        list[Path], typer.Argument(help="Metrics JSONs to interval; each needs a sibling FASTA")
    ],
    against: Annotated[Path | None, typer.Option(
        help="Also run the PAIRED comparison of every cell against this one (same family, same "
             "frame). This is the test to use for a lead: two unpaired intervals overlapping is "
             "not it.")] = None,
    pairs: Annotated[Path, typer.Option()] = Path("data/pretrain/oas_homolog_pairs.tsv.gz"),
    draws: Annotated[list[int], typer.Option(
        help="Bootstrap draw counts; repeatable. Every count is reported, the largest is written.",
    )] = list(DRAWS),
    seed: Annotated[int, typer.Option(help="Bootstrap seed, so the interval is reproducible")] = SEED,
    report: Annotated[Path | None, typer.Option(
        help="Where to write the paired comparison as JSON")] = None,
    check_only: Annotated[bool, typer.Option(
        "--check-only", help="Report what would be written, and write nothing")] = False,
    tolerance: Annotated[float, typer.Option(
        help="Absolute tolerance on the recomputed centre against the one on record")] = 1e-9,
) -> None:
    """Give a committed cell's set-level agreements a template-resampled interval."""
    # A missing seed family is the expected state on a fresh clone (`data/` is git-ignored), and
    # `read_family` already says how to get one -- print that instead of a traceback, as `edit` does.
    try:
        loaded = {path: load_cell(path, pairs) for path in cells}
    except FileNotFoundError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from None
    reference = loaded[against] if against in loaded else (
        load_cell(against, pairs) if against is not None else None)

    first = next(iter(loaded.values()))
    n_templates = len(first.sets[primary_set(first)].joint)
    # ONE generator for the largest count, then prefixes for the smaller ones.
    rng = np.random.default_rng(seed)
    largest = draw_counts(rng, n_templates, max(draws))
    counts_by_draws = {count: largest[:count] for count in sorted(draws)}

    for path, cell in loaded.items():
        by_set = own_intervals(cell, counts_by_draws, seed)
        if check_only:
            for name, per_metric in by_set.items():
                for metric, interval in per_metric.items():
                    logger.info(f"{path.name} [{name}] {metric}: would write "
                                f"[{interval['low']:.4f}, {interval['high']:.4f}]")
            continue
        write_intervals(cell, by_set, tolerance)

    if reference is None:
        return
    records = []
    for path, cell in loaded.items():
        if path == against:
            continue
        records.extend(compare(cell, reference, seed, counts_by_draws))
    for record in records:
        for count, block in record["by_draws"].items():
            verdict = "INCLUDES ZERO" if block["includes_zero"] else "excludes zero"
            logger.info(
                f"{Path(str(record['a']['cell'])).stem} - {Path(str(record['b']['cell'])).stem} "
                f"{record['metric']}: margin {record['margin']:+.4f}, paired 95% at {count} draws "
                f"[{block['paired_diff_ci'][0]:+.4f}, {block['paired_diff_ci'][1]:+.4f}] "
                f"(mean {block['paired_diff_mean']:+.4f}, "
                f"{100 * block['share_of_draws_a_ahead']:.0f}% of draws ahead) -- {verdict}"
            )
    if report is not None and not check_only:
        write_metrics(report, {
            "seed": seed, "draws": sorted(draws), "unit": "template",
            "pairs": str(pairs),
            "method": (
                "Paired percentile bootstrap over the 20 template blocks of each cell's committed "
                "FASTA. Each draw resamples ONE set of template indices and scores BOTH methods on "
                "it, so the variance the two share -- which templates the split handed them -- "
                "cancels. Blocks are matched to templates by nearest edit distance, because "
                "generation_eval walks a seeded permutation of split.templates while the baselines "
                "walk it in order; at seed 0 the recovered mapping is exactly "
                "random.Random(0).sample(range(20), 20) for both families, which is the independent "
                "check on it. Centres are the runs' own emitted values, reproduced from the "
                "committed FASTAs to 1e-9. Read paired_diff_ci; the marginal a_ci/b_ci are SPREADS "
                "shifted ~0.08 below their centres, because a with-replacement draw leaves 12.8 "
                "distinct templates and this estimator moves with the effective sample size."
            ),
            "template_mapping": {
                str(path): {name: list(cell.sets[name].templates) for name in cell.sets}
                for path, cell in ({**loaded, against: reference}).items()
            },
            "comparisons": records,
        })
        logger.info(f"paired comparison: {report}")


if __name__ == "__main__":
    typer.run(main)
