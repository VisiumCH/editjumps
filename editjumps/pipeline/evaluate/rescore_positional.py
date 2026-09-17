"""Recompute a per-position Appendix-B metric for a committed cell, from the sequences it saved."""

import json
from pathlib import Path
from typing import Annotated

import typer

from editjumps.core.family_split import disjoint_members, split_family, usable_members
from editjumps.core.generation_metrics_undefined import (
    js_divergence_positional,
    kl_divergence_positional,
)
from editjumps.core.utils import get_logger
from editjumps.pipeline.evaluate.generation_eval import project_to_template

logger = get_logger(__file__)

#: Metric name -> the function computing it from (aligned generated, aligned reference).
POSITIONAL = {
    "kl_divergence_positional": kl_divergence_positional,
    "js_divergence_positional": js_divergence_positional,
}


def fasta_header(path: Path) -> dict[str, str]:
    """Parse the leading ``; key: value`` provenance comments of a generated FASTA."""
    header = {}
    for line in path.read_text().splitlines():
        if not line.startswith(";"):
            break
        key, _, value = line[1:].partition(":")
        header[key.strip()] = value.strip()
    return header


def fasta_sets(path: Path) -> dict[str, list[str]]:
    """Every generated set in a saved FASTA, keyed by set name, in the order written."""
    sets: dict[str, list[str]] = {}
    name = None
    for line in path.read_text().splitlines():
        if line.startswith(";") or not line.strip():
            continue
        if line.startswith(">"):
            name = line[1:].rsplit("_", 1)[0]
            sets.setdefault(name, [])
        elif name is not None:
            sets[name].append(line.strip())
    return sets


def recompute(fasta: Path, pairs: Path, metrics: tuple[str, ...]) -> dict[str, dict[str, float]]:
    """Recompute per-position metrics for every set one saved run holds."""
    header = fasta_header(fasta)
    family = header.get("family")
    if not family or family == "None":
        raise ValueError(f"{fasta}: header records no family, so no holdout can be rebuilt")
    n_templates, n_variants = int(header["n_templates"]), int(header["n_variants"])
    seed = int(header["seed"])
    holdout_size = int(header.get("holdout_size", 200))

    sets = fasta_sets(fasta)
    # The §4.3 pairing ceiling is as many DISTINCT natural sequences as the model generated, so it
    # is the one set whose size is allowed to fall short when the family pool cannot supply enough.
    for name, sequences in sets.items():
        if len(sequences) != n_templates * n_variants and "pairing" not in name:
            raise ValueError(
                f"{fasta}: set {name!r} holds {len(sequences)} sequences, header predicts "
                f"{n_templates * n_variants}"
            )

    members = usable_members(read_family(Path(family)))
    if header.get("disjoint_from_pairs") not in (None, "None", "False"):
        members = disjoint_members(members, pairs)
    split = split_family(members, n_templates, holdout_size, seed)

    reference_template = split.templates[0]
    # The same reference the evaluators score against: the family holdout, capped the way
    # generation_eval caps it. Not the per-template partners, which are far too few per position.
    cap = max(300, holdout_size)
    aligned_reference = [project_to_template(s, reference_template)
                         for s in split.reference[:cap]]
    out: dict[str, dict[str, float]] = {}
    for set_name, sequences in sets.items():
        aligned = [project_to_template(s, reference_template) for s in sequences]
        out[set_name] = {name: float(POSITIONAL[name](aligned, aligned_reference))
                         for name in metrics}
    return out


def read_family(path: Path) -> list[str]:
    """Sequences from a seed-family FASTA."""
    if not path.exists():
        raise FileNotFoundError(
            f"no seed family at {path}.\n\n"
            "This tool re-reads a committed metrics FASTA, but it also needs the seed family that "
            "run scored against, and `data/` is git-ignored and DVC-backed. Either:\n"
            "  uv run dvc pull data/interim/seed_families   # needs the private remote\n"
            "  uv run editjumps seed-homologs                 # rebuild from the public corpus\n\n"
            "docs/weights.md says which artefacts are obtainable from outside the project."
        )
    sequences, current = [], []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if current:
                sequences.append("".join(current))
                current = []
        else:
            current.append(line.strip())
    if current:
        sequences.append("".join(current))
    return sequences


def main(
    cells: Annotated[
        list[Path], typer.Argument(help="Metrics JSONs to fill; each needs a sibling FASTA")
    ],
    pairs: Annotated[Path, typer.Option()] = Path("data/pretrain/oas_homolog_pairs.tsv.gz"),
    metric: Annotated[
        list[str], typer.Option(help="Which per-position metrics to fill")
    ] = ["kl_divergence_positional"],
    check_only: Annotated[bool, typer.Option(
        "--check-only",
        help="Recompute and compare against any value already present; write nothing")] = False,
    tolerance: Annotated[float, typer.Option(help="Relative tolerance for --check-only")] = 1e-9,
) -> None:
    """Fill per-position metrics into committed cells from the sequences they saved."""
    unknown = set(metric) - set(POSITIONAL)
    if unknown:
        raise ValueError(f"not per-position metrics: {sorted(unknown)}")
    try:
        _rescore(cells, pairs, metric, check_only, tolerance)
    # A missing seed family is the expected state on a fresh clone (`data/` is git-ignored), and
    # `read_family` already says how to get one -- print that instead of a traceback, as `edit` does.
    except FileNotFoundError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from None


def _rescore(cells: list[Path], pairs: Path, metric: list[str], check_only: bool,
             tolerance: float) -> None:
    """Fill or check each cell; see `main` for the arguments."""
    for cell in cells:
        fasta = cell.with_suffix(".fasta")
        if not fasta.exists():
            raise ValueError(f"{cell} has no sibling FASTA, so nothing can be recomputed")
        report = json.loads(cell.read_text())
        by_set = recompute(fasta, pairs, tuple(metric))
        # Which set is the METHOD's own output, as opposed to a model-free baseline the same run scored.
        primary = next(name for name in by_set if not name.startswith("random_"))
        for set_name, values in by_set.items():
            if set_name == primary:
                block = report.setdefault("our_interpretation", {})
            elif set_name in report.get("baselines", {}):
                block = report["baselines"][set_name]
            else:
                logger.info(f"{cell.name}: set {set_name!r} is in the FASTA but not the JSON, skipped")
                continue
            for name, value in values.items():
                existing = block.get(name)
                if existing is not None:
                    agrees = abs(existing - value) <= tolerance * max(1.0, abs(existing))
                    logger.info(f"{cell.name} [{set_name}] {name}: on record {existing:.10g}, "
                                f"recomputed {value:.10g} -- {'agrees' if agrees else 'DISAGREES'}")
                    if not agrees:
                        raise ValueError(
                            f"{cell}: [{set_name}] {name} recomputes to {value:.10g}, not the "
                            f"{existing:.10g} on record; the reconstruction and the run disagree"
                        )
                    continue
                if check_only:
                    logger.info(f"{cell.name} [{set_name}] {name}: absent, would write {value:.10g}")
                    continue
                block[name] = value
                # Provenance, because a recovered number and an emitted one are not the same claim.
                notes = report.setdefault("reading_notes", {})
                notes.setdefault("recomputed", {})[f"{set_name}.{name}"] = (
                    f"recovered from {fasta.name} by editjumps rescore-positional; this evaluator "
                    "does not emit it, and the value was NOT produced by the run that wrote this file"
                )
                logger.info(f"{cell.name} [{set_name}] {name}: wrote {value:.10g}")
        if not check_only:
            cell.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    typer.run(main)
