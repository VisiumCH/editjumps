"""Collect every §4.2 result into one tidy table, with the metadata that decides comparability."""

import csv
import json
import random
from pathlib import Path
from typing import Annotated

import typer

from editjumps.core.utils import get_logger

logger = get_logger(__file__)

#: (method label, family, path, nested baseline key or None).
DISJOINT: tuple[tuple[str, str, str, str | None], ...] = (
    ("our EvoFlows port", "ty1", "metrics/disjoint/editor-ty1.json", None),
    ("our EvoFlows port", "her2vh", "metrics/disjoint/editor-her2vh.json", None),
    ("random mutations", "ty1", "metrics/disjoint/editor-ty1.json", "random_mutations"),
    ("random mutations", "her2vh", "metrics/disjoint/editor-her2vh.json", "random_mutations"),
    ("random pairing", "ty1", "metrics/disjoint/editor-ty1.json", "random_homolog_pairing"),
    ("random pairing", "her2vh", "metrics/disjoint/editor-her2vh.json", "random_homolog_pairing"),
    ("EvoDiff-MSA", "ty1", "metrics/disjoint/evodiff-msa-ty1.json", None),
    ("EvoDiff-MSA", "her2vh", "metrics/disjoint/evodiff-msa-her2vh.json", None),
    ("Evotuning", "ty1", "metrics/disjoint/evotune-ty1.json", None),
    ("Evotuning", "her2vh", "metrics/disjoint/evotune-her2vh.json", None),
    ("Evotuning (forced)", "ty1", "metrics/disjoint/evotune-forced-ty1.json", None),
    ("Evotuning (forced)", "her2vh", "metrics/disjoint/evotune-forced-her2vh.json", None),
    # All eight files share one frame -- that is what makes the cells comparable at all, and.
)

#: The runs the earlier draft tables were built from, kept because dropping them would hide what the.
LEGACY: tuple[tuple[str, str, str, str | None], ...] = (
    ("our EvoFlows port", "ty1", "metrics/aligned/perseq-ty1.json", None),
    ("our EvoFlows port", "her2vh", "metrics/aligned/perseq-her2vh.json", None),
    ("random mutations", "ty1", "metrics/aligned/perseq-ty1.json", "random_mutations"),
    ("random mutations", "her2vh", "metrics/aligned/perseq-her2vh.json", "random_mutations"),
    ("random pairing", "ty1", "metrics/aligned/perseq-ty1.json", "random_homolog_pairing"),
    ("random pairing", "her2vh", "metrics/aligned/perseq-her2vh.json", "random_homolog_pairing"),
    ("EvoDiff-MSA", "ty1", "metrics/evodiff_msa_baseline.json", None),
    ("EvoDiff-MSA", "her2vh", "metrics/evodiff/her2vh-evodiff_msa_baseline.json", None),
    ("Evotuning", "ty1", "metrics/evotune/ty1-evotune_baseline.json", None),
    ("Evotuning", "her2vh", "metrics/evotune/her2vh-evotune_baseline.json", None),
    ("Evotuning (forced)", "ty1", "metrics/evotune/ty1-evotune_baseline_forced.json", None),
    ("Evotuning (forced)", "her2vh", "metrics/evotune/her2vh-evotune_baseline_forced.json", None),
    ("schedule linear", "ty1", "metrics/schedule/schedlin-ty1.json", None),
    ("schedule linear", "her2vh", "metrics/schedule/schedlin-her2vh.json", None),
    ("schedule cubic", "ty1", "metrics/schedule/schedcub-ty1.json", None),
    ("schedule cubic", "her2vh", "metrics/schedule/schedcub-her2vh.json", None),
)

#: Metric -> (json block, key, scope). ``scope`` is the load-bearing column: ``within_study`` metrics depend on.
METRICS: tuple[tuple[str, str, str, str], ...] = (
    ("edits_per_sequence", "defined_by_the_paper", "levenshtein_to_template", "cross_study"),
    ("pairwise_pooled", "defined_by_the_paper", "pairwise_levenshtein_pooled", "cross_study"),
    ("spectrum_mmd", "defined_by_the_paper", "spectrum_mmd", "cross_study"),
    # KL has TWO readings and their scopes are opposite, which is why these names say which is which..
    ("kl_composition_pooled", "defined_by_the_paper", "kl_generated_vs_natural", "within_study"),
    ("kl_positional", "our_interpretation", "kl_divergence_positional", "cross_study"),
    ("covariance_agreement", "defined_by_the_paper", "covariance_agreement", "within_study"),
    ("covariance_ceiling", "defined_by_the_paper", "covariance_agreement_ceiling", "within_study"),
    ("mip_agreement", "defined_by_the_paper", "mip_agreement", "within_study"),
    ("mip_ceiling", "defined_by_the_paper", "mip_agreement_ceiling", "within_study"),
    ("js_positional", "our_interpretation", "js_divergence_positional", "within_study"),
    ("entropy_delta", "our_interpretation", "entropy_delta", "within_study"),
    ("profile_log_likelihood", "our_interpretation", "profile_log_likelihood", "within_study"),
)

#: Metrics a nested model-free baseline legitimately does not carry, so that a reader counting "6 methods x 2.
BASELINES_LACK: tuple[str, ...] = ("covariance_ceiling", "mip_ceiling", "profile_log_likelihood")

#: Metric -> the artefact key holding the per-unit values its centre is a mean over.
RESAMPLE_FROM: dict[str, str] = {
    "edits_per_sequence": "levenshtein_to_template_per_template",
}

#: The same metric, for a row whose run stored per-SEQUENCE values instead.
RESAMPLE_FROM_SEQUENCES: dict[str, str] = {
    "edits_per_sequence": "levenshtein_to_template_per_sequence",
}

#: Suffix under which a RECOVERED interval sits beside the centre it belongs to, as a dict with ``low`` /.
RECOVERED_INTERVAL_SUFFIX = "_ci"

#: ``ci_kind`` values.
CI_FROM_ARTEFACT = "mean_over_templates"
CI_RECOVERED = "template_resample_spread"

#: Where ``editjumps agreement-interval --against`` writes its paired comparison, one file per seed family.
PAIRED_REPORT = "metrics/paired_agreement_bootstrap_{family}.json"

#: Every paired report to render under a family's table, with the reference it was drawn against.
PAIRED_REPORTS: tuple[tuple[str, str], ...] = (
    (PAIRED_REPORT, "the evotuned PLM"),
)

#: Bootstrap draws and seed.
BOOTSTRAP_DRAWS = 20000
BOOTSTRAP_SEED = 0


def interval(values: list[float], draws: int = BOOTSTRAP_DRAWS,
             seed: int = BOOTSTRAP_SEED) -> tuple[float, float]:
    """Percentile bootstrap over the unit the centre is a mean over."""
    # `random.Random`, not numpy: `metrics/disjoint/README.md` once published bounds from.
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choices(values, k=n)) / n for _ in range(draws))
    return means[int(0.025 * draws)], means[int(0.975 * draws)]


def per_template_means(per_sequence: list[float], n_templates: int) -> list[float]:
    """Collapse a template-major per-sequence list into one mean per template."""
    if n_templates <= 0 or len(per_sequence) % n_templates:
        raise ValueError(
            f"{len(per_sequence)} per-sequence values do not divide into {n_templates} templates; "
            "the list is not template-major and its block means would mix templates"
        )
    width = len(per_sequence) // n_templates
    return [sum(per_sequence[i:i + width]) / width for i in range(0, len(per_sequence), width)]


def alignment_length(report: dict) -> int | None:
    """Pull the projected width out of a run's free-text ``alignment`` field."""
    text = str(report.get("alignment", ""))
    if "L=" not in text:
        return None
    return int(text.split("L=")[1].split(")")[0])


def frame_of(report: dict) -> str:
    """Name the comparability frame from the run's OWN recorded overlap."""
    overlap = report.get("reference_in_training_pairs")
    if overlap is None:
        return "unrecorded"
    return "disjoint" if overlap == 0 else f"train_overlap_{overlap}"


def rows_for(method: str, family: str, path: Path, nested: str | None) -> list[dict]:
    """Emit one tidy row per metric this run reports for this method."""
    if not path.exists():
        logger.warning(f"{method}/{family}: {path} absent, skipped")
        return []
    report = json.loads(path.read_text())
    diagnostic = report.get("mmd_diagnostic", {})
    base = report["baselines"][nested] if nested else None
    out = []
    for name, block, key, scope in METRICS:
        source = base if base is not None else report.get(block, {})
        if key not in source:
            continue
        low = high = None
        ci_kind = ""
        resample_key = RESAMPLE_FROM.get(name)
        recovered = source.get(f"{key}{RECOVERED_INTERVAL_SUFFIX}")
        sequence_key = RESAMPLE_FROM_SEQUENCES.get(name)
        if resample_key and isinstance(source.get(resample_key), list):
            low, high = interval([float(v) for v in source[resample_key]])
            ci_kind = CI_FROM_ARTEFACT
        elif sequence_key and isinstance(source.get(sequence_key), list):
            # A model-free baseline: per-sequence values, template-major, so the correct unit is one reshape.
            per_template = per_template_means(
                [float(v) for v in source[sequence_key]],
                int(report.get("n_templates", 0)),
            )
            low, high = interval(per_template)
            ci_kind = CI_FROM_ARTEFACT
        elif isinstance(recovered, dict):
            low, high = float(recovered["low"]), float(recovered["high"])
            ci_kind = CI_RECOVERED
        out.append({
            "method": method, "family": family, "metric": name, "value": source[key],
            "ci_low": low, "ci_high": high, "ci_kind": ci_kind,
            "scope": scope,
            "frame": frame_of(report),
            "alignment_length": alignment_length(report),
            "reference_n": report.get("defined_by_the_paper", {}).get("agreement_reference_n"),
            "n_generated": diagnostic.get("n_generated"),
            "n_reference": diagnostic.get("n_reference"),
            # The generated set's SHAPE, not just its size.
            "n_templates": report.get("n_templates"),
            # Whether this particular value was emitted by the run or recovered afterwards from the
            # sequences it saved. Both are legitimate; conflating them is not.
            "recomputed": was_recomputed(report, key, nested),
            "source_file": str(path),
        })
    return out


def was_recomputed(report: dict, key: str, nested: str | None) -> bool:
    """Say whether this value was recovered from saved sequences rather than emitted by the run.."""
    notes = report.get("reading_notes", {}).get("recomputed", {})
    if nested is not None:
        return f"{nested}.{key}" in notes
    baselines = tuple(f"{name}." for name in report.get("baselines", {}))
    return any(note.endswith(f".{key}") and not note.startswith(baselines) for note in notes)


def paper_rows(path: Path) -> list[dict]:
    """Read the original's own Figure 3 values as means over its six datasets."""
    if not path.exists():
        return []
    figure = json.loads(path.read_text())
    panels = {"panel_0": "edits_per_sequence", "panel_1": "pairwise_pooled",
              "panel_2": "covariance_agreement", "panel_5": "js_positional",
              "panel_6": "kl_positional", "panel_7": "mip_agreement",
              "panel_8": "spectrum_mmd"}
    out = []
    for panel, metric in panels.items():
        by_method = figure.get(panel, {}).get("by_method", {})
        for name, values in by_method.items():
            scope = next(s for m, _, _, s in METRICS if m == metric)
            out.append({
                "method": f"PAPER: {name}", "family": "paper_mean_of_6", "metric": metric,
                "value": sum(values) / len(values), "ci_low": None, "ci_high": None, "ci_kind": "",
                "scope": scope, "frame": "published", "alignment_length": None,
                "reference_n": None, "n_generated": None, "n_reference": None, "n_templates": None,
                "recomputed": False, "source_file": str(path),
            })
    return out


#: How each metric is rendered in the appendix table, and which way is better.
RENDER: tuple[tuple[str, str, str, int], ...] = (
    ("edits_per_sequence", "edits/seq", "matched, not optimised", 2),
    ("pairwise_pooled", "pairwise Lev.", "closer to *random pairing*", 2),
    ("spectrum_mmd", "spectrum MMD", "lower", 4),
    ("kl_positional", "KL (positional)", "lower", 5),
    ("kl_composition_pooled", "KL (pooled)", "lower", 6),
    ("js_positional", "JS (positional)", "lower", 5),
    ("covariance_agreement", "covariance agr.", "higher", 4),
    ("mip_agreement", "MIP agr.", "higher", 4),
    ("entropy_delta", "entropy delta", "closer to 0", 4),
    ("profile_log_likelihood", "profile LL", "higher", 2),
)


def paired_table(family: str, root: Path) -> list[str]:
    """Render one family's paired agreement comparison, if the report has been generated."""
    lines: list[str] = []
    for template, reference in PAIRED_REPORTS:
        path = root / template.format(family=family)
        if not path.exists():
            continue
        report = json.loads(path.read_text())
        draws = [str(d) for d in sorted(report["draws"])]
        lines += [
            f"### {family}: paired difference against {reference}",
            "",
            "| comparison | metric | margin | "
            + " | ".join(f"paired 95% at {d} draws" for d in draws) + " | verdict |",
            "|---|---|---|" + "---|" * (len(draws) + 1),
        ]
        for record in report["comparisons"]:
            name = Path(str(record["a"]["cell"])).stem
            cells = []
            for count in draws:
                block = record["by_draws"][count]
                cells.append(
                    f"[{block['paired_diff_ci'][0]:+.4f}, {block['paired_diff_ci'][1]:+.4f}]")
            largest = record["by_draws"][draws[-1]]
            verdict = "**includes zero**" if largest["includes_zero"] else "excludes zero"
            lines.append(f"| {name} | {record['metric']} | {record['margin']:+.4f} | "
                         + " | ".join(cells) + f" | {verdict} |")
        lines.append("")
    return lines


def appendix_table(rows: list[dict], root: Path = Path(".")) -> str:
    """Render the disjoint frame as the markdown table the paper's appendix carries."""
    cells = {(r["method"], r["family"], r["metric"]): r for r in rows if r["frame"] == "disjoint"}
    methods = list(dict.fromkeys(
        r["method"] for r in rows if r["frame"] == "disjoint"))
    families = list(dict.fromkeys(
        r["family"] for r in rows if r["frame"] == "disjoint"))
    lines = [
        "# §4.2 results, all metrics, one frame",
        "",
        "Every cell below is drawn the same way: 20 templates x 20 variants, holdout 200, agreement",
        "ceiling 300 real homologs, seed 0, and a scoring reference that shares **nothing** with the",
        "editor's training pairs (`reference_in_training_pairs == 0` in every source file).",
        "Generated by `editjumps consolidate --out-md`; the tidy long form, including the runs drawn in",
        "other frames, is in `metrics/all_results.csv`.",
        "",
    ]
    for family in families:
        header = ["method"] + [label for _, label, _, _ in RENDER]
        arrows = [""] + [{"lower": "lower better", "higher": "higher better"}.get(d, d)
                         for _, _, d, _ in RENDER]
        # Delimiter immediately after the header, then the direction legend as an ordinary body row: GFM only.
        lines += [f"## {family}", "",
                  "| " + " | ".join(header) + " |",
                  "|" + "---|" * len(header),
                  "| " + " | ".join(arrows) + " |"]
        for method in methods:
            out = [method]
            for metric, _, _, digits in RENDER:
                row = cells.get((method, family, metric))
                if row is None:
                    out.append("--")
                    continue
                text = f"{row['value']:.{digits}f}"
                if row["ci_low"] is not None:
                    text += f" [{row['ci_low']:.{digits}f}, {row['ci_high']:.{digits}f}]"
                    # The two kinds of interval must not read alike in the table: one contains its
                    # centre and one does not.
                    if row["ci_kind"] == CI_RECOVERED:
                        text += "‡"
                if row["recomputed"]:
                    text += "*"
                out.append(text)
            lines.append("| " + " | ".join(out) + " |")
        lines.append("")
        lines += paired_table(family, root)
    ceiling = {(r["family"], r["metric"]): r["value"] for r in rows
               if r["frame"] == "disjoint" and r["metric"] in ("covariance_ceiling", "mip_ceiling")}
    lines += [
        "## How to read these",
        "",
        "**The two agreements have a ceiling well below 1.0 and it is not the same number for both.**",
        "300 real homologs scored against the same reference reach "
        + ", ".join(f"{f}: covariance {ceiling.get((f, 'covariance_ceiling'), float('nan')):.4f}, "
                    f"MIP {ceiling.get((f, 'mip_ceiling'), float('nan')):.4f}" for f in families)
        + ". Comparing a method's agreement to 1.0 understates it; comparing it to these does not.",
        "",
        "**Only some columns may be placed beside the published Figure 3.** `edits/seq`, "
        "`pairwise Lev.`, `spectrum MMD` and `KL (positional)` are `cross_study` in "
        "`metrics/all_results.csv`; every other column is `within_study` and depends on this "
        "reference set, so it may be differenced across the rows here and must not be compared "
        "with the paper. `KL (pooled)` is the reading that is ~12x below the paper's panel and is "
        "kept only so the two readings can be seen together.",
        "",
        "**`entropy delta` is signed**, generated minus reference, so nearest zero is best and a "
        "negative value means the method is *under*-diversifying rather than doing well. The two "
        "model-free baselines bracket it: random mutations is far too diverse, and the random-pairing row "
        "is real homologs scored against the 20 per-template partners rather than against "
        "themselves, which is why it is not 0.",
        "",
        "**Every interval here is a percentile bootstrap over TEMPLATES**, the unit every centre in "
        "the table is a mean over. Resampling the 400 sequences instead gives an interval about 2.5x "
        "too narrow, because variants of one template are correlated by construction. There are two "
        "kinds, they are marked apart, and the difference matters:",
        "",
        "* **`edits/seq`** (no mark, 20,000 draws, seed 0, `random.Random`) resamples one value per "
        "TEMPLATE and re-averages. It is a coverage interval on a mean and contains its centre. "
        "Every row has one. Four rows record `levenshtein_to_template_per_template` directly; the "
        "two model-free baselines record `levenshtein_to_template_per_sequence`, which is template-major, "
        "so their per-template means are one reshape away and are recovered rather than skipped.",
        "",
        "* **`covariance agr.` and `MIP agr.`** (marked `‡`, 2,000 draws, seed 0) resample the 20 "
        "TEMPLATE BLOCKS of the run's committed FASTA and recompute the agreement from the "
        "sequences (`editjumps agreement-interval`; the centre is still the run's own, reproduced from "
        "the FASTA to 1e-9). **Two thirds of them do not contain their centre, and that is not a bug.** "
        "(16 of the 24. Which cells bracket is not worth a rule: neither the metric, nor the size "
        "of the shift, nor proximity to the ceiling separates the 8 that bracket from the 16 that "
        "do not -- the two NARROWEST intervals in the set bracket and so does the WIDEST, while the "
        "16 that do not are spread across the whole range in between. Read the warning, not a "
        "pattern: none of the 24 is a coverage interval.) A "
        "with-replacement draw of 20 templates leaves 12.8 distinct ones on average, and these two "
        "estimators move with the effective sample size behind the matrix by more than they move "
        "between methods (see `matrix_agreement`), so a resampled set scores lower than the one "
        "measured. Each cell records that as `bootstrap_shift`: 0.077 to 0.132 on the generated "
        "rows, and only 0.004 to 0.017 on the near-ceiling random-pairing rows, where there is "
        "little room. Read `‡` as *how far the value moves when the twenty templates are redrawn*: "
        "0.15 to 0.23 wide on our port's four cells (0.15 to 0.31 across every generated row), "
        "which is 7 to 38x our port's margin over the evotuned PLM.",
        "",
        "**Do not difference two `‡` intervals.** Overlap between them is not a test, and their "
        "shared downward shift means non-overlap would not be one either. Use the paired tables "
        "above: each draw there resamples ONE set of template indices and scores BOTH methods on it, "
        "so the shift and the template-to-template variance cancel. Pairing is by TEMPLATE, not by "
        "position in the FASTA — `generation_eval` walks a seeded permutation of `split.templates` "
        "while the baselines walk it in order, and differencing on block position inflates the "
        "interval about 3x. Every one of our port's four leads over the evotuned PLM includes zero, "
        "at 100 draws and at 2,000; the same test does separate the forced evotuned baseline from the "
        "unforced one on all four, so it is not merely blunt.",
        "",
        "**`*` marks a value recovered from the run's saved sequences** rather than emitted by the "
        "run itself (`editjumps rescore-positional`). The evotune and EvoDiff-MSA evaluators do not "
        "emit the positional KL; it is recomputed from their committed FASTAs, and the "
        "reconstruction reproduces the editor's own emitted value to ten significant figures.",
        "",
        "**`--` is not a missing measurement.** The two model-free baselines carry no "
        + ", ".join(f"`{m}`" for m in BASELINES_LACK)
        + " for the reasons in `BASELINES_LACK`: the two ceilings are properties of the run rather "
        "than of a method, and the profile log-likelihood is defined against a different reference "
        "than the one a saved FASTA can be rescored in.",
        "",
    ]
    return "\n".join(lines)


def main(
    root: Annotated[Path, typer.Option(help="Repository root")] = Path("."),
    out_json: Annotated[Path, typer.Option()] = Path("metrics/all_results.json"),
    out_csv: Annotated[Path, typer.Option()] = Path("metrics/all_results.csv"),
    out_md: Annotated[Path | None, typer.Option(
        help="Also render the disjoint frame as the paper's appendix table")] = None,
    include_legacy: Annotated[bool, typer.Option(
        help="Also emit the pre-disjoint runs, labelled by their own recorded overlap")] = True,
) -> None:
    """Write every §4.2 result to one tidy JSON and CSV."""
    rows: list[dict] = []
    for method, family, rel, nested in DISJOINT + (LEGACY if include_legacy else ()):
        rows.extend(rows_for(method, family, root / rel, nested))
    rows.extend(paper_rows(root / "metrics/evoflows_figure3.json"))

    (root / out_json).write_text(json.dumps(rows, indent=2))
    with (root / out_csv).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    by_frame: dict[str, int] = {}
    for row in rows:
        by_frame[row["frame"]] = by_frame.get(row["frame"], 0) + 1
    logger.info(f"wrote {len(rows)} rows to {out_json} and {out_csv}")
    for frame, count in sorted(by_frame.items()):
        logger.info(f"  {frame}: {count} rows")
    recomputed = sum(1 for row in rows if row["recomputed"])
    if recomputed:
        logger.info(f"  {recomputed} values recovered from saved sequences, not emitted by the run")
    if out_md is not None:
        (root / out_md).write_text(appendix_table(rows, root))
        logger.info(f"  appendix table: {out_md}")


if __name__ == "__main__":
    typer.run(main)
