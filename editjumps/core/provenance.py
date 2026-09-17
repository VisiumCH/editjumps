"""Pipeline stage research claim mapping."""

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

import typer

Claim = Literal["reproduction"]

#: What each claim means, for the generated table's legend.
CLAIMS: dict[Claim, str] = {
    "reproduction": "Implements an EvoFlows/Edit Flows decision. Evidence about the paper.",
}
# One entry, deliberately: see the module docstring. Adding a stage that is not a reproduction
# means widening `Claim` and this table first, which is a visible edit.

#: Claims that reproduce SOMEONE ELSE'S published specification, so ``deviation`` is meaningful for them: there.
PAPER_CLAIMS: frozenset[str] = frozenset({"reproduction"})


@dataclass(frozen=True)
class Provenance:
    """Which research claim a pipeline stage belongs to."""

    claim: Claim
    why: str
    deviation: bool = False


#: Stage name in ``dvc.yaml`` -> which claim it serves. Exhaustive; a test enforces that.
STAGE_PROVENANCE: dict[str, Provenance] = {
    # --- the EvoFlows reproduction -------------------------------------------------------------
    "download_oas": Provenance(
        "reproduction", "§3.1/§4.2 sequence database — they use UniRef30 + a ColabFold "
        "environmental DB; antibody-only OAS is ours", deviation=True),
    "seed_homologs": Provenance(
        "reproduction", "built only to put numbers in their Table 2's columns"),
    "split_corpus": Provenance(
        "reproduction", "§4.2 train / inference / holdout split; their ratios are not stated"),
    "fetch_base_checkpoint": Provenance(
        "reproduction", "§3.4 'the encoder trunk of a pre-trained ESM-2 model'"),
    "pretrain_esm": Provenance(
        "reproduction", "§3.4 states NO domain-adaptive pretraining; our OAS-adapted trunk is "
        "arm A of the 2x2, an addition rather than a reproduction step", deviation=True),
    "build_homolog_pairs": Provenance(
        "reproduction", "§4.2 pair construction + eq 7-8; MMseqs clustering replaces their "
        "iterative profile search, and pairs are capped per family", deviation=True),
    "train_edit_flows": Provenance(
        "reproduction", "eq 6 Bregman rate-matching loss, eq 13-16 architecture"),
    "build_deterministic_pairs": Provenance(
        "reproduction", "§4.1's synthetic dataset — the rules verbatim, applied to natural sequences"),
    "train_deterministic_editor": Provenance(
        "reproduction", "§4.1 trains on those pairs; evaluating an editor that never saw the rules "
        "measures zero-shot transfer, not the paper's experiment"),
    "deterministic_benchmark": Provenance(
        "reproduction", "§4.1 synthetic ground truth — the only evaluation with a known answer"),
    "evotune_esm": Provenance(
        "reproduction", "§2.2 Evotuning (Alley et al., 2019) — the same MLM objective on one "
        "homolog family; the family and its train/holdout split are ours"),
    "evotune_baseline": Provenance(
        "reproduction", "§4.3's 'Evotuned PLM' baseline — eq 12 entropy profile picks positions, "
        "the evotuned MLM infills them"),
    "evotune_baseline_forced": Provenance(
        "reproduction", "§4.3's 'Evotuned PLM with forced substitutions' — same, with the original "
        "amino acid blocked at every masked position"),
    "evodiff_msa_baseline": Provenance(
        "reproduction", "§4.2's 'EvoDiff-MSA (Alamdari et al., 2024)' baseline — their released MSA "
        "model and weights; the mutation-budget masking that makes it §4.2-matched is ours",
        deviation=True),
}


def by_claim(claim: Claim) -> list[str]:
    """Stage names serving one claim, in ``STAGE_PROVENANCE`` registration order."""
    if claim not in CLAIMS:
        raise ValueError(f"unknown claim {claim!r}; options: {sorted(CLAIMS)}")
    return [name for name, prov in STAGE_PROVENANCE.items() if prov.claim == claim]


def markdown_table() -> str:
    """Render the provenance map as the README's table."""
    lines: list[str] = []
    for claim, meaning in CLAIMS.items():
        stages = by_claim(claim)
        lines.append(f"### `{claim}` — {len(stages)} stages\n")
        lines.append(f"{meaning}\n")
        lines.append("| stage | why |")
        lines.append("|---|---|")
        for name in stages:
            prov = STAGE_PROVENANCE[name]
            flag = " **(deviation)**" if prov.deviation else ""
            lines.append(f"| `{name}` | {prov.why}{flag} |")
        lines.append("")
    return "\n".join(lines)



#: Stages that produce a measurement and deliberately open NO MLflow run, each with the reason.
MLFLOW_EXEMPT: dict[str, str] = {
    "figure_extract": (
        "digitises a figure out of SOMEONE ELSE'S PDF. The numbers are the source's, not a "
        "measurement of ours, so there is no model, no params and nothing for a run to link."
    ),
    "standard": (
        "the `editjumps evaluate` front door dispatches to the editor and the three baselines, each of "
        "which now opens its own run. An outer run would duplicate every metric one level up and "
        "make the per-method runs harder to find, which is the opposite of the point."
    ),
    "sampler_step_size": (
        "model-free CPU diagnostic over a fixed synthetic rate field: no checkpoint, no training "
        "data, no artefact. A run would record params that fully determine the output."
    ),
    "calibrate_throughput": (
        "measures s/step and peak memory of the MACHINE, not of a model. Its numbers describe the "
        "hardware a run happened to land on and are not comparable across boxes."
    ),
    "consolidate": (
        "reads committed artefacts and emits a table. Every number in it already belongs to a "
        "tracked run, and the table records the source file for each one, so an outer run would be "
        "a second copy of those numbers attached to no model and no data."
    ),
    "rescore_positional": (
        "recovers a per-position metric from sequences a tracked run already saved, and is fully "
        "determined by that FASTA plus the family split its own header records. It writes the value "
        "into the artefact under `reading_notes.recomputed`, which is where a reader looks for this "
        "provenance -- an MLflow run would put it somewhere they would not. Listed here even though "
        "the coverage check does not flag it: the check keys on the literal `metrics/`, which this "
        "module happens not to contain, and a classification should not rest on that."
    ),
    "agreement_interval": (
        "recovers an INTERVAL, not a measurement, from sequences a tracked run already saved, and is "
        "fully determined by that FASTA plus the family split its own header records and a fixed "
        "seed. Same reasoning as `rescore_positional`, and the same destination: it writes into the "
        "artefact beside the centre it belongs to, under `reading_notes.recomputed_intervals`, which "
        "is where a reader checking whether a number is the run's own will look. A run of its own "
        "would attach an interval to no model and no data, one hop away from the centre it qualifies."
    ),
    "seed_homologs": (
        "a deterministic data build. It reports family sizes into the paper's Table 2 columns, but "
        "those are properties of the corpus and the clustering parameters, both DVC-tracked."
    ),
    # One entry, and it is a debt rather than a clean exemption: it measures a real thing, has
    # params worth recording, and SHOULD be tracked. Recorded here so it stays visible.
    "alignment_scoring": (
        "reproduction side and a genuine gap. Measures what the alignment scoring does to the "
        "edit-op label distribution; it has params worth recording and should be tracked."
    ),
}


#: Our metric key -> the Figure 3 panel it is meant to reproduce, for the range check in.
FIGURE3_PANEL_FOR_METRIC: dict[str, str] = {
    "levenshtein_to_template": "Avg Levenshtein to x0",
    "pairwise_levenshtein_pooled": "Avg pairwise Levenshtein",
    "covariance_agreement": "Covariance",
    "entropy_delta": "Entropy delta",
    "js_divergence_positional": "JS divergence",
    "kl_divergence_positional": "KL divergence",
    "mip_agreement": "MIP",
    "spectrum_mmd": "MMD",
}

#: Metrics whose value is outside its panel's range, with the reason.
FIGURE3_RANGE_EXEMPT: dict[str, str] = {
    "pairwise_levenshtein_pooled": (
        "1.2x below their minimum (24.5 against 30.4). Their panel pools six datasets including "
        "enzymes and a growth factor at 100-360 residues; ours is a 121 aa antibody domain, so a "
        "lower pooled pairwise distance is expected from the sequences rather than the method."
    ),
    "profile_log_likelihood": (
        "ABOVE their best (-64 against -85.4), i.e. our profile is easier to match than theirs. "
        "Undefined in the appendix, so the pseudocount and normalisation are ours; a raw summed "
        "log-likelihood fits their sign and magnitude but not their exact scale. Not in the map "
        "above for that reason -- listed here so the discrepancy is recorded rather than hidden."
    ),
}

#: Markers delimiting the generated block in docs/claims.md.
README_START = "<!-- provenance:start -->"
README_END = "<!-- provenance:end -->"


def readme_block() -> str:
    """Build the exact text that must sit between the README markers."""
    note = "<!-- generated by `make provenance` from editjumps/core/provenance.py — do not edit by hand -->"
    return f"{README_START}\n{note}\n\n{markdown_table()}{README_END}"


def sync_readme(path: Path) -> bool:
    """Rewrite the README's generated block in place."""
    text = path.read_text()
    start, end = text.find(README_START), text.find(README_END)
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"{path}: expected {README_START} ... {README_END} markers")
    updated = text[:start] + readme_block() + text[end + len(README_END):]
    if updated == text:
        return False
    path.write_text(updated)
    return True


def main(
    write: Annotated[bool, typer.Option(help="Rewrite the generated block in place.")] = False,
    readme: Annotated[Path, typer.Option(help="Path to the page holding the generated block.")] =
    Path("docs/claims.md"),
) -> None:
    """Print the reproduction-vs-extension table, or sync it into the README."""
    if write:
        print(f"{readme} updated" if sync_readme(readme) else f"{readme} already current")
    else:
        print(markdown_table())


if __name__ == "__main__":
    typer.run(main)
