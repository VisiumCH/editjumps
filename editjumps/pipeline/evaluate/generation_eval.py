"""Evaluate a trained editor the way EvoFlows evaluates one (§4.2 + Appendix B)."""

import gzip
import random
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import numpy as np
import typer

from editjumps.core.edit_flows.alignment import levenshtein, needleman_wunsch
from editjumps.core.generation_metrics import (
    frequencies,
    kl_divergence,
    matrix_agreement,
    mean_levenshtein_to_template,
    mean_pairwise_levenshtein,
    mutual_information_apc,
    one_hot,
    positional_interaction_strength,
    smoothed_composition,
    spectrum_mmd,
    spectrum_mmd_estimators,
)
from editjumps.core.generation_metrics_undefined import (
    entropy_delta,
    js_divergence,
    js_divergence_positional,
    kl_divergence_positional,
    profile_log_likelihood,
)
from editjumps.core.length_capability import NATIVE, LengthCapability, growth_summary
from editjumps.core.utils import (
    design_space_tags,
    flatten_metrics,
    get_logger,
    start_mlflow_run,
    write_generated_fasta,
    write_metrics,
)
from editjumps.core.variant_seeds import check_variant_budget, variant_seed

if TYPE_CHECKING:
    pass

logger = get_logger(__file__)

GAP = "-"

#: **The editor can change length, and it is the only method in the §4.2 comparison that can.** The CTMC it.
CAN_CHANGE_LENGTH: bool = True

#: A target length is therefore `NATIVE` here: reported as attainment, never enforced.
LENGTH_CAPABILITY: LengthCapability = LengthCapability(
    method="edit_flows_model",
    can_change_length=CAN_CHANGE_LENGTH,
    target_length=NATIVE,
    mechanism=(
        "Insert and delete are events of the simulated CTMC (edit_flows.inference.euler_trace / "
        "gillespie_trace), so the output length is a random variable. --target-length raises the "
        "sampler's max_len so the target is reachable and reports attainment; it is never enforced, "
        "because forcing a length would replace the model's length distribution with the "
        "benchmark's."
    ),
)

#: One MLflow experiment for every row of the §4.2 / Appendix-B comparison: this evaluator plus the three.
APPENDIX_B_EXPERIMENT = "editjumps-appendix-b"

def project_to_template(sequence: str, template: str) -> str:
    """Align ``sequence`` to ``template`` and return it in template coordinates."""
    source, target = needleman_wunsch([ord(c) for c in template], [ord(c) for c in sequence])
    projected = []
    for template_token, seq_token in zip(source, target, strict=True):
        if template_token < 0:            # insertion relative to the template: not a column
            continue
        projected.append(GAP if seq_token < 0 else chr(seq_token))
    return "".join(projected)


def evaluate(model_folder: Path, pairs: Path, n_templates: int, n_variants: int,
             n_steps: int, seed: int, k: int, rate_head: str = "linear", q_head: str = "fresh",
             *,
             allow_train_overlap: bool = False, disjoint_from_pairs: bool = False,
             clock: float | None = None, pll_model: str = "", pll_positions: int = 24,
             family_fasta: Path | None = None, holdout_size: int = 200,
             ceiling_n: int = 300, target_length: int | None = None) -> dict:
    """Generate variants from held-out templates and score them against natural homologs."""
    check_variant_budget(n_variants)
    # The sampler's hard length cap, passed ONLY when a growth target is set so an existing call is.
    length_cap: dict[str, int] = {} if target_length is None else {"max_len": max(400, target_length + 2)}

    # ONE callable, `generate(template, index) -> n_variants sequences`, is the whole of what a method.
    from transformers import AutoTokenizer

    from editjumps.pipeline.train.evoflows import EvoFlowsModel, pick_device, sample_edits

    device = pick_device()
    model = EvoFlowsModel.load_trained(model_folder, rate_head=rate_head, q_head=q_head).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(str(Path(model_folder) / "encoder"))

    def generate(template: str, index: int) -> list[str]:
        """Sample ``n_variants`` variants of one template with the editor."""
        ids = tokenizer(template)["input_ids"]
        out = [sample_edits(model, list(ids), n_steps=n_steps, clock=clock, **length_cap,
                            rng=random.Random(variant_seed(seed, index, v)))
               for v in range(n_variants)]
        return [tokenizer.decode(o, skip_special_tokens=True).replace(" ", "") for o in out]

    if family_fasta is not None:
        # One family, split into templates and a disjoint holdout as §4.2 does: scoring generations against.
        from editjumps.core.family_split import (
            disjoint_members,
            sequences_in_pairs,
            split_family,
            usable_members,
        )
        from editjumps.pipeline.preprocess.pretrain.seed_homologs import read_fasta

        members = usable_members(read_fasta(family_fasta).values())
        if disjoint_from_pairs:
            # Draw the whole split -- templates AND reference -- from family members that do not appear in the.
            kept = disjoint_members(members, pairs)
            logger.info(f"disjoint draw: {len(kept)}/{len(members)} family members are not in "
                        f"{pairs} ({100 * len(kept) / max(1, len(members)):.1f}%)")
            members = kept
        try:
            split = split_family(members, n_templates, holdout_size, seed)
        except ValueError as exc:
            raise ValueError(f"{family_fasta}: {exc}") from exc
        # The editor is scored against `split.reference`.
        train_overlap = set(split.reference) & sequences_in_pairs(pairs)
        if train_overlap and not allow_train_overlap:
            raise ValueError(
                f"{len(train_overlap)} of {len(split.reference)} scoring-reference sequences are "
                f"also in {pairs}, which the editor trained on. The baselines refuse this exact "
                f"overlap, so scoring the editor through it compares two methods under different "
                f"rules. Pass --allow-train-overlap to proceed anyway; the count is recorded in the "
                f"artefact either way."
            )

        inference = split.templates
        # `split.pool` is everything left over, and is what the §4.3 pairing ceiling draws from, so
        # it must not overlap the reference it is scored against.
        holdout = split.reference + split.pool
        rows = [(seq, holdout[i % len(holdout)]) for i, seq in enumerate(inference)]
        family_holdout = split.reference
        family_pool = split.pool
    else:
        with gzip.open(pairs, "rt") as fh:
            rows = [line.rstrip("\n").split("\t") for line in fh if "\t" in line]
        family_holdout = None
        family_pool = None
    # Templates from the END of the file: the trainer's split_pairs_by_family holds out a tail by
    # family, so these are the pairs least likely to have been trained on.
    held_out = rows if family_fasta is not None else rows[-max(n_templates * 4, 1):]
    rng = random.Random(seed)
    chosen = rng.sample(held_out, min(n_templates, len(held_out)))

    generated_all: list[str] = []
    # Without --family-fasta the reference is the whole held-out tail, which spans MANY families while the.
    natural_all: list[str] = list(family_holdout) if family_holdout else [row[1] for row in held_out]
    natural_partners: list[str] = []
    per_template = []
    for index, (template, natural) in enumerate(chosen):
        variants = generate(template, index)
        generated_all.extend(variants)
        natural_partners.append(natural)
        per_template.append({
            "levenshtein_to_template": mean_levenshtein_to_template(variants, template),
            "pairwise_levenshtein": mean_pairwise_levenshtein(variants),
            # Each variant's own distance, kept so an interval can be bootstrapped. See the note on
            # `levenshtein_to_template_per_template` below for which unit to resample.
            "levenshtein_to_template_per_sequence": [
                float(levenshtein(v, template)) for v in variants
            ],
        })
        logger.info(f"template {index + 1}/{len(chosen)}: {len(variants)} variants, "
                    f"mean edit distance {per_template[-1]['levenshtein_to_template']:.1f}")

    # One fixed alignment for everything. `split.templates[0]` when there is a family, NOT
    # `chosen[0][0]`, for two reasons that both cost us numbers.
    #
    # Comparability: evotune_baseline.py and evodiff_msa_baseline.py both align to
    # `split.templates[0]`, so an editor aligned to `chosen[0][0]` sits in a different coordinate
    # system from every baseline it is compared against. Covariance, MIP and positional JS are all
    # per-position, so those comparisons were meaningless -- Ty1 put the editor at L=115 and both
    # baselines at L=121, and the two runs' recorded `alignment` strings said so.
    #
    # Stability: `chosen` is `rng.sample(held_out, n_templates)`, so `chosen[0][0]` moves with the
    # Seed AND with n_templates.
    reference_template = split.templates[0] if family_fasta is not None else chosen[0][0]
    aligned_generated = [project_to_template(s, reference_template) for s in generated_all]
    # Per-position statistics (covariance, MIp, entropy) need the expensive alignment, so they use
    # the template partners; the set-level distances (KL, MMD) use the full reference.
    aligned_natural = [project_to_template(s, reference_template) for s in natural_partners]
    # A second alignment of the FULL reference, for the metrics added here. `aligned_natural` holds one.
    reference_cap = max(300, holdout_size) if family_holdout else 300
    aligned_natural_full = [project_to_template(s, reference_template)
                            for s in natural_all[:reference_cap]]
    f_i_full, f_ij_full, c_ij_full = frequencies(one_hot(aligned_natural_full))
    strength_full = positional_interaction_strength(c_ij_full)
    mip_full = mutual_information_apc(f_i_full, f_ij_full)

    f_i, f_ij, c_ij = frequencies(one_hot(aligned_generated))
    f_i_ref, f_ij_ref, c_ij_ref = frequencies(one_hot(aligned_natural))
    strength, strength_ref = positional_interaction_strength(c_ij), positional_interaction_strength(c_ij_ref)
    mip, mip_ref = mutual_information_apc(f_i, f_ij), mutual_information_apc(f_i_ref, f_ij_ref)

    mmd_estimators = spectrum_mmd_estimators(generated_all, natural_all, k=k)
    defined = {
        "levenshtein_to_template": float(np.mean([p["levenshtein_to_template"] for p in per_template])),
        # BOTH resampling units, because the centre above is a mean over TEMPLATES -- each entry is one.
        "levenshtein_to_template_per_template": [
            float(p["levenshtein_to_template"]) for p in per_template
        ],
        "levenshtein_to_template_per_sequence": [
            p["levenshtein_to_template_per_sequence"] for p in per_template
        ],
        "pairwise_levenshtein": float(np.mean([p["pairwise_levenshtein"] for p in per_template])),
        # Pooled over the whole family rather than averaged within each template's variant set.
        "pairwise_levenshtein_pooled": mean_pairwise_levenshtein(generated_all),
        "covariance_frobenius_generated": float(np.linalg.norm(c_ij)),
        "covariance_frobenius_natural": float(np.linalg.norm(c_ij_ref)),
        "interaction_strength_mean_generated": float(strength.mean()),
        "interaction_strength_mean_natural": float(strength_ref.mean()),
        "mip_mean_generated": float(mip.mean()),
        "mip_mean_natural": float(mip_ref.mean()),
        # The comparable forms of the Covariance and MIP panels.
        "covariance_agreement": matrix_agreement(strength, strength_full),
        "mip_agreement": matrix_agreement(mip, mip_full),
        # Recorded because the two agreements above are only readable next to it: a small reference
        # drags both toward zero regardless of the model.
        "agreement_reference_n": len(aligned_natural_full),
        "kl_generated_vs_natural": kl_divergence(smoothed_composition(generated_all),
                                                 smoothed_composition(natural_all)),
        "spectrum_mmd": mmd_estimators["biased"],
    }
    # The unbiased U-statistic and the sample sizes it was computed at.
    mmd_diagnostic = {
        "spectrum_mmd_unbiased_squared": mmd_estimators["unbiased_squared"],
        "n_generated": mmd_estimators["n_generated"],
        "n_reference": mmd_estimators["n_reference"],
        "comparable_only_at_equal_n": True,
    }
    # The ceiling for the two agreements: real homologs from the same family, held out from the reference.
    ceiling: dict[str, float] = {}
    # The ceiling's OWN sample size stays at 300 when the reference grows, deliberately.
    if family_pool:
        aligned_ceiling = [project_to_template(s, reference_template) for s in family_pool[:ceiling_n]]
        if len(aligned_ceiling) >= 2:
            f_i_c, f_ij_c, c_ij_c = frequencies(one_hot(aligned_ceiling))
            ceiling = {
                "covariance_agreement_ceiling":
                    matrix_agreement(positional_interaction_strength(c_ij_c), strength_full),
                "mip_agreement_ceiling":
                    matrix_agreement(mutual_information_apc(f_i_c, f_ij_c), mip_full),
                "ceiling_n": float(len(aligned_ceiling)),
            }
    defined.update(ceiling)

    ours = {
        "entropy_delta": entropy_delta(aligned_generated, aligned_natural),
        "js_divergence": js_divergence(generated_all, natural_all),
        "js_divergence_positional": js_divergence_positional(aligned_generated, aligned_natural_full),
        # The comparable KL.
        "kl_divergence_positional": kl_divergence_positional(aligned_generated, aligned_natural_full),
        "profile_log_likelihood": profile_log_likelihood(aligned_generated, aligned_natural),
    }
    # Two of the paper's baselines (§4.3), which need no model and give the metrics a floor and a
    # ceiling. Without them "MMD 3.2" means nothing.
    baseline_random = []
    baseline_pairing = []
    rng_base = random.Random(seed + 99991)
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    edits_made = int(round(np.mean([p["levenshtein_to_template"] for p in per_template]))) or 1
    for index, (template, _natural) in enumerate(chosen):
        for _ in range(n_variants):
            # "Random mutations. Applies the same expected mutation count, but samples mutation
            # positions, mutation types and replacement amino acids uniformly." (§4.3)
            seq = list(template)
            for _e in range(edits_made):
                if not seq:
                    break
                pos = rng_base.randrange(len(seq))
                kind = rng_base.choice(("sub", "ins", "del"))
                if kind == "sub":
                    seq[pos] = rng_base.choice(alphabet)
                elif kind == "ins":
                    seq.insert(pos, rng_base.choice(alphabet))
                else:
                    seq.pop(pos)
            baseline_random.append("".join(seq))
    # "Random inference homolog pairing.
    n_wanted = len(baseline_random)
    if family_pool:
        pool = [seq for seq in family_pool if seq not in set(natural_all)]
        baseline_pairing = rng_base.sample(pool, min(n_wanted, len(pool)))
    else:
        # No family file: fall back to the held-out partners, sampling with replacement because the
        # pairs tail is small. Flagged in `caveats` so the ceiling is not read as comparable.
        partners = [row[1] for row in held_out]
        baseline_pairing = [partners[rng_base.randrange(len(partners))] for _ in range(n_wanted)]

    # Template-MAJOR, the order the baseline sets are generated in: entry i of `baseline_random` is a mutated.
    templates_expanded = [c[0] for c in chosen for _ in range(n_variants)]

    def score_set(generated: list[str], aligned_ref: list[str]) -> dict:
        """Score one set the way the model's set is scored, against the same references."""
        aligned = [project_to_template(s, reference_template) for s in generated]
        # The agreements and the positional JS are computed here, not only for the model, because the §4.3.
        f_i_b, f_ij_b, c_ij_b = frequencies(one_hot(aligned))
        # Per-sequence BEFORE averaging.
        per_sequence = [
            mean_levenshtein_to_template([g], t)
            for g, t in zip(generated, templates_expanded, strict=False)
        ]
        return {
            "levenshtein_to_template": float(np.mean(per_sequence)) if generated else 0.0,
            "levenshtein_to_template_per_sequence": [float(v) for v in per_sequence],
            "kl_generated_vs_natural": kl_divergence(smoothed_composition(generated),
                                                     smoothed_composition(natural_all)),
            "spectrum_mmd": spectrum_mmd(generated, natural_all, k=k),
            "entropy_delta": entropy_delta(aligned, aligned_ref),
            "covariance_agreement": matrix_agreement(
                positional_interaction_strength(c_ij_b), strength_full),
            "mip_agreement": matrix_agreement(mutual_information_apc(f_i_b, f_ij_b), mip_full),
            "js_divergence_positional": js_divergence_positional(aligned, aligned_natural_full),
            # Pooled over the whole baseline set, matching `defined["pairwise_levenshtein_pooled"]` and NOT.
            "pairwise_levenshtein_pooled": mean_pairwise_levenshtein(generated),
            "agreement_reference_n": len(aligned_natural_full),
            # The frame the three per-position numbers above were computed in, recorded PER ROW.
            "alignment_length": len(aligned[0]) if aligned else len(reference_template),
            "n_scored": len(generated),
        }

    baselines = {
        "random_mutations": score_set(baseline_random, aligned_natural),
        "random_homolog_pairing": score_set(baseline_pairing, aligned_natural),
    }

    # Diversity vs novelty -- NOT EvoFlows metrics; see editjumps/core/diversity_novelty.py.
    from editjumps.core.diversity_novelty import diversity_novelty as _diversity_novelty

    template_seqs = [c[0] for c in chosen]
    novelty_reference = list(family_pool) if family_pool else [row[1] for row in held_out]
    dn = {"model": _diversity_novelty(generated_all, template_seqs, novelty_reference, seed=seed)}
    for name, group in (("random_mutations", baseline_random),
                        ("random_homolog_pairing", baseline_pairing)):
        dn[name] = _diversity_novelty(group, template_seqs, novelty_reference, seed=seed)
    logger.info(f"  diversity {dn['model']['diversity']:.2f}  novelty {dn['model']['novelty']:.2f} "
                f"(templates {dn['model']['novelty_templates']:.2f}, "
                f"delta {dn['model']['novelty_delta']:+.2f})")

    # B.2's ESM-2 pseudo-log-likelihood, the paper's naturalness axis.
    pll = {}
    if pll_model:
        from editjumps.core.pseudo_likelihood import pseudo_log_likelihood
        for name, group in (("generated", generated_all), ("natural_holdout", natural_all),
                            ("random_mutations", baseline_random),
                            ("random_homolog_pairing", baseline_pairing)):
            pll[name] = pseudo_log_likelihood(group, model_name=pll_model, seed=seed,
                                              max_positions=pll_positions)
            logger.info(f"  PLL {name:<24} {pll[name]['mean']:+.4f} per position (n={pll[name]['n']})")

    report: dict = {
        "model": str(model_folder),
        # The §3.3 clock this run sampled at, or null when unclocked.
        "clock_normalization": clock,
        "baselines": baselines,
        "n_templates": len(chosen), "n_variants_per_template": n_variants, "n_generated": len(generated_all),
        "n_natural_reference": len(natural_all), "n_natural_aligned": len(natural_partners),
        "reference_construction": ("single family, inference/holdout disjoint (paper's §4.2)"
                                  if family_fasta else "held-out tail of the pairs file (many families)"),
        # Wording matches evotune_baseline.py and evodiff_msa_baseline.py exactly.
        "alignment": (f"Needleman-Wunsch projection onto the first template's coordinates "
                      f"(L={len(reference_template)})"),
        # Which key to use when two could plausibly be meant, IN the artefact rather than only in this source.
        "reference_in_training_pairs": (len(train_overlap) if family_fasta is not None else None),
        "reading_notes": {
            "diversity": ("Use pairwise_levenshtein_pooled in any table or cross-row comparison. "
                          "pairwise_levenshtein is the WITHIN-template mean, kept for continuity "
                          "with earlier runs; it is structurally 0 at n_variants=1 and the baseline "
                          "rows do not carry it, so mixing the two across rows compares different "
                          "quantities."),
            "bootstrap_unit": ("Resample levenshtein_to_template_per_template. The centre is a mean "
                               "over templates, so resampling the per-sequence values treats "
                               "variants of one template as independent when they are correlated by "
                               "construction, and returns an interval about 2.5x too narrow."),
            "kl": ("Use our_interpretation.kl_divergence_positional against the paper's KL panel. "
                   "defined_by_the_paper.kl_generated_vs_natural applies the same eq 23-24 formula "
                   "to the POOLED amino-acid composition, which two sets of close homologs share "
                   "almost exactly: it reads ~12x below their panel (0.0005 against 0.0059-0.1237) "
                   "and is comparable across our own arms only. Eq 23-24 define the formula and "
                   "never the representation, so the pooled key is a reading and not the paper's "
                   "number; it is kept because 151 earlier run artefacts report it alone."),
        },
        # Carried out under a private key and POPPED by `main` before the JSON is written: the sequences.
        "_sequences": {
            "model": list(generated_all),
            "random_mutations": list(baseline_random),
            "random_homolog_pairing": list(baseline_pairing),
        },
        "defined_by_the_paper": defined,
        "mmd_diagnostic": mmd_diagnostic,
        "diversity_novelty": dn,
        "esm2_pseudo_log_likelihood": pll,
        "our_interpretation": ours,
        "not_implemented": [] if pll_model else ["esm2_pseudo_log_likelihood (B.2) - pass --pll-model"],
        "caveats": [] if family_fasta else [
            "MMD and KL use a natural reference drawn from the held-out tail, which spans many "
            "families. The paper's holdout is a subsample of the TEMPLATE'S OWN family, so these "
            "two numbers are comparable across our arms but NOT against the paper's figures. "
            "Pass --family-fasta for the paper's construction.",
        ],
    }
    # Added only when asked for, so an artefact from any existing invocation is unchanged key for key..
    if target_length is not None:
        report["length_growth"] = growth_summary(
            LENGTH_CAPABILITY, target_length,
            [len(template) for template in templates_expanded],
            [len(sequence) for sequence in generated_all],
        )
        report["caveats"].append(
            "--target-length is a MEASUREMENT here, not a constraint: the sampler's own insert/"
            "delete rates decide the length and the target only raised the hard cap, so "
            "length_growth.fraction_at_target is a property of the model. The mask-and-infill "
            "baselines cannot reach a target at all without the scaffolded construction of "
            "editjumps.core.length_growth, which is ours and not their published method - a table "
            "putting the two side by side must say which rows are scaffolded."
        )
    return report


def main(
    model_folder: Annotated[Path, typer.Option()] = Path("data/pretrain/edit_flows_baseline"),
    pairs: Annotated[Path, typer.Option()] = Path("data/pretrain/oas_homolog_pairs.tsv.gz"),
    n_templates: Annotated[int, typer.Option(help="Templates to generate from")] = 20,
    n_variants: Annotated[int, typer.Option(help="Variants generated per template")] = 20,
    n_steps: Annotated[int, typer.Option(help="Euler steps per generation")] = 50,
    seed: Annotated[int, typer.Option()] = 0,
    k: Annotated[int, typer.Option(help="Spectrum-kernel k-mer length (not stated in the paper)")] = 3,
    clock: Annotated[float, typer.Option(help="Clock normalization (§3.3); 0 = off, as before")] = 0.0,
    pll_model: Annotated[str, typer.Option(help="Stock ESM-2 for B.2 naturalness; empty skips")] = "",
    pll_positions: Annotated[int, typer.Option(help="Positions scored per sequence for the PLL")] = 24,
    family_fasta: Annotated[Path | None, typer.Option(help="One seed family; enables the paper's holdout")] = None,
    disjoint_from_pairs: Annotated[bool, typer.Option(
        "--disjoint-from-pairs",
        help="Draw templates AND reference only from family members absent from --pairs")] = False,
    allow_train_overlap: Annotated[bool, typer.Option(
        "--allow-train-overlap",
        help="Score anyway when the reference overlaps the training pairs (the baselines refuse)")] = False,
    holdout_size: Annotated[int, typer.Option(help="Natural sequences in the family holdout")] = 200,
    ceiling_n: Annotated[
        int, typer.Option(help="Real homologs behind the agreement ceiling (must match the baselines)")
    ] = 300,
    rate_head: Annotated[str, typer.Option(help="linear | mlp; must match the checkpoint")] = "linear",
    q_head: Annotated[str, typer.Option(help="fresh | esm_lm_head; must match the checkpoint")] = "fresh",
    target_length: Annotated[int | None, typer.Option(
        help="Measure the generations against this length (a growth benchmark); raises the "
             "sampler's cap and reports attainment, never constrains it")] = None,
    metrics_path: Annotated[Path, typer.Option()] = Path("metrics/generation_eval.json"),
) -> None:
    """Score a trained editor's generations against held-out natural homologs."""
    # A missing checkpoint folder and a missing input file are both user-fixable and both already carry a.
    try:
        report = evaluate(model_folder, pairs, n_templates, n_variants, n_steps, seed, k,
                          rate_head=rate_head, q_head=q_head, clock=clock or None,
                          pll_model=pll_model, pll_positions=pll_positions,
                          family_fasta=family_fasta, holdout_size=holdout_size, ceiling_n=ceiling_n,
                          disjoint_from_pairs=disjoint_from_pairs,
                          allow_train_overlap=allow_train_overlap,
                          target_length=target_length)
    except FileNotFoundError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from None
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    sequences = report.pop("_sequences", {})
    fasta = write_generated_fasta(metrics_path, sequences, header={
        "model": model_folder,
        "family": family_fasta,
        "method": "edit_flows_model",
        "clock": clock or None,
        "seed": seed,
        "n_templates": n_templates,
        "n_variants": n_variants,
        "disjoint_from_pairs": disjoint_from_pairs,
    })
    write_metrics(metrics_path, report)
    if fasta:
        logger.info(f"wrote {fasta} ({sum(len(v) for v in sequences.values())} sequences)")
    def summarise(section: str) -> None:
        """Log one metrics section, formatting scalars and describing everything else."""
        for name, value in report[section].items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                logger.info(f"  {name:<40} {value:12.4f}")
            elif isinstance(value, list):
                logger.info(f"  {name:<40} {f'[{len(value)} values]':>12}")
            else:
                logger.info(f"  {name:<40} {str(value):>12}")

    summarise("defined_by_the_paper")
    logger.info("  -- below: metrics the paper plots but never defines, our reading --")
    summarise("our_interpretation")
    logger.info(f"wrote {metrics_path}")

    # Tracking comes AFTER the JSON, deliberately: that file is the durable record (DVC tracks it) and this is.
    tags = {
        **design_space_tags(
            objective="edit_flows",
            pretrain_data=(family_fasta.name if family_fasta is not None else pairs.name),
        ),
        # Which row of the §4.2 table this run is. The baselines set the same tag to their own name.
        "method": "edit_flows_model",
        "section": "4.2",
        "reference_construction": report["reference_construction"],
    }
    if target_length is not None:
        # A growth run answers a different question from a §4.2 row, so it is tagged as one.
        tags["target_length"] = str(target_length)
    run_name = "generation-eval-" + \
        f"{Path(model_folder).name}" + (
        f"-{family_fasta.stem}" if family_fasta is not None else ""
    )
    with start_mlflow_run(APPENDIX_B_EXPERIMENT, run_name=run_name, tags=tags):
        import mlflow

        # The settings that define the run.
        mlflow.log_params({
            "model_folder": str(model_folder),
            "pairs": str(pairs),
            "family_fasta": str(family_fasta) if family_fasta is not None else "",
            "n_templates": n_templates,
            "n_variants": n_variants,
            "n_steps": n_steps,
            "seed": seed,
            "k": k,
            "clock": clock,
            "holdout_size": holdout_size,
            "ceiling_n": ceiling_n,
            "rate_head": rate_head,
            "q_head": q_head,
            "disjoint_from_pairs": disjoint_from_pairs,
            "allow_train_overlap": allow_train_overlap,
            "pll_model": pll_model,
            "pll_positions": pll_positions,
            # 0 means "no growth target", which is what every run so far did. Recorded because a
            # growth row and a §4.2 row are different comparisons and must be filterable apart.
            "target_length": target_length if target_length is not None else 0,
            "metrics_path": str(metrics_path),
        })
        # EVERY scalar in the report, from the whole tree rather than a hand-picked section.
        for path, value in flatten_metrics(report, summarise_lists=True).items():
            mlflow.log_metric(path, value)
        mlflow.log_artifact(str(metrics_path))


if __name__ == "__main__":
    typer.run(main)
