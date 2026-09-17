"""EvoFlows' two evotuned-PLM baselines, generated and scored (§2.2 + §4.3)."""

import gzip
import json
import random
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import typer

from editjumps.core.evotune.profile import (
    position_weights,
)
from editjumps.core.evotune.substitution import (
    Proposer,
    matched_budget,
)
from editjumps.core.family_split import (
    disjoint_members,
    sequences_in_pairs,
    split_family,
    usable_members,
)
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
    spectrum_mmd_estimators,
)
from editjumps.core.generation_metrics_undefined import (
    entropy_delta,
    js_divergence,
    js_divergence_positional,
    profile_log_likelihood,
)
from editjumps.core.length_capability import SCAFFOLDED, LengthCapability, growth_summary
from editjumps.core.length_growth import (
    SLOT_SEED_OFFSET,
    align_to_template,
    allocate_slots,
    grow_by_infilling,
    slot_weights,
)
from editjumps.core.sequences import AA
from editjumps.core.utils import (
    design_space_tags,
    flatten_metrics,
    get_logger,
    load_tokenizer,
    start_mlflow_run,
    tok_attr,
    token_id,
    write_generated_fasta,
    write_metrics,
)
from editjumps.core.variant_seeds import check_variant_budget, variant_seed
from editjumps.pipeline.evaluate.generation_eval import APPENDIX_B_EXPERIMENT, project_to_template

logger = get_logger(__file__)

#: The paper's names for the two baselines, keyed by whether substitutions are forced.
METHOD_NAMES: dict[bool, str] = {
    False: "evotuned_plm",
    True: "evotuned_plm_forced_substitutions",
}

#: **Substitutions only: this baseline cannot make a sequence longer.** ``substitute_by_profile`` builds its.
CAN_CHANGE_LENGTH: bool = False

#: What this module does when handed a target length: `SCAFFOLDED`.
LENGTH_CAPABILITY: LengthCapability = LengthCapability(
    method=METHOD_NAMES[False],
    can_change_length=CAN_CHANGE_LENGTH,
    target_length=SCAFFOLDED,
    mechanism=(
        "Substitution-only: evotune.substitution.substitute_by_profile starts from list(template) and "
        "writes back one residue per masked index, so length is preserved by construction and the "
        "MLM has no insertion operation. --target-length opens L - n masked columns instead (ours, "
        "not the paper's baseline)."
    ),
)


def length_capability(forced: bool) -> LengthCapability:
    """Return the length capability of one of the two evotuned baselines."""
    from dataclasses import replace

    return replace(LENGTH_CAPABILITY, method=METHOD_NAMES[forced])


#: How §4.2's mutation budget is spent. ``realised`` keeps drawing positions until the budget has actually.
BUDGET_MODES: tuple[str, ...] = ("realised", "masked")

#: Residues the MLM may propose: the 20 standard amino acids and nothing else.
PROPOSABLE: tuple[str, ...] = tuple(sorted(AA))


def read_corpus(path: Path) -> list[str]:
    """Read a gzipped one-sequence-per-line corpus."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:  # type: ignore[operator]
        return [line.strip() for line in handle if line.strip()]


def budget_from_metrics(path: Path) -> int:
    """Read §4.2's matched mutation budget out of a ``generation-eval`` metrics file. "Matching the."""
    report = json.loads(path.read_text())
    value = report.get("defined_by_the_paper", {}).get("levenshtein_to_template")
    if value is None:
        raise ValueError(
            f"{path} has no defined_by_the_paper.levenshtein_to_template; §4.2's budget cannot be "
            "matched against it. Pass --mutations explicitly instead."
        )
    return matched_budget(float(value))


def build_profile(train_members: list[str], template: str, profile_size: int, seed: int,
                  eps: float = 1e-12) -> list[float]:
    """Build the eq-12 entropy profile for one template, in that template's own coordinates. **Ours, not."""
    aligned = [project_to_template(member, template) for member in profile_sample(train_members, profile_size, seed)]
    return position_weights(aligned, template, eps=eps)


def profile_sample(train_members: list[str], profile_size: int, seed: int) -> list[str]:
    """Draw the family subsample the profile is built from."""
    if not train_members:
        raise ValueError("no training members: the entropy profile has nothing to be derived from")
    if len(train_members) > profile_size:
        return random.Random(seed).sample(train_members, profile_size)
    return train_members


def build_growth_profile(train_members: list[str], template: str, profile_size: int, seed: int,
                         eps: float = 1e-12) -> tuple[list[float], list[float]]:
    """Build the eq-12 profile and the insertion-slot weights from one pass over the same members."""
    alignments = [align_to_template(member, template)
                  for member in profile_sample(train_members, profile_size, seed)]
    weights = position_weights([a.projected for a in alignments], template, eps=eps)
    return weights, slot_weights(alignments, len(template))


def generate_variants(
    templates: list[str],
    train_members: list[str],
    budget: int,
    propose: Proposer,
    n_variants: int,
    seed: int,
    *,
    forced: bool,
    top_up: bool = True,
    max_rounds: int = 8,
    profile_size: int = 200,
    target_length: int | None = None,
) -> tuple[list[str], list[dict], list[dict]]:
    """Run one baseline over every template — torch-free, given a ``propose`` callable."""
    check_variant_budget(n_variants)
    generated: list[str] = []
    accounting: list[dict] = []
    per_template: list[dict] = []
    for index, template in enumerate(templates):
        if target_length is None:
            weights = build_profile(train_members, template, profile_size, seed + index)
            counts = [0] * (len(template) + 1)
        else:
            # One pass over one subsample for both, so the growth slots and the substitution
            # positions describe the same sequences.
            weights, growth_weights = build_growth_profile(
                train_members, template, profile_size, seed + index
            )
            counts = allocate_slots(len(template), target_length, growth_weights,
                                    random.Random(seed + SLOT_SEED_OFFSET + index))
        variants: list[str] = []
        for variant in range(n_variants):
            rng = random.Random(variant_seed(seed, index, variant))
            sequence, stats = grow_by_infilling(
                template, weights, budget, counts, propose, rng,
                forced=forced, top_up=top_up, max_rounds=max_rounds,
            )
            variants.append(sequence)
            accounting.append(stats)
        generated.extend(variants)
        changed = [a["n_changed"] for a in accounting[-n_variants:]]
        per_template.append({
            "levenshtein_to_template": mean_levenshtein_to_template(variants, template),
            # Kept per SEQUENCE as well as per template: the report exposes both so an interval can be.
            "levenshtein_to_template_per_sequence": [
                mean_levenshtein_to_template([sequence], template) for sequence in variants
            ],
            "pairwise_levenshtein": mean_pairwise_levenshtein(variants),
            "mean_mutations": float(np.mean(changed)) if changed else 0.0,
        })
        logger.info(
            f"template {index + 1}/{len(templates)} (L={len(template)}): {len(variants)} variants, "
            f"{per_template[-1]['mean_mutations']:.2f} mutations (budget {budget}), "
            f"edit distance {per_template[-1]['levenshtein_to_template']:.2f}"
        )
    return generated, accounting, per_template


def score_generations(
    generated: list[str],
    templates_expanded: list[str],
    reference: list[str],
    aligned_reference: list[str],
    alignment_template: str,
    per_template: list[dict],
    k: int = 3,
    family_pool: list[str] | None = None,
    ceiling_n: int = 300,
) -> dict[str, Any]:
    """Assemble the Appendix-B metrics for a generated set — numpy only, no model."""
    aligned_generated = [project_to_template(sequence, alignment_template) for sequence in generated]
    f_i, f_ij, c_ij = frequencies(one_hot(aligned_generated))
    f_i_ref, f_ij_ref, c_ij_ref = frequencies(one_hot(aligned_reference))
    strength, strength_ref = positional_interaction_strength(c_ij), positional_interaction_strength(c_ij_ref)
    mip, mip_ref = mutual_information_apc(f_i, f_ij), mutual_information_apc(f_i_ref, f_ij_ref)
    mmd = spectrum_mmd_estimators(generated, reference, k=k)
    # `aligned_reference` is one partner per template -- 20 sequences at the paper's config -- which is far.
    aligned_full = [project_to_template(sequence, alignment_template) for sequence in reference]
    f_i_full, f_ij_full, c_ij_full = frequencies(one_hot(aligned_full))
    strength_full = positional_interaction_strength(c_ij_full)
    mip_full = mutual_information_apc(f_i_full, f_ij_full)

    defined = {
        "levenshtein_to_template": float(np.mean([p["levenshtein_to_template"] for p in per_template])),
        # The centre above is a mean over TEMPLATES, so an interval on it must resample templates.
        "levenshtein_to_template_per_template": [
            float(p["levenshtein_to_template"]) for p in per_template
        ],
        "levenshtein_to_template_per_sequence": [
            p["levenshtein_to_template_per_sequence"] for p in per_template
            if "levenshtein_to_template_per_sequence" in p
        ],
        "pairwise_levenshtein": float(np.mean([p["pairwise_levenshtein"] for p in per_template])),
        "pairwise_levenshtein_per_template": [
            float(p["pairwise_levenshtein"]) for p in per_template
        ],
        "pairwise_levenshtein_pooled": mean_pairwise_levenshtein(generated),
        "covariance_frobenius_generated": float(np.linalg.norm(c_ij)),
        "covariance_frobenius_natural": float(np.linalg.norm(c_ij_ref)),
        "interaction_strength_mean_generated": float(strength.mean()),
        "interaction_strength_mean_natural": float(strength_ref.mean()),
        "mip_mean_generated": float(mip.mean()),
        "mip_mean_natural": float(mip_ref.mean()),
        "covariance_agreement": matrix_agreement(strength, strength_full),
        "mip_agreement": matrix_agreement(mip, mip_full),
        "agreement_reference_n": float(len(aligned_full)),
        "kl_generated_vs_natural": kl_divergence(smoothed_composition(generated),
                                                 smoothed_composition(reference)),
        "spectrum_mmd": mmd["biased"],
    }
    # The ceiling for the two agreements, in THIS run's own coordinate frame: real homologs from the family's.
    if family_pool:
        aligned_ceiling = [project_to_template(sequence, alignment_template)
                           for sequence in family_pool[:ceiling_n]]
        if len(aligned_ceiling) >= 2:
            f_i_c, f_ij_c, c_ij_c = frequencies(one_hot(aligned_ceiling))
            defined.update({
                "covariance_agreement_ceiling":
                    matrix_agreement(positional_interaction_strength(c_ij_c), strength_full),
                "mip_agreement_ceiling":
                    matrix_agreement(mutual_information_apc(f_i_c, f_ij_c), mip_full),
                "ceiling_n": float(len(aligned_ceiling)),
            })
    # Levenshtein against the RIGHT template, cross-checking the template-major ordering: for a.
    defined["levenshtein_to_own_template"] = float(np.mean([
        mean_levenshtein_to_template([sequence], template)
        for sequence, template in zip(generated, templates_expanded, strict=True)
    ])) if generated else 0.0
    return {
        "defined_by_the_paper": defined,
        "mmd_diagnostic": {
            "spectrum_mmd_unbiased_squared": mmd["unbiased_squared"],
            "n_generated": mmd["n_generated"],
            "n_reference": mmd["n_reference"],
            "comparable_only_at_equal_n": True,
        },
        "our_interpretation": {
            "entropy_delta": entropy_delta(aligned_generated, aligned_reference),
            "js_divergence": js_divergence(generated, reference),
            "js_divergence_positional": js_divergence_positional(aligned_generated, aligned_full),
            "profile_log_likelihood": profile_log_likelihood(aligned_generated, aligned_reference),
        },
        # Same wording as generation_eval's, so a baseline row and the editor row of one table are
        # bracketed by intervals over the same unit.
        "reading_notes": {
            "bootstrap_unit": (
                "Resample levenshtein_to_template_per_template (or, for the pooled/set-level "
                "metrics, the template blocks of the sibling .fasta: entry i came from template "
                "i // n_variants_per_template). The centre is a mean over templates, so resampling "
                "the per-sequence values treats variants of one template as independent when they "
                "are correlated by construction, and returns an interval about 2.5x too narrow."
            ),
            "diversity": (
                "Use pairwise_levenshtein_pooled in any table or cross-row comparison. "
                "pairwise_levenshtein is the WITHIN-template mean, kept for continuity with earlier "
                "runs; the two are different quantities and must not be mixed across rows."
            ),
        },
    }


def mlm_proposer(model_folder: Path, temperature: float, rng: random.Random) -> Proposer:
    """Build the network half: an evotuned MLM infilling one masked position, temperature-scaled."""
    if temperature <= 0:
        raise ValueError(f"temperature={temperature}; must be > 0")
    import torch  # ty: ignore[unresolved-import]
    from transformers import AutoModelForMaskedLM  # ty: ignore[unresolved-import]

    from editjumps.pipeline.train.evoflows import pick_device

    tokenizer = load_tokenizer(model_folder)
    model = AutoModelForMaskedLM.from_pretrained(str(model_folder))
    device = pick_device()
    model.to(device).eval()

    if tok_attr(tokenizer, "mask_token_id") is None:
        raise ValueError(f"{model_folder}: tokenizer has no mask token, so nothing can be infilled")
    residue_id = {residue: token_id(tokenizer, residue) for residue in PROPOSABLE}
    missing = [r for r, i in residue_id.items() if i is None or i == tok_attr(tokenizer, "unk_token_id")]
    if missing:
        raise ValueError(f"{model_folder}: tokenizer maps {missing} to <unk>; cannot infill residues")
    prefix, suffix = [tok_attr(tokenizer, "cls_token_id")], [tok_attr(tokenizer, "eos_token_id")]

    def propose(working: list[str | None], position: int, blocked: str) -> str:
        ids = prefix + [
            tok_attr(tokenizer, "mask_token_id") if residue is None else residue_id[residue] for residue in working
        ] + suffix
        with torch.no_grad():
            logits = model(input_ids=torch.tensor([ids], device=device)).logits[0, position + len(prefix)]
        candidates = [residue for residue in PROPOSABLE if residue != blocked]
        scaled = torch.tensor([logits[residue_id[r]].item() for r in candidates]) / temperature
        weights = torch.softmax(scaled, dim=0).tolist()
        return rng.choices(candidates, weights=weights, k=1)[0]

    return propose


def evaluate(
    model_folder: Path,
    family_fasta: Path,
    train_corpus: Path,
    n_templates: int,
    n_variants: int,
    budget: int,
    seed: int,
    *,
    disjoint_from_pairs: Path | None = None,
    forced: bool,
    temperature: float = 1.0,
    holdout_size: int = 200,
    profile_size: int = 200,
    top_up: bool = True,
    max_rounds: int = 8,
    k: int = 3,
    ceiling_n: int = 300,
    pll_model: str = "",
    pll_positions: int = 24,
    target_length: int | None = None,
) -> dict[str, Any]:
    """Generate one evotuned-PLM baseline over a family and score it against the family holdout."""
    from editjumps.pipeline.preprocess.pretrain.seed_homologs import read_fasta

    members = usable_members(read_fasta(family_fasta).values())
    if disjoint_from_pairs:
        # Draw from the same population the editor's disjoint evaluation draws from.
        kept = disjoint_members(members, disjoint_from_pairs)
        logger.info(f"disjoint draw: {len(kept)}/{len(members)} family members are not in "
                    f"{disjoint_from_pairs}")
        members = kept
    split = split_family(members, n_templates, holdout_size, seed)
    train_members = read_corpus(train_corpus)
    leaked = set(train_members) & set(split.reference)
    if leaked:
        raise ValueError(
            f"{len(leaked)} sequence(s) are in BOTH the evotuning corpus and the scoring holdout. "
            "The `evotune` stage and this one must be run with the same --seed, --n-templates and "
            "--holdout-size; otherwise the baseline is scored against its own training data."
        )
    logger.info(
        f"{METHOD_NAMES[forced]}: {len(split.templates)} templates, {len(split.reference)} holdout, "
        f"{len(train_members)} train sequences for the profile, budget {budget} mutations"
    )

    # Refused rather than silently ignored: a scaffolded run is a different method from the
    # published baseline, so the declaration is consulted before a GPU is spent.
    capability = length_capability(forced)
    capability.check(target_length)

    rng = random.Random(seed + 7919)
    propose = mlm_proposer(model_folder, temperature, rng)
    generated, accounting, per_template = generate_variants(
        split.templates, train_members, budget, propose, n_variants, seed,
        forced=forced, top_up=top_up, max_rounds=max_rounds, profile_size=profile_size,
        target_length=target_length,
    )

    alignment_template = split.templates[0]
    # The same natural sequences generation_eval aligns: the front of the holdout, one per
    # template. Set-level metrics use the whole holdout.
    aligned_reference = [project_to_template(sequence, alignment_template)
                         for sequence in split.reference[:n_templates]]
    templates_expanded = [template for template in split.templates for _ in range(n_variants)]
    report = score_generations(generated, templates_expanded, split.reference, aligned_reference,
                              alignment_template, per_template, k=k, family_pool=split.pool,
                              ceiling_n=ceiling_n)

    from editjumps.core.diversity_novelty import diversity_novelty

    report["diversity_novelty"] = diversity_novelty(
        generated, split.templates, split.pool or split.reference, seed=seed
    )
    pll = {}
    if pll_model:
        from editjumps.core.pseudo_likelihood import pseudo_log_likelihood

        for name, group in (("generated", generated), ("natural_holdout", split.reference)):
            pll[name] = pseudo_log_likelihood(group, model_name=pll_model, seed=seed,
                                              max_positions=pll_positions)
            logger.info(f"  PLL {name:<18} {pll[name]['mean']:+.4f} per position (n={pll[name]['n']})")

    changed = [a["n_changed"] for a in accounting]
    masked = [a["n_masked"] for a in accounting]
    report.update({
        "method": METHOD_NAMES[forced],
        "model": str(model_folder),
        "family": str(family_fasta),
        "temperature": temperature,
        "forced_substitutions": forced,
        "mutation_budget": {
            "target": budget,
            "budget_mode": "realised (top up until the budget mutates)" if top_up
                           else "masked (one round of `budget` masks)",
            "mean_realised": float(np.mean(changed)) if changed else 0.0,
            "sd_realised": float(np.std(changed)) if changed else 0.0,
            "mean_masked": float(np.mean(masked)) if masked else 0.0,
            "hit_budget_fraction": float(np.mean([a["hit_budget"] for a in accounting])) if accounting else 0.0,
        },
        "n_templates": len(split.templates),
        "n_variants_per_template": n_variants,
        "n_generated": len(generated),
        "n_natural_reference": len(split.reference),
        "n_train_for_profile": len(train_members),
        # The number the disjoint table is read against, recorded per artefact rather than argued from the.
        "reference_in_training_pairs": (
            len(set(split.reference) & sequences_in_pairs(disjoint_from_pairs))
            if disjoint_from_pairs is not None else None
        ),
        "alignment": (f"Needleman-Wunsch projection onto the first template's coordinates "
                      f"(L={len(alignment_template)})"),
        # Popped by `main` into a sibling FASTA: pooled diversity, the spectrum MMD and the composition KL are.
        "_sequences": {METHOD_NAMES[forced]: generated},
        "esm2_pseudo_log_likelihood": pll,
        "not_implemented": [] if pll_model else ["esm2_pseudo_log_likelihood (B.2) - pass --pll-model"],
        "caveats": [
            "Set-level metrics (levenshtein, KL, MMD, diversity/novelty, PLL) are comparable to a "
            "generation-eval run on the same family with the same --seed, --n-templates, "
            "--n-variants and --holdout-size, and only then: the biased MMD V-statistic is not "
            "comparable across template counts.",
            "Alignment-dependent metrics (entropy_delta, covariance, MIp) use the FIRST template of "
            "the split as their coordinate system, while generation_eval uses whichever template "
            "its own permutation puts first. Comparable between the two evotuned baselines; only "
            "approximately so against the editor.",
            "This baseline's evotuning excludes the scoring holdout. The edit-flow model it is "
            "compared against draws its pairs from corpus-wide clustering, not from this "
            "per-family split, so it carries no such guarantee — the comparison is conservative in "
            "the baseline's disfavour, not in its favour.",
        ],
    })
    # Only when asked for, so an artefact from any existing invocation is unchanged key for key.
    if target_length is not None:
        report["length_growth"] = growth_summary(
            capability, target_length,
            [len(template) for template in templates_expanded],
            [len(sequence) for sequence in generated],
        )
        report["caveats"].append(
            "--target-length is OURS, not the paper's. This baseline is substitution-only "
            "(CAN_CHANGE_LENGTH is False); the run opened L - n masked columns of each template and "
            "let the MLM fill them, so the row is a SCAFFOLDED variant of the baseline and not the "
            "published one. The per-position metrics still project onto the first template's "
            "un-widened coordinates, so the opened columns are dropped from them, and "
            "levenshtein_to_own_template no longer equals the realised mutation count."
        )
    return report


def main(
    model_folder: Annotated[Path, typer.Option(help="Evotuned MLM from `editjumps evotune`")] = Path(
        "data/pretrain/esm2_evotuned"
    ),
    family_fasta: Annotated[Path, typer.Option(help="The seed family this was evotuned on")] = Path(
        "data/interim/seed_families/Anti-SARS-CoV-2_VHH_Ty1.fasta"
    ),
    disjoint_from_pairs: Annotated[Path | None, typer.Option(
        help="Draw the split only from members absent from this pairs file; match the "
             "editor's run exactly or the two are not comparable")] = None,
    train_corpus: Annotated[Path, typer.Option(help="`evotune`'s train corpus - the profile source")] = Path(
        "data/pretrain/evotune_family.train.txt.gz"
    ),
    forced: Annotated[bool, typer.Option(help="Block the original residue (the forced baseline)")] = False,
    n_templates: Annotated[int, typer.Option()] = 20,
    n_variants: Annotated[int, typer.Option()] = 20,
    mutations: Annotated[int, typer.Option(help="§4.2's matched mutation budget per sequence")] = 4,
    mutations_from: Annotated[
        Path | None,
        typer.Option(help="Read the budget from a generation-eval metrics JSON instead (§4.2 matching)"),
    ] = None,
    temperature: Annotated[float, typer.Option(help="MLM softmax temperature; the paper gives no value")] = 1.0,
    holdout_size: Annotated[int, typer.Option(help="Must match the `evotune` run")] = 200,
    profile_size: Annotated[int, typer.Option(help="Family members aligned per template")] = 200,
    budget_mode: Annotated[
        str, typer.Option(help="realised (top up until the budget mutates) | masked (one round)")
    ] = "realised",
    max_rounds: Annotated[int, typer.Option(help="Top-up round cap")] = 8,
    seed: Annotated[int, typer.Option(help="Must match the `evotune` run")] = 0,
    k: Annotated[int, typer.Option(help="Spectrum-kernel k-mer length")] = 3,
    ceiling_n: Annotated[
        int, typer.Option(help="Real homologs behind the agreement ceiling (must match the editor)")
    ] = 300,
    pll_model: Annotated[str, typer.Option(help="Stock ESM-2 for B.2 naturalness; empty skips")] = "",
    pll_positions: Annotated[int, typer.Option()] = 24,
    target_length: Annotated[int | None, typer.Option(
        help="Grow every variant to this width by opening masked columns (OURS, not the paper's "
             "baseline, which is substitution-only); omit for the published method")] = None,
    metrics_path: Annotated[Path, typer.Option()] = Path("metrics/evotune_baseline.json"),
) -> None:
    """Run one of EvoFlows' two evotuned-PLM baselines and write its Appendix-B metrics."""
    if budget_mode not in BUDGET_MODES:
        raise ValueError(f"budget_mode={budget_mode!r}; options: {BUDGET_MODES}")
    budget = budget_from_metrics(mutations_from) if mutations_from is not None else mutations
    if mutations_from is not None:
        logger.info(f"budget {budget} mutations, matched to {mutations_from} (§4.2)")
    if target_length is not None:
        logger.info(f"target length {target_length}: {length_capability(forced).method} is "
                    f"substitution-only, so this run OPENS masked columns (ours, not the paper's)")
    report = evaluate(
        model_folder, family_fasta, train_corpus, n_templates, n_variants, budget, seed,
        disjoint_from_pairs=disjoint_from_pairs,
        forced=forced, temperature=temperature, holdout_size=holdout_size, profile_size=profile_size,
        top_up=budget_mode == "realised", max_rounds=max_rounds, k=k, ceiling_n=ceiling_n,
        pll_model=pll_model,
        pll_positions=pll_positions,
        target_length=target_length,
    )
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    sequences = report.pop("_sequences", {})
    fasta = write_generated_fasta(metrics_path, sequences, header={
        "model": model_folder,
        "family": family_fasta,
        "seed": seed,
        "n_templates": n_templates,
        "n_variants": n_variants,
        "disjoint_from_pairs": disjoint_from_pairs,
    })
    write_metrics(metrics_path, report)
    if fasta:
        logger.info(f"wrote {fasta} ({sum(len(v) for v in sequences.values())} sequences)")
    budget_block = report["mutation_budget"]
    logger.info(
        f"{report['method']}: {budget_block['mean_realised']:.2f} +/- {budget_block['sd_realised']:.2f} "
        f"mutations realised against a budget of {budget_block['target']} "
        f"({budget_block['mean_masked']:.2f} positions masked)"
    )
    # `{value:12.4f}` raises TypeError on the per-template and per-sequence arrays, and it would do so AFTER.
    def log_section(section: str) -> None:
        """Log one metrics section, formatting scalars and describing arrays."""
        for name, value in report[section].items():
            if isinstance(value, int | float):
                logger.info(f"  {name:<40} {value:12.4f}")
            elif isinstance(value, list):
                logger.info(f"  {name:<40} {f'[{len(value)} values]':>12}")

    log_section("defined_by_the_paper")
    logger.info("  -- below: metrics the paper plots but never defines, our reading --")
    log_section("our_interpretation")
    logger.info(f"wrote {metrics_path}")

    # After the JSON, which is the durable record; `start_mlflow_run` has already installed
    # `make_tracking_nonfatal`, so a tracking failure below cannot fail the baseline.
    tags = {
        **design_space_tags(
            # An evotuned masked LM, so the objective axis is `mlm` (as in `pretrain_esm`) and the
            # weights started from a checkpoint that was then evotuned on this family.
            objective="mlm",
            weight_init="evotuned",
            pretrain_data=family_fasta.name,
        ),
        "method": report["method"],  # the §4.2 row: forced or unforced
        "section": "4.2",
    }
    if target_length is not None:
        # A scaffolded run is not the published baseline, so it must be a DIFFERENT row rather than
        # one that averages in with §4.2's.
        tags["method"] = f"{report['method']}_scaffolded"
        tags["target_length"] = str(target_length)
    with start_mlflow_run(
        APPENDIX_B_EXPERIMENT,
        run_name=f"evotune-baseline-{'forced' if forced else 'unforced'}-{family_fasta.stem}",
        tags=tags,
    ):
        import mlflow

        mlflow.log_params({
            "model_folder": str(model_folder),
            "family_fasta": str(family_fasta),
            "train_corpus": str(train_corpus),
            "disjoint_from_pairs": str(disjoint_from_pairs) if disjoint_from_pairs is not None else "",
            "forced": forced,
            "n_templates": n_templates,
            "n_variants": n_variants,
            # Both the requested budget and where it came from: `--mutations-from` overrides
            # `--mutations`, so logging only the latter would misdescribe a §4.2-matched run.
            "mutations": budget,
            "mutations_requested": mutations,
            "mutations_from": str(mutations_from) if mutations_from is not None else "",
            "temperature": temperature,
            "holdout_size": holdout_size,
            "profile_size": profile_size,
            "budget_mode": budget_mode,
            "max_rounds": max_rounds,
            "seed": seed,
            "k": k,
            "ceiling_n": ceiling_n,
            "pll_model": pll_model,
            "pll_positions": pll_positions,
            # 0 means "no target", which is what every §4.2 run does; a non-zero value marks the
            # SCAFFOLDED variant, and the two must be filterable apart in the same experiment.
            "target_length": target_length if target_length is not None else 0,
            "metrics_path": str(metrics_path),
        })
        # The whole report, so nothing is hand-picked: the nesting becomes the metric path, which keeps.
        for path, value in flatten_metrics(report, summarise_lists=True).items():
            mlflow.log_metric(path, value)
        mlflow.log_artifact(str(metrics_path))


if __name__ == "__main__":
    typer.run(main)
