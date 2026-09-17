"""EvoFlows' EvoDiff-MSA baseline, generated and scored (§4.2)."""

import json
import random
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import typer

from editjumps.core.evodiff_msa.alignment import write_a3m
from editjumps.core.evodiff_msa.proposer import MODELS, MsaProposer
from editjumps.core.evodiff_msa.sampling import uniform_weights
from editjumps.core.evotune.substitution import matched_budget
from editjumps.core.family_split import (
    disjoint_members,
    sequences_in_pairs,
    split_family,
    usable_members,
)
from editjumps.core.generation_metrics import mean_levenshtein_to_template, mean_pairwise_levenshtein
from editjumps.core.length_capability import (
    REFUSED,
    SCAFFOLDED,
    LengthCapability,
    growth_summary,
)
from editjumps.core.length_growth import (
    SLOT_SEED_OFFSET,
    align_to_template,
    allocate_slots,
    grow_by_infilling,
    query_row,
    slot_weights,
    widen_row,
)
from editjumps.core.utils import (
    design_space_tags,
    flatten_metrics,
    get_logger,
    start_mlflow_run,
    write_generated_fasta,
    write_metrics,
)
from editjumps.core.variant_seeds import check_variant_budget, template_seed, variant_seed
from editjumps.pipeline.evaluate.evotune_baseline import BUDGET_MODES, score_generations
from editjumps.pipeline.evaluate.generation_eval import APPENDIX_B_EXPERIMENT, project_to_template

logger = get_logger(__file__)

#: The paper's name for this baseline, as it appears in §4.2's list of six methods.
METHOD_NAME = "evodiff_msa"

#: How ``x1`` is produced. ``inpaint`` is the budget-matched mode §4.2 requires and is ours; ``unconditional``.
MODES: tuple[str, ...] = ("inpaint", "unconditional")

#: **EvoDiff-MSA cannot make a sequence longer.** It fills *columns of an alignment*: the width is the.
CAN_CHANGE_LENGTH: bool = False

#: What ``--target-length`` does in the default (``inpaint``) mode: `SCAFFOLDED`.
LENGTH_CAPABILITY: LengthCapability = LengthCapability(
    method=METHOD_NAME,
    can_change_length=CAN_CHANGE_LENGTH,
    target_length=SCAFFOLDED,
    mechanism=(
        "Fixed-width infilling: the model fills columns of an alignment and evodiff opens none, and "
        "build_alignment projects members onto the template so the width IS the template's length. "
        "--target-length instead builds the alignment at width L, with L - n columns opened where "
        "family members actually insert, and asks the model to fill them (ours, not theirs)."
    ),
)


def length_capability(mode: str) -> LengthCapability:
    """Return the length capability of this baseline in one of its two modes."""
    if mode not in MODES:
        raise ValueError(f"mode={mode!r}; options: {MODES}")
    if mode == "inpaint":
        return LENGTH_CAPABILITY
    from dataclasses import replace

    return replace(
        LENGTH_CAPABILITY,
        target_length=REFUSED,
        mechanism=(
            "--mode unconditional calls evodiff's generate_query_oadm_msa_simple verbatim: it "
            "decodes the entire query row over the alignment's width and the row is gap-stripped "
            "afterwards, so its length is whatever the model produces and nothing can target one. "
            "Use --mode inpaint for a scaffolded growth run, or record this row as inapplicable."
        ),
    )


def run_workdir(
    workdir: Path,
    family_fasta: Path,
    mode: str,
    seed: int,
    disjoint_from_pairs: Path | None,
    target_length: int | None,
) -> Path:
    """Give one run its own alignment directory, keyed by everything that changes the alignments."""
    parts = [family_fasta.stem, mode, f"seed{seed}"]
    if disjoint_from_pairs is not None:
        parts.append("disjoint")
    if target_length is not None:
        parts.append(f"L{target_length}")
    return workdir / "-".join(parts)


def build_alignment(template: str, members: list[str], msa_size: int, seed: int) -> list[str]:
    """Project family members onto one template's coordinates — the MSA rows, minus the query. **Ours."""
    return [project_to_template(member, template) for member in alignment_sample(members, msa_size, seed)]


def alignment_sample(members: list[str], msa_size: int, seed: int) -> list[str]:
    """Draw the family subsample the MSA is built from."""
    if not members:
        raise ValueError("no family members to align: an MSA baseline needs an MSA")
    if len(members) > msa_size:
        return random.Random(seed).sample(members, msa_size)
    return members


def build_widened_alignment(
    template: str, members: list[str], msa_size: int, seed: int, target_length: int,
    slot_rng: random.Random,
) -> tuple[str, list[str], list[int]]:
    """Build a width-``target_length`` alignment: the query row, the member rows and the opened columns.."""
    alignments = [align_to_template(member, template)
                  for member in alignment_sample(members, msa_size, seed)]
    counts = allocate_slots(len(template), target_length, slot_weights(alignments, len(template)), slot_rng)
    return query_row(template, counts), [widen_row(a, counts) for a in alignments], counts


def parse_slice(text: str) -> range | None:
    """Read the ``--generate-only A:B`` shard argument."""
    if not text:
        return None
    start, separator, stop = text.partition(":")
    if not separator or not start.isdigit() or not stop.isdigit() or int(start) >= int(stop):
        raise ValueError(f"--generate-only={text!r}; expected A:B with 0 <= A < B, e.g. 0:5")
    return range(int(start), int(stop))


def variants_cache(workdir: Path, index: int) -> Path:
    """Where one template's finished variants are kept, beside its alignment."""
    return workdir / f"template_{index:04d}.variants.json"


def cached_variants(
    workdir: Path, index: int, template: str, fingerprint: dict[str, Any]
) -> tuple[list[str], list[dict], dict] | None:
    """Read one template's variants back, but only if they were made the same way. **Why a cache at."""
    path = variants_cache(workdir, index)
    if not path.exists():
        return None
    try:
        entry = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        # A half-written file from a killed run is a miss, not a crash: the point of the cache is
        # to survive interruptions, and an interruption during the write is one of them.
        logger.warning(f"{path} is unreadable; regenerating template {index}")
        return None
    if entry.get("template") != template or entry.get("fingerprint") != fingerprint:
        logger.info(f"{path} was made with different settings; regenerating template {index}")
        return None
    return entry["variants"], entry["accounting"], entry["per_template"]


def write_variants(
    workdir: Path, index: int, template: str, fingerprint: dict[str, Any],
    variants: list[str], accounting: list[dict], per_template: dict,
) -> None:
    """Save one template's variants so an interrupted run does not repeat them."""
    path = variants_cache(workdir, index)
    payload = {"template": template, "fingerprint": fingerprint, "variants": variants,
               "accounting": accounting, "per_template": per_template}
    temporary = path.with_suffix(".json.partial")
    temporary.write_text(json.dumps(payload))
    temporary.replace(path)


def generate_variants(
    templates: list[str],
    train_members: list[str],
    budget: int,
    n_variants: int,
    seed: int,
    workdir: Path,
    *,
    mode: str = "inpaint",
    forced: bool = False,
    top_up: bool = True,
    max_rounds: int = 8,
    msa_size: int = 256,
    n_sequences: int = 64,
    model: str = "msa-oa-dm-maxsub",
    selection_type: str = "",
    temperature: float = 1.0,
    device: str = "auto",
    shim: str = "",
    penalty_value: float = 2.0,
    target_length: int | None = None,
    only: range | None = None,
) -> tuple[list[str], list[dict], list[dict]]:
    """Run the baseline over every template: one alignment, one model process, ``n_variants`` draws."""
    if mode not in MODES:
        raise ValueError(f"mode={mode!r}; options: {MODES}")
    check_variant_budget(n_variants)
    length_capability(mode).check(target_length)
    generated: list[str] = []
    accounting: list[dict] = []
    per_template: list[dict] = []
    workdir.mkdir(parents=True, exist_ok=True)

    fingerprint = {
        "budget": budget, "n_variants": n_variants, "seed": seed, "mode": mode, "forced": forced,
        "top_up": top_up, "max_rounds": max_rounds, "msa_size": msa_size,
        "n_sequences": n_sequences, "model": model, "selection_type": selection_type,
        "temperature": temperature, "penalty_value": penalty_value, "target_length": target_length,
    }
    for index, template in enumerate(templates):
        if only is not None and index not in only:
            continue
        cached = cached_variants(workdir, index, template, fingerprint)
        if cached is not None:
            variants, template_accounting, summary = cached
            generated.extend(variants)
            accounting.extend(template_accounting)
            per_template.append(summary)
            logger.info(f"template {index + 1}/{len(templates)} (L={len(template)}): "
                        f"{len(variants)} variants read from cache")
            continue
        if target_length is None:
            aligned = build_alignment(template, train_members, msa_size, seed + index)
            row0, counts = template, [0] * (len(template) + 1)
        else:
            # Width L, with the extra columns where family members actually insert. `counts` is carried into.
            row0, aligned, counts = build_widened_alignment(
                template, train_members, msa_size, seed + index, target_length,
                random.Random(seed + SLOT_SEED_OFFSET + index),
            )
        a3m = workdir / f"template_{index:04d}.a3m"
        rows = write_a3m(a3m, row0, aligned)
        rows_used = min(n_sequences, rows)
        kwargs: dict[str, Any] = {"n_sequences": rows_used, "model": model,
                                  "selection_type": selection_type, "max_seq_len": len(row0),
                                  "temperature": temperature, "device": device,
                                  # The runner reads the a3m by path in another process; this is
                                  # what it must find there.
                                  "expect_query": row0}
        if shim:
            kwargs["shim"] = shim
        variants: list[str] = []
        with MsaProposer(a3m, template_seed(seed, index), **kwargs) as propose:
            for variant in range(n_variants):
                if mode == "unconditional":
                    sequence = propose.generate_query(penalty_value)
                    stats = {"budget": 0, "n_masked": len(template), "rounds": 1,
                             "n_changed": int(mean_levenshtein_to_template([sequence], template)),
                             "hit_budget": False}
                else:
                    rng = random.Random(variant_seed(seed, index, variant))
                    sequence, stats = grow_by_infilling(
                        template, uniform_weights(template), budget, counts, propose, rng,
                        forced=forced, top_up=top_up, max_rounds=max_rounds,
                    )
                variants.append(sequence)
                accounting.append(stats)
            forwards = propose.n_calls
        generated.extend(variants)
        changed = [a["n_changed"] for a in accounting[-n_variants:]]
        per_template.append({
            "levenshtein_to_template": mean_levenshtein_to_template(variants, template),
            # Per SEQUENCE as well, for the same reason evotune_baseline records it: the report has
            # to carry the arrays an interval is bootstrapped from, not just the centre.
            "levenshtein_to_template_per_sequence": [
                mean_levenshtein_to_template([sequence], template) for sequence in variants
            ],
            "pairwise_levenshtein": mean_pairwise_levenshtein(variants),
            "mean_mutations": float(np.mean(changed)) if changed else 0.0,
            "msa_rows": rows_used,
            "model_forward_passes": forwards,
        })
        write_variants(workdir, index, template, fingerprint,
                       variants, accounting[-n_variants:], per_template[-1])
        logger.info(
            f"template {index + 1}/{len(templates)} (L={len(template)}): {len(variants)} variants "
            f"from {rows_used} MSA rows, {per_template[-1]['mean_mutations']:.2f} mutations "
            f"(budget {budget}), edit distance {per_template[-1]['levenshtein_to_template']:.2f}"
        )
    return generated, accounting, per_template


def evaluate(
    family_fasta: Path,
    n_templates: int,
    n_variants: int,
    budget: int,
    seed: int,
    workdir: Path,
    *,
    disjoint_from_pairs: Path | None = None,
    mode: str = "inpaint",
    forced: bool = False,
    holdout_size: int = 200,
    msa_size: int = 256,
    n_sequences: int = 64,
    model: str = "msa-oa-dm-maxsub",
    selection_type: str = "",
    temperature: float = 1.0,
    device: str = "auto",
    top_up: bool = True,
    max_rounds: int = 8,
    penalty_value: float = 2.0,
    k: int = 3,
    ceiling_n: int = 300,
    pll_model: str = "",
    pll_positions: int = 24,
    shim: str = "",
    target_length: int | None = None,
    generate_only: range | None = None,
) -> dict[str, Any]:
    """Generate the EvoDiff-MSA baseline over one family and score it against the family holdout."""
    from editjumps.pipeline.preprocess.pretrain.seed_homologs import read_fasta

    members = usable_members(read_fasta(family_fasta).values())
    if disjoint_from_pairs:
        # Draw from the same population the editor's disjoint evaluation draws from.
        kept = disjoint_members(members, disjoint_from_pairs)
        logger.info(f"disjoint draw: {len(kept)}/{len(members)} family members are not in "
                    f"{disjoint_from_pairs}")
        members = kept
    split = split_family(members, n_templates, holdout_size, seed)
    train_members = split.pool
    leaked = set(train_members) & set(split.reference)
    if leaked:
        raise ValueError(
            f"{len(leaked)} sequence(s) are in BOTH the MSA source and the scoring holdout; "
            "the generator would be conditioned on its own reference distribution"
        )
    if not train_members:
        raise ValueError(
            f"{family_fasta} leaves no train part at n_templates={n_templates}, "
            f"holdout_size={holdout_size}: there is no MSA to condition on. Lower --holdout-size."
        )
    logger.info(
        f"{METHOD_NAME} ({mode}): {len(split.templates)} templates, {len(split.reference)} holdout, "
        f"{len(train_members)} train sequences for the MSA, budget {budget} mutations"
    )

    # Refused before the checkpoint is downloaded rather than after: `unconditional` has no width to
    # target, and a same-length answer to a growth question is worse than no answer.
    capability = length_capability(mode)
    capability.check(target_length)

    generated, accounting, per_template = generate_variants(
        split.templates, train_members, budget, n_variants, seed, run_workdir(
            workdir, family_fasta, mode, seed, disjoint_from_pairs, target_length,
        ),
        mode=mode, forced=forced, top_up=top_up, max_rounds=max_rounds, msa_size=msa_size,
        n_sequences=n_sequences, model=model, selection_type=selection_type,
        temperature=temperature, device=device, shim=shim, penalty_value=penalty_value,
        target_length=target_length, only=generate_only,
    )
    # A shard fills the cache and stops.
    if generate_only is not None:
        return {"generated_only": [generate_only.start, generate_only.stop],
                "n_generated": len(generated)}

    alignment_template = split.templates[0]
    # The same natural sequences the other two align: the front of the holdout, one per template.
    # Set-level metrics use the whole holdout.
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
    report.update({
        "method": METHOD_NAME,
        "mode": mode,
        "model": model,
        "family": str(family_fasta),
        "temperature": temperature,
        "forced_substitutions": forced,
        "msa": {
            "source": "the family's TRAIN part (§4.2), never the scoring holdout",
            "n_available": len(train_members),
            "n_aligned_per_template": min(msa_size, len(train_members)),
            "n_rows_conditioned_on": [p["msa_rows"] for p in per_template][:1] or [0],
            "selection_type": selection_type or MODELS[model],
            "alignment": ("Needleman-Wunsch projection onto each template's own coordinates "
                          "(ours, not theirs - see build_alignment)"),
        },
        "mutation_budget": {
            "target": budget if mode == "inpaint" else None,
            "matched": mode == "inpaint",
            "budget_mode": ("realised (top up until the budget mutates)" if top_up
                            else "masked (one round of `budget` masks)") if mode == "inpaint"
                           else "none - their unconditional entry point decodes the whole query row",
            "mean_realised": float(np.mean(changed)) if changed else 0.0,
            "sd_realised": float(np.std(changed)) if changed else 0.0,
            "hit_budget_fraction": (float(np.mean([a["hit_budget"] for a in accounting]))
                                    if accounting else 0.0),
        },
        "n_templates": len(split.templates),
        "n_variants_per_template": n_variants,
        "n_generated": len(generated),
        "n_natural_reference": len(split.reference),
        "model_forward_passes": int(sum(p["model_forward_passes"] for p in per_template)),
        # Recorded per artefact rather than argued from the flag: `--disjoint-from-pairs` is meant to drive.
        "reference_in_training_pairs": (
            len(set(split.reference) & sequences_in_pairs(disjoint_from_pairs))
            if disjoint_from_pairs is not None else None
        ),
        "alignment": (f"Needleman-Wunsch projection onto the first template's coordinates "
                      f"(L={len(alignment_template)})"),
        # Popped by `main` into a sibling FASTA: the pooled diversity, the spectrum MMD and the composition KL.
        "_sequences": {f"{METHOD_NAME}_{mode}": generated},
        "esm2_pseudo_log_likelihood": pll,
        "not_implemented": [] if pll_model else ["esm2_pseudo_log_likelihood (B.2) - pass --pll-model"],
        "ours_not_theirs": [
            "Where the budget goes: EvoDiff-MSA has no positional model and §4.2 does not say, so "
            "positions are drawn UNIFORMLY, matching the paper's own random-mutation baseline.",
            "Inpainting at all: evodiff ships no MSA-inpainting entry point, only whole-query "
            "generation, which cannot be budget-matched. Masking a subset of x0's positions and "
            "decoding only those is ours.",
            "The candidate set (20 standard amino acids), the temperature and the forced-"
            "substitution switch. Their own per-step rule is unchanged in --mode unconditional.",
            "That the MSA is built from the family's train part, and aligned by Needleman-Wunsch "
            "projection onto the template rather than by a profile aligner.",
        ],
        "theirs_not_ours": [
            f"The model, the released weights ({model}) and the tokenizer: evodiff 1.1.2, run in "
            "its own environment (editjumps/core/evodiff_msa/install_evodiff.sh), never reimplemented here.",
            "The MSA subsampling: evodiff.data.subsample_msa, unchanged, including MaxHamming.",
            "In --mode unconditional, the whole decoding loop: "
            "evodiff.generate_msa.generate_query_oadm_msa_simple, called verbatim.",
        ],
        "caveats": [
            "Set-level metrics (levenshtein, KL, MMD, diversity/novelty, PLL) are comparable to a "
            "generation-eval or evotune-baseline run on the same family with the same --seed, "
            "--n-templates, --n-variants and --holdout-size, and only then: the biased MMD "
            "V-statistic is not comparable across template counts.",
            "Alignment-dependent metrics (entropy_delta, covariance, MIp) use the FIRST template of "
            "the split as their coordinate system, the same choice evotune_baseline makes. "
            "Comparable between the three model-based baselines; only approximately so against the "
            "editor, whose own permutation puts a different template first.",
            "--mode unconditional is NOT budget-matched: its mutation count is whatever the model "
            "produces, so its distances to the holdout answer a different question from every "
            "other row of the §4.2 comparison.",
            "This baseline never sees the scoring holdout, in the MSA or anywhere else. The edit-"
            "flow model it is compared against draws its pairs from corpus-wide clustering rather "
            "than from this per-family split, so it carries no such guarantee - the comparison is "
            "conservative in the baseline's disfavour, not in its favour.",
        ],
    })
    # Only when asked for, so an artefact from any existing invocation is unchanged key for key.
    if target_length is not None:
        report["length_growth"] = growth_summary(
            capability, target_length,
            [len(template) for template in templates_expanded],
            [len(sequence) for sequence in generated],
        )
        report["ours_not_theirs"].append(
            "The width itself: --target-length builds the alignment at width L with L - n columns "
            "opened where family members insert, and asks the model to fill them. EvoDiff-MSA "
            "cannot change length (CAN_CHANGE_LENGTH is False) and evodiff opens no column."
        )
        report["caveats"].append(
            "This is a SCAFFOLDED growth run, not the published baseline. The per-position metrics "
            "still project onto the first template's un-widened coordinates, so the opened columns "
            "are dropped from them."
        )
    return report


def main(
    family_fasta: Annotated[Path, typer.Option(help="The seed family to generate from")] = Path(
        "data/interim/seed_families/Anti-SARS-CoV-2_VHH_Ty1.fasta"
    ),
    disjoint_from_pairs: Annotated[Path | None, typer.Option(
        help="Draw the split only from members absent from this pairs file; match the "
             "editor's run exactly or the two are not comparable")] = None,
    mode: Annotated[str, typer.Option(help="inpaint (budget-matched, §4.2) | unconditional (theirs)")] = "inpaint",
    model: Annotated[str, typer.Option(help="Released EvoDiff MSA checkpoint")] = "msa-oa-dm-maxsub",
    forced: Annotated[bool, typer.Option(help="Block the original residue (ours, not the paper's)")] = False,
    n_templates: Annotated[int, typer.Option()] = 20,
    n_variants: Annotated[int, typer.Option()] = 20,
    mutations: Annotated[int, typer.Option(help="§4.2's matched mutation budget per sequence")] = 4,
    mutations_from: Annotated[
        Path | None,
        typer.Option(help="Read the budget from a generation-eval metrics JSON instead (§4.2 matching)"),
    ] = None,
    holdout_size: Annotated[int, typer.Option(help="Must match the other baselines")] = 200,
    msa_size: Annotated[int, typer.Option(help="Family members aligned per template")] = 256,
    n_sequences: Annotated[int, typer.Option(help="MSA rows the model conditions on, query included")] = 64,
    selection_type: Annotated[
        str, typer.Option(help="MSA subsampling; empty = the checkpoint's training-time choice")
    ] = "",
    temperature: Annotated[float, typer.Option(help="Softmax temperature; 1.0 is the model's own")] = 1.0,
    device: Annotated[str, typer.Option(help="auto | cpu | cuda | mps, resolved in the runner")] = "auto",
    budget_mode: Annotated[
        str, typer.Option(help="realised (top up until the budget mutates) | masked (one round)")
    ] = "realised",
    max_rounds: Annotated[int, typer.Option(help="Top-up round cap")] = 8,
    penalty_value: Annotated[float, typer.Option(help="unconditional only: their gap penalty")] = 2.0,
    seed: Annotated[int, typer.Option(help="Must match the other baselines and the editor")] = 0,
    k: Annotated[int, typer.Option(help="Spectrum-kernel k-mer length")] = 3,
    ceiling_n: Annotated[
        int, typer.Option(help="Real homologs behind the agreement ceiling (must match the editor)")
    ] = 300,
    pll_model: Annotated[str, typer.Option(help="Stock ESM-2 for B.2 naturalness; empty skips")] = "",
    pll_positions: Annotated[int, typer.Option()] = 24,
    target_length: Annotated[int | None, typer.Option(
        help="Build the alignment at this width and open the extra columns for the model to fill "
             "(OURS, not theirs; refused with --mode unconditional); omit for the published method")] = None,
    workdir: Annotated[Path, typer.Option(help="Where per-template alignments are written")] = Path(
        "data/interim/evodiff_msa"
    ),
    generate_only: Annotated[str, typer.Option(
        help="A:B — generate only templates [A, B) into the run's cache and write no metrics. Every "
             "draw is seeded from the template index, so shards join into the same run; scoring is "
             "the unsharded command afterwards, which reads them all back")] = "",
    shim: Annotated[str, typer.Option(help="Override the `evodiff-msa` executable (testing)")] = "",
    metrics_path: Annotated[Path, typer.Option()] = Path("metrics/evodiff_msa_baseline.json"),
) -> None:
    """Run EvoFlows' EvoDiff-MSA baseline (§4.2) and write its Appendix-B metrics."""
    if budget_mode not in BUDGET_MODES:
        raise ValueError(f"budget_mode={budget_mode!r}; options: {BUDGET_MODES}")
    if mode not in MODES:
        raise ValueError(f"mode={mode!r}; options: {MODES}")
    if model not in MODELS:
        raise ValueError(f"model={model!r}; options: {sorted(MODELS)}")
    only = parse_slice(generate_only)
    budget = mutations
    if mutations_from is not None:
        from editjumps.pipeline.evaluate.evotune_baseline import budget_from_metrics

        budget = budget_from_metrics(mutations_from)
        logger.info(f"budget {budget} mutations, matched to {mutations_from} (§4.2)")
    else:
        budget = matched_budget(float(mutations))
    report = evaluate(
        family_fasta, n_templates, n_variants, budget, seed, workdir,
        disjoint_from_pairs=disjoint_from_pairs,
        mode=mode, forced=forced, holdout_size=holdout_size, msa_size=msa_size,
        n_sequences=n_sequences, model=model, selection_type=selection_type,
        temperature=temperature, device=device, top_up=budget_mode == "realised",
        max_rounds=max_rounds, penalty_value=penalty_value, k=k, ceiling_n=ceiling_n,
        pll_model=pll_model,
        pll_positions=pll_positions, shim=shim,
        target_length=target_length, generate_only=only,
    )
    if only is not None:
        logger.info(f"cached templates {only.start}:{only.stop} ({report['n_generated']} "
                    f"sequences); run without --generate-only to score the whole cell")
        return
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    sequences = report.pop("_sequences", {})
    fasta = write_generated_fasta(metrics_path, sequences, header={
        "model": model,
        "family": family_fasta,
        "mode": mode,
        "seed": seed,
        "n_templates": n_templates,
        "n_variants": n_variants,
        "disjoint_from_pairs": disjoint_from_pairs,
    })
    write_metrics(metrics_path, report)
    if fasta:
        logger.info(f"wrote {fasta} ({sum(len(v) for v in sequences.values())} sequences)")
    budget_report = report["mutation_budget"]
    logger.info(
        f"wrote {metrics_path}: {report['n_generated']} sequences, "
        f"{budget_report['mean_realised']:.2f} +/- {budget_report['sd_realised']:.2f} mutations "
        f"(target {budget_report['target']}), {report['model_forward_passes']} forward passes"
    )

    # After the JSON, which is the durable record; `start_mlflow_run` installs
    # `make_tracking_nonfatal`, so nothing here can fail a run that has already generated.
    tags = {
        **design_space_tags(
            objective="msa_diffusion",   # order-agnostic autoregressive diffusion over an MSA
            backbone=model,              # their released checkpoint, not one of ours
            weight_init="released",
            pretrain_data=family_fasta.name,
        ),
        # `--mode unconditional` is NOT budget-matched, so it is a different §4.2 row from
        # `inpaint` and must be filterable as one rather than averaged in with it.
        "method": f"evodiff_msa_{mode}",
        "section": "4.2",
    }
    if target_length is not None:
        # A scaffolded run is not the published baseline, so it is a DIFFERENT row.
        tags["method"] = f"evodiff_msa_{mode}_scaffolded"
        tags["target_length"] = str(target_length)
    with start_mlflow_run(
        APPENDIX_B_EXPERIMENT,
        run_name=f"evodiff-msa-{mode}-{family_fasta.stem}",
        tags=tags,
    ):
        import mlflow

        mlflow.log_params({
            "family_fasta": str(family_fasta),
            "disjoint_from_pairs": str(disjoint_from_pairs) if disjoint_from_pairs is not None else "",
            "mode": mode,
            "model": model,
            "forced": forced,
            "n_templates": n_templates,
            "n_variants": n_variants,
            # The budget as resolved (`matched_budget`, or read from an editor's run) and as asked
            # for, plus where it was read from; only the resolved one describes the generations.
            "mutations": budget,
            "mutations_requested": mutations,
            "mutations_from": str(mutations_from) if mutations_from is not None else "",
            "holdout_size": holdout_size,
            "msa_size": msa_size,
            "n_sequences": n_sequences,
            "selection_type": selection_type,
            "temperature": temperature,
            "device": device,
            "budget_mode": budget_mode,
            "max_rounds": max_rounds,
            "penalty_value": penalty_value,
            "seed": seed,
            "k": k,
            "ceiling_n": ceiling_n,
            "pll_model": pll_model,
            "pll_positions": pll_positions,
            # 0 means "no target", which is what every §4.2 run does; a non-zero value marks the
            # SCAFFOLDED variant, and the two must be filterable apart in the same experiment.
            "target_length": target_length if target_length is not None else 0,
            "workdir": str(workdir),
            "metrics_path": str(metrics_path),
        })
        # Whole tree, nesting as the metric path, so nothing is hand-picked and no repeated key name collides..
        for path, value in flatten_metrics(report, summarise_lists=True).items():
            mlflow.log_metric(path, value)
        mlflow.log_artifact(str(metrics_path))


if __name__ == "__main__":
    typer.run(main)
