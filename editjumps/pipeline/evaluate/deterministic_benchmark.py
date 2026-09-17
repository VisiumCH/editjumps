"""EvoFlows §4.1's deterministic benchmark, scored — the only ground-truth evaluation we have."""

from pathlib import Path
from typing import Annotated, Any, NamedTuple

import typer

from editjumps.core.edit_flows.alignment import levenshtein, needleman_wunsch
from editjumps.core.edit_flows.deterministic import (
    EDIT_CLASSES,
    apply_deterministic_edits,
    class_counts,
    deterministic_edits,
    ground_truth_labels,
)
from editjumps.core.edit_flows.inference import EditEvent, apply_event
from editjumps.core.edit_flows.targets import edit_targets
from editjumps.core.sequences import AA, PAIR_SEP
from editjumps.core.utils import (
    decode_one,
    design_space_tags,
    encode_one,
    flatten_metrics,
    get_logger,
    load_tokenizer,
    start_mlflow_run,
    tokens_of,
    write_metrics,
)

logger = get_logger(__file__)

#: MLflow experiment for §4.1. Separate from the Appendix-B one because the two evaluations share no metric.
EXPERIMENT_NAME = "editjumps-deterministic-benchmark"

#: Stable letter<->int mapping for the alignment (Needleman-Wunsch works on token ids).
ALPHABET: tuple[str, ...] = tuple(sorted(AA))
_TO_INT = {letter: i for i, letter in enumerate(ALPHABET)}

#: The residue each rule writes, for scoring identity (§4.1: Sub -> H, Ins -> S).
CLASS_RESIDUE: dict[str, str] = {"substitution": "H", "insertion": "S"}


def _encode(seq: str) -> list[int]:
    """Map a residue string to token ids for alignment, dropping anything off-alphabet."""
    return [_TO_INT[c] for c in seq if c in _TO_INT]


def predicted_edit_labels(z0: str, generated: str) -> tuple[list[set[str]], dict[int, dict[str, str]]]:
    """Recover which edit classes landed on each position of ``z0``, by alignment."""
    # Z0 indexes the label list, so a character the alphabet drops shifts every position after it and silently.
    off_alphabet = sorted(set(z0) - set(ALPHABET))
    if off_alphabet:
        raise ValueError(
            f"z0 contains non-alphabet character(s) {off_alphabet}; score one chain at a time "
            "(§4.1 is defined on a single natural protein sequence, not a joined VH.VL string)"
        )
    # A shared synthetic BOS on both sides before aligning. `edit_targets` needs a real token in column 0 (eq.
    bos = len(ALPHABET)
    source, target = [bos, *_encode(z0)], [bos, *_encode(generated)]
    # The DEFAULT ② scoring, and it must stay matched to the checkpoint's: recovery re-derives the realised.
    aligned_0, aligned_1 = needleman_wunsch(source, target)

    labels: list[set[str]] = [set() for _ in range(len(z0))]
    residues: dict[int, dict[str, str]] = {}
    for edit in edit_targets(aligned_0, aligned_1):
        # insertions attach *after* edit.index; §4.1 counts them at the position they precede.
        # Both are then shifted back by the synthetic BOS occupying stripped index 0.
        position = edit.index if edit.op == "insert" else edit.index - 1
        if not 0 <= position < len(z0):
            continue  # an insertion past the final residue has no §4.1 position to score at
        cls = {"insert": "insertion", "delete": "deletion", "substitute": "substitution"}[edit.op]
        labels[position].add(cls)
        if 0 <= edit.token < len(ALPHABET):
            residues.setdefault(position, {})[cls] = ALPHABET[edit.token]
    return labels, residues


class ProvenanceTrace(NamedTuple):
    """One generation's per-token provenance, decoded to residues so scoring stays torch-free."""

    input_residues: list[str | None]
    output_residues: list[str | None]
    provenance: list[int | None]


#: The bookkeeping a provenance run cannot score, reported rather than dropped: output material with no input.
PROVENANCE_STAT_KEYS: tuple[str, ...] = (
    "inserted_tokens", "insertions_unattributable", "substitutions_to_non_residue",
)


def residues_from_tokens(tokens: list[str]) -> list[str | None]:
    """Map tokenizer token strings to residue letters, with None for anything that is not one."""
    return [tok if tok in _TO_INT else None for tok in tokens]


def predicted_edit_labels_from_provenance(
    z0: str, trace: ProvenanceTrace,
) -> tuple[list[set[str]], dict[int, dict[str, str]], dict[str, int]]:
    """Read which edit classes landed on each position of ``z0`` from the sampler's own provenance."""
    residue_slots = [i for i, res in enumerate(trace.input_residues) if res is not None]
    spelled = "".join(str(trace.input_residues[i]) for i in residue_slots)
    if spelled != z0:
        raise ValueError(
            f"provenance input does not spell z0 ({len(spelled)} residues vs {len(z0)}); the "
            "tokenizer used for generation must be the one whose tokens are passed here"
        )
    to_z0 = {slot: k for k, slot in enumerate(residue_slots)}

    labels: list[set[str]] = [set() for _ in range(len(z0))]
    residues: dict[int, dict[str, str]] = {}
    stats = dict.fromkeys(PROVENANCE_STAT_KEYS, 0)

    survived: set[int] = set()
    run: list[str | None] = []
    last_z: int | None = None

    def flush(following_z: int | None) -> None:
        """Attribute a finished run of origin-less tokens to a §4.1 insertion position."""
        if not run:
            return
        position = last_z + 1 if last_z is not None else following_z
        if position is None or not 0 <= position < len(z0):
            stats["insertions_unattributable"] += 1
        else:
            labels[position].add("insertion")
            written = run[-1]  # the inserted residue immediately preceding `position`
            if written is not None:
                residues.setdefault(position, {})["insertion"] = written
        run.clear()

    for out_i, origin in enumerate(trace.provenance):
        if origin is None:
            stats["inserted_tokens"] += 1
            run.append(trace.output_residues[out_i])
            continue
        z = to_z0.get(origin)
        if z is None:
            continue  # a surviving special token (BOS/EOS): anchors no z0 position
        flush(z)
        survived.add(z)
        last_z = z
        written = trace.output_residues[out_i]
        if written is None:
            stats["substitutions_to_non_residue"] += 1
        elif written != z0[z]:
            labels[z].add("substitution")
            residues.setdefault(z, {})["substitution"] = written
    flush(None)

    for z in range(len(z0)):
        if z not in survived:
            labels[z].add("deletion")
    return labels, residues, stats


def oracle_provenance(z0: str) -> ProvenanceTrace:
    """Build a perfect editor's output + provenance for §4.1's rules (the ceiling control)."""
    off_alphabet = sorted(set(z0) - set(ALPHABET))
    if off_alphabet:
        raise ValueError(f"z0 contains non-alphabet character(s) {off_alphabet}")
    bos, eos = len(ALPHABET), len(ALPHABET) + 1
    ids = [bos, *_encode(z0), eos]
    provenance: list[int | None] = list(range(len(ids)))

    # `deterministic_edits` already resolves the ordering (a deletion suppresses the substitution
    # at that position), so a perfect editor is exactly one that applies these labels.
    truth = deterministic_edits(z0)
    by_class = {
        cls: sorted(p for p, kinds in truth.items() if cls in kinds)
        for cls in ("insertion", "deletion", "substitution")
    }

    def slot_of(z: int) -> int:
        """Locate z0 position ``z`` in the current sequence (its origin index is ``z + 1``)."""
        return provenance.index(z + 1)

    def step(kind: str, slot: int, token: int | None) -> None:
        """Apply one edit through the sampler's own bookkeeping, keeping the trace in step."""
        nonlocal ids, provenance
        ids, _, updated = apply_event(ids, EditEvent(kind, slot), token, provenance=provenance)
        provenance = updated if updated is not None else provenance

    for z in by_class["insertion"]:
        # Ins(z, S) writes the new residue BEFORE position z, i.e. after the slot preceding it.
        step("insert", slot_of(z) - 1, _TO_INT["S"])
    for z in by_class["deletion"]:
        step("delete", slot_of(z), None)
    for z in by_class["substitution"]:
        step("substitute", slot_of(z), _TO_INT["H"])
    to_letter: list[str | None] = [ALPHABET[t] if t < len(ALPHABET) else None for t in ids]
    input_residues: list[str | None] = [None, *list(z0), None]
    return ProvenanceTrace(input_residues, to_letter, provenance)


def score_classes(
    truth: list[list[set[str]]],
    predicted: list[list[set[str]]],
    residues: list[dict[int, dict[str, str]]],
) -> dict[str, dict[str, float]]:
    """Per-class precision/recall over a set of sequences, plus identity accuracy. ``no_op`` is scored."""
    if len(truth) != len(predicted) or len(truth) != len(residues):
        raise ValueError(f"shape mismatch: {len(truth)} truth vs {len(predicted)} predicted")

    out: dict[str, dict[str, float]] = {}
    for cls in EDIT_CLASSES:
        tp = fp = fn = 0
        id_ok = id_total = 0
        for seq_truth, seq_pred, seq_res in zip(truth, predicted, residues, strict=True):
            if len(seq_truth) != len(seq_pred):
                raise ValueError(f"sequence length mismatch: {len(seq_truth)} vs {len(seq_pred)}")
            for i, (t_set, p_set) in enumerate(zip(seq_truth, seq_pred, strict=True)):
                # no_op is the absence of every other class, not a member of the label sets
                in_truth = (not t_set) if cls == "no_op" else cls in t_set
                in_pred = (not p_set) if cls == "no_op" else cls in p_set
                if in_truth and in_pred:
                    tp += 1
                    if cls in CLASS_RESIDUE:
                        id_total += 1
                        if seq_res.get(i, {}).get(cls) == CLASS_RESIDUE[cls]:
                            id_ok += 1
                elif in_pred:
                    fp += 1
                elif in_truth:
                    fn += 1

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        stats = {
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "tp": float(tp), "fp": float(fp), "fn": float(fn), "support": float(tp + fn),
        }
        if cls in CLASS_RESIDUE:
            stats["identity_correct"] = float(id_ok)
            stats["identity_total"] = float(id_total)
            stats["identity_accuracy"] = id_ok / id_total if id_total else 0.0
        out[cls] = stats
    return out


def sequence_level(z0s: list[str], generated: list[str]) -> dict[str, float]:
    """Whole-sequence agreement with the unique ``z1`` the rules define."""
    exact = 0
    distances, baselines = [], []
    for z0, gen in zip(z0s, generated, strict=True):
        z1 = apply_deterministic_edits(z0)
        exact += int(gen == z1)
        distances.append(levenshtein(gen, z1))
        baselines.append(levenshtein(z0, z1))
    n = len(z0s)
    return {
        "exact_match_rate": exact / n if n else 0.0,
        "mean_levenshtein_to_z1": sum(distances) / n if n else 0.0,
        "mean_levenshtein_baseline": sum(baselines) / n if n else 0.0,
        "n": float(n),
    }


def score_generations(
    z0s: list[str], generated: list[str], traces: list[ProvenanceTrace] | None = None,
) -> dict[str, Any]:
    """Score a set of (source, output) pairs — the torch-free core, so it is unit-testable."""
    truth = [ground_truth_labels(z0) for z0 in z0s]
    predicted, residues = [], []
    stats = dict.fromkeys(PROVENANCE_STAT_KEYS, 0)
    if traces is None:
        for z0, gen in zip(z0s, generated, strict=True):
            labels, res = predicted_edit_labels(z0, gen)
            predicted.append(labels)
            residues.append(res)
    else:
        for z0, trace in zip(z0s, traces, strict=True):
            labels, res, seq_stats = predicted_edit_labels_from_provenance(z0, trace)
            predicted.append(labels)
            residues.append(res)
            for key, value in seq_stats.items():
                stats[key] += value
    report: dict[str, Any] = {
        "per_class": score_classes(truth, predicted, residues),
        "sequence": sequence_level(z0s, generated),
        # so a reader can see whether the rare classes have enough support to trust
        "ground_truth_counts": class_counts(z0s),
    }
    if traces is not None:
        report["provenance_stats"] = stats
    return report


def main(
    model_folder: Annotated[Path, typer.Option()] = Path("data/pretrain/edit_flows"),
    sequences: Annotated[Path, typer.Option()] = Path("data/pretrain/oas_corpus.val.txt.gz"),
    n: Annotated[int, typer.Option()] = 50,
    n_steps: Annotated[int, typer.Option()] = 50,
    clocks: Annotated[str, typer.Option(help="Comma-separated clock values to sweep.")] = "25,40,60",
    rate_head: Annotated[str, typer.Option()] = "linear",
    q_head: Annotated[str, typer.Option()] = "fresh",
    sampler: Annotated[str, typer.Option(help="⑤ sampler: euler | gillespie.")] = "euler",
    provenance: Annotated[
        bool, typer.Option(help="Score from the sampler's own edit decisions, not from an alignment.")
    ] = False,
    seed: Annotated[int, typer.Option()] = 0,
    metrics_path: Annotated[Path, typer.Option()] = Path("metrics/deterministic_benchmark.json"),
) -> None:
    """Run §4.1's benchmark across a clock sweep and write per-class scores."""
    import gzip
    import random

    from editjumps.core.edit_flows.stages import resolve_sampler
    from editjumps.pipeline.train.evoflows import EvoFlowsModel

    sample = resolve_sampler(sampler)
    model = EvoFlowsModel.load_trained(model_folder, rate_head=rate_head, q_head=q_head)
    model.eval()
    tokenizer = model.tokenizer if hasattr(model, "tokenizer") else None
    if tokenizer is None:
        tokenizer = load_tokenizer(model_folder / "encoder")

    opener = gzip.open if sequences.suffix == ".gz" else open
    with opener(sequences, "rt") as handle:  # type: ignore[operator]
        raw = [line.strip() for _, line in zip(range(n), handle, strict=False) if line.strip()]
    # One chain, not the joined VH.VL string: §4.1's rules are defined on "a natural protein sequence", and an.
    z0s = [seq.split(PAIR_SEP)[0] for seq in raw]
    logger.info(f"{len(z0s)} source sequences from {sequences} (heavy chain, split on {PAIR_SEP!r})")

    # The control: this scorer, given the true z1. Per-class numbers are read against this, never
    # against 1.0 — see the module docstring.
    ceiling = score_generations(z0s, [apply_deterministic_edits(z) for z in z0s])
    ceil_pc = ceiling["per_class"]
    logger.info("oracle ceiling (true z1 through this scorer) — per-class numbers cannot exceed it:")
    for cls in EDIT_CLASSES:
        st = ceil_pc[cls]
        logger.info(f"  {cls:13s} P={st['precision']:.3f} R={st['recall']:.3f} F1={st['f1']:.3f}")

    report: dict[str, Any] = {
        "n_sequences": len(z0s), "n_steps": n_steps, "sampler": sampler, "provenance": provenance,
        "oracle_ceiling": ceiling, "by_clock": {},
    }
    if provenance:
        # The same control on the provenance path, reported unconditionally: "the ceiling is
        # gone" is a measurement, not an argument.
        prov_ceiling = score_generations(
            z0s, [apply_deterministic_edits(z) for z in z0s], [oracle_provenance(z) for z in z0s],
        )
        report["oracle_ceiling_provenance"] = prov_ceiling
        logger.info("oracle ceiling on the PROVENANCE path (same sequences, same scorer):")
        for cls in EDIT_CLASSES:
            st = prov_ceiling["per_class"][cls]
            logger.info(f"  {cls:13s} P={st['precision']:.3f} R={st['recall']:.3f} F1={st['f1']:.3f}")

    for clock in [float(c) for c in clocks.split(",")]:
        rng = random.Random(seed)
        generated, traces = [], []
        for z0 in z0s:
            input_ids = encode_one(tokenizer, z0)
            out = sample(model, input_ids, n_steps=n_steps, rng=rng, clock=clock, provenance=provenance)
            out_ids, prov = out if provenance else (out, None)
            generated.append(decode_one(tokenizer, out_ids))
            if prov is not None:
                traces.append(ProvenanceTrace(
                    residues_from_tokens(tokens_of(tokenizer, input_ids)),
                    residues_from_tokens(tokens_of(tokenizer, out_ids)),
                    prov,
                ))
        scored = score_generations(z0s, generated, traces if provenance else None)
        report["by_clock"][str(clock)] = scored
        per_class = scored["per_class"]
        seq = scored["sequence"]
        logger.info(
            f"clock={clock:g}: exact={seq['exact_match_rate']:.3f} "
            f"lev={seq['mean_levenshtein_to_z1']:.2f} (no-op baseline {seq['mean_levenshtein_baseline']:.2f})"
        )
        for cls in EDIT_CLASSES:
            stats = per_class[cls]
            extra = ""
            if "identity_accuracy" in stats:
                extra = f" identity={stats['identity_accuracy']:.3f} (n={stats['identity_total']:.0f})"
            logger.info(
                f"  {cls:13s} P={stats['precision']:.3f} R={stats['recall']:.3f} "
                f"F1={stats['f1']:.3f} support={stats['support']:.0f}{extra}"
            )

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    write_metrics(metrics_path, report)
    logger.info(f"wrote {metrics_path}")

    # Tracking after the JSON, which is the durable record. `start_mlflow_run` installs.
    tags = {
        **design_space_tags(objective="edit_flows", pretrain_data=sequences.name),
        "section": "4.1",
        "sampler": sampler,
        # Which of the two scoring paths produced the per-class block.
        "scoring_path": "provenance" if provenance else "alignment",
    }
    with start_mlflow_run(
        EXPERIMENT_NAME,
        run_name=f"deterministic-benchmark-{model_folder.name}-{sampler}",
        tags=tags,
    ):
        import mlflow

        mlflow.log_params({
            "model_folder": str(model_folder),
            "sequences": str(sequences),
            "n": n,
            "n_steps": n_steps,
            "clocks": clocks,
            "rate_head": rate_head,
            "q_head": q_head,
            "sampler": sampler,
            "provenance": provenance,
            "seed": seed,
            "metrics_path": str(metrics_path),
        })
        # The whole report: the oracle ceilings and every clock's per-class block.
        for path, value in flatten_metrics(report, summarise_lists=True).items():
            mlflow.log_metric(path, value)
        mlflow.log_artifact(str(metrics_path))


if __name__ == "__main__":
    typer.run(main)
