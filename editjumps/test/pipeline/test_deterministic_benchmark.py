"""The §4.1 deterministic benchmark: its oracle, its alignment ceiling and its provenance scoring."""

import pytest


def test_deterministic_benchmark_oracle_is_perfect_at_sequence_level() -> None:
    """The true z1 must score exact-match 1.0 — otherwise the harness, not the model, is wrong."""
    import random

    from editjumps.core.edit_flows.deterministic import apply_deterministic_edits
    from editjumps.pipeline.evaluate.deterministic_benchmark import score_generations

    rng = random.Random(0)
    z0s = ["".join(rng.choice("ACDEFGHIKLMNPQRSTVWY") for _ in range(80)) for _ in range(10)]
    scored = score_generations(z0s, [apply_deterministic_edits(z) for z in z0s])

    assert scored["sequence"]["exact_match_rate"] == 1.0
    assert scored["sequence"]["mean_levenshtein_to_z1"] == 0.0
    # the rules must actually do something, or the benchmark is vacuous
    assert scored["sequence"]["mean_levenshtein_baseline"] > 0.0


def test_deterministic_benchmark_per_class_ceiling_is_below_one() -> None:
    """§4.1's z1 is unique but its edit script is not, so alignment-based scoring has a ceiling."""
    from editjumps.core.edit_flows.deterministic import apply_deterministic_edits, ground_truth_labels
    from editjumps.pipeline.evaluate.deterministic_benchmark import predicted_edit_labels

    # the measured case from the module docstring: GHG -> SHC
    z0 = "MWILGHGCYKSDDFFCDVPTKTIHWQWKRSNDMYESWMHI"
    z1 = apply_deterministic_edits(z0)
    truth = ground_truth_labels(z0)
    predicted, _ = predicted_edit_labels(z0, z1)

    assert z1[4:7] == "SHC" and z0[4:7] == "GHG"
    assert truth != predicted, "if these agree, the ceiling finding no longer holds"
    assert truth[4] == {"deletion"} and predicted[4] == {"substitution"}


def test_deterministic_benchmark_insertion_index_convention() -> None:
    """An insertion is scored at the position it precedes, matching §4.1's Ins(p, S)."""
    from editjumps.pipeline.evaluate.deterministic_benchmark import predicted_edit_labels

    # inserting S before position 2 of a run with no other edits available
    z0 = "DDDD"
    labels, residues = predicted_edit_labels(z0, "DDSDD")
    fired = [i for i, s in enumerate(labels) if "insertion" in s]
    assert fired, "the insertion was not detected at all"
    # it must land on a real z0 position and carry the residue actually written
    assert all(0 <= i < len(z0) for i in fired)
    assert residues[fired[0]]["insertion"] == "S"


def test_deterministic_benchmark_no_op_recall_catches_an_idle_editor() -> None:
    """An editor that changes nothing must not look good: no_op recall is the control."""
    import random

    from editjumps.pipeline.evaluate.deterministic_benchmark import score_generations

    rng = random.Random(1)
    z0s = ["".join(rng.choice("ACDEFGHIKLMNPQRSTVWY") for _ in range(80)) for _ in range(10)]
    idle = score_generations(z0s, list(z0s))  # returns its input unchanged

    assert idle["sequence"]["exact_match_rate"] == 0.0
    assert idle["per_class"]["no_op"]["recall"] == 1.0      # it calls everything a no-op
    assert idle["per_class"]["substitution"]["recall"] == 0.0  # and finds no real edit
    assert idle["per_class"]["insertion"]["recall"] == 0.0


def test_deterministic_benchmark_ceiling_holds_up_on_real_antibody_sequences() -> None:
    """On single chains the alignment ceiling is high; only the separator bug made it look low."""
    from editjumps.core.edit_flows.deterministic import apply_deterministic_edits
    from editjumps.pipeline.evaluate.deterministic_benchmark import score_generations

    # two real VH sequences (heavy chain only, no separator)
    z0s = [
        "EVQLQQSGPDLVKPGASVKISCKASGYSFTDYFMNWVKQSPEKSLEWIGEINPSTGGTTYNQKFKGKATLTVDKSSSTAYMELRSLTSEDSAVYYCAR",
        "QVQLKESGPGLVAPSQSLSITCTVSGFSLINYAISWVRQPPGKGLEWLGVIWTGGGTNYNSALMSRLSISKDNSKSQVFLKMNSLQTDDTAMYYCAR",
    ]
    scored = score_generations(z0s, [apply_deterministic_edits(z) for z in z0s])
    assert scored["sequence"]["exact_match_rate"] == 1.0
    for cls in ("substitution", "deletion"):
        assert scored["per_class"][cls]["precision"] > 0.8, cls


#: Real antibody heavy chains, used where a *real* sequence matters — §4.1's alignment ceiling is a property of.
_REAL_HEAVY_CHAINS = [
    "EVQLQQSGPDLVKPGASVKISCKASGYSFTDYFMNWVKQSPEKSLEWIGEINPSTGGTTYNQKFKGKATLTVDKSSSTAYMELRSLTSEDSAVYYCAR",
    "QVQLKESGPGLVAPSQSLSITCTVSGFSLINYAISWVRQPPGKGLEWLGVIWTGGGTNYNSALMSRLSISKDNSKSQVFLKMNSLQTDDTAMYYCAR",
    "EVQLLESGGEVKKPGASVKVSCRASGYTFRNYGLTWVRQAPGQGLEWMGWISAYNGNTNYAQKFQGRVTLTTDTSTSTAYMELRSLRSDDTAVYFCAR"
    "DVPGHGAAFMDVWGTGTTVTVSS",
    "EVQLVESGGGLVQPGRSLRLSCAASGFTFDDYAMHWVRQAPGKGLEWVSAITWNSGHIDYADSVEGRFTISRDNAKNSLYLQMNSLRAEDTAVYYCAK"
    "VSYLSTASSLDYWGQGTLVTVSS",
    "QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYWMHWVRQAPGQGLEWIGYINPSTGYTEYNQKFKD",
]


def test_provenance_removes_the_alignment_ceiling_on_real_chains() -> None:
    """The point of the whole exercise: per-class scoring stops being ceiling-limited. §4.1's scorer, run."""
    from editjumps.core.edit_flows.deterministic import EDIT_CLASSES, apply_deterministic_edits
    from editjumps.pipeline.evaluate.deterministic_benchmark import oracle_provenance, score_generations

    z1s = [apply_deterministic_edits(z) for z in _REAL_HEAVY_CHAINS]
    traces = [oracle_provenance(z) for z in _REAL_HEAVY_CHAINS]
    # the trace must actually spell z1, or it is not a perfect editor's trace
    for trace, z1 in zip(traces, z1s, strict=True):
        assert "".join(r for r in trace.output_residues if r is not None) == z1

    aligned = score_generations(_REAL_HEAVY_CHAINS, z1s)["per_class"]
    from_prov = score_generations(_REAL_HEAVY_CHAINS, z1s, traces)

    for cls in EDIT_CLASSES:
        stats = from_prov["per_class"][cls]
        assert stats["precision"] == 1.0 and stats["recall"] == 1.0, f"{cls} is still ceiling-limited"
        if "identity_accuracy" in stats:
            assert stats["identity_accuracy"] == 1.0
    # The ceiling it replaces is real: the alignment path misses on all three classes.
    for cls in ("insertion", "substitution", "deletion"):
        assert aligned[cls]["precision"] < 1.0
    # Nothing dropped: only the inserted tokens are unscoreable, and those are reported.
    stats = from_prov["provenance_stats"]
    assert stats["insertions_unattributable"] == 0 and stats["substitutions_to_non_residue"] == 0
    assert stats["inserted_tokens"] == from_prov["ground_truth_counts"]["insertion"]


def test_provenance_scoring_reports_what_it_cannot_place() -> None:
    """A trailing insertion has no §4.1 position, and is counted rather than dropped."""
    from editjumps.pipeline.evaluate.deterministic_benchmark import (
        ProvenanceTrace,
        predicted_edit_labels_from_provenance,
    )

    z0 = "ACD"
    inputs: list[str | None] = [None, "A", "C", "D", None]  # BOS + residues + EOS
    # output: BOS, A, (inserted S), D, (inserted S), EOS  -- C deleted, one insertion trailing D
    outputs: list[str | None] = [None, "A", "S", "D", "S", None]
    prov: list[int | None] = [0, 1, None, 3, None, 4]
    labels, residues, stats = predicted_edit_labels_from_provenance(z0, ProvenanceTrace(inputs, outputs, prov))

    assert labels[0] == set()                       # A survived unchanged
    assert labels[1] == {"insertion", "deletion"}   # insertion anchored after A -> position 1; C gone
    assert labels[2] == set()                       # D survived unchanged
    assert residues[1]["insertion"] == "S"
    assert stats["inserted_tokens"] == 2
    assert stats["insertions_unattributable"] == 1  # the one past the final residue
    # a tokenizer mismatch must fail loudly rather than shift every position
    with pytest.raises(ValueError):
        predicted_edit_labels_from_provenance("ACDE", ProvenanceTrace(inputs, outputs, prov))
