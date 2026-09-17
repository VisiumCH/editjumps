"""The §4.2/§4.3 evaluator: each method scored in its own frame, on the model's own columns."""

import random
from pathlib import Path

import pytest

from editjumps.test.fakes import _stub_the_gpu_half_of_generation_eval


def test_baselines_carry_the_ceiling_the_agreements_must_be_read_against() -> None:
    """`score_set` must score the model-free baselines on the three metrics that need a ceiling."""
    import inspect

    from editjumps.pipeline.evaluate import generation_eval

    body = inspect.getsource(generation_eval.evaluate)
    score_set = body[body.index("def score_set"):body.index("baselines = {")]
    for metric in ("covariance_agreement", "mip_agreement", "js_divergence_positional"):
        assert metric in score_set, (
            f"score_set must report {metric} for the baselines; without it the §4.3 random-pairing "
            "ceiling is not recorded and the normalised table cannot be checked against the run"
        )
    # Against the FULL reference, not the 20 partners: a 20-sequence reference biases both
    # agreements toward zero, which is the bug that made a working editor read as broken.
    assert "strength_full" in score_set and "mip_full" in score_set
    assert "aligned_natural_full" in score_set
    # And the reference size travels with it, because two ceilings at different n are not comparable.
    assert "agreement_reference_n" in score_set
    # THE SAME ALIGNMENT as the score, which is the defect this whole key exists to make impossible.
    assert "project_to_template(s, reference_template)" in score_set, (
        "the baseline must be projected onto reference_template, the SAME template the model's own "
        "score uses; a ceiling in another alignment is not a ceiling for that score"
    )


def _synthetic_family(n_members: int, seed: int) -> list[str]:
    """Build a family with real coupling structure, so the agreement scores measure something."""
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    parent = alphabet * 3
    rng = random.Random(seed)
    members = []
    for index in range(n_members):
        seq = list(parent)
        clade = alphabet[index % 4]
        seq[5] = seq[25] = clade          # the coupling: both positions move together
        for _ in range(3):                # plus private drift, so members are distinct
            seq[rng.randrange(len(seq))] = rng.choice(alphabet)
        for _ in range(index % 5):        # and length variation, deterministic across seeds
            seq.pop(rng.randrange(30, len(seq)))
        members.append("".join(seq))
    return list(dict.fromkeys(members))


def test_baseline_rows_carry_the_models_columns_in_the_models_own_frame(
    monkeypatch: "pytest.MonkeyPatch", tmp_path: Path
) -> None:
    """The model-free baselines must report the model's metrics, in the model's alignment."""
    import numpy as np

    from editjumps.core.family_split import split_family, usable_members
    from editjumps.core.generation_metrics import (
        frequencies,
        matrix_agreement,
        one_hot,
        positional_interaction_strength,
    )
    from editjumps.pipeline.evaluate import generation_eval

    members = _synthetic_family(90, seed=0)
    fasta = tmp_path / "family.fasta"
    fasta.write_text("".join(f">m{i}\n{seq}\n" for i, seq in enumerate(members)))

    _stub_the_gpu_half_of_generation_eval(monkeypatch)
    report = generation_eval.evaluate(
        tmp_path / "model", tmp_path / "unused.tsv.gz", n_templates=4, n_variants=5, n_steps=5,
        seed=0, k=3, family_fasta=fasta, holdout_size=20,
    )

    # Claim 1: every column the model's row carries, on both baseline rows.
    defined, ours = report["defined_by_the_paper"], report["our_interpretation"]
    from_defined = ("levenshtein_to_template", "pairwise_levenshtein_pooled",
                    "covariance_agreement", "mip_agreement", "kl_generated_vs_natural",
                    "spectrum_mmd", "agreement_reference_n")
    from_ours = ("entropy_delta", "js_divergence_positional")
    for name, row in report["baselines"].items():
        for key in from_defined:
            assert key in defined, f"the model's own row lost {key}"
            assert key in row, (
                f"baseline {name} does not report {key}, so that table cell is absent rather than "
                "pending: no job fills it, only a code change"
            )
        for key in from_ours:
            assert key in ours and key in row, f"baseline {name} does not report {key}"
        assert row["n_scored"] == report["n_generated"], (
            f"baseline {name} is scored at a different sample size from the model; both agreements "
            "move with sample size by more than any model difference this evaluation measures"
        )

    # Claim 2: the frame, read off the numbers the run recorded, not off the source.
    split = split_family(usable_members(members), 4, 20, 0)
    reference_template = split.templates[0]
    # Asserted FIRST because it is what gives the width assertions their power: the candidate templates differ.
    assert len({len(t) for t in split.templates}) > 1, (
        "the fixture must offer templates of differing widths, or alignment_length cannot "
        "distinguish the right template from a wrong one"
    )
    model_length = int(report["alignment"].split("L=")[1].rstrip(")"))
    assert model_length == len(reference_template), (
        "the model's own alignment is not split.templates[0]; every baseline is projected onto that "
        "template, so the per-position columns would again compare different coordinate systems"
    )
    for name, row in report["baselines"].items():
        assert row["alignment_length"] == model_length, (
            f"baseline {name} was scored at L={row['alignment_length']} against the model's "
            f"L={model_length}. This is the Ty1 bug (L=121 vs L=115): the covariance, MIP and "
            "positional-JS cells are then individually correct and jointly meaningless"
        )

    # And why the number above is the only thing that could catch it: score the SAME sequences in a.
    def agreement(template: str) -> float:
        """Covariance agreement of the family pool against the holdout, in one frame."""
        left = [generation_eval.project_to_template(s, template) for s in split.pool]
        right = [generation_eval.project_to_template(s, template) for s in split.reference]
        _, _, c_left = frequencies(one_hot(left))
        _, _, c_right = frequencies(one_hot(right))
        return matrix_agreement(positional_interaction_strength(c_left),
                                positional_interaction_strength(c_right))

    wrong_frame = reference_template[:-6]
    assert len(wrong_frame) != len(reference_template)
    right_value, wrong_value = agreement(reference_template), agreement(wrong_frame)
    assert right_value != wrong_value, "the wrong frame must actually change the number"
    for value in (right_value, wrong_value):
        assert 0.0 < value < 1.0, (
            "both frames give an ordinary-looking agreement, which is the point: the value cannot "
            "flag the mismatch, only the recorded alignment_length can"
        )

    floor, ceiling = report["baselines"]["random_mutations"], report["baselines"]["random_homolog_pairing"]
    # The floor/ceiling ORDERING on the two agreements is deliberately not asserted here, and the reason is a.
    for row in (floor, ceiling):
        for key in ("covariance_agreement", "mip_agreement", "js_divergence_positional"):
            assert np.isfinite(row[key]), f"{key} must be a real number, not NaN"
        # Pooled diversity needs no alignment, so it is the one added column with no frame risk --
        # it must simply be present and non-degenerate.
        assert row["pairwise_levenshtein_pooled"] > 0.0
