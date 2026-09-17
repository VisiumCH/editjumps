"""One definition of the §4.2 train/inference/holdout split, shared by every evaluator."""

import inspect

import pytest


def test_family_split_is_disjoint_deterministic_and_shared() -> None:
    """One definition of §4.2's train/inference/holdout split, because MMD depends on it."""
    from editjumps.core.family_split import split_family, usable_members
    from editjumps.pipeline.evaluate import generation_eval

    members = [f"SEQ{i}" + "A" * 60 for i in range(50)]
    split = split_family(members, 5, 20, seed=0)
    assert len(split.templates) == 5 and len(split.reference) == 20 and len(split.pool) == 25
    everything = split.templates + split.reference + split.pool
    assert len(set(everything)) == len(members), "the three parts must partition the family exactly"
    assert not set(split.templates) & set(split.reference), "generating from the scoring holdout"
    assert not set(split.reference) & set(split.pool), "the pairing ceiling must not sit in the holdout"

    assert split_family(members, 5, 20, seed=0) == split_family(members, 5, 20, seed=0)
    assert split_family(members, 5, 20, seed=1) != split
    # It must not mutate its argument: the code this replaced shuffled the caller's list in place.
    assert members[0] == "SEQ0" + "A" * 60

    with pytest.raises(ValueError, match="need >"):
        split_family(members[:6], 5, 20, seed=0)

    assert usable_members(["A" * 51, "A" * 50, "A" * 10]) == ["A" * 51]
    assert "split_family(" in inspect.getsource(generation_eval.evaluate), (
        "generation_eval must share the split, not re-implement it"
    )

    # Disjoint by SEQUENCE, not list position: OAS returns the same sequence under several accessions, so.
    duplicated = [seq for seq in members[:20] for _ in range(3)]
    deduped = split_family(duplicated, 5, 8, seed=0)
    assert len(deduped.templates) == 5 and len(deduped.reference) == 8 and len(deduped.pool) == 7
    for left, right in ((deduped.templates, deduped.reference), (deduped.reference, deduped.pool),
                        (deduped.templates, deduped.pool)):
        assert not set(left) & set(right), "the same sequence must not appear in two parts"
    # The floor counts distinct members, so twenty copies of one sequence are not an n of twenty.
    with pytest.raises(ValueError, match="distinct usable members"):
        split_family([members[0]] * 50, 5, 20, seed=0)
