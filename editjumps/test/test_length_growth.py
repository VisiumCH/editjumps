"""The scaffolded construction that lets a substitution-only method reach a longer target."""

import random

import pytest

from editjumps.test.fakes import _stub_proposer


def test_the_scaffolded_construction_reaches_the_target_on_real_insertion_evidence() -> None:
    """Where the opened columns go, that they are filled, and that the no-growth path is unchanged."""
    from editjumps.core.evotune.substitution import substitute_by_profile
    from editjumps.core.length_growth import (
        align_to_template,
        allocate_slots,
        column_origins,
        grow_by_infilling,
        query_row,
        slot_weights,
        widen_row,
    )
    from editjumps.pipeline.evaluate.generation_eval import project_to_template

    template = "QVQLVESGGGLVQPGGSLRLSCAASGFTFS"
    # Every member inserts "PPP" after template position 10 and nowhere else.
    members = [template[:10] + "PPP" + template[10:] for _ in range(6)] + [template]

    # 1: one aligner, two views of it.
    for member in members:
        alignment = align_to_template(member, template)
        assert alignment.projected == project_to_template(member, template)
        assert len(alignment.insertions) == len(template) + 1

    alignments = [align_to_template(member, template) for member in members]
    weights = slot_weights(alignments, len(template))
    assert abs(sum(weights) - 1.0) < 1e-12
    # 2: all the evidence is at one slot, so every opened column lands there.
    assert sum(1 for weight in weights if weight > 0) == 1
    slot = weights.index(max(weights))

    counts = allocate_slots(len(template), len(template) + 3, weights, random.Random(0))
    assert sum(counts) == 3 and counts[slot] == 3

    # The widened alignment is rectangular, and the opened columns carry the members' own residues.
    row0 = query_row(template, counts)
    widened = [widen_row(alignment, counts) for alignment in alignments]
    assert {len(row) for row in [row0, *widened]} == {len(template) + 3}
    origins = column_origins(len(template), counts)
    opened = [i for i, origin in enumerate(origins) if origin is None]
    assert [row0[i] for i in opened] == ["-", "-", "-"], "the query must read the opened columns as gaps"
    assert [widened[0][i] for i in opened] == ["P", "P", "P"], "a member's insertion must land there"
    assert widened[-1][opened[0]] == "-", "a member that does not insert there keeps a gap"

    # 3: filled to the target, and forcing leaves an opened column alone.
    profile = [1.0 / len(template)] * len(template)
    for forced in (False, True):
        grown, stats = grow_by_infilling(
            template, profile, 2, counts, _stub_proposer(), random.Random(7), forced=forced
        )
        assert len(grown) == len(template) + 3 == stats["target_length"]
        assert stats["n_inserted"] == 3 and stats["at_target"]
        assert stats["length_in"] == len(template) and stats["length_out"] == len(grown)
        assert [grown[i] for i in opened] == ["W", "W", "W"], (
            "an opened column has no original residue, so `forced` has nothing to block there"
        )

    # 4: no growth, no change - the same sequence and the same accounting as the published path.
    zero = [0] * (len(template) + 1)
    grown, stats = grow_by_infilling(template, profile, 3, zero, _stub_proposer(), random.Random(11))
    plain, plain_stats = substitute_by_profile(template, profile, 3, _stub_proposer(), random.Random(11))
    assert grown == plain
    assert {key: stats[key] for key in plain_stats} == plain_stats
    assert stats["n_inserted"] == 0

    # A shrink target has no construction here: opening columns cannot delete a residue.
    with pytest.raises(ValueError, match="below the template's own"):
        allocate_slots(len(template), len(template) - 1, weights, random.Random(0))
