"""The per-position supervision, and the one column type the two readings disagree on."""

import pytest


def test_edit_targets_reads_off_the_right_edits() -> None:
    """Edit_targets emits insert/delete/substitute at the correct x_t indices, skipping no-ops."""
    from editjumps.core.edit_flows.path import EPS
    from editjumps.core.edit_flows.targets import EditTarget, edit_targets

    bos, a, b, c, d, x = 0, 1, 2, 3, 4, 9
    z_t = [bos, a, EPS, c, d, EPS]
    z_1 = [bos, a, b, x, EPS, EPS]
    assert edit_targets(z_t, z_1) == [
        EditTarget(1, "insert", b),
        EditTarget(2, "substitute", x),
        EditTarget(3, "delete", EPS),
    ]

def test_edit_targets_rejects_a_leading_gap() -> None:
    """A leading gap must raise: eq 13's ins(x, i, a) cannot represent an attach index of -1."""
    from editjumps.core.edit_flows.path import EPS
    from editjumps.core.edit_flows.targets import edit_targets

    with pytest.raises(ValueError, match="index -1"):
        edit_targets([EPS, 1, 2], [3, 1, 2])
    with pytest.raises(ValueError, match="index -1"):
        edit_targets([], [])
    # the well-formed case still works
    assert edit_targets([1, 2, 3], [1, 9, 3]) == [(1, "substitute", 9)]

def test_figure13_supervision_differs_from_eq23_only_on_gap_gap_columns() -> None:
    """The two readings of the delete branch differ on exactly one column type, and nowhere else."""
    from editjumps.core.edit_flows.path import EPS
    from editjumps.core.edit_flows.targets import EditTarget, edit_targets

    bos, a, b, c, d, x = 0, 1, 2, 3, 4, 9
    z_t = [bos, a, EPS, c, d, EPS]
    z_1 = [bos, a, b, x, EPS, EPS]

    eq23 = edit_targets(z_t, z_1, supervision="eq23")
    fig13 = edit_targets(z_t, z_1, supervision="figure13")
    assert eq23 == edit_targets(z_t, z_1), "eq23 must be the default; every recorded run used it"

    # Figure 13 emits everything eq 23 does, plus one delete on the (EPS, EPS) column.
    from collections import Counter

    assert len(fig13) == len(eq23) + 1
    surplus = Counter(fig13) - Counter(eq23)
    assert surplus == Counter({EditTarget(3, "delete", EPS): 1}), surplus
    assert all(t.op == "delete" for t in surplus), "the readings may only differ on deletions"
    assert Counter(fig13)[EditTarget(3, "delete", EPS)] == 2, "Figure 13 deletes the same index twice"

    # With no (EPS, EPS) column anywhere, the two readings must be identical.
    clean_t, clean_1 = [bos, a, EPS, c], [bos, a, b, x]
    assert edit_targets(clean_t, clean_1, "eq23") == edit_targets(clean_t, clean_1, "figure13")

    with pytest.raises(ValueError, match="supervision must be one of"):
        edit_targets(z_t, z_1, supervision="whatever")
