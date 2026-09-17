"""Which edit each position of ``x_t`` needs to reach ``z_1`` (Edit Flows arXiv 2506.09018, eq. 23).."""

from typing import NamedTuple

from editjumps.core.edit_flows.path import EPS

#: The two readings of the paper's delete branch. See :func:`edit_targets`.
EDIT_SUPERVISION = ("eq23", "figure13")


class EditTarget(NamedTuple):
    """One edit at a position of the epsilon-stripped ``x_t``; insertions attach *after* ``index``."""

    index: int
    op: str
    token: int


def edit_targets(z_t: list[int], z_1: list[int], supervision: str = "eq23") -> list[EditTarget]:
    """Read off the edit each position of ``x_t`` needs to reach the target ``z_1``, no-ops omitted.."""
    if supervision not in EDIT_SUPERVISION:
        raise ValueError(f"supervision must be one of {sorted(EDIT_SUPERVISION)}, got {supervision!r}")
    if not z_t or z_t[0] == EPS:
        raise ValueError(
            "z_t must be non-empty and start with a real token; a leading gap yields an "
            "insertion at index -1, which eq 13's ins(x, i, a) cannot represent. Prepend BOS."
        )
    targets: list[EditTarget] = []
    x_index = -1
    for tok_t, tok_1 in zip(z_t, z_1, strict=True):
        if tok_t != EPS:
            x_index += 1
        if tok_t == EPS and tok_1 != EPS:
            targets.append(EditTarget(x_index, "insert", tok_1))
        elif tok_t != EPS and tok_1 == EPS:
            targets.append(EditTarget(x_index, "delete", EPS))
        elif tok_t == EPS and tok_1 == EPS and supervision == "figure13":
            # The whole difference between the two readings lives on this one column type.
            targets.append(EditTarget(x_index, "delete", EPS))
        elif tok_t != EPS and tok_1 != EPS and tok_t != tok_1:
            targets.append(EditTarget(x_index, "substitute", tok_1))
    return targets
