"""Torch-free measurements about an edit run: the ⑤ clock setting, and the op mix of one edit.."""

from __future__ import annotations


def resolve_clock(clock: float | None, params: dict | None = None, block: str = "edit_flows") -> float | None:
    """Resolve the ⑤ clock-normalization setting for one evaluation run."""
    from_params = ((params or {}).get(block) or {}).get("clock")
    value = clock if clock is not None else from_params
    if value is None:
        return None
    return float(value) if float(value) > 0 else None


def clock_scale(clock: float | None, length: int) -> float:
    """Return the multiplier clock normalization applies to every edit rate: ``clock / length``."""
    if clock is None or length <= 0:
        return 1.0
    return clock / length


def edit_ops(template: str, generated: str) -> dict[str, int]:
    """Split the edits between ``template`` and ``generated`` into."""
    from editjumps.core.edit_flows.alignment import needleman_wunsch

    source, target = needleman_wunsch([ord(c) for c in template], [ord(c) for c in generated])
    ops = {"substitutions": 0, "insertions": 0, "deletions": 0}
    for left, right in zip(source, target, strict=True):
        if left < 0:
            ops["insertions"] += 1
        elif right < 0:
            ops["deletions"] += 1
        elif left != right:
            ops["substitutions"] += 1
    ops["total"] = sum(ops.values())
    return ops
