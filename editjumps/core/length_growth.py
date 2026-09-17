"""Growing a sequence with a model that cannot insert: the width-``L`` scaffold, shared."""

import random
from collections.abc import Sequence
from typing import NamedTuple

from editjumps.core.edit_flows.alignment import needleman_wunsch
from editjumps.core.evotune.profile import GAP
from editjumps.core.evotune.substitution import Proposer, substitute_by_profile

#: Offset for the RNG stream that allocates growth slots, so it collides with none of the streams the baselines.
SLOT_SEED_OFFSET = 104729


class Alignment(NamedTuple):
    """One sequence aligned to a template, with the insertion columns kept."""

    projected: str
    insertions: tuple[str, ...]


def align_to_template(sequence: str, template: str) -> Alignment:
    """Align ``sequence`` to ``template``, returning both the projection and what it dropped."""
    source, target = needleman_wunsch([ord(c) for c in template], [ord(c) for c in sequence])
    projected: list[str] = []
    insertions: list[str] = [""] * (len(template) + 1)
    for template_token, seq_token in zip(source, target, strict=True):
        if template_token < 0:
            # An insertion relative to the template: not a column of the projection, but it is the
            # evidence that this slot is one a real homolog grows at.
            if seq_token >= 0:
                insertions[len(projected)] += chr(seq_token)
            continue
        projected.append(GAP if seq_token < 0 else chr(seq_token))
    return Alignment("".join(projected), tuple(insertions))


def slot_weights(alignments: Sequence[Alignment], length: int) -> list[float]:
    """Weight each insertion slot by how many family members actually insert there."""
    counts = [0.0] * (length + 1)
    for alignment in alignments:
        if len(alignment.insertions) != length + 1:
            raise ValueError(
                f"alignment carries {len(alignment.insertions)} slots for a template of {length}; "
                "every alignment must be against the same template"
            )
        for slot, inserted in enumerate(alignment.insertions):
            if inserted:
                counts[slot] += 1.0
    total = sum(counts)
    if total <= 0.0:
        return [1.0 / (length + 1)] * (length + 1)
    return [count / total for count in counts]


def allocate_slots(length: int, target_length: int, weights: Sequence[float], rng: random.Random) -> list[int]:
    """Decide how many columns to open at each slot to reach ``target_length``."""
    if target_length < length:
        raise ValueError(
            f"target_length={target_length} is below the template's own {length}; a mask-and-infill "
            "method can open columns but cannot delete residues, so a shrink target has no "
            "construction here"
        )
    if len(weights) != length + 1:
        raise ValueError(f"{len(weights)} slot weights for a template of {length}; expected {length + 1}")
    counts = [0] * (length + 1)
    extra = target_length - length
    if extra == 0:
        return counts
    for slot in rng.choices(range(length + 1), weights=list(weights), k=extra):
        counts[slot] += 1
    return counts


def column_origins(length: int, counts: Sequence[int]) -> list[int | None]:
    """Map each column of the widened problem back to a template position, or to nothing."""
    if len(counts) != length + 1:
        raise ValueError(f"{len(counts)} slot counts for a template of {length}; expected {length + 1}")
    origins: list[int | None] = []
    for position in range(length + 1):
        origins.extend([None] * counts[position])
        if position < length:
            origins.append(position)
    return origins


def scaffold_row(template: str, counts: Sequence[int]) -> list[str | None]:
    """Build the working row the proposer is handed: template residues, ``None`` at opened columns.."""
    origins = column_origins(len(template), counts)
    return [None if origin is None else template[origin] for origin in origins]


def query_row(template: str, counts: Sequence[int]) -> str:
    """Render the template as a row of the widened alignment, gaps at the opened columns."""
    return "".join(GAP if residue is None else residue for residue in scaffold_row(template, counts))


def widen_row(alignment: Alignment, counts: Sequence[int]) -> str:
    """Place one family member into the widened alignment, insertions in the opened columns."""
    length = len(alignment.projected)
    if len(counts) != length + 1:
        raise ValueError(f"{len(counts)} slot counts for a template of {length}; expected {length + 1}")
    row: list[str] = []
    for position in range(length + 1):
        inserted = alignment.insertions[position][: counts[position]]
        row.append(inserted + GAP * (counts[position] - len(inserted)))
        if position < length:
            row.append(alignment.projected[position])
    return "".join(row)


def lift_weights(weights: Sequence[float], counts: Sequence[int]) -> list[float]:
    """Lift a template-coordinate profile onto the widened problem, zero at the opened columns."""
    origins = column_origins(len(weights), counts)
    return [0.0 if origin is None else weights[origin] for origin in origins]


def grow_by_infilling(
    template: str,
    weights: Sequence[float],
    budget: int,
    counts: Sequence[int],
    propose: Proposer,
    rng: random.Random,
    *,
    forced: bool = False,
    top_up: bool = True,
    max_rounds: int = 8,
) -> tuple[str, dict]:
    """Grow ``template`` to the scaffolded width, then spend the mutation budget on it."""
    if len(weights) != len(template):
        raise ValueError(
            f"{len(weights)} weights for a template of {len(template)}; the profile must be in "
            "ungapped template coordinates (see evotune.profile.ungapped_weights)"
        )
    extra = sum(counts)
    if extra == 0:
        variant, stats = substitute_by_profile(
            template, weights, budget, propose, rng, forced=forced, top_up=top_up, max_rounds=max_rounds
        )
    else:
        residues = scaffold_row(template, counts)
        opened = [column for column, residue in enumerate(residues) if residue is None]
        # Random order, so the last column filled is not systematically the C-terminal one, and each
        # choice conditions on the ones already revealed.
        for column in rng.sample(opened, len(opened)):
            residues[column] = propose(residues, column, "")
        unfilled = [column for column, residue in enumerate(residues) if residue is None]
        if unfilled:
            raise ValueError(
                f"the proposer left column(s) {unfilled[:8]} unfilled; a scaffolded run must fill "
                "every opened column or its output is not the target length"
            )
        grown = "".join(residue for residue in residues if residue is not None)
        variant, stats = substitute_by_profile(
            grown, lift_weights(weights, counts), budget, propose, rng,
            forced=forced, top_up=top_up, max_rounds=max_rounds,
        )
    target = len(template) + extra
    return variant, {
        **stats,
        "length_in": len(template),
        "length_out": len(variant),
        "target_length": target,
        "n_inserted": extra,
        "at_target": len(variant) >= target,
    }
