"""The alignment file the baseline conditions on: an a3m with the query as row 0."""

from collections.abc import Sequence
from pathlib import Path

#: Gap character, matching :mod:`editjumps.core.generation_metrics` and evodiff's own ``GAP``.
GAP = "-"


def write_a3m(path: Path, query: str, aligned_members: Sequence[str]) -> int:
    """Write ``query`` as row 0 and ``aligned_members`` after it; return the row count."""
    if not query:
        raise ValueError("the query row is empty; there is no sequence to edit")
    ragged = sorted({len(row) for row in aligned_members if len(row) != len(query)})
    if ragged:
        raise ValueError(
            f"alignment is ragged: query is {len(query)} but rows of lengths {ragged} were given; "
            "every row must already be projected onto the query's coordinates"
        )
    rows = [query, *aligned_members]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f">row{i}\n{row}\n" for i, row in enumerate(rows)))
    return len(rows)
