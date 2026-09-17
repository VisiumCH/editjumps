"""Row 0 is the query, and a ragged alignment is refused rather than truncated."""

from pathlib import Path

import pytest


def test_evodiff_a3m_writes_the_query_first_and_refuses_a_ragged_alignment(tmp_path: Path) -> None:
    """Row 0 of the a3m IS the sequence being edited, and a ragged row is a silent frame shift.."""
    from editjumps.core.evodiff_msa.alignment import write_a3m
    from editjumps.core.evodiff_msa.proposer import MASK_CHAR, row_from_working
    from editjumps.core.evodiff_msa.sampling import uniform_weights

    query = "ACDEFGHIKL"
    members = ["ACDEFGHIKM", "AC-EFGHIKL"]
    path = tmp_path / "family.a3m"
    assert write_a3m(path, query, members) == 3, "the row count is the ceiling on --n-sequences"
    rows = [line for line in path.read_text().splitlines() if not line.startswith(">")]
    assert rows[0] == query, "row 0 must be the query, not a family member"
    assert rows[1:] == members

    with pytest.raises(ValueError, match="ragged"):
        write_a3m(tmp_path / "bad.a3m", query, ["ACDEFGHIK"])
    with pytest.raises(ValueError, match="query row is empty"):
        write_a3m(tmp_path / "bad.a3m", "", members)

    # EvoDiff-MSA has no positional model and §4.2 does not say where a matched budget goes, so
    # uniform is OUR choice — the same one the paper's random-mutation baseline makes.
    weights = uniform_weights(query)
    assert weights == [0.1] * 10 and sum(weights) == pytest.approx(1.0)
    assert uniform_weights("") == []

    # `None` in substitute_by_profile's working list is a still-masked column on the wire.
    assert row_from_working(["A", None, "C"], 3) == f"A{MASK_CHAR}C"
    with pytest.raises(ValueError, match="alignment is 4 columns"):
        row_from_working(["A", None, "C"], 4)
