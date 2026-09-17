"""The inference trace: which rows are shown, and what each row must carry."""

from pathlib import Path

import pytest


def test_read_template_takes_a_fasta_record_and_a_pairs_line(tmp_path: Path) -> None:
    """Both evaluation-dataset shapes resolve, and a pairs line yields the SOURCE's heavy half."""
    from editjumps.core.sequences import PAIR_SEP
    from editjumps.pipeline.evaluate.inference_trace import read_template

    fasta = tmp_path / "family.fasta"
    fasta.write_text(">h0\nACDE\nFGHI\n>h1\nKLMN\n>h2\nPQRS\n")
    assert read_template(str(fasta), 0) == "ACDEFGHI", "a wrapped record must be joined"
    assert read_template(str(fasta), 1) == "KLMN"
    assert read_template(str(fasta), 2) == "PQRS", "the last record needs no trailing '>' to close it"
    with pytest.raises(ValueError, match="no record 3"):
        read_template(str(fasta), 3)

    pairs = tmp_path / "pairs.tsv"
    pairs.write_text(f"AAAA{PAIR_SEP}BBBB\tCCCC{PAIR_SEP}DDDD\nEEEE\tFFFF\n")
    assert read_template(str(pairs), 0) == "AAAA", "the source's heavy half, not the joined pair"
    assert read_template(str(pairs), 1) == "EEEE"


def test_the_trace_rows_carry_the_length_distance_and_op_mix() -> None:
    """A row must show the four things a reader needs, and the rows must span the trajectory."""
    from editjumps.pipeline.evaluate.inference_trace import format_rows, pick_rows

    assert pick_rows(0) == []
    assert pick_rows(1) == [0], "a one-snapshot trajectory prints one row"
    picked = pick_rows(101)
    assert picked[0] == 0 and picked[-1] == 100, "the endpoints are always printed"
    assert picked == sorted(set(picked)), "de-duplicated and in order"

    template = "ACDEFGHIKLMNPQRSTVWY"
    rows = format_rows(template, [(0.0, template), (0.5, "ACDEFGHIKLMNPQRSTVW"), (1.0, "AGDEFGHIKLMNPQRSTVW")])
    assert rows[0].startswith("  template") and template in rows[0]
    assert "d=  0" in rows[1] and "sub=  0" in rows[1], "the first snapshot is the template itself"
    assert "del=  1" in rows[2], "a shortened snapshot reads as a deletion"
    assert "sub=  1" in rows[3] and "del=  1" in rows[3], "the op mix is cumulative against the template"


def test_provenance_ops_counts_what_the_editor_sampler_actually_did() -> None:
    """The exact op counts come off the sampler's per-token origins, not off a re-alignment."""
    from editjumps.pipeline.evaluate.inference_trace import provenance_ops

    # BOS, then five residues. Output: position 1 substituted, position 3 deleted, one insertion.
    input_ids = [0, 10, 11, 12, 13, 14]
    output_ids = [0, 99, 11, 13, 14, 77]
    origins: list[int | None] = [0, 1, 2, 4, 5, None]
    assert provenance_ops(input_ids, output_ids, origins) == {
        "substitutions": 1, "insertions": 1, "deletions": 1, "total": 3,
    }
    assert provenance_ops(input_ids, input_ids, list(range(len(input_ids)))) == {
        "substitutions": 0, "insertions": 0, "deletions": 0, "total": 0,
    }
    assert provenance_ops(input_ids, output_ids, []) == {}, "no provenance, no exact counts"
