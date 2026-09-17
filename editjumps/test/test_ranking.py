"""`editjumps.ranking`: sort direction and the input index, with the scorer stubbed."""

from pathlib import Path

import pytest


def test_rank_orders_by_score_and_keeps_the_input_index(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`rank` sorts most-natural first, reports 1-based ranks, and remembers where each came from."""
    from editjumps import ranking

    calls = {}

    def fake_pll(sequences: list[str], model_name: str = "", seed: int = 0,
                 max_positions: int | None = None, batch_size: int = 16) -> dict:
        calls["sequences"] = list(sequences)
        calls["max_positions"] = max_positions
        # Deliberately not monotone in input order, so a no-op "sort" cannot pass.
        table = {"AAAA": -2.0, "CCCC": -0.5, "DDDD": -1.0}
        per = [table[s] for s in sequences]
        return {"mean": sum(per) / len(per), "sum_mean": 0.0, "n": len(per),
                "per_sequence": per, "per_sequence_sum": per}

    monkeypatch.setattr("editjumps.core.pseudo_likelihood.pseudo_log_likelihood", fake_pll)

    out = ranking.rank(["AAAA", "CCCC", "DDDD"], max_positions=7)
    assert [r.sequence for r in out] == ["CCCC", "DDDD", "AAAA"], "must be most-natural FIRST"
    assert [r.rank for r in out] == [1, 2, 3]
    assert [r.index for r in out] == [1, 2, 0], "index must point back into the input"
    assert out[0].score == -0.5
    assert calls["max_positions"] == 7

    # Whitespace and case are normalised before scoring, not after.
    ranking.rank([" cccc ", "aaaa"], max_positions=1)
    assert calls["sequences"] == ["CCCC", "AAAA"]

    # A blank candidate scores as nothing rather than badly and would sort into the middle.
    with pytest.raises(ValueError, match="empty candidate"):
        ranking.rank(["CCCC", "   "])

    assert ranking.rank([]) == []

    # A FASTA path is accepted in place of a list.
    fasta = tmp_path / "candidates.fasta"
    fasta.write_text(">one\nCCCC\n>two\nAA\nAA\n")
    assert ranking.read_fasta(fasta) == ["CCCC", "AAAA"]
    assert [r.sequence for r in ranking.rank(fasta, max_positions=1)] == ["CCCC", "AAAA"]
