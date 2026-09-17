"""What a run reports about its own edit: the clock it used, the distance, the op mix."""




def test_edit_ops_splits_the_distance_into_substitutions_and_indels() -> None:
    """The op split must agree with the edit distance it is a decomposition of."""
    from editjumps.core.edit_flows.alignment import levenshtein
    from editjumps.core.edit_flows.metrics import edit_ops

    assert edit_ops("ACDEF", "ACDEF") == {"substitutions": 0, "insertions": 0, "deletions": 0, "total": 0}
    assert edit_ops("ACDEF", "AGDEF")["substitutions"] == 1
    assert edit_ops("ACDEF", "ACDEFG")["insertions"] == 1
    assert edit_ops("ACDEF", "ACEF")["deletions"] == 1
    for template, generated in (("ACDEFGHIK", "AGDEFGHIKL"), ("QVQLVESGG", "QVQLESGGW"), ("MKT", "TKM")):
        ops = edit_ops(template, generated)
        assert ops["total"] == levenshtein(template, generated), (
            f"{template} -> {generated}: the op split must sum to the distance it decomposes"
        )
