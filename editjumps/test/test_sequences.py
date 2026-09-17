"""The one definition of an amino-acid sequence: alphabet, validity, chain join and key."""




def test_sequence_primitives_are_the_single_definition() -> None:
    """One alphabet, one validity rule, one join key — the thing five copies used to disagree on."""
    import pandas as pd
    import pytest as _pytest

    from editjumps.core.sequences import (
        AA,
        KEY_SEP,
        PAIR_SEP,
        is_valid_domain,
        join_chains,
        normalise,
        sequence_key,
        sequence_keys,
        split_chains,
    )

    assert len(AA) == 20 and "B" not in AA and "X" not in AA

    # Normalisation strips AND upper-cases; a copy that only stripped made these keys differ.
    assert sequence_key("  qvql ", "diqm") == sequence_key("QVQL", "DIQM")
    assert sequence_key("QVQL", "DIQM") == f"QVQL{KEY_SEP}DIQM"
    assert normalise(None) == "" and normalise(float("nan")) == ""

    vh, vl = "Q" * 120, "D" * 110
    assert is_valid_domain(vh)
    assert not is_valid_domain("<pending>" + "Q" * 120)  # placeholder
    assert not is_valid_domain("Q" * 40)                 # too short
    assert not is_valid_domain("Q" * 100 + "BXZ")        # non-standard residues
    assert not is_valid_domain(None) and not is_valid_domain(3.5)

    # Vectorised form must agree with the scalar one, row for row.
    df = pd.DataFrame({"Heavy_Seq": [" qvql ", vh], "Light_Seq": ["diqm", vl]})
    assert list(sequence_keys(df)) == [sequence_key(" qvql ", "diqm"), sequence_key(vh, vl)]
    # A missing column must raise, not silently produce an all-empty key that joins everything.
    with _pytest.raises(KeyError):
        sequence_keys(df.drop(columns=["Light_Seq"]))

    # The corpus join form is a different separator from the key form, on purpose.
    assert PAIR_SEP != KEY_SEP
    assert join_chains("qvql", "diqm") == f"QVQL{PAIR_SEP}DIQM"
    assert list(split_chains(f"QVQL{PAIR_SEP}DIQM")) == ["QVQL", "DIQM"]
    assert list(split_chains("QVQL")) == ["QVQL"]  # separator edited away by the sampler


