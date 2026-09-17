"""The framework/CDR lock picks out the right token positions."""

import pytest


def test_build_edit_mask_locks_the_right_region() -> None:
    """Build_edit_mask frees only the CDRs (cdr), only the framework (framework), or all."""
    from editjumps.core.edit_flows.mask import build_edit_mask

    # VH=EVQ, VL=DIK; CDR-H3=VQ (tokens 2,3), CDR-L3=IK (tokens 6,7); '.' is the VH/VL sep
    tokens = ["<cls>", "E", "V", "Q", ".", "D", "I", "K", "<eos>"]
    assert build_edit_mask(tokens, "VQ", "IK", "cdr") == [
        False, False, True, True, False, False, True, True, False
    ]
    # framework = every residue EXCEPT the CDRs (and never the specials/'.')
    assert build_edit_mask(tokens, "VQ", "IK", "framework") == [
        False, True, False, False, False, True, False, False, False
    ]
    assert build_edit_mask(tokens, "VQ", "IK", "all") == [True] * len(tokens)
    with pytest.raises(ValueError):
        build_edit_mask(tokens, "VQ", "IK", "loops")
