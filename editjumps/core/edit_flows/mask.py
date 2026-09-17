"""Per-position edit masks for restricting sampling regions (all, cdr, framework)."""

from editjumps.core.sequences import AA

REGIONS = ("all", "cdr", "framework")


def _locate(residues: list[str], token_index: list[int], sub: str) -> set[int]:
    """Token indices of the first occurrence of ``sub`` in ``residues`` (empty if absent/blank)."""
    if not sub:
        return set()
    hay = "".join(residues)
    start = hay.find(sub)
    if start < 0:
        return set()
    return set(token_index[start : start + len(sub)])


def build_edit_mask(tokens: list[str], cdr_h3: str, cdr_l3: str, region: str = "all") -> list[bool]:
    """Mark which token positions an edit run may touch, for the given ``region``. ``tokens`` is the."""
    if region not in REGIONS:
        raise ValueError(f"region {region!r} unknown; options: {REGIONS}")
    if region == "all":
        return [True] * len(tokens)

    # Split residues into heavy (before the '.' separator) and light (after), tracking token idx.
    sep_at = next((i for i, tok in enumerate(tokens) if tok == "."), None)
    heavy_res, heavy_idx, light_res, light_idx = [], [], [], []
    for i, tok in enumerate(tokens):
        if len(tok) == 1 and tok in AA:
            if sep_at is None or i < sep_at:
                heavy_res.append(tok)
                heavy_idx.append(i)
            else:
                light_res.append(tok)
                light_idx.append(i)

    cdr = _locate(heavy_res, heavy_idx, cdr_h3.upper()) | _locate(light_res, light_idx, cdr_l3.upper())
    if region == "cdr":
        return [i in cdr for i in range(len(tokens))]
    aa_positions = set(heavy_idx) | set(light_idx)  # region == "framework"
    return [i in aa_positions and i not in cdr for i in range(len(tokens))]
