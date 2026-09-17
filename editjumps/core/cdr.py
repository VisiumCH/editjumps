"""IMGT numbering of a variable domain — which residues are CDR, and what position each one is."""

from collections.abc import Sequence
from functools import lru_cache

#: IMGT CDR position ranges, inclusive.
IMGT_CDR_RANGES: tuple[tuple[int, int], ...] = ((27, 38), (56, 65), (105, 117))

#: Accepted ``chain`` arguments, mapped to the domain types ANARCI is allowed to assign.
CHAIN_TYPES: dict[str, set[str]] = {"heavy": {"H"}, "light": {"K", "L"}}


@lru_cache(maxsize=4096)
def _numbered(seq: str, chain: str) -> tuple[int, ...] | None:
    """Return a chain's cached ANARCI numbering — each miss shells out to ``hmmscan``, at ~14 ms."""
    import shutil

    from anarci import run_anarci

    if chain not in CHAIN_TYPES:
        raise ValueError(f"chain={chain!r} unknown; options: {sorted(CHAIN_TYPES)}")
    if shutil.which("hmmscan") is None:
        raise RuntimeError("hmmscan not found - ANARCI needs HMMER "
                           "(brew install hmmer / apt-get install hmmer)")

    domains = run_anarci([("q", seq)], scheme="imgt", allow=CHAIN_TYPES[chain])[1][0]
    if not domains:
        return None
    # ANARCI emits gaps as '-', which occupy an IMGT position but no sequence position, so only
    # real residues are recorded.
    out = tuple(position for (position, _insertion), residue in domains[0][0] if residue != "-")
    # A partial numbering would silently misalign every index that follows it, so refuse it.
    return out if len(out) == len(seq) else None


def cdr_positions(positions: "Sequence[int]") -> set[int]:
    """Return which indices of a numbered chain fall inside an IMGT CDR."""
    return {i for i, position in enumerate(positions)
            if any(low <= position <= high for low, high in IMGT_CDR_RANGES)}


def imgt_positions(seq: str, chain: str) -> list[int] | None:
    """Return the IMGT position of every residue in a variable domain, in sequence order."""
    numbered = _numbered(seq, chain)
    return list(numbered) if numbered is not None else None


def cdr_indices(seq: str, chain: str) -> set[int]:
    """Return the 0-based sequence positions that fall in one of the six IMGT CDRs."""
    positions = _numbered(seq, chain)
    return cdr_positions(positions) if positions is not None else set()
