"""Shared antibody-sequence primitives — the alphabet, validity, and the join key."""

# Annotations reference names imported only under TYPE_CHECKING below.
from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

#: The 20 standard amino acids.
AA = frozenset("ACDEFGHIKLMNPQRSTVWY")

#: Separator between VH and VL in the joined corpus form (``params.yaml: oas.pair_sep``).
PAIR_SEP = "."

#: Separator for the (VH, VL) de-duplication key.
KEY_SEP = "|"

#: Minimum length for a V-domain to be considered a real sequence rather than a stub.
MIN_DOMAIN_LEN = 90



def is_valid_domain(seq: object, min_len: int = MIN_DOMAIN_LEN) -> bool:
    """Whether a value is a plausible amino-acid V-domain sequence."""
    if not isinstance(seq, str):
        return False
    cleaned = seq.strip().upper()
    if len(cleaned) < min_len or cleaned.startswith("<"):
        return False
    return set(cleaned) <= AA


def normalise(seq: object) -> str:
    """Canonical form of one chain: stripped and upper-cased."""
    # NaN-safe without pandas: NaN is the only float that is not equal to itself.
    if seq is None or (isinstance(seq, float) and seq != seq):
        return ""
    return str(seq).strip().upper()


def sequence_key(heavy: object, light: object) -> str:
    """Canonical ``VH|VL`` identity key for one antibody."""
    return f"{normalise(heavy)}{KEY_SEP}{normalise(light)}"


def sequence_keys(df: pd.DataFrame, heavy: str = "Heavy_Seq", light: str = "Light_Seq") -> pd.Series:
    """Vectorised `sequence_key` over a frame's VH/VL columns."""
    import pandas as pd  # noqa: F401  (lazy: keeps this module dependency-free to import)

    missing = [c for c in (heavy, light) if c not in df.columns]
    if missing:
        raise KeyError(f"cannot build a sequence key: missing column(s) {missing}")
    left = df[heavy].astype(str).str.strip().str.upper()
    right = df[light].astype(str).str.strip().str.upper()
    return left + KEY_SEP + right


def join_chains(heavy: object, light: object, sep: str = PAIR_SEP) -> str:
    """Join VH and VL into the corpus's single-line form."""
    return f"{normalise(heavy)}{sep}{normalise(light)}"


def split_chains(joined: str, sep: str = PAIR_SEP) -> Iterable[str]:
    """Split a joined ``VH<sep>VL`` line into its non-empty chains."""
    return [part for part in str(joined).split(sep) if part]
