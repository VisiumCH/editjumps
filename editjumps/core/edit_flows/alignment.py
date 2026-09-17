"""The aligners: two real sequences -> one aligned pair ``(z_0, z_1)`` (EvoFlows arXiv 2603.11703."""

from typing import NamedTuple

from editjumps.core.edit_flows.path import EPS

#: Alignment costs.
GAP_COST = 1.0
MISMATCH_COST = 1.0
MATCH_COST = 0.0


def needleman_wunsch(source: list[int], target: list[int]) -> tuple[list[int], list[int]]:
    """Global-align two sequences at unit gap/mismatch cost; the default ② path. **DEVIATION 2 — the."""
    n, m = len(source), len(target)
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i * GAP_COST
    for j in range(1, m + 1):
        dp[0][j] = j * GAP_COST
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diag = dp[i - 1][j - 1] + (MATCH_COST if source[i - 1] == target[j - 1] else MISMATCH_COST)
            delete = dp[i - 1][j] + GAP_COST
            insert = dp[i][j - 1] + GAP_COST
            dp[i][j] = min(diag, delete, insert)

    z_0: list[int] = []
    z_1: list[int] = []
    i, j = n, m
    while i > 0 or j > 0:
        diag_ok = (
            i > 0
            and j > 0
            and dp[i][j] == dp[i - 1][j - 1] + (MATCH_COST if source[i - 1] == target[j - 1] else MISMATCH_COST)
        )
        if diag_ok:
            z_0.append(source[i - 1])
            z_1.append(target[j - 1])
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + GAP_COST:
            z_0.append(source[i - 1])
            z_1.append(EPS)  # deletion
            i -= 1
        else:
            z_0.append(EPS)  # insertion
            z_1.append(target[j - 1])
            j -= 1
    z_0.reverse()
    z_1.reverse()
    return z_0, z_1


# --- Affine gaps + BLOSUM62: the standard reading of what §3.2 cites -----------------------------
# Why both scorings are kept and neither is the paper's: see :func:`needleman_wunsch`.

#: BLOSUM62 log-odds substitution scores, one row per amino acid (Henikoff & Henikoff 1992, as distributed by.
BLOSUM62_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
_BLOSUM62_ROWS = {
    "A": " 4  0 -2 -1 -2  0 -2 -1 -1 -1 -1 -2 -1 -1 -1  1  0  0 -3 -2",
    "C": " 0  9 -3 -4 -2 -3 -3 -1 -3 -1 -1 -3 -3 -3 -3 -1 -1 -1 -2 -2",
    "D": "-2 -3  6  2 -3 -1 -1 -3 -1 -4 -3  1 -1  0 -2  0 -1 -3 -4 -3",
    "E": "-1 -4  2  5 -3 -2  0 -3  1 -3 -2  0 -1  2  0  0 -1 -2 -3 -2",
    "F": "-2 -2 -3 -3  6 -3 -1  0 -3  0  0 -3 -4 -3 -3 -2 -2 -1  1  3",
    "G": " 0 -3 -1 -2 -3  6 -2 -4 -2 -4 -3  0 -2 -2 -2  0 -2 -3 -2 -3",
    "H": "-2 -3 -1  0 -1 -2  8 -3 -1 -3 -2  1 -2  0  0 -1 -2 -3 -2  2",
    "I": "-1 -1 -3 -3  0 -4 -3  4 -3  2  1 -3 -3 -3 -3 -2 -1  3 -3 -1",
    "K": "-1 -3 -1  1 -3 -2 -1 -3  5 -2 -1  0 -1  1  2  0 -1 -2 -3 -2",
    "L": "-1 -1 -4 -3  0 -4 -3  2 -2  4  2 -3 -3 -2 -2 -2 -1  1 -2 -1",
    "M": "-1 -1 -3 -2  0 -3 -2  1 -1  2  5 -2 -2  0 -1 -1 -1  1 -1 -1",
    "N": "-2 -3  1  0 -3  0  1 -3  0 -3 -2  6 -2  0  0  1  0 -3 -4 -2",
    "P": "-1 -3 -1 -1 -4 -2 -2 -3 -1 -3 -2 -2  7 -1 -2 -1 -1 -2 -4 -3",
    "Q": "-1 -3  0  2 -3 -2  0 -3  1 -2  0  0 -1  5  1  0 -1 -2 -2 -1",
    "R": "-1 -3 -2  0 -3 -2  0 -3  2 -2 -1  0 -2  1  5 -1 -1 -3 -3 -2",
    "S": " 1 -1  0  0 -2  0 -1 -2  0 -2 -1  1 -1  0 -1  4  1 -2 -3 -2",
    "T": " 0 -1 -1 -1 -2 -2 -2 -1 -1 -1 -1  0 -1 -1 -1  1  5  0 -2 -2",
    "V": " 0 -1 -3 -2 -1 -3 -3  3 -2  1  1 -3 -2 -2 -3 -2  0  4 -3 -1",
    "W": "-3 -2 -4 -3  1 -2 -2 -3 -3 -2 -1 -4 -4 -2 -3 -3 -2 -3 11  2",
    "Y": "-2 -2 -3 -2  3 -3  2 -1 -2 -1 -1 -2 -3 -1 -2 -2 -2 -1  2  7",
}
BLOSUM62: dict[tuple[str, str], int] = {
    (a, b): int(score)
    for a, row in _BLOSUM62_ROWS.items()
    for b, score in zip(BLOSUM62_ALPHABET, row.split(), strict=True)
}

#: UNSPECIFIED 2.
BLOSUM62_GAP_OPEN = 11.0
BLOSUM62_GAP_EXTEND = 1.0

#: UNSPECIFIED 3.
NON_RESIDUE_MATCH_SCORE = 11.0
NON_RESIDUE_MISMATCH_SCORE = -4.0

_NEG_INF = float("-inf")
# Traceback states, and the pointer values stored per cell. `_SUB` = both tokens consumed (a.
_SUB, _DEL, _INS = 0, 1, 2


class AffineScoring(NamedTuple):
    """A substitution table plus affine gap penalties, in the score (maximise) convention."""

    name: str
    substitution: list[list[float]] | None
    gap_open: float
    gap_extend: float


#: The scoring `needleman_wunsch` uses, in `AffineScoring` form: unit mismatch, unit gap, ``gap_open = 0`` so.
UNIT_SCORING = AffineScoring(name="unit", substitution=None, gap_open=0.0, gap_extend=1.0)


def blosum62_scoring(
    tokens: dict[str, int],
    vocab_size: int,
    gap_open: float = BLOSUM62_GAP_OPEN,
    gap_extend: float = BLOSUM62_GAP_EXTEND,
) -> AffineScoring:
    """Build token-id-indexed BLOSUM62 scoring for `needleman_wunsch_affine`."""
    matrix = [[NON_RESIDUE_MISMATCH_SCORE] * vocab_size for _ in range(vocab_size)]
    for token_id in range(vocab_size):
        matrix[token_id][token_id] = NON_RESIDUE_MATCH_SCORE
    residues = {t: i for t, i in tokens.items() if t in BLOSUM62_ALPHABET and 0 <= i < vocab_size}
    for token_a, id_a in residues.items():
        row = matrix[id_a]
        for token_b, id_b in residues.items():
            row[id_b] = float(BLOSUM62[token_a, token_b])
    return AffineScoring(name="blosum62", substitution=matrix, gap_open=gap_open, gap_extend=gap_extend)


def _align_affine(
    source: list[int], target: list[int], scoring: AffineScoring
) -> tuple[list[int], list[int]]:
    """Gotoh three-state affine-gap global alignment (the DP; orientation handled by the caller)."""
    n, m = len(source), len(target)
    sub = scoring.substitution
    open_gap, extend_gap = -(scoring.gap_open + scoring.gap_extend), -scoring.gap_extend

    best_sub = [[_NEG_INF] * (m + 1) for _ in range(n + 1)]
    best_del = [[_NEG_INF] * (m + 1) for _ in range(n + 1)]
    best_ins = [[_NEG_INF] * (m + 1) for _ in range(n + 1)]
    from_sub = [[_SUB] * (m + 1) for _ in range(n + 1)]
    from_del = [[_SUB] * (m + 1) for _ in range(n + 1)]
    from_ins = [[_SUB] * (m + 1) for _ in range(n + 1)]

    best_sub[0][0] = 0.0
    for i in range(1, n + 1):
        best_del[i][0] = open_gap + (i - 1) * extend_gap
        from_del[i][0] = _DEL if i > 1 else _SUB
    for j in range(1, m + 1):
        best_ins[0][j] = open_gap + (j - 1) * extend_gap
        from_ins[0][j] = _INS if j > 1 else _SUB

    for i in range(1, n + 1):
        token_i = source[i - 1]
        sub_row = sub[token_i] if sub is not None else None
        prev_sub, prev_del, prev_ins = best_sub[i - 1], best_del[i - 1], best_ins[i - 1]
        cur_sub, cur_del, cur_ins = best_sub[i], best_del[i], best_ins[i]
        ptr_sub, ptr_del, ptr_ins = from_sub[i], from_del[i], from_ins[i]
        for j in range(1, m + 1):
            token_j = target[j - 1]
            score = sub_row[token_j] if sub_row is not None else (0.0 if token_i == token_j else -1.0)
            # substitution column: any state may precede it
            best, state = prev_sub[j - 1], _SUB
            if prev_del[j - 1] > best:
                best, state = prev_del[j - 1], _DEL
            if prev_ins[j - 1] > best:
                best, state = prev_ins[j - 1], _INS
            cur_sub[j], ptr_sub[j] = best + score, state
            # deletion column: open from a substitution/insertion run, or extend a deletion run
            best, state = prev_sub[j] + open_gap, _SUB
            candidate = prev_del[j] + extend_gap
            if candidate > best:
                best, state = candidate, _DEL
            candidate = prev_ins[j] + open_gap
            if candidate > best:
                best, state = candidate, _INS
            cur_del[j], ptr_del[j] = best, state
            # insertion column: the mirror of the above along j
            best, state = cur_sub[j - 1] + open_gap, _SUB
            candidate = cur_ins[j - 1] + extend_gap
            if candidate > best:
                best, state = candidate, _INS
            candidate = cur_del[j - 1] + open_gap
            if candidate > best:
                best, state = candidate, _DEL
            cur_ins[j], ptr_ins[j] = best, state

    z_0: list[int] = []
    z_1: list[int] = []
    i, j = n, m
    state = _SUB
    if best_del[i][j] > best_sub[i][j]:
        state = _DEL
    if best_ins[i][j] > max(best_sub[i][j], best_del[i][j]):
        state = _INS
    while i > 0 or j > 0:
        if state == _SUB:
            z_0.append(source[i - 1])
            z_1.append(target[j - 1])
            state = from_sub[i][j]
            i -= 1
            j -= 1
        elif state == _DEL:
            z_0.append(source[i - 1])
            z_1.append(EPS)  # deletion
            state = from_del[i][j]
            i -= 1
        else:
            z_0.append(EPS)  # insertion
            z_1.append(target[j - 1])
            state = from_ins[i][j]
            j -= 1
    z_0.reverse()
    z_1.reverse()
    return z_0, z_1


def needleman_wunsch_affine(
    source: list[int],
    target: list[int],
    scoring: AffineScoring = UNIT_SCORING,
    symmetric: bool = True,
) -> tuple[list[int], list[int]]:
    """Global-align two sequences under selectable scoring, optionally orientation-symmetric."""
    if symmetric and target < source:
        aligned_target, aligned_source = _align_affine(target, source, scoring)
        return aligned_source, aligned_target
    return _align_affine(source, target, scoring)


def levenshtein(a: str, b: str) -> int:
    """Return the edit distance between two sequences, i.e. how *minimal* an edit was."""
    prev = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        curr = [i]
        for j, char_b in enumerate(b, start=1):
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + (char_a != char_b)))
        prev = curr
    return prev[-1]
