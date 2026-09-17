"""EvoFlows §4.1's deterministic benchmark: synthetic edits with a known ground truth. §4.1 verbatim."""

from editjumps.core.utils import get_logger

logger = get_logger(__file__)

#: The four per-position classes the paper's Figure 2 reports precision and recall for.
EDIT_CLASSES = ("no_op", "insertion", "substitution", "deletion")


def deterministic_edits(z0: str) -> dict[int, set[str]]:
    """Label every position of ``z0`` with the edits §4.1's rules apply to it, sparsely. **Multi-label."""
    n = len(z0)
    edits: dict[int, set[str]] = {}

    def mark(position: int, kind: str) -> None:
        """Add ``kind`` to the label set at ``position``."""
        edits.setdefault(position, set()).add(kind)

    # Insertions first, matching the stated order. Ins(i-2, S) fires for each C.
    for i, residue in enumerate(z0):
        if residue == "C" and i - 2 >= 0:
            mark(i - 2, "insertion")

    # Deletions next. "z0[j] = L and z0[k] = K for some j < i < k" -- an L strictly before and a K
    # strictly after; the first L and the last K settle it in O(1) per position.
    lowest_l, highest_k = z0.find("L"), z0.rfind("K")
    for i, residue in enumerate(z0):
        if residue == "G" and lowest_l != -1 and lowest_l < i and highest_k != -1 and i < highest_k:
            mark(i, "deletion")

    # Substitutions last, so a deletion on the same position wins and the substitution never lands.
    for i, residue in enumerate(z0):
        target = i + 5
        if residue == "A" and target < n and "deletion" not in edits.get(target, ()):
            mark(target, "substitution")
    return edits


def apply_deterministic_edits(z0: str) -> str:
    """Build the unique ``z1`` that §4.1's rules produce from ``z0``."""
    n = len(z0)
    inserts = {i - 2 for i, r in enumerate(z0) if r == "C" and i - 2 >= 0}
    lowest_l, highest_k = z0.find("L"), z0.rfind("K")
    deletes = {
        i for i, r in enumerate(z0)
        if r == "G" and lowest_l != -1 and lowest_l < i and highest_k != -1 and i < highest_k
    }
    subs = {i + 5 for i, r in enumerate(z0) if r == "A" and i + 5 < n}

    out: list[str] = []
    for i, residue in enumerate(z0):
        if i in inserts:
            out.append("S")            # Ins(i, S): the new residue precedes position i
        if i in deletes:
            continue                   # deletions precede substitutions, so deletion wins
        out.append("H" if i in subs else residue)
    return "".join(out)


def ground_truth_labels(z0: str) -> list[set[str]]:
    """Dense per-position label sets for ``z0``, for scoring a model's predictions."""
    sparse = deterministic_edits(z0)
    return [set(sparse.get(i, set())) for i in range(len(z0))]


def class_counts(sequences: list[str]) -> dict[str, int]:
    """Count ground-truth classes over a set of sequences, to check the benchmark is not degenerate."""
    counts = dict.fromkeys(EDIT_CLASSES, 0)
    for z0 in sequences:
        for labels in ground_truth_labels(z0):
            if not labels:
                counts["no_op"] += 1
            for label in labels:
                counts[label] += 1
    return counts
