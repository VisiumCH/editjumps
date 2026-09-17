"""The two aligners, and that the affine one reproduces the unit-cost one column for column."""

import pytest


def test_needleman_wunsch_aligns_a_known_indel() -> None:
    """A single deletion between source and target aligns to one gap column."""
    from editjumps.core.edit_flows.alignment import needleman_wunsch
    from editjumps.core.edit_flows.path import EPS, strip_epsilon

    z0, z1 = needleman_wunsch([1, 2, 3, 4], [1, 2, 4])  # '3' deleted
    assert len(z0) == len(z1)
    assert strip_epsilon(z0) == [1, 2, 3, 4]
    assert strip_epsilon(z1) == [1, 2, 4]
    # exactly one deletion column (real in z0, gap in z1); no insertions
    assert sum(1 for a, b in zip(z0, z1, strict=True) if b == EPS and a != EPS) == 1
    assert sum(1 for a in z0 if a == EPS) == 0

def test_needleman_wunsch_takes_no_cost_parameters() -> None:
    """The cost knobs are gone: they were unused and carried the module's only reachable crash."""
    import inspect

    from editjumps.core.edit_flows.alignment import needleman_wunsch

    assert list(inspect.signature(needleman_wunsch).parameters) == ["source", "target"]
    with pytest.raises(TypeError):
        needleman_wunsch([1, 2, 3, 4, 5, 6], [], gap_cost=0.7)  # ty: ignore[unknown-argument]

def test_affine_aligner_reproduces_the_unit_cost_aligner_exactly() -> None:
    """The new Gotoh DP under UNIT_SCORING must equal `needleman_wunsch` column for column."""
    import random

    from editjumps.core.edit_flows.alignment import (
        UNIT_SCORING,
        blosum62_scoring,
        needleman_wunsch,
        needleman_wunsch_affine,
    )
    from editjumps.core.edit_flows.path import (
        EPS,
        strip_epsilon,
    )

    rng = random.Random(0)
    scoring = blosum62_scoring({residue: ord(residue) for residue in "ACDEFGHIKLMNPQRSTVWY"}, 128)
    for _ in range(500):
        a = [rng.randrange(5) for _ in range(rng.randint(0, 12))]
        b = [rng.randrange(5) for _ in range(rng.randint(0, 12))]
        assert needleman_wunsch_affine(a, b, UNIT_SCORING, symmetric=False) == needleman_wunsch(a, b)

    for _ in range(200):
        a = [ord(rng.choice("ACDEFGHIKLMNPQRSTVWY")) for _ in range(rng.randint(0, 15))]
        b = [ord(rng.choice("ACDEFGHIKLMNPQRSTVWY")) for _ in range(rng.randint(0, 15))]
        for candidate in (UNIT_SCORING, scoring):
            for symmetric in (True, False):
                z_0, z_1 = needleman_wunsch_affine(a, b, candidate, symmetric=symmetric)
                assert len(z_0) == len(z_1)
                assert strip_epsilon(z_0) == a and strip_epsilon(z_1) == b
                assert not any(t_0 == EPS and t_1 == EPS for t_0, t_1 in zip(z_0, z_1, strict=True))

def test_affine_traceback_survives_non_integer_gap_penalties() -> None:
    """The pointer matrix must remove the float-`==` crash class, not re-document it."""
    import random

    from editjumps.core.edit_flows.alignment import AffineScoring, needleman_wunsch_affine
    from editjumps.core.edit_flows.path import strip_epsilon

    rng = random.Random(1)
    for _ in range(2000):
        scoring = AffineScoring(
            name="fuzz", substitution=None,
            gap_open=rng.uniform(0.0, 3.0), gap_extend=rng.uniform(0.1, 1.3),
        )
        a = [rng.randrange(4) for _ in range(rng.randint(0, 9))]
        b = [rng.randrange(4) for _ in range(rng.randint(0, 9))]
        z_0, z_1 = needleman_wunsch_affine(a, b, scoring, symmetric=False)
        assert strip_epsilon(z_0) == a and strip_epsilon(z_1) == b

def test_blosum62_table_is_the_published_matrix() -> None:
    """The embedded BLOSUM62 must be the real one — a typo here silently re-weights every label."""
    from editjumps.core.edit_flows.alignment import (
        BLOSUM62,
        BLOSUM62_ALPHABET,
        NON_RESIDUE_MATCH_SCORE,
        NON_RESIDUE_MISMATCH_SCORE,
    )

    assert len(BLOSUM62) == 400
    for a in BLOSUM62_ALPHABET:
        for b in BLOSUM62_ALPHABET:
            assert BLOSUM62[a, b] == BLOSUM62[b, a]
    # What a unit mismatch cost cannot tell apart: aromatic-for-aromatic vs aromatic-for-proline.
    assert BLOSUM62["W", "F"] == 1 > BLOSUM62["W", "P"] == -4
    assert max(BLOSUM62.values()) == NON_RESIDUE_MATCH_SCORE == 11
    assert min(BLOSUM62.values()) == NON_RESIDUE_MISMATCH_SCORE == -4

    matrices = pytest.importorskip("Bio.Align").substitution_matrices
    published = matrices.load("BLOSUM62")
    for (a, b), score in BLOSUM62.items():
        assert score == int(published[a, b]), f"BLOSUM62[{a},{b}] is {score}, published {published[a, b]}"

def test_blosum62_scoring_maps_the_tokenizer_vocab_and_anchors_non_residues() -> None:
    """Scores are token-id indexed, built from the vocab, with a stated rule for non-residues."""
    from editjumps.core.edit_flows.alignment import (
        BLOSUM62,
        NON_RESIDUE_MATCH_SCORE,
        NON_RESIDUE_MISMATCH_SCORE,
        blosum62_scoring,
    )

    tokens = {"<cls>": 0, "W": 1, "F": 2, "P": 3, ".": 4, "B": 99}  # 99 is out of range on purpose
    scoring = blosum62_scoring(tokens, vocab_size=6, gap_open=11.0, gap_extend=1.0)
    matrix = scoring.substitution
    assert matrix is not None and len(matrix) == 6
    assert matrix[1][2] == float(BLOSUM62["W", "F"]) and matrix[1][3] == float(BLOSUM62["W", "P"])
    assert matrix[1][1] == float(BLOSUM62["W", "W"]) == 11.0
    # Non-residues anchor as hard as a conserved tryptophan; crossing one costs the matrix's worst.
    assert matrix[4][4] == NON_RESIDUE_MATCH_SCORE and matrix[0][0] == NON_RESIDUE_MATCH_SCORE
    assert matrix[4][1] == NON_RESIDUE_MISMATCH_SCORE and matrix[0][4] == NON_RESIDUE_MISMATCH_SCORE
    assert (scoring.gap_open, scoring.gap_extend) == (11.0, 1.0)

def test_canonical_orientation_makes_the_alignment_direction_independent() -> None:
    """EvoFlows' coupling is symmetric (eq 8), so the edit labels must not depend on draw order.."""
    import random

    from editjumps.core.edit_flows.alignment import UNIT_SCORING, needleman_wunsch, needleman_wunsch_affine

    rng = random.Random(0)
    asymmetric_found = 0
    for _ in range(1500):
        a = [rng.randrange(4) for _ in range(rng.randint(4, 10))]
        b = [rng.randrange(4) for _ in range(rng.randint(4, 10))]
        forward = needleman_wunsch_affine(a, b, UNIT_SCORING, symmetric=True)
        reverse = needleman_wunsch_affine(b, a, UNIT_SCORING, symmetric=True)
        assert forward == (reverse[1], reverse[0]), (a, b)
        legacy_forward, legacy_reverse = needleman_wunsch(a, b), needleman_wunsch(b, a)
        asymmetric_found += legacy_forward != (legacy_reverse[1], legacy_reverse[0])
    assert asymmetric_found > 0, "the legacy aligner is symmetric here, so this test proves nothing"

    # the review's own case, which must now mirror
    source = [ord(c) for c in "DDCBAAC"]
    target = [ord(c) for c in "BAACCCA"]
    mirrored = needleman_wunsch_affine(target, source, UNIT_SCORING)
    assert needleman_wunsch_affine(source, target, UNIT_SCORING) == (mirrored[1], mirrored[0])

def test_affine_alignment_is_optimal_against_biopython() -> None:
    """An independent optimality check: our Gotoh DP must find biopython's optimal score."""
    import random

    align_module = pytest.importorskip("Bio.Align")

    from editjumps.core.edit_flows.alignment import blosum62_scoring, needleman_wunsch_affine
    from editjumps.core.edit_flows.path import EPS

    aligner = align_module.PairwiseAligner()
    aligner.substitution_matrix = align_module.substitution_matrices.load("BLOSUM62")
    aligner.mode = "global"
    aligner.open_gap_score = -12.0
    aligner.extend_gap_score = -1.0

    letters = "ACDEFGHIKLMNPQRSTVWY"
    scoring = blosum62_scoring({letter: ord(letter) for letter in letters}, 128)
    matrix = scoring.substitution
    assert matrix is not None

    def score(z_0: list[int], z_1: list[int]) -> float:
        """Score one of our alignments under our own convention."""
        total, in_gap = 0.0, False
        for token_0, token_1 in zip(z_0, z_1, strict=True):
            if token_0 == EPS or token_1 == EPS:
                total -= scoring.gap_extend if in_gap else scoring.gap_open + scoring.gap_extend
                in_gap = True
            else:
                total += matrix[token_0][token_1]
                in_gap = False
        return total

    rng = random.Random(0)
    for _ in range(60):
        a = "".join(rng.choice(letters) for _ in range(rng.randint(1, 25)))
        b = "".join(rng.choice(letters) for _ in range(rng.randint(1, 25)))
        ours = score(*needleman_wunsch_affine([ord(c) for c in a], [ord(c) for c in b], scoring))
        assert ours == aligner.score(a, b), f"suboptimal alignment for {a} / {b}"


def test_affine_gaps_keep_an_indel_contiguous_where_linear_gaps_fragment_it() -> None:
    """Gap *opening* is the mechanism: without it, contiguous and split indels tie on cost."""
    from editjumps.core.edit_flows.alignment import UNIT_SCORING, blosum62_scoring, needleman_wunsch_affine
    from editjumps.core.edit_flows.path import EPS
    from editjumps.pipeline.evaluate.alignment_scoring import RESIDUE_TOKENS, VOCAB_SIZE, column_counts

    source = [ord(c) for c in "DDCBAAC"]
    target = [ord(c) for c in "BAACCCA"]
    affine = blosum62_scoring(RESIDUE_TOKENS, VOCAB_SIZE)

    linear_runs = column_counts(*needleman_wunsch_affine(source, target, UNIT_SCORING))["gap_runs"]
    affine_runs = column_counts(*needleman_wunsch_affine(source, target, affine))["gap_runs"]
    assert affine_runs < linear_runs, "a gap-opening charge must not leave the indel fragmented"

    z_0, z_1 = needleman_wunsch_affine(source, target, affine)
    gap_positions = [i for i, token in enumerate(z_0) if token == EPS]
    assert gap_positions == list(range(gap_positions[0], gap_positions[0] + len(gap_positions)))
