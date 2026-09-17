"""Cluster- and family-level splitting: no cluster or sequence may straddle train/val."""

from editjumps.core.cluster_split import split_by_cluster


def test_split_by_cluster_no_leakage() -> None:
    """Every cluster lands entirely in one split (the leakage guarantee)."""
    cluster_ids = [0, 0, 0, 1, 2, 2, 3, 4, 5, 6]
    assignment = split_by_cluster(cluster_ids, val_frac=0.3)
    side_of_cluster: dict[int, str] = {}
    for cid, side in zip(cluster_ids, assignment, strict=True):
        side_of_cluster.setdefault(cid, side)
        assert side_of_cluster[cid] == side, f"cluster {cid} straddles the split"


def test_split_by_cluster_approaches_target_fraction() -> None:
    """With many singleton clusters the val fraction lands near the target."""
    cluster_ids = list(range(100))  # 100 singleton clusters
    assignment = split_by_cluster(cluster_ids, val_frac=0.2)
    assert abs(assignment.count("val") / len(assignment) - 0.2) <= 0.01


def test_split_by_cluster_large_cluster_goes_to_train() -> None:
    """A cluster bigger than the val quota cannot fit in val, so it stays in train."""
    cluster_ids = [0] * 9 + [1]  # one 9-item cluster, one singleton; target 1 of 10
    assignment = split_by_cluster(cluster_ids, val_frac=0.1)
    assert assignment[:9] == ["train"] * 9
    assert assignment[9] == "val"


def test_split_pairs_by_family_is_sequence_disjoint_and_beats_a_random_split() -> None:
    """The held-out set must share no sequence with train, or val_loss is meaningless."""
    from editjumps.core.cluster_split import split_pairs_by_family

    # Interleaved so a prefix split would cut through both families.
    fam_a = [("a0", "a1"), ("a1", "a2"), ("a2", "a3"), ("a0", "a3")]
    fam_b = [("b0", "b1"), ("b1", "b2"), ("b2", "b3"), ("b0", "b3")]
    pairs = [p for ab in zip(fam_a, fam_b, strict=True) for p in ab]

    train_idx, val_idx = split_pairs_by_family(pairs, val_frac=0.5, seed=0)
    assert sorted(train_idx + val_idx) == list(range(len(pairs))), "every pair must be assigned once"
    assert val_idx, "a 50% target over two families must hold one out"

    train_seqs = {s for i in train_idx for s in pairs[i]}
    val_seqs = {s for i in val_idx for s in pairs[i]}
    assert not (train_seqs & val_seqs), f"sequence leaked across the split: {train_seqs & val_seqs}"

    # The whole family moved together — not a few of its pairs.
    assert val_seqs in ({"a0", "a1", "a2", "a3"}, {"b0", "b1", "b2", "b3"})

    # Contrast: the naive split this replaces shares sequences, which is the bug being prevented.
    naive_train, naive_val = pairs[: len(pairs) // 2], pairs[len(pairs) // 2 :]
    naive_overlap = {s for p in naive_train for s in p} & {s for p in naive_val for s in p}
    assert naive_overlap, "the counter-example no longer demonstrates the hazard"


def test_split_pairs_by_family_disables_cleanly_and_is_deterministic() -> None:
    """Val_frac=0 must return everything as train, and a fixed seed must reproduce the split."""
    from editjumps.core.cluster_split import split_pairs_by_family

    pairs = [(f"s{i}", f"s{i}x") for i in range(20)]
    train_idx, val_idx = split_pairs_by_family(pairs, val_frac=0.0)
    assert train_idx == list(range(20)) and val_idx == []

    first = split_pairs_by_family(pairs, val_frac=0.25, seed=7)
    assert first == split_pairs_by_family(pairs, val_frac=0.25, seed=7), "same seed must reproduce"
    # 20 singleton families, 25% target -> a nonempty held-out set well under half.
    assert 0 < len(first[1]) <= 10
