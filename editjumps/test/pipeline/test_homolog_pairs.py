"""Homolog-pair construction: family grouping and pair sampling."""


def test_group_families_groups_by_cluster_and_dedups() -> None:
    """Lines group by cluster id; exact-duplicate sequences within a family collapse."""
    from editjumps.pipeline.preprocess.pretrain.homolog_pairs import group_families

    cluster_ids = [0, 0, 1, 0, 1]
    lines = ["A", "B", "C", "A", "C"]  # "A" repeats in cluster 0; "C" repeats in cluster 1
    families = group_families(cluster_ids, lines)
    assert families == {0: ["A", "B"], 1: ["C"]}  # dedup within family, first-seen order


def test_sample_pairs_enumerates_small_caps_large_skips_singletons() -> None:
    """Small families enumerate all unordered pairs; large ones are capped; singletons yield none."""
    import random

    from editjumps.pipeline.preprocess.pretrain.homolog_pairs import sample_pairs

    families = {0: ["a", "b", "c"], 1: ["x"], 2: [f"s{i}" for i in range(10)]}
    pairs = sample_pairs(families, max_pairs_per_family=5, rng=random.Random(0))
    # family 0: C(3,2)=3 pairs (all), family 1: singleton -> 0, family 2: capped to 5
    fam0 = [p for p in pairs if set(p) <= {"a", "b", "c"}]
    fam2 = [p for p in pairs if p[0].startswith("s")]
    assert sorted(fam0) == [("a", "b"), ("a", "c"), ("b", "c")]
    assert len(fam2) == 5
    assert all(p[0] != p[1] for p in pairs)  # never pairs a sequence with itself
    assert len({(min(p), max(p)) for p in fam2}) == 5  # distinct unordered pairs


