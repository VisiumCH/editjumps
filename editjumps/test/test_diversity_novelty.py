"""`editjumps/core/diversity_novelty.py`: what the novelty summary's keys actually hold."""

from editjumps.core.diversity_novelty import nearest_neighbour_distance


def test_the_summary_keys_describe_the_statistics_they_hold() -> None:
    """`min` is the smallest nearest-neighbour distance, not a minimum over any set of means."""
    reference = ["AAAA", "CCCC"]
    # Distances to the nearest reference entry: 0 (exact hit), 1, 2.
    queries = ["AAAA", "AAAC", "AACC"]

    summary = nearest_neighbour_distance(queries, reference)

    assert summary["min"] == 0.0, "an exact match against the reference is zero away from it"
    assert summary["median"] == 1.0
    assert summary["mean"] == 1.0
    assert summary["min"] <= summary["mean"], "a minimum cannot exceed the mean it is drawn from"
    assert (summary["n_queries"], summary["n_reference"]) == (3, 2)


def test_min_tracks_the_single_closest_query() -> None:
    """Moving one query nearer the reference moves `min` and leaves the others' contribution alone."""
    reference = ["AAAAAAAA"]
    far = nearest_neighbour_distance(["CCCCCCCC", "CCCCCCCA"], reference)
    near = nearest_neighbour_distance(["CCCCCCCC", "AAAAAAAA"], reference)

    assert near["min"] == 0.0
    assert far["min"] > near["min"]


def test_the_empty_case_offers_the_same_keys() -> None:
    """A caller reading the summary must not have to branch on whether there was anything to score."""
    empty = nearest_neighbour_distance([], ["AAAA"])
    scored = nearest_neighbour_distance(["AAAA"], ["AAAA"])
    assert set(empty) == set(scored)
    assert "min_of_means" not in empty, "the misleading key must not come back"
