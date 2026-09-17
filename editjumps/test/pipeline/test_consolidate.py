"""The consolidated appendix table: one comparability frame, with its directions labelled."""

from pathlib import Path


def test_the_appendix_table_is_one_frame_and_labels_its_directions() -> None:
    """The consolidated table must not mix comparability frames, or label a direction wrongly."""
    from editjumps.pipeline.evaluate.consolidate import (
        DISJOINT,
        METRICS,
        RENDER,
        appendix_table,
        frame_of,
        rows_for,
    )

    assert frame_of({"reference_in_training_pairs": 0}) == "disjoint"
    assert frame_of({"reference_in_training_pairs": 113}) == "train_overlap_113"
    assert frame_of({}) == "unrecorded"

    rows = []
    for method, family, rel, nested in DISJOINT:
        rows.extend(rows_for(method, family, Path(rel), nested))
    assert rows, "no disjoint cells found"
    assert {row["frame"] for row in rows} == {"disjoint"}, (
        "a cell in metrics/disjoint/ records a reference that overlaps the training pairs"
    )
    # One frame means one alignment width per family, which is the evidence that the three
    # evaluators drew the same partition rather than merely the same counts.
    for family in {row["family"] for row in rows}:
        widths = {row["alignment_length"] for row in rows if row["family"] == family}
        assert len(widths) == 1, f"{family} spans alignment widths {widths}"

    directions = {metric: direction for metric, _, direction, _ in RENDER}
    assert "lower" not in directions["entropy_delta"], (
        "entropy_delta is signed; 'lower better' claims -0.5 beats 0.0"
    )
    assert directions["entropy_delta"] == "closer to 0"
    assert directions["spectrum_mmd"] == "lower"
    assert directions["covariance_agreement"] == "higher"
    # Every rendered metric must exist in METRICS, or a column would silently render as all "--".
    known = {metric for metric, _, _, _ in METRICS}
    assert {metric for metric, _, _, _ in RENDER} <= known

    table = appendix_table(rows)
    assert "train_overlap" not in table, "the appendix table leaked a non-disjoint row"
    for _, label, _, _ in RENDER:
        assert label in table


def test_the_committed_appendix_table_is_what_the_generator_produces() -> None:
    """`metrics/appendix_table.md` must equal what `consolidate` renders from the committed metrics."""
    from editjumps.pipeline.evaluate.consolidate import (
        DISJOINT,
        LEGACY,
        appendix_table,
        paper_rows,
        rows_for,
    )

    root = Path(__file__).parents[3]
    rows: list[dict] = []
    for method, family, rel, nested in DISJOINT + LEGACY:
        rows.extend(rows_for(method, family, root / rel, nested))
    rows.extend(paper_rows(root / "metrics/evoflows_figure3.json"))

    committed = (root / "metrics/appendix_table.md").read_text()
    assert committed == appendix_table(rows, root), (
        "metrics/appendix_table.md is stale: it differs from what consolidate renders from the "
        "committed metrics. Run `make consolidate`."
    )
