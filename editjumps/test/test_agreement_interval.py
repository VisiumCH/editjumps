"""Tests for `editjumps.pipeline.evaluate.agreement_interval`."""

import json
import random
from pathlib import Path

import numpy as np
import pytest

from editjumps.core.generation_metrics import (
    ALPHABET,
    frequencies,
    levenshtein,
    mutual_information_apc,
    one_hot,
    positional_interaction_strength,
)
from editjumps.pipeline.evaluate.agreement_interval import (
    AGREEMENTS,
    assign_blocks,
    check_fast_path,
    compact,
    coupling_matrices,
    draw_counts,
    percentile_interval,
    segment_sum,
    template_of_each_block,
)

ROOT = Path(__file__).parents[2]


def synthetic_set(n_templates: int = 6, block_size: int = 5, length: int = 24,
                  seed: int = 7) -> tuple[list[str], list[str]]:
    """Build a template-major generated set the way the evaluators lay one out."""
    rng = random.Random(seed)
    # A conserved framework with a few variable columns, which is what a homolog family looks like
    # and what makes the occupied-slot restriction worth anything.
    framework = [rng.choice("ACDEFGHIKLMNPQRSTVWY") for _ in range(length)]
    variable = rng.sample(range(length), 8)
    templates, sequences = [], []
    for _ in range(n_templates):
        template = list(framework)
        for position in rng.sample(variable, 4):
            template[position] = rng.choice("ACDEFGHIKLMNPQRSTVWY")
        templates.append("".join(template))
        for _ in range(block_size):
            variant = list(template)
            for position in rng.sample(variable, 2):
                variant[position] = rng.choice(ALPHABET)
            sequences.append("".join(variant))
    return templates, sequences


def library_matrices(aligned: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Compute the two coupling matrices the canonical way."""
    f_i, f_ij, c_ij = frequencies(one_hot(aligned))
    return positional_interaction_strength(c_ij), mutual_information_apc(f_i, f_ij)


def test_the_block_regrouped_path_reproduces_the_library_on_the_identity_draw() -> None:
    """The fast path is an exact regrouping, not an approximation, or every interval is wrong."""
    _, sequences = synthetic_set()
    blocks = compact("model", sequences, 5, ())
    want_strength, want_mip = library_matrices(sequences)
    got_strength, got_mip = coupling_matrices(blocks, np.ones(6))
    assert np.abs(want_strength - got_strength).max() < 1e-12
    assert np.abs(want_mip - got_mip).max() < 1e-12
    check_fast_path(sequences, blocks)                 # the module's own guard, same claim


def test_a_resampled_draw_equals_rebuilding_the_set_from_the_drawn_blocks() -> None:
    """A draw must be the metric on the resampled SET, not an interpolation between block scores."""
    _, sequences = synthetic_set()
    blocks = compact("model", sequences, 5, ())
    rng = np.random.default_rng(3)
    for counts in draw_counts(rng, 6, 5):
        rebuilt = [seq for index, repeats in enumerate(counts)
                   for _ in range(int(repeats))
                   for seq in sequences[index * 5:(index + 1) * 5]]
        want_strength, want_mip = library_matrices(rebuilt)
        got_strength, got_mip = coupling_matrices(blocks, counts)
        assert np.abs(want_strength - got_strength).max() < 1e-12, counts
        assert np.abs(want_mip - got_mip).max() < 1e-12, counts


def test_segment_sum_contracts_the_flat_axis_the_way_the_estimators_do() -> None:
    """The flat-slot contraction is the estimators' ``sum(axis=(2, 3))`` and nothing else."""
    starts = np.array([0, 2, 5])
    matrix = np.arange(49, dtype=float).reshape(7, 7)
    got = segment_sum(matrix, starts)
    assert got.shape == (3, 3)
    assert got[0, 0] == matrix[0:2, 0:2].sum()
    assert got[1, 2] == matrix[2:5, 5:7].sum()


def test_blocks_are_matched_to_templates_by_content_not_by_position() -> None:
    """Block order is not template order, and the recovery must not assume it is. ``generation_eval``."""
    templates, sequences = synthetic_set()
    order = list(random.Random(0).sample(range(6), 6))
    shuffled = [seq for index in order for seq in sequences[index * 5:(index + 1) * 5]]
    assert template_of_each_block(shuffled, templates, 5) == tuple(order)
    # And a set whose blocks belong to no template at all -- the random-homolog-pairing set -- must be
    # refused rather than assigned something, because a wrong assignment is a silently wrong pairing.
    with pytest.raises(ValueError, match="not a permutation"):
        template_of_each_block(templates[:1] * 6, templates, 1)


def test_a_method_that_edits_past_its_neighbours_is_still_matched_to_its_own_template() -> None:
    """Stage 1 collides for a method with no mutation budget; stage 2 has to recover the mapping."""
    templates, sequences = synthetic_set(n_templates=6, block_size=6, length=40, seed=11)
    rng = random.Random(5)
    heavy = []
    for index in range(6):
        for variant in sequences[index * 6:(index + 1) * 6]:
            residues = list(variant)
            # 12 substitutions in a 40-column frame, against templates that differ in 4 columns:
            # every block is now much further from its own template than the templates are apart.
            for position in rng.sample(range(len(residues)), 12):
                residues[position] = rng.choice(ALPHABET.replace("-", ""))
            heavy.append("".join(residues))
    order = list(random.Random(1).sample(range(6), 6))
    shuffled = [seq for index in order for seq in heavy[index * 6:(index + 1) * 6]]

    per_block_argmin = [
        min(range(6), key=lambda i: levenshtein(shuffled[block * 6], templates[i]))
        for block in range(6)
    ]
    assert sorted(per_block_argmin) != list(range(6)), (
        "this set no longer exercises the fallback -- stage 1 resolves it, so the test is vacuous"
    )
    assert template_of_each_block(shuffled, templates, 6) == tuple(order)


def test_a_set_that_belongs_to_no_template_is_refused_rather_than_assigned_one() -> None:
    """An assignment always succeeds, so the random-pairing set must be refused by a check.."""
    templates, _ = synthetic_set(n_templates=6, block_size=6, length=40, seed=13)
    rng = random.Random(17)
    unrelated = ["".join(rng.choice(ALPHABET.replace("-", "")) for _ in range(40))
                 for _ in range(36)]
    with pytest.raises(ValueError, match="split-half check"):
        template_of_each_block(unrelated, templates, 6)


def test_the_assignment_is_global_and_not_a_greedy_pass() -> None:
    """Minimum TOTAL cost, because greedy leaves a later block only wrong choices."""
    assert assign_blocks(np.array([[1.0, 2.0], [0.0, 9.0]])) == (1, 0)


def test_draws_are_nested_so_two_counts_are_one_experiment_at_two_resolutions() -> None:
    """The 100-draw and 2,000-draw answers must differ only in resolution, not in seed."""
    counts = draw_counts(np.random.default_rng(0), 20, 2000)
    assert counts.shape == (2000, 20)
    assert (counts.sum(axis=1) == 20).all(), "a resample changed the generated set's size"
    again = draw_counts(np.random.default_rng(0), 20, 2000)
    assert (counts == again).all(), "the same seed gave a different bootstrap"
    assert (counts[:100] == again[:100]).all()


def test_percentile_interval_uses_order_statistics() -> None:
    """No interpolation, matching ``consolidate.interval``: 100 draws do not support more."""
    values = np.arange(1000, dtype=float)
    low, high = percentile_interval(values)
    assert (low, high) == (25.0, 975.0)


def test_the_disjoint_cells_carry_a_recovered_interval_that_says_what_it_is() -> None:
    """The two agreements must not go back to a bare centre, and their interval must be labelled."""
    for path in sorted((ROOT / "metrics" / "disjoint").glob("*.json")):
        report = json.loads(path.read_text())
        for metric, block_name in AGREEMENTS:
            block = report[block_name]
            interval = block.get(f"{metric}_ci")
            assert isinstance(interval, dict), (
                f"{path.name}: {metric} has no recovered interval; run "
                f"`editjumps agreement-interval {path}`"
            )
            assert interval["unit"] == "template", path.name
            assert interval["draws"] >= 2000, f"{path.name}: {interval['draws']} draws is coarse"
            assert "SPREAD" in interval["reads_as"], path.name
            assert interval["low"] < interval["high"], path.name
            # Every cell must DESCRIBE its own shift correctly, whichever way it went.
            shift = interval["bootstrap_mean"] - block[metric]
            assert interval["bootstrap_shift"] == pytest.approx(shift, abs=1e-9), path.name
            direction = "BELOW" if shift < 0 else "ABOVE"
            assert direction in interval["reads_as"], (
                f"{path.name}: {metric}'s reads_as does not state the direction its own numbers "
                f"show ({direction}, by {abs(shift):.4f})"
            )
            brackets = interval["low"] <= block[metric] <= interval["high"]
            assert interval["contains_centre"] == brackets, path.name
            assert ("does NOT contain the centre" in interval["reads_as"]) == (not brackets), (
                f"{path.name}: {metric}'s reads_as claims the wrong thing about whether the "
                f"interval brackets its centre ({block[metric]:.4f} in "
                f"[{interval['low']:.4f}, {interval['high']:.4f}] is {brackets})"
            )
            # And the ORIGINAL claim is kept exactly where it was established: the twelve EvoFlows-comparison.

def test_the_paired_reports_are_the_test_the_appendix_table_leans_on() -> None:
    """Our port's four nominal leads must be recorded as paired differences, and include zero."""
    from editjumps.pipeline.evaluate.consolidate import PAIRED_REPORT

    for family in ("ty1", "her2vh"):
        path = ROOT / PAIRED_REPORT.format(family=family)
        assert path.exists(), f"no paired agreement report for {family}"
        report = json.loads(path.read_text())
        assert 2000 in report["draws"] and 100 in report["draws"]
        ours = [r for r in report["comparisons"] if "editor" in r["a"]["cell"]]
        assert len(ours) == len(AGREEMENTS), f"{family}: editor rows {len(ours)}"
        for record in ours:
            assert record["margin"] > 0, f"{family} {record['metric']}: no lead to test"
            for count, block in record["by_draws"].items():
                assert block["includes_zero"], (
                    f"{family} {record['metric']} at {count} draws now EXCLUDES zero "
                    f"({block['paired_diff_ci']}); docs/findings.md says otherwise"
                )
            # The mispaired variant is kept as a diagnostic, and it must stay wider -- that gap is
            # the evidence for pairing by template rather than by block position.
            largest = record["by_draws"][str(max(report["draws"]))]
            paired = largest["paired_diff_ci"]
            mispaired = largest["if_paired_on_raw_block_index"]["paired_diff_ci"]
            assert (mispaired[1] - mispaired[0]) > 2 * (paired[1] - paired[0]), (
                f"{family} {record['metric']}: pairing by template no longer narrows the interval, "
                "so either the block-to-template recovery broke or the FASTAs changed order"
            )


def test_consolidate_labels_the_two_kinds_of_interval_apart() -> None:
    """A recovered spread and a coverage interval on a mean must not render alike."""
    from editjumps.pipeline.evaluate.consolidate import (
        CI_FROM_ARTEFACT,
        CI_RECOVERED,
        DISJOINT,
        appendix_table,
        rows_for,
    )

    rows = []
    for method, family, rel, nested in DISJOINT:
        rows.extend(rows_for(method, family, ROOT / rel, nested))
    by_metric = {(r["metric"], r["method"], r["family"]): r for r in rows}

    edits = by_metric[("edits_per_sequence", "our EvoFlows port", "ty1")]
    assert edits["ci_kind"] == CI_FROM_ARTEFACT
    assert edits["ci_low"] < edits["value"] < edits["ci_high"], (
        "a coverage interval on a mean must contain its centre"
    )
    for metric, _ in AGREEMENTS:
        row = by_metric[(metric, "our EvoFlows port", "ty1")]
        assert row["ci_kind"] == CI_RECOVERED
        assert row["ci_high"] < row["value"], (
            "the recovered spread is expected to sit below its centre; if that changed, the "
            "appendix table's explanation of `‡` is now wrong"
        )

    table = appendix_table(rows, ROOT)
    assert "‡" in table, "the recovered intervals render indistinguishably from the plain ones"
    assert "paired difference against the evotuned PLM" in table
    assert "includes zero" in table
