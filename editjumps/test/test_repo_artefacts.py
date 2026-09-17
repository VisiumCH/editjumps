"""What the committed artefacts under metrics/ and figures/ must say about themselves."""

import json
from pathlib import Path

import pytest


def test_the_disjoint_table_cells_all_share_one_frame() -> None:
    """Every committed cell of `metrics/disjoint/` is drawn the same way, or it is not a table."""
    cells = sorted(Path("metrics/disjoint").glob("*.json"))
    assert cells, "metrics/disjoint holds no cells"
    frames: dict[str, set[str]] = {}
    for path in cells:
        report = json.loads(path.read_text())
        assert report["n_templates"] == 20, path
        assert report["n_variants_per_template"] == 20, path
        assert report["n_natural_reference"] == 200, path
        assert report["reference_in_training_pairs"] == 0, (
            f"{path} was scored against a reference overlapping the editor's training pairs"
        )
        assert report["defined_by_the_paper"]["ceiling_n"] == 300.0, path
        # The interval and the centre have to come out of the same artefact, and the unit is
        # templates: resampling the 400 sequences returns an interval about 2.5x too narrow.
        assert len(report["defined_by_the_paper"]["levenshtein_to_template_per_template"]) == 20, path
        assert "template" in report["reading_notes"]["bootstrap_unit"], path
        # Pooled diversity, the MMD and the KL are set-level, so their template-resampled interval
        # needs the sequences; they ship beside the metrics.
        assert path.with_suffix(".fasta").exists(), f"{path} has no sibling FASTA"
        frames.setdefault(path.stem.rsplit("-", 1)[-1], set()).add(report["alignment"])
    # Every evaluator projects onto ITS OWN split.templates[0], so one alignment width per family is evidence.
    for family, alignments in frames.items():
        assert len(alignments) == 1, f"{family} cells disagree about the frame: {alignments}"


def test_the_disjoint_editor_target_reproduces_the_committed_editor_cells() -> None:
    """`make disjoint-editor`'s flags must equal what the two editor cells say about themselves."""
    import re

    root = Path(__file__).parents[2]
    makefile = (root / "Makefile").read_text()

    # The recipe is one `for` loop over the two families, line-continued. Take the target's block
    # (its own line plus every following tab-indented line) and flatten the continuations.
    block = re.search(r"^disjoint-editor:.*?(?=\n[^\t\n])", makefile, re.M | re.S)
    assert block, "no disjoint-editor recipe in the Makefile"
    recipe = block.group(0).replace("\\\n", " ")
    # The negative lookahead matters: --disjoint-from-pairs is a BARE flag here, and without it
    # the next flag is captured as its value.
    flags = dict(re.findall(r"--([a-z0-9-]+)(?:[ \t]+(?!--)(\S+))?", recipe))

    # The recipe names the EDITOR variable rather than repeating the path, and this resolves its
    # default -- a target hardcoding a second copy of the path is the thing to avoid.
    assert flags["model-folder"] == "$(EDITOR)", flags["model-folder"]
    editor = re.search(r"^EDITOR \?= (\S+)", makefile, re.M)
    assert editor, "no EDITOR default in the Makefile"

    for tag in ("ty1", "her2vh"):
        cell = root / "metrics" / "disjoint" / f"editor-{tag}.json"
        report = json.loads(cell.read_text())
        # `; key: value` provenance header that generation_eval writes beside the metrics.
        header = dict(
            line[2:].split(": ", 1)
            for line in cell.with_suffix(".fasta").read_text().splitlines()
            if line.startswith("; ")
        )

        assert editor.group(1) == report["model"], (editor.group(1), report["model"])
        assert int(flags["n-templates"]) == report["n_templates"]
        assert int(flags["n-variants"]) == report["n_variants_per_template"]
        assert int(flags["holdout-size"]) == report["n_natural_reference"]
        assert float(flags["ceiling-n"]) == report["defined_by_the_paper"]["ceiling_n"]
        assert float(flags["clock"]) == report["clock_normalization"]
        assert flags["seed"] == header["seed"], (flags["seed"], header["seed"])

        # The flag is not the evidence -- `reference_in_training_pairs` is. A run that accepted
        # --disjoint-from-pairs and ignored it would look identical from the outside, and once did.
        assert "disjoint-from-pairs" in flags
        assert report["reference_in_training_pairs"] == 0, cell
        assert header["disjoint_from_pairs"] == "True", header

        # Appendix B.2's PLL is empty in both cells, so passing --pll-model would not reproduce them.
        assert "pll-model" not in flags
        assert not report["esm2_pseudo_log_likelihood"], cell

        assert f"editor-{tag}.json" in recipe.replace("$$tag", tag)

    # Eval_B_stock_appA is the mlp / esm_lm_head arm (metrics/inputform/README.md, metrics/clock/README.md.
    assert flags["rate-head"] == "mlp", flags
    assert flags["q-head"] == "esm_lm_head", flags
    assert flags["n-steps"] == "50", flags
    assert flags["group"] == "train", "generation-eval needs torch"

def test_our_metrics_land_in_the_papers_plotted_ranges() -> None:
    """Each metric we compare with Figure 3 must fall inside that panel's extracted range."""
    import json

    from editjumps.core.provenance import FIGURE3_PANEL_FOR_METRIC, FIGURE3_RANGE_EXEMPT

    root = Path(__file__).parents[2]
    figure = json.loads((root / "metrics" / "evoflows_figure3.json").read_text())
    ranges = {}
    for panel in figure.values():
        values = [v for group in panel["by_method"].values() for v in group]
        ranges[panel["title"]] = (min(values), max(values))

    # Every mapped panel must exist in the extraction, or the map has rotted.
    for metric, title in FIGURE3_PANEL_FOR_METRIC.items():
        assert title in ranges, f"{metric} maps to panel {title!r}, which is not in the extraction"

    report = json.loads((root / "metrics" / "disjoint" / "editor-ty1.json").read_text())
    values = {**report["defined_by_the_paper"], **report.get("our_interpretation", {})}

    # Every mapped metric must be PRESENT in the artefact.
    absent = [m for m in FIGURE3_PANEL_FOR_METRIC if not isinstance(values.get(m), (int, float))]
    assert not absent, (
        f"mapped metrics missing from the artefact: {absent}. Either the evaluator does not emit "
        f"them yet (regenerate metrics/disjoint/editor-ty1.json) or the map names the wrong key."
    )

    outside = []
    for metric, title in FIGURE3_PANEL_FOR_METRIC.items():
        value = values[metric]
        low, high = ranges[title]
        if not low <= value <= high and metric not in FIGURE3_RANGE_EXEMPT:
            outside.append(f"    {metric} = {value:.5g}, outside {title} range {low:.4g}..{high:.4g}")
    assert not outside, (
        "metrics outside the paper's own plotted range for the panel they claim to reproduce:\n"
        + "\n".join(outside)
        + "\nEither the reading is wrong, or add it to FIGURE3_RANGE_EXEMPT with the reason."
    )

    # An exemption that now passes is stale and should be deleted.
    for metric in FIGURE3_RANGE_EXEMPT:
        title = FIGURE3_PANEL_FOR_METRIC.get(metric)
        value = values.get(metric)
        if title and isinstance(value, (int, float)):
            low, high = ranges[title]
            assert not (low <= value <= high), (
                f"{metric} is exempt but now lands inside {title}'s range; drop its entry"
            )


def test_generation_eval_artefacts_say_which_of_each_ambiguous_pair_to_use() -> None:
    """The vendored artefacts must name the right key where two could plausibly be meant."""
    import json

    root = Path(__file__).parents[2]
    for name in ("perseq-ty1.json", "perseq-her2vh.json"):
        report = json.loads((root / "metrics" / "aligned" / name).read_text())
        notes = report.get("reading_notes", {})
        assert "pairwise_levenshtein_pooled" in notes.get("diversity", ""), name
        assert "levenshtein_to_template_per_template" in notes.get("bootstrap_unit", ""), name
        assert "kl_divergence_positional" in notes.get("kl", ""), name
        # The note is only true if BOTH readings are actually in the file. A run that emits the note
        # and one key would send the reader looking for a number that is not there.
        assert isinstance(report["our_interpretation"].get("kl_divergence_positional"), float), name
        assert isinstance(report["defined_by_the_paper"].get("kl_generated_vs_natural"), float), name

        model = report["defined_by_the_paper"]
        # The pair the note is about must actually be present and actually differ, or the note is
        # describing a hazard this file does not have.
        assert model["pairwise_levenshtein_pooled"] > model["pairwise_levenshtein"] > 0, name
        assert len(model["levenshtein_to_template_per_template"]) == report["n_templates"], name
        # Baselines carry the pooled diversity and NOT the within-template one, which is exactly why
        # filling a table's rows from different keys silently compares different quantities.
        for row in report["baselines"].values():
            assert "pairwise_levenshtein_pooled" in row, name


def test_figure3_log_panel_agrees_with_the_linear_panel_on_random_pairing() -> None:
    """Panel 0 and panel 1 must report the same number for random pairing, and once did not."""
    import json
    import statistics

    report = json.loads((Path(__file__).parents[2] / "metrics" / "evoflows_figure3.json").read_text())
    to_x0, pairwise = report["panel_0"], report["panel_1"]
    assert "Levenshtein to" in to_x0["title"], to_x0["title"]
    assert "pairwise Levenshtein" in pairwise["title"], pairwise["title"]

    left = statistics.mean(to_x0["by_method"]["Random pairing"])
    right = statistics.mean(pairwise["by_method"]["Random pairing"])
    assert abs(left - right) / right < 0.05, (
        f"panel 0 says {left:.2f} and panel 1 says {right:.2f} for the same quantity; "
        f"if this fires, suspect the log-axis decade reconstruction before believing either"
    )

    # The log panel must carry the restored decades, not the printed glyphs.
    assert to_x0["log_decades_restored"] is True
    assert to_x0["y_ticks"] == [2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0]
    assert to_x0["y_tick_labels_as_printed"] == [2.0, 5.0, 10.0, 100.0]
    # The linear panel must be left alone: reconstruction firing there would be a false positive.
    assert pairwise["log_decades_restored"] is False


def test_restore_log_decades_only_fires_on_repeated_labels() -> None:
    """The reconstruction rebuilds page 9's axis and declines every axis that does not need it."""
    from editjumps.pipeline.evaluate.figure_extract import restore_log_decades

    # Page 9 panel 0's real geometry: y grows downward, so the first entry is the TOP of the axis.
    printed = [(111.05, 2.0), (121.28, 100.0), (131.40, 5.0), (144.84, 2.0),
               (155.07, 10.0), (165.19, 5.0), (178.63, 2.0)]
    restored = restore_log_decades(printed)
    assert restored is not None
    assert [v for _, v in sorted(restored, key=lambda t: -t[0])] == [2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0]

    # Declines: no repeats (a linear axis), and non-positive values (never a log axis).
    assert restore_log_decades([(10.0, 50.0), (20.0, 100.0), (30.0, 150.0)]) is None
    assert restore_log_decades([(10.0, -1.0), (20.0, -1.0), (30.0, 2.0)]) is None
    assert restore_log_decades([(10.0, 2.0), (20.0, 2.0)]) is None      # under three ticks


def test_figure3_extraction_carries_the_method_labels() -> None:
    """Every panel of the tracked Figure 3 read-out names the same six methods, in x order."""
    import json

    expected = ["Random pairing", "EvoFlow (ours)", "EvoDiff-MSA", "Evotune",
                "Evotune (forced)", "Random"]
    report = json.loads((Path(__file__).parents[2] / "metrics" / "evoflows_figure3.json").read_text())
    assert len(report) == 10, f"expected 10 panels, got {len(report)}"
    for name, panel in report.items():
        assert panel["methods"] == expected, f"{name}: {panel['methods']}"
        assert set(panel["by_method"]) == set(expected), f"{name} by_method keys"
        # 6 datasets per method, and the colour and method views must hold the same numbers.
        assert all(len(v) == 6 for v in panel["by_method"].values()), name
        assert (sorted(map(tuple, panel["by_method"].values()))
                == sorted(map(tuple, panel["by_colour"].values()))), f"{name}: views disagree"
    # Random mutations is worst on distributional quality and random pairing best -- the paper's
    # caption says so, and it is what pins the two ends of the mapping.
    mmd = report["panel_8"]["by_method"]
    assert max(mmd["Random"]) > max(mmd["EvoFlow (ours)"]) > 0
    assert max(mmd["Random pairing"]) < max(mmd["EvoFlow (ours)"])


def test_consolidated_results_carry_their_comparability_metadata() -> None:
    """Every row of the consolidated table says what it may be compared with."""
    import csv

    path = Path(__file__).parents[2] / "metrics" / "all_results.csv"
    if not path.exists():                       # generated by `make consolidate`, not committed CI state
        pytest.skip("metrics/all_results.csv not generated")
    rows = list(csv.DictReader(path.open()))
    assert rows, "consolidated table is empty"

    for row in rows:
        assert row["scope"] in {"within_study", "cross_study"}, row
        # Our own runs must always say which frame and reference they used; the paper's rows cannot,
        # because it states neither, and that asymmetry is the finding rather than missing data.
        if not row["method"].startswith("PAPER"):
            if row["metric"] in {"covariance_agreement", "mip_agreement", "js_positional"}:
                assert row["alignment_length"], f"per-position metric with no alignment: {row}"
                assert row["reference_n"], f"agreement metric with no reference size: {row}"
            if row["metric"] == "spectrum_mmd":
                assert row["n_reference"], f"MMD with no reference count: {row}"

    # The scope split must not collapse: if every metric became cross_study someone has quietly
    # widened what may be compared against the published paper.
    scopes = {row["scope"] for row in rows}
    assert scopes == {"within_study", "cross_study"}, f"scope column degenerate: {scopes}"


def test_no_committed_cell_disparages_the_frame_its_own_fields_prove() -> None:
    """A cell's prose must not claim a comparability defect its own `alignment` field disproves."""
    cells = sorted(Path("metrics/disjoint").glob("*.json"))
    assert cells, "metrics/disjoint holds no cells"

    widths: dict[str, set[int]] = {}
    for path in cells:
        report = json.loads(path.read_text())
        family = "ty1" if path.stem.endswith("ty1") else "her2vh"
        text = str(report.get("alignment", ""))
        assert "L=" in text, path
        widths.setdefault(family, set()).add(int(text.split("L=")[1].split(")")[0]))

        for caveat in report.get("caveats", []):
            assert "approximately so against the editor" not in caveat, (
                f"{path} claims it is only approximately comparable with the editor, but every "
                f"cell of its family reports the same alignment width. Either the claim is stale "
                f"prose or the frames really do differ -- and if they differ, this file's own "
                f"`alignment` field is wrong too."
            )

    for family, seen in widths.items():
        assert len(seen) == 1, (
            f"{family} cells span alignment widths {sorted(seen)}; a caveat about approximate "
            f"comparability would then be TRUE and the frame test is what should fail"
        )


def test_the_disjoint_directory_holds_exactly_the_eight_documented_cells() -> None:
    """`metrics/disjoint/` must contain the eight JSON/FASTA pairs `consolidate.DISJOINT` names."""
    import subprocess

    from editjumps.pipeline.evaluate.consolidate import DISJOINT

    root = Path(__file__).parents[2]
    expected = {(root / rel).name for _, _, rel, _ in DISJOINT}
    expected |= {name.replace(".json", ".fasta") for name in expected}
    # TRACKED files, not everything on disk: a local experiment writing a scratch cell into this directory is.
    listed = subprocess.run(["git", "ls-files", "metrics/disjoint"], cwd=root,
                            capture_output=True, text=True, check=True).stdout.split()
    found = {Path(p).name for p in listed if p.endswith((".json", ".fasta"))}
    assert found == expected, (
        "metrics/disjoint/ does not hold exactly the cells consolidate.DISJOINT names.\n"
        f"  unexpected: {sorted(found - expected)}\n  missing: {sorted(expected - found)}\n"
        "Every file there is one cell of one table; anything else changes the counts the docs "
        "state over this directory."
    )
