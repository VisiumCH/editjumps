"""The EvoDiff-MSA §4.2 baseline, driven end to end against a fake runner."""

import json
from pathlib import Path

import pytest

from editjumps.test.fakes import _fake_shim


def test_evodiff_msa_baseline_hits_the_matched_budget_and_reveals_iteratively(tmp_path: Path) -> None:
    """The whole generation loop against a fake runner: budget, ordering and progressive reveal."""
    from editjumps.pipeline.evaluate.evodiff_msa_baseline import generate_variants

    shim, log = _fake_shim(tmp_path / "shim")
    # No W anywhere, so the fake's favourite residue is always a visible mutation.
    stem = "ACDEFGHIKLMNPQRSTVY" * 3
    members: list[str] = [stem[:i] + "A" + stem[i + 1:] for i in range(40)]
    templates: list[str] = [stem, "A" + stem[1:]]

    generated, accounting, per_template = generate_variants(
        templates, members, 3, n_variants=2, seed=0, workdir=tmp_path / "msa",
        msa_size=32, n_sequences=8, shim=shim,
    )

    assert len(generated) == 4 and len(accounting) == 4 and len(per_template) == 2
    for index, sequence in enumerate(generated):
        template = templates[index // 2]      # template-MAJOR, matching evotune_baseline
        assert len(sequence) == len(template), "a substitution baseline must preserve length"
        assert sum(a != b for a, b in zip(sequence, template, strict=True)) == 3
        assert sequence.count("W") == 3
    assert all(stats["n_changed"] == 3 and stats["hit_budget"] for stats in accounting)
    assert all(p["mean_mutations"] == 3.0 for p in per_template)
    # One forward pass per filled position, checkpoint loaded once per template — why the seam
    # is a server and not a CLI.
    assert [p["model_forward_passes"] for p in per_template] == [6, 6]
    assert [p["msa_rows"] for p in per_template] == [8, 8]

    requests = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(requests) == 12, "one fill per position per variant"
    # Progressive reveal: mask the drawn set, fill one at a time, so mask counts go 3, 2, 1.
    # A single joint pass would show 3, 3, 3.
    assert [r["masks"] for r in requests[:3]] == [3, 2, 1]
    assert all(r["query"][r["position"]] == "#" for r in requests)

    # The alignment is kept, not deleted: it is the input the whole baseline hinges on.
    written = sorted((tmp_path / "msa").glob("*.a3m"))
    assert [p.name for p in written] == ["template_0000.a3m", "template_0001.a3m"]
    first = [line for line in written[0].read_text().splitlines() if not line.startswith(">")]
    assert first[0] == templates[0]
    assert len(first) == 33, "query + msa_size aligned members"


def test_evodiff_shards_join_into_the_run_they_would_have_produced(tmp_path: Path) -> None:
    """Two shards plus the cache give the same sequences as one run, and no model call repeats."""
    from editjumps.pipeline.evaluate.evodiff_msa_baseline import generate_variants, parse_slice

    stem = "ACDEFGHIKLMNPQRSTVY" * 3
    members: list[str] = [stem[:i] + "A" + stem[i + 1:] for i in range(40)]
    templates: list[str] = [stem, "A" + stem[1:], "C" + stem[1:], "D" + stem[1:]]

    shim, whole_log = _fake_shim(tmp_path / "whole")
    whole, _accounting, whole_per = generate_variants(
        templates, members, 3, n_variants=2, seed=0, workdir=tmp_path / "a", shim=shim,
        msa_size=32, n_sequences=8)

    shim, shard_log = _fake_shim(tmp_path / "sharded")
    for text in ("2:4", "0:2"):            # out of order on purpose: order must not matter
        generate_variants(templates, members, 3, n_variants=2, seed=0, workdir=tmp_path / "b",
                          shim=shim, msa_size=32, n_sequences=8, only=parse_slice(text))
    calls_after_generation = len(shard_log.read_text().splitlines())
    joined, _accounting, joined_per = generate_variants(
        templates, members, 3, n_variants=2, seed=0, workdir=tmp_path / "b", shim=shim,
        msa_size=32, n_sequences=8)

    assert joined == whole, "a sharded run must produce the sequence set the whole run does"
    assert [p["levenshtein_to_template"] for p in joined_per] == \
        [p["levenshtein_to_template"] for p in whole_per]
    assert calls_after_generation == len(whole_log.read_text().splitlines())
    assert len(shard_log.read_text().splitlines()) == calls_after_generation, (
        "the scoring pass re-ran the model instead of reading the cache"
    )

    # A cache entry made with different settings is a MISS: same index, different budget must not
    # report the old sequences under the new one.
    rerun, _accounting, _per = generate_variants(
        templates, members, 2, n_variants=2, seed=0, workdir=tmp_path / "b", shim=shim,
        msa_size=32, n_sequences=8)
    assert rerun != whole
    assert len(shard_log.read_text().splitlines()) > calls_after_generation

    for text in ("", "3", "5:2", "2:2", "a:b", "-1:2"):
        if text == "":
            assert parse_slice(text) is None
        else:
            with pytest.raises(ValueError, match="generate-only"):
                parse_slice(text)


def test_evodiff_forced_substitutions_are_ours_and_do_change_the_output(tmp_path: Path) -> None:
    """--forced blocks the residue already there, which for this baseline is our addition."""
    from editjumps.pipeline.evaluate.evodiff_msa_baseline import generate_variants

    shim, _log = _fake_shim(tmp_path / "shim")
    template = "W" * 40
    members = ["W" * 20 + "A" * 20, "A" * 40, *["W" * 39 + "A"] * 8]

    for forced, expected in ((False, 0), (True, 2)):
        _generated, accounting, _per = generate_variants(
            [template], members, 2, n_variants=1, seed=0, workdir=tmp_path / f"msa_{forced}",
            msa_size=16, n_sequences=4, forced=forced, top_up=False, shim=shim,
        )
        assert accounting[0]["n_changed"] == expected, f"forced={forced}"
        assert accounting[0]["n_masked"] == 2, "both variants mask the same number of positions"


def test_evodiff_msa_report_is_complete_and_labels_what_is_ours(tmp_path: Path) -> None:
    """The whole runner end to end, with only the checkpoint replaced by a fake."""
    from editjumps.core.family_split import split_family, usable_members
    from editjumps.pipeline.evaluate import evodiff_msa_baseline

    shim, _log = _fake_shim(tmp_path / "shim")
    letters = "ACDEFGHIKLMNPQRSTVY"
    members = [letters[i // 19] + letters[i % 19] + letters * 4 for i in range(40)]
    fasta = tmp_path / "family.fasta"
    fasta.write_text("".join(f">m{i}\n{seq}\n" for i, seq in enumerate(members)))

    report = evodiff_msa_baseline.evaluate(
        fasta, 4, 3, 2, 0, tmp_path / "msa",
        holdout_size=12, msa_size=16, n_sequences=8, shim=shim,
    )

    assert report["method"] == "evodiff_msa" and report["mode"] == "inpaint"
    assert report["n_generated"] == 12 and report["n_natural_reference"] == 12
    assert report["model_forward_passes"] == 24, "budget 2 x 3 variants x 4 templates"
    # §4.2's matched quantity, and the flag that says whether this mode matches it at all.
    assert report["mutation_budget"]["target"] == 2
    assert report["mutation_budget"]["matched"] is True
    assert report["mutation_budget"]["mean_realised"] == 2.0
    assert report["msa"]["n_rows_conditioned_on"] == [8]
    assert "TRAIN part" in report["msa"]["source"], "the MSA must never be the scoring holdout"

    defined = report["defined_by_the_paper"]
    # Substitutions only, so the realised mutation count IS the edit distance to the right
    # template; a mismatch means the template-major ordering came apart.
    assert defined["levenshtein_to_own_template"] == 2.0
    assert set(defined) >= {"spectrum_mmd", "kl_generated_vs_natural", "mip_mean_generated",
                            "covariance_frobenius_generated", "pairwise_levenshtein",
                            "pairwise_levenshtein_pooled", "covariance_agreement", "mip_agreement"}
    assert set(report["our_interpretation"]) == {"entropy_delta", "js_divergence",
                                                "js_divergence_positional", "profile_log_likelihood"}
    assert report["mmd_diagnostic"]["n_generated"] == 12, "MMD is only comparable at equal n"
    assert "diversity" in report["diversity_novelty"]
    # The honesty contract: both lists ship with the numbers, and the model is on the THEIRS side.
    assert len(report["ours_not_theirs"]) == 4 and len(report["theirs_not_ours"]) == 3
    assert any("released weights" in line for line in report["theirs_not_ours"])
    assert any("UNIFORMLY" in line for line in report["ours_not_theirs"])
    assert len(report["caveats"]) == 4
    json.dumps(report)  # the stage writes this straight out; it must be serialisable

    # A family too small to leave a train part must say so, not condition on nothing.
    with pytest.raises(ValueError, match="no train part"):
        evodiff_msa_baseline.evaluate(fasta, 4, 3, 2, 0, tmp_path / "msa2",
                                      holdout_size=36, shim=shim)
    # sanity: that really is the whole holdout, i.e. the guard fires for the stated reason
    assert split_family(usable_members(members), 4, 36, 0).pool == []


def test_evodiff_unconditional_mode_is_theirs_and_is_flagged_as_unmatched(tmp_path: Path) -> None:
    """Their own entry point is runnable, and the report refuses to call it budget-matched.."""
    from editjumps.pipeline.evaluate import evodiff_msa_baseline

    shim, log = _fake_shim(tmp_path / "shim")
    letters = "ACDEFGHIKLMNPQRSTVY"
    members = [letters[i // 19] + letters[i % 19] + letters * 4 for i in range(40)]
    fasta = tmp_path / "family.fasta"
    fasta.write_text("".join(f">m{i}\n{seq}\n" for i, seq in enumerate(members)))

    report = evodiff_msa_baseline.evaluate(
        fasta, 3, 2, 4, 0, tmp_path / "msa", mode="unconditional",
        holdout_size=10, msa_size=16, n_sequences=8, shim=shim,
    )
    assert report["mutation_budget"]["matched"] is False
    assert report["mutation_budget"]["target"] is None
    assert "unconditional" in report["mutation_budget"]["budget_mode"]
    assert any("NOT budget-matched" in line for line in report["caveats"])
    # One call to their function per variant, and no per-position fills at all.
    ops = [json.loads(line)["op"] for line in log.read_text().splitlines()]
    assert ops == ["generate_query"] * 6

    with pytest.raises(ValueError, match="mode='nonsense'"):
        evodiff_msa_baseline.generate_variants([], [], 1, 1, 0, tmp_path, mode="nonsense")


def test_evodiff_alignments_of_two_runs_never_share_a_directory(tmp_path: Path) -> None:
    """Every input that changes the alignments changes the directory they are written to."""
    from editjumps.pipeline.evaluate.evodiff_msa_baseline import run_workdir

    ty1 = Path("data/interim/seed_families/Anti-SARS-CoV-2_VHH_Ty1.fasta")
    her2 = Path("data/interim/seed_families/Anti-HER2_scFv_VH_trastuzumab.fasta")
    pairs = Path("data/pretrain/oas_homolog_pairs.tsv.gz")
    directories = {
        run_workdir(tmp_path, family, mode, seed, disjoint, target)
        for family in (ty1, her2)
        for mode in ("inpaint", "unconditional")
        for seed in (0, 1)
        for disjoint in (None, pairs)
        for target in (None, 130)
    }
    assert len(directories) == 2 * 2 * 2 * 2 * 2
    # Deterministic: the same configuration must reuse its own directory rather than accumulate one
    # per run, since the alignments are kept to be inspected.
    assert run_workdir(tmp_path, ty1, "inpaint", 0, pairs, None) == \
        run_workdir(tmp_path, ty1, "inpaint", 0, pairs, None)


def test_a_scaffolded_evodiff_run_grows_to_the_target_with_a_stubbed_msa_model(tmp_path: Path) -> None:
    """The whole EvoDiff growth path, end to end, against the fake runner: width, fill, accounting."""
    from editjumps.pipeline.evaluate import evodiff_msa_baseline

    shim, log = _fake_shim(tmp_path / "fake")
    template = "QVQLVESGGGLVQPGGSLRLSCAASGFTFS"
    members = [template[:10] + "PPP" + template[10:] for _ in range(4)] + [template.replace("G", "A")]
    target = len(template) + 4

    generated, accounting, per_template = evodiff_msa_baseline.generate_variants(
        [template], members, 2, 2, 0, tmp_path / "work",
        n_sequences=2, msa_size=8, shim=shim, target_length=target,
    )
    assert len(generated) == 2
    for sequence, stats in zip(generated, accounting, strict=True):
        assert len(sequence) == target, "a scaffolded run must return the target length"
        assert stats["n_inserted"] == 4 and stats["at_target"]
    assert per_template[0]["model_forward_passes"] > 0

    # The alignment handed to the model is the widened one, query first and rectangular.
    a3m = (tmp_path / "work" / "template_0000.a3m").read_text().splitlines()
    rows = [line for line in a3m if not line.startswith(">")]
    assert {len(row) for row in rows} == {target}
    assert rows[0].count("-") >= 4, "the query row must read the opened columns as gaps"

    # Every fill the model was asked for was at the widened width.
    filled = [json.loads(line) for line in log.read_text().splitlines()]
    assert filled, "no forward pass reached the runner"
    assert {len(entry["query"]) for entry in filled} == {target}

    # Without a target the same call is the fixed-width baseline, unchanged.
    same, _, _ = evodiff_msa_baseline.generate_variants(
        [template], members, 2, 1, 0, tmp_path / "plain",
        n_sequences=2, msa_size=8, shim=shim,
    )
    assert len(same[0]) == len(template)
