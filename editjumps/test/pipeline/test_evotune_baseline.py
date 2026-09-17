"""The evotuned §4.2 baseline: a matched budget, and a profile that never sees the holdout."""

import json
from pathlib import Path

import pytest


def test_evotune_matched_budget_is_the_editors_own_edit_distance(tmp_path: Path) -> None:
    """§4.2's matching rule is read off the editor's run, never hand-copied. "Matching the expected."""
    from editjumps.core.evotune.substitution import matched_budget
    from editjumps.pipeline.evaluate.evotune_baseline import budget_from_metrics

    assert matched_budget(3.4) == 3 and matched_budget(3.6) == 4
    assert matched_budget(0.2) == 1, "a comparison at zero mutations compares nothing"

    path = tmp_path / "generation_eval.json"
    path.write_text(json.dumps({"defined_by_the_paper": {"levenshtein_to_template": 5.7}}))
    assert budget_from_metrics(path) == 6

    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"defined_by_the_paper": {}}))
    with pytest.raises(ValueError, match="cannot be matched"):
        budget_from_metrics(empty)


def test_evotune_corpus_never_trains_on_the_scoring_holdout(tmp_path: Path) -> None:
    """The evotuned baseline must not be scored against sequences it memorised."""
    import gzip

    from editjumps.core.family_split import split_family, usable_members
    from editjumps.pipeline.train.evotune import family_corpus, training_members

    # 40 members written twice: dedup must halve the corpus without moving the split. Residue
    # prefixes, not digits -- `training_members` rightly drops any member carrying one.
    letters = "ACDEFGHIKLMNPQRSTVWY"
    members = [letters[i // 20] + letters[i % 20] + letters * 3 for i in range(40)]
    fasta = tmp_path / "family.fasta"
    fasta.write_text("".join(f">m{i}_{copy}\n{seq}\n" for copy in range(2)
                             for i, seq in enumerate(members)))

    train_path, val_path = tmp_path / "c.train.txt.gz", tmp_path / "c.val.txt.gz"
    n_train, n_val = family_corpus(fasta, train_path, val_path, n_templates=5, holdout_size=20,
                                   split_seed=0, val_fraction=0.2, seed=0)

    with gzip.open(train_path, "rt") as fh:
        trained_on = {line.strip() for line in fh if line.strip()}
    with gzip.open(val_path, "rt") as fh:
        validated_on = {line.strip() for line in fh if line.strip()}

    from editjumps.pipeline.preprocess.pretrain.seed_homologs import read_fasta

    split = split_family(usable_members(read_fasta(fasta).values()), 5, 20, 0)
    assert not trained_on & set(split.reference), "trained on the sequences it will be scored against"
    assert not trained_on & set(split.templates), "trained on the sequences it generates from"
    assert not trained_on & validated_on
    assert n_train + n_val == len(set(split.pool)), "duplicates must collapse, once, after the split"
    assert n_val > 0 and n_train > 0

    # Off-alphabet members would tokenize to <unk> and teach the model nothing.
    assert training_members(["ACDE", "ACDE", "ACXE", ""]) == ["ACDE"]

    with pytest.raises(ValueError, match="nothing to train on"):
        family_corpus(fasta, train_path, val_path, n_templates=5, holdout_size=100, split_seed=0)
    with pytest.raises(ValueError, match="val_fraction"):
        family_corpus(fasta, train_path, val_path, val_fraction=1.0)


def test_evotune_baseline_refuses_a_profile_built_from_the_holdout(tmp_path: Path) -> None:
    """The two stages share three params; when they disagree the run must fail, not report a number.."""
    import gzip

    from editjumps.core.family_split import split_family, usable_members
    from editjumps.pipeline.evaluate.evotune_baseline import evaluate

    letters = "ACDEFGHIKLMNPQRSTVWY"
    members = [letters[i // 20] + letters[i % 20] + letters * 3 for i in range(40)]
    fasta = tmp_path / "family.fasta"
    fasta.write_text("".join(f">m{i}\n{seq}\n" for i, seq in enumerate(members)))
    split = split_family(usable_members(members), 5, 20, 0)

    leaky = tmp_path / "leaky.train.txt.gz"
    with gzip.open(leaky, "wt") as out:
        out.write("\n".join([*split.pool, split.reference[0]]) + "\n")

    # Raised before any model is loaded, so this holds in the lean env with no torch installed.
    with pytest.raises(ValueError, match="BOTH the evotuning corpus and the scoring holdout"):
        evaluate(tmp_path / "model", fasta, leaky, 5, 2, 3, 0, forced=False, holdout_size=20)


def test_evotune_baseline_generates_from_the_profile_without_a_model() -> None:
    """The whole generation loop is exercisable with a stub proposer -- no GPU, no torch."""
    from editjumps.pipeline.evaluate.evotune_baseline import generate_variants

    # A family that varies only in its last five positions: the profile must concentrate there.
    stem = "ACDEFGHIKLMNPQRSTVWY" * 3
    train = [stem + tail for tail in ("AAAAA", "CCCCC", "DDDDD", "AAACC", "CCDDA")]
    templates = [stem + "AAAAA", stem + "CCCCC"]

    def always_w(working: list[str | None], position: int, blocked: str) -> str:
        """Write W everywhere, so every mask is a visible mutation."""
        return "W"

    generated, accounting, per_template = generate_variants(
        templates, train, 3, always_w, n_variants=4, seed=0, forced=False, profile_size=10)

    assert len(generated) == 8 and len(accounting) == 8 and len(per_template) == 2
    for index, sequence in enumerate(generated):
        # Template-MAJOR: entry i came from template i // n_variants. Backwards, every baseline
        # edit distance was measured against the wrong template.
        template = templates[index // 4]
        assert len(sequence) == len(template), "a substitution baseline must preserve length"
        assert sum(a != b for a, b in zip(sequence, template, strict=True)) == 3
        # The stem carries no entropy, so eq 12 must not substitute there while weight remains.
        assert sequence[: len(stem)] == stem, "substituted inside a perfectly conserved region"
    assert all(stats["n_changed"] == 3 for stats in accounting)
    assert all(p["mean_mutations"] == 3.0 for p in per_template)


def test_evotune_baseline_report_is_complete_with_a_stubbed_forward_pass(
    monkeypatch: "pytest.MonkeyPatch", tmp_path: Path
) -> None:
    """The whole runner, end to end, with only the MLM forward pass stubbed out."""
    import gzip

    from editjumps.core.family_split import split_family, usable_members
    from editjumps.pipeline.evaluate import evotune_baseline

    letters = "ACDEFGHIKLMNPQRSTVWY"
    members = [letters[i // 20] + letters[i % 20] + letters * 4 for i in range(40)]
    fasta = tmp_path / "family.fasta"
    fasta.write_text("".join(f">m{i}\n{seq}\n" for i, seq in enumerate(members)))

    split = split_family(usable_members(members), 4, 12, 0)
    corpus = tmp_path / "train.txt.gz"
    with gzip.open(corpus, "wt") as out:
        out.write("\n".join(split.pool) + "\n")

    def stub_proposer(model_folder: Path, temperature: float, rng: "object") -> object:
        """Stand in for the evotuned MLM: always writes W, so every mask is a visible mutation."""
        assert temperature == 0.5, "the temperature must reach the model, not be dropped en route"
        return lambda working, position, blocked: "W"

    monkeypatch.setattr(evotune_baseline, "mlm_proposer", stub_proposer)
    report = evotune_baseline.evaluate(
        tmp_path / "model", fasta, corpus, 4, 3, 2, 0,
        forced=True, temperature=0.5, holdout_size=12, profile_size=10,
    )

    assert report["method"] == "evotuned_plm_forced_substitutions"
    assert report["forced_substitutions"] is True
    assert report["n_generated"] == 12 and report["n_natural_reference"] == 12
    # Forcing makes the budget exact: the one configuration where it is not an expectation.
    assert report["mutation_budget"]["mean_realised"] == 2.0
    assert report["mutation_budget"]["hit_budget_fraction"] == 1.0

    defined = report["defined_by_the_paper"]
    # Substitutions only, so the realised mutation count IS the edit distance to the right
    # template; a mismatch means the template-major ordering came apart.
    assert defined["levenshtein_to_own_template"] == 2.0
    assert set(defined) >= {"spectrum_mmd", "kl_generated_vs_natural", "mip_mean_generated",
                            "covariance_frobenius_generated", "pairwise_levenshtein",
                            "pairwise_levenshtein_pooled", "covariance_agreement", "mip_agreement"}
    assert all(isinstance(v, float | list) for v in defined.values())
    # The arrays an interval is bootstrapped from ship WITH the centre, or the interval in a table comes from.
    assert defined["levenshtein_to_template_per_template"] == [2.0] * 4
    assert defined["pairwise_levenshtein_per_template"] == [pytest.approx(defined["pairwise_levenshtein"])] * 4
    assert defined["levenshtein_to_template_per_sequence"] == [[2.0] * 3] * 4
    assert "template" in report["reading_notes"]["bootstrap_unit"]
    # Set-level metrics need the sequences to be resampled by template, so they are kept too.
    assert len(report["_sequences"]["evotuned_plm_forced_substitutions"]) == 12
    assert set(report["our_interpretation"]) == {"entropy_delta", "js_divergence",
                                                "js_divergence_positional", "profile_log_likelihood"}
    # MMD is only comparable at equal n, so the sample sizes must be recorded on every run.
    assert report["mmd_diagnostic"]["n_generated"] == 12
    assert report["mmd_diagnostic"]["n_reference"] == 12
    assert "diversity" in report["diversity_novelty"]
    assert report["esm2_pseudo_log_likelihood"] == {} and report["not_implemented"]
    assert len(report["caveats"]) == 3, "the comparability caveats must ship with the numbers"
    json.dumps(report)  # the stage writes this straight out; it must be serialisable
