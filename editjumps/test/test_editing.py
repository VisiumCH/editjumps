"""`editjumps edit` / `from editjumps import edit`: sequence in, edited variants out."""

import json
import random
from pathlib import Path

import pytest

#: A real therapeutic antibody VH domain (trastuzumab heavy-chain variable region), used so the end-to-end.
TRASTUZUMAB_VH = (
    "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRW"
    "GGDGFYAMDYWGQGTLVTVSS"
)


def _stub_the_gpu_half_of_edit(monkeypatch: "pytest.MonkeyPatch", edits_per_call: int = 5) -> None:
    """Replace the checkpoint-dependent half of `editing.edit_report`."""
    import sys
    import types

    alphabet = "ACDEFGHIKLMNPQRSTVWY"

    class _Tokenizer:
        """Character-level stand-in: one token per residue, so decode(encode(s)) == s."""

        def __call__(self, text: str) -> dict:
            return {"input_ids": [ord(c) for c in text]}

        def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
            # Spaces because the caller strips them, matching the real tokenizer's output.
            return " ".join(chr(i) for i in ids)

    class _TokenizerFactory:
        @staticmethod
        def from_pretrained(path: str) -> "_Tokenizer":
            return _Tokenizer()

    class _Model:
        def to(self, device: object) -> "_Model":
            return self

        def eval(self) -> "_Model":
            return self

        @classmethod
        def load_trained(cls, folder: object, rate_head: str = "linear", q_head: str = "fresh") -> "_Model":
            return cls()

    def _sample_edits(model: object, ids: list[int], n_steps: int = 50, clock: float | None = None,
                      rng: "random.Random | None" = None) -> list[int]:
        # Substitutions only, so the realised edit distance is exactly `edits_per_call` and the
        # tests can assert on it. The real sampler also inserts and deletes.
        rng = rng or random.Random(0)
        out = list(ids)
        for position in rng.sample(range(len(out)), edits_per_call):
            current = chr(out[position])
            out[position] = ord(rng.choice([a for a in alphabet if a != current]))
        return out

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.__dict__["AutoTokenizer"] = _TokenizerFactory
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    fake_evoflows = types.ModuleType("editjumps.pipeline.train.evoflows")
    fake_evoflows.__dict__.update({
        "EvoFlowsModel": _Model,
        "pick_device": lambda: "cpu",
        "sample_edits": _sample_edits,
    })
    monkeypatch.setitem(sys.modules, "editjumps.pipeline.train.evoflows", fake_evoflows)


def _fake_model_folder(tmp_path: Path) -> Path:
    """Create a folder that passes `resolve_model_folder`'s artifact check."""
    folder = tmp_path / "editor"
    (folder / "encoder").mkdir(parents=True)
    (folder / "evoflows_model.pt").write_bytes(b"")
    return folder


def test_edit_is_importable_as_a_one_liner_from_the_package_root() -> None:
    """`from editjumps import edit` works, and works without the optional `train` group."""
    import subprocess
    import sys

    import editjumps

    assert callable(editjumps.edit)
    assert {"edit", "Variant", "EditReport", "CheckpointNotFoundError"} <= set(editjumps.__all__)

    # Checked in a FRESH interpreter, not against this one's `sys.modules`: by the time this test runs.
    probe = subprocess.run(
        [sys.executable, "-c",
         "import sys, editjumps; assert callable(editjumps.edit);"
         " print(sorted(m for m in sys.modules if m in {'torch', 'transformers'}))"],
        capture_output=True, text=True, check=False,
    )
    assert probe.returncode == 0, probe.stderr
    assert probe.stdout.strip() == "[]", f"importing editjumps pulled in the train group: {probe.stdout}"


def test_edit_without_a_checkpoint_says_exactly_what_to_do_and_does_not_traceback(
    monkeypatch: "pytest.MonkeyPatch", tmp_path: Path
) -> None:
    """The no-weights path - the one every new user outside the org hits first."""
    from typer.testing import CliRunner

    from editjumps import edit
    from editjumps.editing import CHECKPOINT_BUCKET, CheckpointNotFoundError
    from editjumps.main import app

    monkeypatch.chdir(tmp_path)  # so the default model folder is genuinely absent

    with pytest.raises(CheckpointNotFoundError) as raised:
        edit(TRASTUZUMAB_VH, n=2)
    message = str(raised.value)

    assert "restore-editor" in message, "the message must name the command that fixes this"
    assert "faithful-appendixa" in message, "and which run tag to restore"
    assert CHECKPOINT_BUCKET in message, "and the bucket, whatever DVC_BUCKET is set to"
    assert "PRIVATE" in message, "and that the bucket is private, so the reader stops retrying"
    assert "train-edit-flows" in message, "and the way out for someone outside the org"
    assert "data/pretrain/edit_flows_restored" in message, "and where it looked"

    # The CLI turns that into an exit code and a message, not a traceback.
    result = CliRunner().invoke(app, ["edit", "--sequence", TRASTUZUMAB_VH, "--n", "2"])
    assert result.exit_code == 1, "a missing checkpoint is a clean failure, not a crash"
    assert "Traceback" not in result.output
    assert "restore-editor" in result.output
    # The refusal is total: no sequences were emitted as a consolation prize.
    assert TRASTUZUMAB_VH not in result.output


def test_edit_returns_variants_with_their_realised_edit_distance(
    monkeypatch: "pytest.MonkeyPatch", tmp_path: Path
) -> None:
    """End to end on a real antibody VH, with the checkpoint stubbed at the `load_trained` boundary."""
    from editjumps import edit

    _stub_the_gpu_half_of_edit(monkeypatch, edits_per_call=5)
    folder = _fake_model_folder(tmp_path)

    variants = edit(TRASTUZUMAB_VH, n=6, edits=5, model=folder)

    assert len(variants) == 6
    for variant in variants:
        assert set(variant.sequence) <= set("ACDEFGHIKLMNPQRSTVWY")
        assert variant.sequence != TRASTUZUMAB_VH, "an unedited 'variant' is not a variant"
        # The stub substitutes exactly 5 residues, so the measured distance must recover that -
        # i.e. the distance is measured against the input, not carried over from `edits`.
        assert variant.edit_distance == 5
    assert len({v.sequence for v in variants}) > 1, "every variant identical means the seed is not advancing"


def test_edit_is_reproducible_and_a_smaller_n_is_a_prefix(
    monkeypatch: "pytest.MonkeyPatch", tmp_path: Path
) -> None:
    """Variant `i` is seeded `seed + i`, so runs repeat and `n` only ever truncates."""
    from editjumps import edit

    _stub_the_gpu_half_of_edit(monkeypatch)
    folder = _fake_model_folder(tmp_path)

    ten = [v.sequence for v in edit(TRASTUZUMAB_VH, n=10, model=folder, seed=7)]
    three = [v.sequence for v in edit(TRASTUZUMAB_VH, n=3, model=folder, seed=7)]

    assert three == ten[:3], "a smaller n must be a prefix, not a fresh draw"
    assert [v.sequence for v in edit(TRASTUZUMAB_VH, n=10, model=folder, seed=7)] == ten
    assert [v.sequence for v in edit(TRASTUZUMAB_VH, n=10, model=folder, seed=8)] != ten


def test_the_edit_budget_is_a_clock_translation_and_says_it_is_approximate(
    monkeypatch: "pytest.MonkeyPatch", tmp_path: Path
) -> None:
    """`--edits` becomes `clock = edits / lambda_bar`, and an explicit clock overrides it."""
    from editjumps import editing

    _stub_the_gpu_half_of_edit(monkeypatch)
    folder = _fake_model_folder(tmp_path)

    # The relationship documented in docs/findings.md: edits ~= clock * lambda_bar.
    assert editing.clock_for_edits(5, lambda_bar=0.1) == pytest.approx(50.0)
    low, high = editing.LAMBDA_BAR_RANGE
    assert low <= editing.DEFAULT_LAMBDA_BAR <= high, "the default must sit inside the measured spread"
    assert high / low > 2, "the spread is wide; if it ever narrows, the 'approximate' wording can soften"

    asked = editing.edit_report(TRASTUZUMAB_VH, n=2, edits=4, model=folder)
    assert asked.clock == pytest.approx(4 / editing.DEFAULT_LAMBDA_BAR)
    assert asked.requested_edits == 4

    # An explicit clock is the mechanism itself, so it is passed straight through and no
    # translation is recorded.
    given = editing.edit_report(TRASTUZUMAB_VH, n=2, clock=40.0, edits=4, model=folder)
    assert given.clock == 40.0, "an explicit clock must win over --edits"
    assert given.requested_edits is None

    # The approximation is stated where a user will read it, not only in the commit message.
    for text in (editing.edit.__doc__, editing.main.__doc__):
        assert text is not None
        assert "APPROXIMATE" in text.upper()


def test_edit_promises_no_quality_score_anywhere_it_could_be_mistaken_for_one(
    monkeypatch: "pytest.MonkeyPatch", tmp_path: Path
) -> None:
    """No confidence number is returned, and the JSON says so rather than omitting the field. `edit`."""
    from editjumps import editing

    _stub_the_gpu_half_of_edit(monkeypatch)
    folder = _fake_model_folder(tmp_path)

    payload = editing.edit_report(TRASTUZUMAB_VH, n=2, model=folder).as_dict()
    assert payload["quality_score"] is None
    assert "holdout" in str(payload["quality_score_note"])
    # No field that a caller could mistake for a fitness/confidence reading.
    assert not {"score", "confidence", "fitness", "likelihood", "probability"} & set(payload)

    for text in (editing.edit.__doc__, editing.main.__doc__):
        assert text is not None
        assert "holdout" in text, "the reason there is no score belongs where the user reads"

    variant_fields = set(editing.Variant.__dataclass_fields__)
    assert variant_fields == {"sequence", "edit_distance"}, f"a variant is these two things; got {variant_fields}"


def test_edit_accepts_a_fasta_path_as_well_as_a_raw_string(
    monkeypatch: "pytest.MonkeyPatch", tmp_path: Path
) -> None:
    """Sequence in means either form, and an ambiguous FASTA is refused rather than guessed."""
    from editjumps import edit, editing

    _stub_the_gpu_half_of_edit(monkeypatch)
    folder = _fake_model_folder(tmp_path)

    one = tmp_path / "one.fasta"
    one.write_text(f">trastuzumab_VH description here\n{TRASTUZUMAB_VH[:60]}\n{TRASTUZUMAB_VH[60:]}\n")

    # Wrapped lines are joined, and the record id is carried through to the report.
    sequence, source_id = editing.read_sequence(one)
    assert sequence == TRASTUZUMAB_VH
    assert source_id == "trastuzumab_VH"
    from_file = [v.sequence for v in edit(one, n=3, model=folder, seed=1)]
    from_string = [v.sequence for v in edit(TRASTUZUMAB_VH, n=3, model=folder, seed=1)]
    assert from_file == from_string, "the same sequence by either route must give the same variants"

    two = tmp_path / "two.fasta"
    two.write_text(f">VH\n{TRASTUZUMAB_VH}\n>VL\n{TRASTUZUMAB_VH}\n")
    with pytest.raises(ValueError, match="exactly one FASTA record"):
        edit(two, n=1, model=folder)

    # A joined VH.VL pair is a legal single sequence: `.` is the trained-on separator.
    joined, _ = editing.read_sequence(f"{TRASTUZUMAB_VH}.{TRASTUZUMAB_VH}")
    assert joined.count(".") == 1

    # A non-existent path is not silently treated as a sequence.
    with pytest.raises(ValueError, match="not an amino-acid sequence"):
        edit("no/such/file.fasta", n=1, model=folder)
    with pytest.raises(ValueError, match="empty sequence"):
        edit("   ", n=1, model=folder)


def test_edit_json_is_consumable_without_parsing_prose(
    monkeypatch: "pytest.MonkeyPatch", tmp_path: Path
) -> None:
    """`--json` emits one JSON object on stdout, carrying the settings that explain the numbers."""
    from typer.testing import CliRunner

    from editjumps.main import app

    _stub_the_gpu_half_of_edit(monkeypatch, edits_per_call=5)
    folder = _fake_model_folder(tmp_path)

    result = CliRunner().invoke(
        app, ["edit", "--sequence", TRASTUZUMAB_VH, "--n", "4", "--edits", "5",
              "--model", str(folder), "--json"]
    )
    assert result.exit_code == 0, result.output

    payload = json.loads(result.stdout)
    assert payload["sequence"] == TRASTUZUMAB_VH
    assert payload["n_variants"] == 4 == len(payload["variants"])
    assert [v["edit_distance"] for v in payload["variants"]] == [5, 5, 5, 5]
    assert payload["edit_distance_mean"] == pytest.approx(5.0)
    assert payload["requested_edits"] == 5
    assert payload["clock"] == pytest.approx(5 / 0.0990, rel=1e-3)
    assert payload["rate_head"] == "mlp" and payload["q_head"] == "esm_lm_head"
    assert payload["model"] == str(folder)

    # The human path stays human: it reports what was realised, not what was asked for.
    plain = CliRunner().invoke(
        app, ["edit", "--sequence", TRASTUZUMAB_VH, "--n", "4", "--model", str(folder)]
    )
    assert plain.exit_code == 0, plain.output
    assert "realised mean 5.0" in plain.output
    assert "approximate" in plain.output
    assert "No quality score" in plain.output


def test_the_default_heads_match_the_run_tag_the_paper_reports() -> None:
    """The documented default checkpoint and its two head names agree with the Makefile. `load_trained`."""
    from editjumps import editing
    from editjumps.core.edit_flows.stages import Q_HEADS, RATE_HEADS

    assert editing.DEFAULT_RATE_HEAD in RATE_HEADS
    assert editing.DEFAULT_Q_HEAD in Q_HEADS

    makefile = Path("Makefile").read_text()
    target = makefile.split("jobs-train-faithful:")[1].split("\n\n")[0]
    assert editing.DEFAULT_RUN_TAG in target, "the default run tag must be the one that target trains"
    assert f"RATE_HEAD={editing.DEFAULT_RATE_HEAD}" in target
    assert f"Q_HEAD={editing.DEFAULT_Q_HEAD}" in target


def test_a_half_written_model_folder_is_refused_before_torch_sees_it(tmp_path: Path) -> None:
    """A folder missing either artifact fails with the restore message, not a torch error."""
    from editjumps.editing import CheckpointNotFoundError, resolve_model_folder

    complete = _fake_model_folder(tmp_path)
    assert resolve_model_folder(complete) == complete

    no_state_dict = tmp_path / "partial"
    (no_state_dict / "encoder").mkdir(parents=True)
    with pytest.raises(CheckpointNotFoundError, match="restore-editor"):
        resolve_model_folder(no_state_dict)

    no_encoder = tmp_path / "partial2"
    no_encoder.mkdir()
    (no_encoder / "evoflows_model.pt").write_bytes(b"")
    with pytest.raises(CheckpointNotFoundError, match="restore-editor"):
        resolve_model_folder(no_encoder)

    with pytest.raises(CheckpointNotFoundError, match="restore-editor"):
        resolve_model_folder(tmp_path / "nothing-here")
