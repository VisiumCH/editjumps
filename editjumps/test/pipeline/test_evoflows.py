"""The Edit Flows training stage: dataset, model, loss, sampler and the validation loop."""

import inspect
import random
from pathlib import Path

import pytest


def test_homolog_pair_dataset_yields_a_consistent_aligned_example() -> None:
    """The dataset tokenizes+aligns a pair so x_t == strip(z_t) and z_t, z_1 line up."""
    pytest.importorskip("torch")
    import gzip
    import tempfile
    from pathlib import Path

    from editjumps.core.edit_flows.path import strip_epsilon
    from editjumps.core.edit_flows.stages import EditFlowConfig
    from editjumps.pipeline.train.evoflows import HomologPairDataset

    class FakeTokenizer:
        """Char-level tokenizer with BOS=0 / EOS=1, enough for the dataset contract."""

        def __call__(self, seq: str) -> dict:
            return {"input_ids": [0] + [ord(c) % 20 + 2 for c in seq] + [1]}

    with tempfile.TemporaryDirectory() as d:
        pairs_path = Path(d) / "pairs.tsv.gz"
        with gzip.open(pairs_path, "wt") as fh:
            fh.write("ACDEF\tACDF\n")  # a single-deletion homolog pair
        ex = HomologPairDataset(pairs_path, FakeTokenizer(), EditFlowConfig(), vocab_size=25, seed=0)[0]

    assert ex["x_t"] == strip_epsilon(ex["z_t"])  # current seq is z_t with gaps removed
    assert len(ex["z_t"]) == len(ex["z_1"])  # aligned to equal length
    assert ex["dkappa"] == 1.0 and 0.0 <= ex["kappa"] <= 1.0


def test_evoflows_model_forward_shapes_and_loss_backprop() -> None:
    """A tiny (untrained) EvoFlows model produces correctly-shaped rates that the loss can backprop."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers import EsmConfig, EsmModel  # ty: ignore[unresolved-import]

    from editjumps.core.edit_flows.loss import edit_flow_loss
    from editjumps.core.edit_flows.path import EPS, strip_epsilon
    from editjumps.pipeline.train.evoflows import EvoFlowsModel

    vocab = 8
    cfg = EsmConfig(
        vocab_size=vocab, hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64,
        pad_token_id=1,  # ESM position-id logic needs a padding index (real checkpoints set it)
    )
    model = EvoFlowsModel(EsmModel(cfg), hidden_size=32, vocab_size=vocab, time_dim=16)

    # Aligned columns: insert 5 after x_t idx1, delete x_t idx2. x_t = [0,2,4].
    z_t = [0, 2, EPS, 4]
    z_1 = [0, 2, 5, EPS]
    x_t = strip_epsilon(z_t)
    il, iq, dl, sl, sq = model(torch.tensor(x_t), t=0.5)
    length = len(x_t)
    assert il.shape == (length,) and dl.shape == (length,) and sl.shape == (length,)
    assert iq.shape == (length, vocab) and sq.shape == (length, vocab)

    loss = edit_flow_loss(z_t, z_1, 0.5, 1.0, il, iq, dl, sl, sq)
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads)  # gradient reached the model


def test_batched_forward_gives_the_same_gradients_as_accumulating_one_at_a_time() -> None:
    """Batching must be a speedup, not a change of objective."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers import EsmConfig, EsmModel  # ty: ignore[unresolved-import]

    from editjumps.core.edit_flows.loss import edit_flow_loss
    from editjumps.core.edit_flows.path import EPS, strip_epsilon
    from editjumps.pipeline.train.evoflows import EvoFlowsModel, pad_batch

    vocab, pad_id = 8, 1
    cfg = EsmConfig(vocab_size=vocab, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                    intermediate_size=64, pad_token_id=pad_id)
    torch.manual_seed(0)
    model = EvoFlowsModel(EsmModel(cfg), hidden_size=32, vocab_size=vocab, time_dim=16)
    # eval(), because ESM dropout makes the two paths draw different masks in train mode — so a
    # batched training run is not step-for-step identical to an accumulating one.
    model.eval()

    # Deliberately different lengths (so padding is exercised) and different times per row.
    examples: list[dict] = [
        {"z_t": [0, 2, EPS, 4], "z_1": [0, 2, 5, EPS], "t": 0.25, "kappa": 0.25, "dkappa": 1.0},
        {"z_t": [0, 3, 4, 6, EPS], "z_1": [0, 3, EPS, 6, 7], "t": 0.80, "kappa": 0.80, "dkappa": 1.0},
        {"z_t": [0, 5], "z_1": [0, 6], "t": 0.50, "kappa": 0.50, "dkappa": 1.0},
    ]
    for example in examples:
        example["x_t"] = strip_epsilon(example["z_t"])
    assert len({len(e["x_t"]) for e in examples}) > 1, "the batch must be ragged or padding is untested"

    def grads_from(loss: "torch.Tensor") -> list["torch.Tensor"]:
        model.zero_grad()
        loss.backward()
        return [p.grad.detach().clone() for p in model.parameters() if p.grad is not None]

    # Reference: one sequence per forward.
    accumulated = torch.zeros(())
    for example in examples:
        rates = model(torch.tensor(example["x_t"]), example["t"])
        accumulated = accumulated + edit_flow_loss(
            example["z_t"], example["z_1"], example["kappa"], example["dkappa"], *rates
        )
    reference_loss = accumulated / len(examples)
    reference_grads = grads_from(reference_loss)

    # Batched: one padded forward, per-row slices into the same loss.
    input_ids, attention_mask, lengths = pad_batch(examples, pad_id, "cpu")
    batched_rates = model.forward_batch(input_ids, attention_mask, [e["t"] for e in examples])
    batched = torch.zeros(())
    for row, (example, n) in enumerate(zip(examples, lengths, strict=True)):
        batched = batched + edit_flow_loss(
            example["z_t"], example["z_1"], example["kappa"], example["dkappa"],
            *(rate[row, :n] for rate in batched_rates),
        )
    batched_loss = batched / len(examples)
    batched_grads = grads_from(batched_loss)

    assert torch.allclose(reference_loss, batched_loss, atol=1e-5), (
        f"loss changed: {reference_loss.item()} vs {batched_loss.item()}"
    )
    assert len(reference_grads) == len(batched_grads)
    for reference, batched_grad in zip(reference_grads, batched_grads, strict=True):
        assert torch.allclose(reference, batched_grad, atol=1e-5), "gradients differ - not a pure speedup"


def test_training_job_ships_the_model_folder_not_only_the_checkpoint(tmp_path: Path) -> None:
    """A run's weights must leave the VM in the layout that can be loaded, not only as resume state.."""
    import inspect

    yaml_text = Path("deploy/gcp/train.sky.yaml").read_text()
    assert "data/pretrain/edit_flows_$RUN_TAG" in yaml_text, "the trained model folder must be copied off the VM"
    assert "models/edit_flows/$RUN_TAG" in yaml_text

    pytest.importorskip("torch")
    from editjumps.pipeline.train.evoflows import EvoFlowsModel, restore_model_folder

    written = inspect.getsource(restore_model_folder)
    read = inspect.getsource(EvoFlowsModel.load_trained.__func__)
    for artefact in ("encoder", "evoflows_model.pt"):
        assert artefact in written and artefact in read, f"{artefact} must be both written and read"
    # The tokenizer is what makes the folder self-contained; eval reloads it from there.
    assert "AutoTokenizer" in written, "restore must save the tokenizer, as the training path does"


def test_restore_rejects_a_file_that_is_not_a_training_checkpoint(tmp_path: Path) -> None:
    """Restoring from the wrong .pt must fail on the missing key, not deep inside model building."""
    torch = pytest.importorskip("torch")
    from editjumps.pipeline.train.evoflows import restore_model_folder

    not_a_checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"state_dict": {}, "epoch": 3}, not_a_checkpoint)  # plausible, but not ours
    with pytest.raises(KeyError, match="not a training-state checkpoint"):
        restore_model_folder(not_a_checkpoint, tmp_path / "out", "some/trunk")


def test_sample_edits_preserves_bos_and_terminates() -> None:
    """Sampling from a tiny model returns a valid token list, keeps BOS, and respects max_len."""
    import random

    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers import EsmConfig, EsmModel  # ty: ignore[unresolved-import]

    from editjumps.pipeline.train.evoflows import EvoFlowsModel, sample_edits

    vocab = 8
    cfg = EsmConfig(
        vocab_size=vocab, hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64,
        pad_token_id=1,
    )
    model = EvoFlowsModel(EsmModel(cfg), hidden_size=32, vocab_size=vocab, time_dim=16)
    torch.manual_seed(0)
    out = sample_edits(model, [0, 2, 3, 4, 5], n_steps=10, rng=random.Random(0), max_len=50)
    assert isinstance(out, list) and len(out) >= 1
    assert out[0] == 0  # BOS preserved at position 0
    assert all(0 <= tok < vocab for tok in out)  # valid token ids
    assert len(out) <= 50


def test_training_state_save_load_round_trips(tmp_path: Path) -> None:
    """Save/load the full training state: step + a parameter tensor round-trip (local only, no GCS)."""
    torch = pytest.importorskip("torch")
    from editjumps.pipeline.train.evoflows import load_training_state, save_training_state

    model = torch.nn.Linear(4, 3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model(torch.zeros(2, 4)).sum().backward()
    optimizer.step()  # populate non-trivial optimizer state
    saved_weight = model.weight.detach().clone()

    path = tmp_path / "checkpoint.pt"
    save_training_state(path, step=7, model=model, optimizer=optimizer)

    with torch.no_grad():
        model.weight.add_(1.0)  # mutate so a successful load must overwrite it
    assert not torch.equal(model.weight, saved_weight)

    resumed = load_training_state(path, model, optimizer)
    assert resumed.step == 7
    assert torch.equal(model.weight, saved_weight)


def test_training_state_carries_early_stopping_across_resume(tmp_path: Path) -> None:
    """best-val tracking and the patience counter survive a save/load, so preemption cannot reset them."""
    torch = pytest.importorskip("torch")
    from editjumps.pipeline.train.evoflows import ResumeState, load_training_state, save_training_state

    model = torch.nn.Linear(4, 3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_weights = {k: v.detach().clone() for k, v in model.state_dict().items()}

    path = tmp_path / "checkpoint.pt"
    save_training_state(
        path, step=12, model=model, optimizer=optimizer,
        best=ResumeState(step=12, best_val=0.25, best_step=8, best_state=best_weights,
                         patience_ref=0.25, stale_evals=2),
    )

    resumed = load_training_state(path, model, optimizer)
    assert (resumed.best_val, resumed.best_step) == (0.25, 8)
    assert (resumed.patience_ref, resumed.stale_evals) == (0.25, 2)
    assert resumed.best_state is not None
    assert torch.equal(resumed.best_state["weight"], best_weights["weight"])


def test_training_state_reads_a_checkpoint_written_before_early_stopping(tmp_path: Path) -> None:
    """A checkpoint from an older run lacks the early-stopping keys and must load with fresh defaults."""
    torch = pytest.importorskip("torch")
    from editjumps.pipeline.train.evoflows import load_training_state

    model = torch.nn.Linear(4, 3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    path = tmp_path / "legacy.pt"
    torch.save(  # exactly the five keys the pre-ResumeState writer produced
        {
            "step": 3,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "python_rng": random.getstate(),
            "torch_rng": torch.get_rng_state(),
        },
        path,
    )

    resumed = load_training_state(path, model, optimizer)
    assert resumed.step == 3
    assert resumed.best_val == float("inf") and resumed.best_step == -1
    assert resumed.best_state is None and resumed.stale_evals == 0


def test_sample_edits_respects_the_framework_mask() -> None:
    """A frozen position is never deleted/substituted and takes no insertion, even at huge rates."""
    import random

    torch = pytest.importorskip("torch")
    from editjumps.pipeline.train.evoflows import sample_edits

    class EditEverythingModel:
        """Rates that delete+insert everywhere, so only the mask can hold ground."""

        def parameters(self) -> object:
            return iter([torch.zeros(1)])

        def eval(self) -> None:
            return None

        def train(self) -> None:
            return None

        def __call__(self, ids: list[int], t: float) -> tuple:
            length, vocab = len(ids), 5
            hot = torch.zeros(length, vocab)
            hot[:, 4] = 1.0  # any insert/substitute emits token 4
            return torch.full((length,), 9.0), hot, torch.full((length,), 9.0), torch.zeros(length), hot

    x = [0, 1, 2, 3]  # 0 = BOS (auto-protected)
    mask = [False, True, False, True]  # additionally freeze position 2
    model = EditEverythingModel()
    out = sample_edits(model, x, n_steps=1, rng=random.Random(0), mask=mask)  # ty: ignore[invalid-argument-type]

    assert out[0] == 0  # BOS preserved at the front
    assert 2 in out  # frozen residue survived the edit-everything rates
    assert 1 not in out and 3 not in out  # editable originals were deleted
    assert set(out) <= {0, 2, 4}  # only frozen tokens + inserted token 4 can appear


def test_a_larger_clock_realises_more_edits() -> None:
    """The clock must change the number of edits that actually land, not just be recorded."""
    import random

    torch = pytest.importorskip("torch")
    from editjumps.pipeline.train.evoflows import sample_edits

    class SubstituteOnlyModel:
        """Λ_sub = 1 everywhere, no insertions or deletions; every substitution writes token 5."""

        def parameters(self) -> object:
            return iter([torch.zeros(1)])

        def eval(self) -> None:
            return None

        def train(self) -> None:
            return None

        def __call__(self, ids: list[int], t: float) -> tuple:
            length, vocab = len(ids), 8
            hot = torch.zeros(length, vocab)
            hot[:, 5] = 1.0
            zeros = torch.zeros(length)
            return zeros, hot, zeros, torch.ones(length), hot

    x = [0] + [1, 2, 3, 4] * 10  # 0 = BOS (protected), 40 editable positions
    model = SubstituteOnlyModel()

    def edited_positions(clock: float | None) -> float:
        """Mean number of changed positions over five seeds (the counts are Binomial-noisy)."""
        counts = []
        for seed in range(5):
            out = sample_edits(model, x, n_steps=50, rng=random.Random(seed), clock=clock)  # ty: ignore[invalid-argument-type]
            assert len(out) == len(x)  # substitution-only: length is preserved
            counts.append(sum(token == 5 for token in out))
        return sum(counts) / len(counts)

    small, large, unclocked = edited_positions(1.0), edited_positions(20.0), edited_positions(None)
    assert small < large, f"clock 20 made no more edits than clock 1 ({large} vs {small})"
    assert large < unclocked, f"unclocked did not out-edit clock 20 ({unclocked} vs {large})"
    # Bands, not point values (~1 / ~14 / ~26 of 40). A fake model: only the knob is measured.
    assert small <= 5 and large >= 5
    assert unclocked >= 20  # unscaled rates rewrite most of the sequence - the pre-clock default


def test_sample_edits_provenance_is_opt_in_and_both_samplers_share_a_signature() -> None:
    """With torch present: `provenance=False` returns a bare list, `True` returns (ids, origins)."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers import EsmConfig, EsmModel  # ty: ignore[unresolved-import]

    from editjumps.pipeline.train.evoflows import EvoFlowsModel, sample_edits, sample_edits_gillespie

    vocab = 8
    cfg = EsmConfig(
        vocab_size=vocab, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
        intermediate_size=64, pad_token_id=1,
    )
    model = EvoFlowsModel(EsmModel(cfg), hidden_size=32, vocab_size=vocab, time_dim=16)
    torch.manual_seed(0)
    x0 = [0, 2, 3, 4, 5, 6, 7, 2, 3, 4]
    for sampler in (sample_edits, sample_edits_gillespie):
        plain = sampler(model, x0, n_steps=8, rng=random.Random(0), clock=10.0, max_len=40)
        assert isinstance(plain, list) and plain[0] == 0
        traced = sampler(model, x0, n_steps=8, rng=random.Random(0), clock=10.0, max_len=40,
                         provenance=True)
        assert isinstance(traced, tuple)
        ids, prov = traced
        assert ids == plain  # turning provenance on must not consume a single extra RNG draw
        assert len(prov) == len(ids)
        origins = [p for p in prov if p is not None]
        assert origins == sorted(origins) and all(0 <= o < len(x0) for o in origins)


def test_train_edit_flows_cli_is_lean_and_has_sane_defaults() -> None:
    """The edit-flow training CLI imports without torch and exposes positive-step defaults."""
    from editjumps.pipeline.train.train_edit_flows import main

    sig = inspect.signature(main)
    assert sig.parameters["max_steps"].default > 0
    assert sig.parameters["batch_size"].default > 0
    assert str(sig.parameters["pairs"].default).endswith("oas_homolog_pairs.tsv.gz")


def test_train_edit_flows_exposes_validation_and_logs_it_to_mlflow() -> None:
    """Pin that val_loss reaches MLflow and that its examples are drawn once, not per call."""
    import inspect

    pytest.importorskip("torch")  # evoflows imports torch at module scope
    from editjumps.pipeline.train import train_edit_flows as cli
    from editjumps.pipeline.train.evoflows import train_edit_flows

    signature = inspect.signature(train_edit_flows)
    for knob in ("val_frac", "eval_steps", "val_pairs"):
        assert knob in signature.parameters, f"{knob} must be settable"
        assert knob in inspect.signature(cli.main).parameters, f"{knob} must be a CLI option"

    source = inspect.getsource(train_edit_flows)
    assert 'mlflow.log_metric("val_loss"' in source, "val_loss must be logged to MLflow"
    assert 'mlflow.log_metric("val_loss_best"' in source
    # Train batches must come from the train split, never the whole dataset.
    assert "train_idx[random.randrange(len(train_idx))]" in source
    assert "dataset[random.randrange(len(dataset))]" not in source, "training would sample held-out pairs"
    # Materialised once: re-deriving per evaluation makes the curve fresh noise, not a signal.
    assert "val_examples: list[dict] = []" in source
    assert "torch.no_grad()" in source


def test_early_stopping_is_wired_to_restore_the_best_weights_not_just_stop() -> None:
    """Early stopping must ship the best checkpoint, or it only saves time and not quality."""
    import inspect

    pytest.importorskip("torch")
    from editjumps.pipeline.train import train_edit_flows as cli
    from editjumps.pipeline.train.evoflows import train_edit_flows

    for knob in ("patience", "min_delta"):
        assert knob in inspect.signature(train_edit_flows).parameters
        assert knob in inspect.signature(cli.main).parameters

    source = inspect.getsource(train_edit_flows)
    assert "if patience and stale_evals >= patience:" in source, "patience must be able to stop the loop"
    assert "best_state = {" in source, "the best weights must be captured"
    assert "model.load_state_dict(" in source, "the best weights must be restored"
    # Restore must happen BEFORE the model is written, or the saved artifact is the wrong one.
    assert source.index("model.load_state_dict(") < source.index('torch.save(model.state_dict()')
    # And the final val must be measured BEFORE the restore, else it describes restored weights.
    assert source.index("final_val = val_loss()") < source.index("model.load_state_dict(")
    assert 'mlflow.log_metric("stopped_early"' in source
    assert 'mlflow.log_metric("saved_weights_from_step"' in source


def test_nan_comparisons_cannot_silently_ship_diverged_weights() -> None:
    """A NaN final loss must force the best-weight restore, not skip it. `best_val < final_val` is False."""
    import inspect
    import math

    pytest.importorskip("torch")
    from editjumps.pipeline.train.evoflows import train_edit_flows

    # The language-level trap this guards, stated outright.
    assert not (31.978 < float("nan")), "NaN comparisons are False - that was the bug"
    assert not (float("nan") < 31.978)
    assert not math.isfinite(float("nan")) and not math.isfinite(float("inf"))

    source = inspect.getsource(train_edit_flows)
    assert "final_broken" in source, "non-finite finals must be handled explicitly"
    assert "math.isfinite(final_val)" in source
    assert "(final_broken or best_val < final_val)" in source, "restore must trigger on non-finite"
    # And a diverged run must stop rather than spend the remaining budget producing NaN.
    assert "math.isfinite(loss_value)" in source
    assert 'mlflow.log_metric("diverged"' in source


def test_divergence_is_caught_before_the_optimizer_step() -> None:
    """The finite check must precede `optimizer.step()`.

    Stepping first writes NaN into the Adam moments as well as the weights. Restoring the best
    weights afterwards does not clean the moments, so the end-of-run checkpoint carries them and the
    next resume re-corrupts the restored weights on its first step.
    """
    import inspect

    pytest.importorskip("torch")
    from editjumps.pipeline.train.evoflows import train_edit_flows

    source = inspect.getsource(train_edit_flows)
    guard = source.index("if not math.isfinite(loss_value)")
    step = source.index("optimizer.step()", source.index("batch_loss.backward()"))
    assert guard < step, "the non-finite guard must come before optimizer.step(), not after it"
    # The gradient can go non-finite while the loss still reads finite; both poison the moments.
    assert "math.isfinite(grad_norm)" in source


def test_select_chain_pairs_takes_one_named_chain_out_of_both_members() -> None:
    """Heavy/light take that chain from BOTH members; joined returns the input untouched."""
    pytest.importorskip("torch")
    from editjumps.core.sequences import PAIR_SEP
    from editjumps.pipeline.preprocess.pretrain.seed_homologs import CHAIN_CHOICES
    from editjumps.pipeline.train.evoflows import select_chain_pairs

    joined = [
        (f"QVQL{PAIR_SEP}DIQM", f"QVKL{PAIR_SEP}DIQL"),
        (f"EVQL{PAIR_SEP}DVLM", f"EVKL{PAIR_SEP}DVLL"),
    ]

    heavy, skipped = select_chain_pairs(joined, "heavy")
    assert heavy == [("QVQL", "QVKL"), ("EVQL", "EVKL")]
    assert skipped == {"no_separator": 0, "empty_chain": 0}

    light, skipped = select_chain_pairs(joined, "light")
    assert light == [("DIQM", "DIQL"), ("DVLM", "DVLL")]
    assert skipped == {"no_separator": 0, "empty_chain": 0}

    passthrough, skipped = select_chain_pairs(joined, "joined")
    assert passthrough == joined, "the historical form must be returned unchanged"
    assert skipped == {"no_separator": 0, "empty_chain": 0}

    # The split must actually shorten the construct, or the option would be inert on real data:
    # this is the property the 1.4x edit-rate gap turns on, not the equality above.
    assert all(len(h) < len(j) for (h, _), (j, _) in zip(heavy, joined, strict=True))

    # Neither member may keep a separator afterwards -- a surviving `.` is a chimeric example.
    assert not any(PAIR_SEP in s for pair in (*heavy, *light) for s in pair)

    # The registry is shared, not restated, so the editor cannot drift from the stage that builds
    # the evaluation families.
    assert CHAIN_CHOICES == ("heavy", "light", "joined")
    with pytest.raises(ValueError, match="options"):
        select_chain_pairs(joined, "vhh")


def test_select_chain_pairs_counts_the_malformed_lines_instead_of_dropping_them_silently() -> None:
    """A pair with no separator, or an empty chain, is skipped with a counted reason."""
    pytest.importorskip("torch")
    from editjumps.core.sequences import PAIR_SEP
    from editjumps.pipeline.train.evoflows import select_chain_pairs

    rows = [
        (f"QVQL{PAIR_SEP}DIQM", f"QVKL{PAIR_SEP}DIQL"),  # well formed
        ("QVQLNOSEP", f"QVKL{PAIR_SEP}DIQL"),            # x0 has no separator
        (f"QVQL{PAIR_SEP}DIQM", "QVKLNOSEP"),            # x1 has no separator
        (f"{PAIR_SEP}DIQM", f"QVKL{PAIR_SEP}DIQL"),      # x0's heavy chain is empty
    ]

    heavy, skipped = select_chain_pairs(rows, "heavy")
    assert heavy == [("QVQL", "QVKL")], "only the well-formed pair survives"
    assert skipped == {"no_separator": 2, "empty_chain": 1}
    assert len(heavy) + sum(skipped.values()) == len(rows), "every input row is accounted for"

    # `light` skips the unseparated lines for the SAME reason rather than silently keeping the
    # single chain it happens to hold.
    light, skipped = select_chain_pairs(rows, "light")
    assert light == [("DIQM", "DIQL"), ("DIQM", "DIQL")]
    assert skipped == {"no_separator": 2, "empty_chain": 0}

    # `joined` accounts for the same rows without inspecting them.
    passthrough, skipped = select_chain_pairs(rows, "joined")
    assert len(passthrough) == len(rows) and sum(skipped.values()) == 0


def test_the_dataset_and_the_trainer_carry_the_chain_through_to_the_examples(tmp_path: Path) -> None:
    """`HomologPairDataset(chain=...)` splits on load and reports what it dropped."""
    pytest.importorskip("torch")
    import gzip

    from editjumps.core.edit_flows.stages import EditFlowConfig
    from editjumps.core.sequences import PAIR_SEP
    from editjumps.pipeline.train.evoflows import HomologPairDataset

    class FakeTokenizer:
        """Char-level tokenizer with BOS=0 / EOS=1, as in the dataset contract test above."""

        def __call__(self, seq: str) -> dict:
            return {"input_ids": [0] + [ord(c) % 20 + 2 for c in seq] + [1]}

    pairs_path = tmp_path / "pairs.tsv.gz"
    with gzip.open(pairs_path, "wt") as fh:
        fh.write(f"ACDEFGH{PAIR_SEP}KLMNP\tACDFGH{PAIR_SEP}KLMNP\n")
        fh.write("ACDEFGH\tACDFGH\n")  # no separator: skipped, and counted

    joined = HomologPairDataset(pairs_path, FakeTokenizer(), EditFlowConfig(), vocab_size=25, chain="joined")
    assert len(joined) == 2 and joined.n_pairs_read == 2
    assert sum(joined.skipped_pairs.values()) == 0

    heavy = HomologPairDataset(pairs_path, FakeTokenizer(), EditFlowConfig(), vocab_size=25, chain="heavy")
    assert heavy.pairs == [("ACDEFGH", "ACDFGH")]
    assert heavy.n_pairs_read == 2 and heavy.skipped_pairs == {"no_separator": 1, "empty_chain": 0}
    assert heavy[0]["x_t"], "a split pair still produces a usable example"

    # The default is the historical construct, so an existing call site is unchanged.
    assert inspect.signature(HomologPairDataset.__init__).parameters["chain"].default == "joined"


def test_chain_is_a_cli_option_a_tracked_dvc_param_and_flips_the_design_tag() -> None:
    """The construct must be selectable, recorded, and visible in the run's own metadata."""
    pytest.importorskip("torch")
    import yaml

    from editjumps.pipeline.train import train_edit_flows as cli
    from editjumps.pipeline.train.evoflows import train_edit_flows

    # Default is `joined` on BOTH layers: the historical behaviour of every reported checkpoint.
    assert inspect.signature(cli.main).parameters["chain"].default == "joined"
    assert inspect.signature(train_edit_flows).parameters["chain"].default == "joined"

    stage = yaml.safe_load(Path("dvc.yaml").read_text())["stages"]["train_edit_flows"]
    assert "--chain ${edit_flows.chain}" in stage["cmd"], "the stage must name the construct"
    assert "edit_flows.chain" in stage["params"], "an untracked construct cannot invalidate the stage"
    block = yaml.safe_load(Path("params.yaml").read_text())["edit_flows"]
    assert block["chain"] in ("heavy", "light", "joined")
    # The whole point of the change: the stage trains on the construct §4.2 evaluates.
    assert block["chain"] == "heavy", "the §4.2 families are single domains; see the params comment"

    source = inspect.getsource(train_edit_flows)
    assert '"chain": chain' in source, "the construct must reach MLflow's params"
    assert 'chain_handling="separate" if chain != "joined" else "combine"' in source

    # The CLI must split the sample by chain before it reports on it, so the log describes the pairs the model.
    cli_source = inspect.getsource(cli.main)
    assert "select_chain_pairs(sample, chain)" in cli_source
    assert cli_source.index("select_chain_pairs(sample, chain)") < cli_source.index(
        "pairs: {len(sample)} sampled, unoriented"
    ), "the sample must be split before it is reported"


def test_the_committed_pairs_file_is_joined_so_chain_heavy_is_a_real_change() -> None:
    """Guard the premise that the committed pairs file is joined."""
    pytest.importorskip("torch")
    import gzip

    from editjumps.core.sequences import PAIR_SEP
    from editjumps.pipeline.train.evoflows import select_chain_pairs

    pairs_path = Path("data/pretrain/oas_homolog_pairs.tsv.gz")
    if not pairs_path.exists():
        pytest.skip("oas_homolog_pairs.tsv.gz is not pulled; `dvc pull` to check the premise")

    sample = []
    with gzip.open(pairs_path, "rt") as fh:
        for i, line in enumerate(fh):
            if i >= 200:
                break
            x0, tab, x1 = line.rstrip("\n").partition("\t")
            if tab:
                sample.append((x0, x1))

    assert sample, "the pairs file must have two-column lines"
    assert all(PAIR_SEP in x0 and PAIR_SEP in x1 for x0, x1 in sample), "the file is joined VH.VL"
    heavy, skipped = select_chain_pairs(sample, "heavy")
    assert sum(skipped.values()) == 0, "no malformed line in the committed file"
    # Roughly halves the construct: 231 -> 122 is what makes the training and eval lengths agree.
    assert sum(len(x0) for x0, _ in heavy) < 0.7 * sum(len(x0) for x0, _ in sample)
