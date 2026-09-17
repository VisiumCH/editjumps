"""The training loop actually run, not read.

Every other assertion about `train_edit_flows` in this suite is `inspect.getsource` against its
text, which passes whenever the right words are present in the wrong order. These execute the loop,
so they fail when the behaviour is wrong rather than when the wording changes.

The trunk is built here rather than pulled: the trainer's default is a DVC-tracked 35M checkpoint
that CI does not have, and a 25k-parameter stand-in exercises the same loop in a fraction of the
time.
"""

import gzip
import shutil
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def tiny_trunk(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Save a minimal ESM masked-LM and its tokenizer, as `train_edit_flows` expects to load them."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers import EsmConfig, EsmForMaskedLM, EsmTokenizer  # ty: ignore[unresolved-import]

    folder = tmp_path_factory.mktemp("trunk")
    vocab = ["<cls>", "<pad>", "<eos>", "<unk>", *"LAGVSERTIDPKQNFYMHWC", "<mask>"]
    (folder / "vocab.txt").write_text("\n".join(vocab) + "\n")

    out = folder / "esm"
    EsmForMaskedLM(
        EsmConfig(vocab_size=len(vocab), hidden_size=32, num_hidden_layers=2,
                  num_attention_heads=2, intermediate_size=64, max_position_embeddings=64,
                  pad_token_id=1, mask_token_id=len(vocab) - 1)
    ).save_pretrained(out)
    EsmTokenizer(vocab_file=str(folder / "vocab.txt")).save_pretrained(out)
    return out


def _write_pairs(path: Path, n: int = 24) -> None:
    """Write a small homolog-pair corpus: each pair is one deletion away from its partner."""
    with gzip.open(path, "wt") as handle:
        for i in range(n):
            left = "ACDEFGHIKL"[: 6 + i % 4] + "M"
            handle.write(f"{left}\t{left[:-2]}\n")


def _fake_gcs(monkeypatch: pytest.MonkeyPatch, remote: Path) -> None:
    """Point the checkpoint mirror at a local directory instead of GCS."""
    import editjumps.core.gcs_checkpoints as gcs

    remote.mkdir(parents=True, exist_ok=True)

    def upload(local: Path, uri: str) -> bool:
        shutil.copy(local, remote / Path(uri).name)
        return True

    def download(uri: str, local: Path) -> bool:
        source = remote / Path(uri).name
        if not source.exists():
            return False
        Path(local).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(source, local)
        return True

    monkeypatch.setattr(gcs, "gcs_upload", upload)
    monkeypatch.setattr(gcs, "gcs_download", download)


def test_a_diverged_run_writes_a_finite_optimizer_state(
    tiny_trunk: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The checkpoint a diverged run leaves behind must be resumable.

    Stepping the optimizer on a non-finite loss writes NaN into the Adam moments. Restoring the best
    weights afterwards does not clean them, so the checkpoint carries NaN and the next resume
    destroys the run on its first step.

    The NaN is injected at a known loss call rather than provoked with a large learning rate: an
    absurd `lr` diverges on some runs and not others, which would make this test flaky and its
    failures meaningless.
    """
    torch = pytest.importorskip("torch")
    from editjumps.core.edit_flows import stages
    from editjumps.pipeline.train.evoflows import train_edit_flows

    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    remote = tmp_path / "remote"
    _fake_gcs(monkeypatch, remote)

    # batch_size=2 and no validation, so the loop makes exactly two loss calls per step: the first
    # step completes and the second goes non-finite.
    batch_size, healthy_steps = 2, 1
    real_resolve = stages.resolve_loss

    def resolve_with_a_nan_after_the_first_step(name: str):  # noqa: ANN202 - mirrors resolve_loss
        real_loss, calls = real_resolve(name), {"n": 0}

        def loss(*args: object, **kwargs: object):  # noqa: ANN202 - mirrors the resolved loss
            calls["n"] += 1
            value = real_loss(*args, **kwargs)
            # Multiply rather than replace, so the value stays attached to the graph and `backward`
            # propagates the NaN into the gradients exactly as a real divergence would.
            return value * float("nan") if calls["n"] > batch_size * healthy_steps else value

        return loss

    monkeypatch.setattr(stages, "resolve_loss", resolve_with_a_nan_after_the_first_step)

    pairs = tmp_path / "pairs.tsv.gz"
    _write_pairs(pairs)

    train_edit_flows(
        pairs=pairs, model_name=str(tiny_trunk), output_folder=tmp_path / "out", max_steps=6,
        batch_size=batch_size, lr=1e-4, save_steps=0, eval_steps=0, val_frac=0.0,
        checkpoint_uri="gs://fake-bucket/ckpt",
    )

    checkpoint = remote / "checkpoint.pt"
    assert checkpoint.exists(), "a diverged run must still leave a checkpoint"
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)

    diverged_at = int(state["step"])
    assert diverged_at == healthy_steps, "the injected NaN must stop the loop on the step it lands"

    # Adam counts its own calls. The loop runs `diverged_at` iterations that complete and one that
    # does not, so an optimizer that stepped `diverged_at` times skipped the diverged batch and one
    # that stepped `diverged_at + 1` times did not. Asserting on the count rather than on NaN in the
    # moments: whether a diverged batch actually poisons them depends on whether the loss overflowed
    # to inf (clip_grad_norm_ then zeroes the gradients and the step is harmless) or went to NaN
    # (which propagates). The ordering is the invariant; the poisoning is one of its consequences.
    counts = {int(entry["step"]) for entry in state["optimizer"]["state"].values()}
    assert counts == {diverged_at}, (
        f"optimizer stepped {counts} times but only {diverged_at} iterations completed: it stepped "
        f"on the diverged batch, which is what writes NaN into the Adam moments and makes the next "
        f"resume corrupt the restored weights"
    )

    for parameter, entry in state["optimizer"]["state"].items():
        for key, value in entry.items():
            if torch.is_tensor(value):
                assert torch.isfinite(value).all(), (
                    f"optimizer state {key} for parameter {parameter} is non-finite; resuming from "
                    f"this checkpoint would corrupt the weights"
                )


def test_a_resumed_run_carries_the_best_weights_and_the_patience_counter(
    tiny_trunk: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second run reading the first's checkpoint starts from its early-stopping state, not from scratch."""
    torch = pytest.importorskip("torch")
    from editjumps.pipeline.train.evoflows import (
        EvoFlowsModel,
        load_training_state,
        train_edit_flows,
    )

    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    remote = tmp_path / "remote"
    _fake_gcs(monkeypatch, remote)

    pairs = tmp_path / "pairs.tsv.gz"
    _write_pairs(pairs)

    def run(output_folder: Path) -> None:
        """One run of the trainer against the shared fake checkpoint mirror."""
        train_edit_flows(
            pairs=pairs, model_name=str(tiny_trunk), output_folder=output_folder, max_steps=6,
            batch_size=2, lr=1e-4, save_steps=0, val_frac=0.5, eval_steps=2, val_pairs=4,
            checkpoint_uri="gs://fake-bucket/ckpt",
        )

    run(tmp_path / "first")

    model = EvoFlowsModel.from_esm(str(tiny_trunk))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    after_first = load_training_state(remote / "checkpoint.pt", model, optimizer)

    assert after_first.best_step >= 0, "the first run evaluated, so it has a best step to carry"
    assert after_first.best_state is not None, "the best weights must be in the checkpoint"
    assert after_first.best_val < float("inf")

    # The second run resumes past max_steps, so it takes no steps: whatever it writes back is what
    # it read, which is exactly the state a preempted job must not lose.
    run(tmp_path / "second")

    after_second = load_training_state(remote / "checkpoint.pt", model, optimizer)
    assert after_second.best_step == after_first.best_step
    assert after_second.best_val == after_first.best_val
    assert after_second.stale_evals == after_first.stale_evals


def test_a_resume_past_max_steps_exits_instead_of_crashing(
    tiny_trunk: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A preempted job that restarts after finishing must exit cleanly.

    Everything after the training loop reads `step`, which the loop binds. A resumed run at or past
    `max_steps` enters no iteration, so `step` was never bound and the run died with
    UnboundLocalError on the one path preemption resilience exists to serve.
    """
    pytest.importorskip("torch")
    from editjumps.pipeline.train.evoflows import train_edit_flows

    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    remote = tmp_path / "remote"
    _fake_gcs(monkeypatch, remote)

    pairs = tmp_path / "pairs.tsv.gz"
    _write_pairs(pairs)

    def run(output_folder: Path) -> Path:
        """One run of the trainer, with the same budget both times."""
        return train_edit_flows(
            pairs=pairs, model_name=str(tiny_trunk), output_folder=output_folder, max_steps=2,
            batch_size=2, lr=1e-4, save_steps=0, val_frac=0.0, eval_steps=0,
            checkpoint_uri="gs://fake-bucket/ckpt",
        )

    run(tmp_path / "first")
    # Same budget, so the resumed run starts at step 2 and the loop body never runs.
    out = run(tmp_path / "second")

    assert out.exists(), "the resumed run must still write its model folder"
