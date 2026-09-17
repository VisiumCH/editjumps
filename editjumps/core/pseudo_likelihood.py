"""ESM-2 pseudo-log-likelihood, the one Appendix-B metric that needs a model (B.2)."""

import random

from editjumps.core.utils import get_logger

logger = get_logger(__file__)


def pseudo_log_likelihood(
    sequences: list[str],
    model_name: str = "facebook/esm2_t33_650M_UR50D",
    seed: int = 0,
    max_positions: int | None = None,
    batch_size: int = 16,
) -> dict:
    """Score sequences by ESM-2 pseudo-log-likelihood under B.2's estimator."""
    # Before the torch import, not after: scoring nothing should not require a deep-learning stack,
    # and a caller that filtered its list down to empty should not be punished for it.
    if not sequences:
        return {"mean": 0.0, "sum_mean": 0.0, "n": 0, "per_sequence": [], "per_sequence_sum": []}

    try:
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer
    except ImportError as missing:  # pragma: no cover - exercised by the lean env
        # `edit` guards this class of failure carefully and `rank` used to raise a bare traceback for it.
        raise ImportError(
            f"{missing.name or 'torch'} is not installed, and scoring sequences needs it.\n"
            "\n"
            "This entry point loads a stock ESM-2 to score naturalness, so it needs the `train`\n"
            "dependency group, which the lean install deliberately leaves out:\n"
            "\n"
            "    uv sync --locked --all-extras --group train\n"
            "    uv run --group train editjumps rank --sequence ...\n"
            "\n"
            "The first run also downloads the scorer (~2.5 GB for the 650M default; pass a smaller\n"
            "--model, e.g. facebook/esm2_t12_35M_UR50D, to avoid it)."
        ) from missing

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device).eval()
    mask_id = tokenizer.mask_token_id

    per_sequence_means, per_sequence_sums = [], []
    with torch.no_grad():
        for index, sequence in enumerate(sequences):
            ids = tokenizer(sequence, return_tensors="pt")["input_ids"][0]
            # Interior positions only: the BOS/EOS tokens are not residues and masking them scores
            # the tokenizer's scaffolding rather than the protein.
            positions = list(range(1, len(ids) - 1))
            rng = random.Random(seed + index)
            rng.shuffle(positions)                      # the "single random order" of B.2
            if max_positions is not None:
                positions = positions[:max_positions]
            if not positions:
                continue

            logs = []
            for start in range(0, len(positions), batch_size):
                chunk = positions[start:start + batch_size]
                batch = ids.repeat(len(chunk), 1).to(device)
                for row, pos in enumerate(chunk):
                    batch[row, pos] = mask_id
                logits = model(input_ids=batch).logits
                for row, pos in enumerate(chunk):
                    log_probs = torch.log_softmax(logits[row, pos], dim=-1)
                    logs.append(float(log_probs[ids[pos]]))
            per_sequence_means.append(sum(logs) / len(logs))
            per_sequence_sums.append(sum(logs))

    if not per_sequence_means:
        return {"mean": 0.0, "sum_mean": 0.0, "n": 0, "per_sequence": [], "per_sequence_sum": []}
    return {
        "mean": sum(per_sequence_means) / len(per_sequence_means),
        "sum_mean": sum(per_sequence_sums) / len(per_sequence_sums),
        "n": len(per_sequence_means),
        # Per-sequence values, in input order.
        "per_sequence": per_sequence_means,
        "per_sequence_sum": per_sequence_sums,
    }
