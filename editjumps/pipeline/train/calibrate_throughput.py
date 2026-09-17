"""Measure what the GPU is actually doing during editor training."""

import time
from pathlib import Path
from typing import Annotated

import typer

from editjumps.core.utils import get_logger, write_metrics

logger = get_logger(__file__)

# 6*N*T is the standard training-FLOPs estimate (2 forward, 4 backward) per token.
FLOPS_PER_PARAM_TOKEN = 6
# Marketing peak, fp16 tensor cores, no sparsity. Only used to express MFU as a percentage.
PEAK_FLOPS = {"L4": 121e12, "A100": 312e12, "A100-80GB": 312e12, "H100": 990e12}


def peak_flops_for(device_name: str) -> float:
    """Look up a device's fp16 peak, matching on substring so "NVIDIA L4" finds "L4"."""
    for key, peak in sorted(PEAK_FLOPS.items(), key=lambda kv: -len(kv[0])):
        if key.lower() in device_name.lower():
            return peak
    return 0.0


def main(
    pairs: Annotated[Path, typer.Option()] = Path("data/pretrain/oas_homolog_pairs.tsv.gz"),
    model_name: Annotated[str, typer.Option()] = "data/pretrain/esm2_oas",
    batch_sizes: Annotated[str, typer.Option(help="Comma-separated batch sizes to time")] = "16,64,256",
    steps: Annotated[int, typer.Option(help="Timed steps per configuration")] = 10,
    warmup: Annotated[int, typer.Option(help="Untimed steps first (kernel autotuning, allocator)")] = 3,
    compare_accumulated: Annotated[bool, typer.Option(help="Also time the pre-batching path")] = True,
    metrics_path: Annotated[Path, typer.Option()] = Path("metrics/calibration.json"),
) -> None:
    """Time the training step across batch sizes and report MFU."""
    import random

    import torch
    from transformers import AutoTokenizer

    from editjumps.core.edit_flows.loss import edit_flow_loss
    from editjumps.core.edit_flows.stages import EditFlowConfig
    from editjumps.core.utils import load_params
    from editjumps.pipeline.train.evoflows import EvoFlowsModel, HomologPairDataset, pad_batch, pick_device

    device = pick_device()
    device_name = torch.cuda.get_device_name() if device == "cuda" else device
    peak = peak_flops_for(device_name)

    def sync() -> None:
        """Wait for queued GPU work; without it we time dispatch, not completion."""
        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()

    config = EditFlowConfig.from_params(load_params())
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = HomologPairDataset(pairs, tokenizer, config, vocab_size=tokenizer.vocab_size, seed=0)
    model = EvoFlowsModel.from_esm(model_name, rate_head=config.rate_head, q_head=config.q_head).to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    n_params = sum(p.numel() for p in model.parameters())
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else model.encoder.config.pad_token_id
    logger.info(f"device={device_name} params={n_params/1e6:.0f}M peak={peak/1e12:.0f} TFLOPS")

    def draw(batch: int) -> list[dict]:
        return [e for e in (dataset[random.randrange(len(dataset))] for _ in range(batch)) if e["x_t"]]

    def batched_step(examples: list[dict]) -> None:
        input_ids, attention_mask, lengths = pad_batch(examples, pad_token_id, device)
        rates = model.forward_batch(input_ids, attention_mask, [e["t"] for e in examples])
        loss = torch.zeros((), device=device)
        for row, (example, n) in enumerate(zip(examples, lengths, strict=True)):
            loss = loss + edit_flow_loss(example["z_t"], example["z_1"], example["kappa"],
                                         example["dkappa"], *(rate[row, :n] for rate in rates))
        (loss / len(examples)).backward()
        optimizer.step()
        optimizer.zero_grad()

    def accumulated_step(examples: list[dict]) -> None:
        loss = torch.zeros((), device=device)
        for example in examples:
            rates = model(torch.tensor(example["x_t"], device=device), example["t"])
            loss = loss + edit_flow_loss(example["z_t"], example["z_1"], example["kappa"],
                                         example["dkappa"], *rates)
        (loss / len(examples)).backward()
        optimizer.step()
        optimizer.zero_grad()

    results = []
    for batch in [int(b) for b in batch_sizes.split(",") if b.strip()]:
        # Drawn once per batch size and reused, so data prep stays out of the timed region (it is
        # measured separately); timing it per step would blend two bottlenecks.
        random.seed(0)
        prep_start = time.time()
        drawn = [draw(batch) for _ in range(warmup + steps)]
        prep_per_example = (time.time() - prep_start) / max(sum(len(d) for d in drawn), 1)

        for label, step_fn in (("batched", batched_step),) + (
            (("accumulated", accumulated_step),) if compare_accumulated else ()
        ):
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            try:
                for examples in drawn[:warmup]:
                    step_fn(examples)
                sync()
                start = time.time()
                for examples in drawn[warmup:]:
                    step_fn(examples)
                sync()
                elapsed = (time.time() - start) / steps
            except torch.OutOfMemoryError:  # a real answer about this batch size, not a crash
                logger.warning(f"batch={batch} {label}: OUT OF MEMORY")
                results.append({"batch": batch, "path": label, "oom": True})
                model.zero_grad(set_to_none=True)
                if device == "cuda":
                    torch.cuda.empty_cache()
                continue

            tokens = sum(len(e["x_t"]) for e in drawn[warmup]) # one representative step
            flops = FLOPS_PER_PARAM_TOKEN * n_params * tokens
            entry = {
                "batch": batch, "path": label, "s_per_step": round(elapsed, 4),
                "seq_per_s": round(batch / elapsed, 1), "tokens_per_s": round(tokens / elapsed, 0),
                "tflops": round(flops / elapsed / 1e12, 2),
                "mfu_percent": round(flops / elapsed / peak * 100, 2) if peak else None,
                "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2) if device == "cuda" else None,
                "data_prep_ms_per_example": round(prep_per_example * 1000, 2),
            }
            results.append(entry)
            logger.info(f"batch={batch:<4} {label:<12} {entry['s_per_step']:.3f} s/step  "
                        f"{entry['seq_per_s']:>6.1f} seq/s  {entry['tflops']:>5.2f} TFLOPS  "
                        f"MFU {entry['mfu_percent']}%  peak {entry['peak_mem_gb']} GB")

    payload = {
        "device": device_name, "params_millions": round(n_params / 1e6, 1),
        "peak_tflops": peak / 1e12,
        # The number every result here is compared against.
        "job21_reference": {"batch": 16, "s_per_step": 1.09, "mfu_percent": 0.6, "device": "NVIDIA L4"},
        "results": results,
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    write_metrics(metrics_path, payload)
    logger.info(f"wrote {metrics_path}")


if __name__ == "__main__":
    typer.run(main)
