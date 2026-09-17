# ESM-2 Pretraining Throughput Options

Analysis of local throughput characteristics and scaling options for ESM-2 masked language model pretraining on local CPU and MPS hardware.

## Baseline Throughput (CPU, FP32)

Benchmark results for `esm2_t12_35M_UR50D` across batch sizes and sequence lengths on standard CPU architectures (epoch estimates relative to 2.46M unique sequence keys):

| Batch Size | Max Length | Seqs / Sec | Est. Hours / Epoch (2.46M sequences) |
|---|---|---|---|
| 8 | 160 | 11.56 | 59 |
| 8 | 280 | 8.40 | 81 |
| 16 | 160 | **14.88** | **46** |
| 16 | 280 | 6.71 | 102 |
| 32 | 160 | 10.30 | 66 |
| 32 | 280 | 5.96 | 115 |

Key empirical observations:
- Sequence length scaling: Length 280 incurs a ~1.5–2× overhead relative to length 160 due to attention complexity.
- Batch sizing: Batch 16 provides the optimal CPU throughput profile.

## Optimization Options

1. **Cluster-aware subsampling:** The corpus contains 2.46M unique CDR keys across 2.08M clusters at 90% sequence identity (~16% redundancy). Subsampling one sequence per cluster via `split_corpus.py` preserves diversity while reducing per-epoch iteration time.
2. **Model scale reduction:** Comparing `esm2_t6_8M` with `esm2_t12_35M` under identical configs yields **27.2 vs. 6.7 seqs/sec** (a **4.05×** throughput improvement), directly mirroring parameter reduction.
3. **Quantization:** Dynamic quantization (`torch.quantization.quantize_dynamic`) is suited for inference serving rather than backward-pass training loops.
4. **Sequence packing:** Paired VH.VL sequence lengths cluster tightly (padding overhead is under 2%), making sequence packing unnecessary.
5. **Precision selection:** On CPU architectures lacking native BF16 matrix multiplication hardware, BF16 software emulation is significantly slower than native FP32 execution. For Apple Silicon (MPS) or CUDA platforms, native half-precision or BF16 execution is recommended.

## Recommendations for Local Development

1. Use `esm2_t6_8M_UR50D` for rapid local iteration and debugging.
2. Use cluster-representative subsampling for development runs.
3. Configure step-based checkpointing (`--save-steps 500`) to guarantee progress persistence.
