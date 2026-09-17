# Performance

Hardware configurations and empirical throughput benchmarks across pipeline stages.

## Contents

- [Hardware in use](#hardware-in-use)
- [Training](#training)
- [Evaluation and baselines](#evaluation-and-baselines)
- [Data-build stages](#data-build-stages)
- [Region and quota constraints](#region-and-quota-constraints)
- [Checkpointing and preemption](#checkpointing-and-preemption)
- [Compute efficiency](#compute-efficiency)
- [CPU-only reference points](#cpu-only-reference-points)
- [CPU fallback anomaly reference](#cpu-fallback-anomaly-reference)
- [Unmeasured quantities](#unmeasured-quantities)

## Hardware in use

GPU workloads execute on GCP via SkyPilot:

| Hardware | Machine Type | Region Priority | Typical Workloads |
|---|---|---|---|
| **L4:1** (24 GB) | `g2-standard-8` | `europe-west6` (Zurich), `europe-west4`, `europe-west1` | 35M jump-process editor, `pretrain_esm`, `evotune_esm` |
| **A100:1** (40 GB) | `a2-highgpu-1g` | `europe-west4` | 650M pretraining (~16–18 GB in bf16), 650M editor (~20 GB in fp32), evotuned baselines |

Zurich (`europe-west6`) is prioritized for L4 workloads to co-locate with GCS storage buckets and MLflow tracking services. A100 workloads route to `europe-west4`.

## Training

### ESM-2 35M Edit-Flow Editor (L4)

Benchmark parameters: `max_steps: 20000`, `batch_size: 16`, `lr: 1e-4` (see [`findings.md`](findings.md)):

| Metric | Measurement |
|---|---|
| Throughput | **~1.05 s/step** |
| 20,000 steps wall clock | **~5.8 h** |

Training loss stabilizes near step 6,000 (~175). Early stopping defaults to `patience: 6` at `eval_steps: 500`.

### `pretrain_esm` (L4)

| Metric | Measurement |
|---|---|
| Full execution (`make jobs-repro`) | **1 h 0 m 16 s** (0 spot preemptions) |
| Final evaluation loss | 1.178 (`epochs: 0.16`, bounded by `max_steps`) |
| Final evaluation throughput | 203.6 samples/s (633.5 s evaluation runtime; `metrics/pretrain_esm.json`) |

### Evotuning ESM-2-650M (A100)

Evaluated across target families (`metrics/evotune/ty1-evotune_esm.json`, `her2vh-evotune_esm.json`):

| Metric | Ty1 | HER2-VH |
|---|---|---|
| Training throughput | **5.78 it/s** | — |
| Final evaluation loss | 0.28149 | 0.28732 |
| Evaluation throughput | **155.941 samples/s** | **150.042 samples/s** |
| Evaluation wall clock | 12.29 s | 14.20 s |

An A100 is required due to memory constraints: 650M bf16 requires ~16–18 GB and 650M fp32 requires ~20 GB, exceeding L4 24 GB limits once activations are included.

### 650M Learning-Rate Sweep (A100)

Grid parameters: `MAX_STEPS=3000`, `BATCH_SIZE=16` (`make jobs-lr-sweep`):

| Metric | Measurement |
|---|---|
| Per trial | **~80 min** |
| Concurrent grid (`ASYNC=1`) | ~1.5 h |
| Serial grid | ~5.5 h |

### 20,000-Step Editor Runs (A100)

Evaluation of schedule configurations across 20,000 steps on an A100 GPU:

| Job | Schedule | Duration | Throughput |
|---|---|---|---|
| 76 | Linear | **3 h 0 m 43 s** | **0.54 s/step** (step 19,990) |
| 83 | Cubic | **3 h 31 m** | ~0.63 s/step |

Compared to ~1.05 s/step on an L4, an A100 provides approximately 2× speedup for this training configuration.

## Evaluation and baselines

### EvoDiff-MSA

The only per-template GPU timing in the repo, from `editjumps/core/evodiff_msa/install_evodiff.sh`'s incident note:

| | |
|---|---|
| GPU | **~2 minutes per template** |
| CPU (the bug) | 54 minutes without finishing the first of 20 templates — roughly **25× slower** |

Model-level throughput, measured on CPU on a 40-member synthetic VHH family (a stand-in, not the
project's own seed family) — [`findings.md`](findings.md), "EvoDiff-MSA baseline":

| | |
|---|---|
| `inpaint` | 8 forward passes in **~8 s wall**, including checkpoint load, 8×125 MSA |
| `unconditional` | one 125-residue query row in **~30 s**, same MSA |
| MSA checkpoint download | ~380 MB, once, from Zenodo record 8045076 |

### Generation Evaluation

Generation evaluation (`make generation-eval`, `make eval-sweep`) performs sampling without backpropagation:
- **GPU (L4):** 20 templates × 20 variants at holdout size 200 executes in **~12.5 min** (jobs 84 and 85).
- **CPU:** Used for small sweep testing on low-memory environments.

### IMGT Numbering

| Operation | Benchmark |
|---|---|
| ANARCI / `hmmscan` IMGT numbering | **~14 ms** per chain on cache misses (cached via `functools.lru_cache` in `editjumps/core/cdr.py`; see [`installation.md`](installation.md#hmmscan-hmmer)). |

### Calibration Profiling

`make jobs-calibrate` profiles step latency and model FLOP utilization (MFU) across batch sizes (~20 min execution on an L4).

## Data-build stages

| Stage | Benchmark |
|---|---|
| `editjumps/measurements/homolog_cap_bias.py` | **~6 min** locally, reconstructing families from `oas_corpus.txt.gz` |
| MMseqs2 stages (`split_corpus`, `build_homolog_pairs`) | Heavy RAM requirement over ~2.6M sequences. Run via cloud jobs (`make jobs-repro STAGES=split_corpus`) or set `EDITJUMPS_ALLOW_LOCAL_HEAVY=1`. |

## Region and quota constraints

- **A100 instances:** Available in GCP `europe-west4` (Netherlands). SkyPilot configurations specify regional fallback lists.
- **Quota specifications:** Workloads target `A100:1` (40 GB), which has active quota allocations.
- Spot availability: For on-demand launches during spot preemptions, use `--retry-until-up --down`.

## Checkpointing and preemption

Managed spot instances provide significant cost reductions:
- Full model checkpoint state: **~391 MB**.
- Checkpoint interval `save_steps: 1000` bounds maximum lost compute upon spot preemption to **~17 min**.

## Compute efficiency

Job 21 profiled at approximately 0.6% Model FLOPs Utilization (MFU) across a 6.4 h training run. The step time is primarily bounded by CPU-GPU launch synchronization rather than raw tensor compute capability. Workloads should be profiled via `make jobs-calibrate` prior to allocating higher-tier GPU resources.

## CPU-only reference points

Local benchmarking of `esm2_t12_35M_UR50D` pretraining on CPU (fp32) for architectural reference (see [`throughput_options.md`](throughput_options.md)):

| Batch Size | Max Length | Seqs / Sec | Est. Hours / Epoch (2.46M sequences) |
|---|---|---|---|
| 8 | 160 | 11.56 | 59 |
| 16 | 160 | **14.88** | **46** |
| 32 | 160 | 10.30 | 66 |
| 16 | 280 | 6.71 | 102 |

Relative scaling characteristics:
- Doubling sequence length from 160 to 280 increases step time by ~1.5–2×.
- Scaling from `esm2_t6_8M` (27.2 seqs/s) to `esm2_t12_35M` (6.7 seqs/s) represents a **4.05×** parameter scaling penalty.
- Note: On CPU architectures without native BF16 matrix instructions, BF16 emulation can be up to 14× slower than FP32 execution.

## CPU fallback anomaly reference

Historical log values reflecting unintended CPU fallback prior to environment guards:

| Metric | Context |
|---|---|
| **85 s/step** | Job 58 (`evotune_esm`) executing on host CPU due to missing CUDA driver linkage |
| **~16 s/step** | Editor training loop falling back to CPU execution |
| **0.638 samples/s** | Job 58 evaluation throughput on CPU (~50 min per evaluation cycle) |

CUDA initialization checks are now enforced at runtime start; see [`known_issues.md`](known_issues.md).

## Unmeasured quantities

The following values are not tracked as structured metric artifacts:
- End-to-end task wall-clock times are logged in cloud job registries rather than pipeline metric JSONs.
- CPU generation evaluation latency varies by host core architecture and is not committed.
- Detailed peak VRAM allocation profiles across diverse sequence lengths.

Profiling can be performed using `make jobs-calibrate`.
## Where to go next
| Document | Description |
|---|---|
| [`reproducing.md`](reproducing.md) | Reproduction workflows and comparability controls |
| [`gcp_setup.md`](gcp_setup.md) | Cloud infrastructure and instance configuration |
| [`known_issues.md`](known_issues.md) | Diagnostic symptoms and operational caveats |
