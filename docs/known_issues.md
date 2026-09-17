# Known Issues and Pitfalls

Common failure modes, diagnostic symptoms, and configuration pitfalls across local and cloud environments.

## Contents

- [Silent CPU fallback](#silent-cpu-fallback)
- [Isolated environment torch configuration](#isolated-environment-torch-configuration)
- [Cloud credential expiration and upload limits](#cloud-credential-expiration-and-upload-limits)
- [Metric comparability across runs](#metric-comparability-across-runs)
- [Checkpoint persistence and export](#checkpoint-persistence-and-export)
- [Memory limits on local runs](#memory-limits-on-local-runs)
- [Cloud GPU capacity and regions](#cloud-gpu-capacity-and-regions)
- [Intentionally unrefreshed pipeline stages](#intentionally-unrefreshed-pipeline-stages)
- [Additional operational caveats](#additional-operational-caveats)
- [Reference documentation](#reference-documentation)

## Silent CPU fallback

**Symptom:** GPU jobs proceed without error but execute orders of magnitude slower than expected (e.g., ~85 s/step instead of ~5.8 it/s).

**Mechanism:** `pyproject.toml` pins `torch>=2.12.1` without forcing a specific CUDA build to preserve macOS MPS/CPU compatibility. When sub-commands re-resolve dependencies, `uv` may select a CUDA-13 wheel that is incompatible with the host driver (e.g., driver 535 / CUDA 12.2), causing PyTorch to fall back silently to CPU execution.

| Stage | CPU execution | GPU execution | Speedup |
|---|---|---|---|
| Training | 85 s/step | 5.78 it/s | ~490× |
| Evaluation throughput | 0.638 samples/s | 145.5 samples/s | ~228× |
| Evaluation wall-clock | 3,005 s | 13.2 s | ~228× |

**Resolution:** Cloud execution specifications set `export UV_NO_SYNC=1` during job execution to prevent dependency re-resolution during DVC stage runs. Setup scripts also verify CUDA availability in the target execution environment.

## Isolated environment torch configuration

**Symptom:** EvoDiff-MSA evaluation falls back to CPU (~25× slower, ~54 minutes per template instead of ~2 minutes).

**Mechanism:** `.evodiff_env` is an isolated virtual environment pinned to `numpy<2`. Host environment CUDA configurations do not automatically propagate to `.evodiff_env`.

**Resolution:** `editjumps/core/evodiff_msa/install_evodiff.sh` passes `--torch-backend=auto` to detect host drivers via `nvidia-smi`. Remote execution configs verify CUDA availability within `.evodiff_env` prior to baseline evaluation.

## Cloud credential expiration and upload limits

**Symptom:** Long-running jobs fail with HTTP 401 Unauthorized or HTTP 413 Payload Too Large during MLflow metric logging.

**Mechanisms:**
1. **OAuth token expiration:** Cloud authentication tokens expire after 60 minutes.
2. **Payload body size limit:** Cloud Run tracking endpoints enforce a 32 MB request body limit, rejecting direct artifact uploads of large models (such as 650M parameter checkpoints).

**Resolution:**
- `editjumps/core/utils.py` automatically refreshes authentication tokens in-process every 40 minutes.
- MLflow reporting calls are wrapped to be non-fatal so logging errors do not abort running jobs.
- Large model weights are saved directly to cloud storage rather than uploaded as MLflow artifacts when file sizes exceed `MLFLOW_ARTIFACT_FILE_LIMIT` (24 MB).

## Metric comparability across runs

Evaluation metrics can be sensitive to sample size, reference size, and alignment coordinates.

### Reference holdout size affects agreement metrics

`covariance_agreement` and `mip_agreement` scale with the size of the held-out reference set. Normalizing by the real-homolog ceiling does not eliminate this dependence:

| Two-family mean | @200 holdout | @800 holdout | Change |
|---|---|---|---|
| Covariance / ceiling | 84.3% | 80.8% | −3.5 pp |
| MIP / ceiling | 78.9% | 80.0% | +1.1 pp |

All metrics files record `agreement_reference_n`. Compare agreement numbers only when computed against identical reference set sizes.

### Spectrum MMD depends on template count and sample size

Spectrum MMD is computed using the biased V-statistic from the EvoFlows paper, which carries positive finite-sample bias:

| Configuration | Templates × Variants | Total generated | MMD |
|---|---|---|---|
| `arm-B-c40-n100u` | 10 × 10 | 100 | 1.633 |
| `arm-B-c40-n200` | 20 × 10 | 200 | 1.107 |
| `arm-B-c40-n300` | 30 × 10 | 300 | 1.038 |
| `arm-B-c40-n500` | 25 × 20 | 500 | 0.995 |
| `arm-B-c40-n500u` | 50 × 10 | 500 | 1.117 |
| `arm-B-c40-n1000` | 50 × 20 | 1000 | 1.118 |

Template count strongly influences MMD. Compare MMD only across runs with matching `n_generated`, `n_reference`, and `n_templates`.

### Coordinate alignment frame differences

Per-position metrics (`covariance_agreement`, `mip_agreement`, `js_divergence_positional`) are projected onto an alignment frame. Each run records its alignment configuration in `alignment`; verify frame consistency before comparing runs directly.

### Replicate variation

Replicate runs under identical settings (`sched-linear` vs `faithful-appendixa`) show run-to-run variation up to 0.0343 in Ty1 MIP agreement. Effects smaller than this threshold fall within empirical variance.

### Loss comparison across schedules

The edit-flow loss weights edits by `kappa_dot / (1 - kappa)`. Linear and cubic schedules evaluate different objectives at intermediate time steps; evaluate held-out generation metrics rather than training losses to compare schedules.

## Checkpoint persistence and export

**Symptom:** Training completes and writes `checkpoint.pt`, but downstream inference fails to find model weights.

**Mechanism:** Training jobs save intermediate resume state (`checkpoint.pt`). Loadable model directories (`encoder/` and `evoflows_model.pt`) must be explicitly generated.

**Resolution:** Run `make restore-editor` to convert a training checkpoint into an exportable inference model folder.

## Memory limits on local runs

`split_corpus` and `build_homolog_pairs` run MMseqs2 over the ~2.6M sequence OAS corpus. MMseqs2 sizes its index and prefilter to available system memory, which can exhaust RAM on local machines.

Two protections exist:
- `params.yaml: mmseqs.split_memory_limit` bounds memory consumption per run.
- Local execution pre-checks block unconstrained runs unless overridden via:

```bash
make jobs-repro STAGES=split_corpus       # execute on cloud instance (recommended)
EDITJUMPS_ALLOW_LOCAL_HEAVY=1 make repro STAGES=   # local execution override
```

## Cloud GPU capacity and regions

- **GPU regional availability:** `europe-west4` provides A100 availability; Zurich (`europe-west6`) lacks A100 instances.
- **Instance configuration:** Standard quota supports `A100:1` (40 GB) instances (`a2-highgpu-1g`).
- **Spot instance availability:** Launch spot runs with `--retry-until-up --down` to handle capacity preemptions.

## Intentionally unrefreshed pipeline stages

The committed `dvc.lock` tracks smoke-test runs. Stages requiring full GPU execution (`pretrain_esm` and `train_edit_flows`) remain unrefreshed in the git checkout. The smoke checkpoint in `data/pretrain/edit_flows` predates symmetric pair clustering; reported evaluation benchmarks were generated on symmetric pairs (see [`edit-flows-editor.md`](model_cards/edit-flows-editor.md)).

## Additional operational caveats

- **File upload exclusions:** Ensure build artifacts and virtual environments are excluded in `.skyignore` to prevent uploading unnecessary directories during job staging.
- **Training divergence:** Learning rates above 0.01 on ESM-2 trunks can produce non-finite losses early in training. Divergence checks ensure only valid weights are persisted.
- **Validation loss scope:** Train and validation splits are sequence-disjoint but not homology-disjoint. Validation loss serves as an early-stopping diagnostic rather than a generalization metric.
- **CDR region definitions:** `editjumps/core/cdr.py` numbers all six IMGT CDRs using ANARCI (~47 residues), while `editjumps/core/edit_flows/mask.py` matches CDR-H3/L3 by substring (~20 residues) to avoid external HMMER dependencies during sampling.

## Reference documentation

| Document | Scope |
|---|---|
| [`findings.md`](findings.md) | Measured experimental logs and empirical records |
| [`reproducing.md`](reproducing.md) | Benchmark reproduction procedures and comparability rules |
| [`performance.md`](performance.md) | Throughput, batch sizes, and resource requirements |
| [`gcp_setup.md`](gcp_setup.md) | Cloud authentication, quotas, and instance configuration |
