# Model Training Guide

Instructions for running local smoke tests and scaling to full cloud training runs.

## 1. Local Smoke Run

A minimal local verification pipeline on CPU or Apple Silicon (MPS) executing small-scale data preparation, training, and sampling:

```bash
# Sync environment dependencies
uv sync --all-groups

# Fetch stock ESM-2 35M base checkpoint
uv run editjumps fetch-base-checkpoint

# Download a minimal single-unit OAS sample
uv run editjumps download-oas --max-units 1 --pair-chains --emit-cdr-keys

# Install MMseqs2 and generate homolog pairs
bash editjumps/core/install_mmseqs.sh
uv run editjumps build-homolog-pairs

# Configure local tracking (optional: defaults to sqlite:///mlflow.db)
make mlflow-ui &   # launches dashboard at http://127.0.0.1:5555

# Run a 50-step smoke training test
uv run --group train editjumps train-edit-flows \
  --pairs data/pretrain/oas_homolog_pairs.tsv.gz \
  --model-name data/pretrain/base_checkpoints/esm2_t12_35M_UR50D \
  --output-folder data/pretrain/edit_flows \
  --rate-head linear --q-head fresh \
  --max-steps 50 --batch-size 2

# Or run via DVC stage
uv run dvc repro -s train_edit_flows

# Test generation with the trained model
uv run --group train editjumps edit \
  --sequence QVQLVESGGGLVKPGGSLRLSCAASGFTFSDYYMSWIRQAPGKGLEWVSYISSSSSYTNYADSVKGRFTISRDNAKNSLYLQMNSLRAEDTAVYYCAREGILGSGYYSVDYWGQGTLVTVSS \
  --model data/pretrain/edit_flows --rate-head linear --q-head fresh
```

### Parameter and Head Alignment

- `train-edit-flows` uses configuration flags `--rate-head linear` and `--q-head fresh` by default in local smoke settings.
- The `edit` CLI defaults to `--rate-head mlp` and `--q-head esm_lm_head` (matching the main reported experimental checkpoint). When evaluating a locally trained model, pass `--rate-head linear --q-head fresh` to match parameter dimensions.
- `--chain`: Specify `--chain heavy` to train on heavy chains exclusively, or `--chain joined` for joined `VH.VL` pairs.

## 2. Trainable Components

| Command | Component | Target Specification |
|---|---|---|
| `train-edit-flows` | Discrete Jump-Process Editor | EvoFlows §3.4 |
| `pretrain` | ESM-2 Masked LM on OAS Corpus | Continuous Pretraining Trunk |
| `evotune` | Family-Specific Masked LM Adaptation | Baseline (EvoFlows §4.2) |

## 3. Output Management and Tracking

- Model artifacts are written to `--output-folder`.
- Experiment parameters and step metrics are logged to MLflow (`MLFLOW_TRACKING_URI`).
- Provide `--run-tag` and distinct `--output-folder` paths across successive local runs to prevent overwriting artifacts.

## 4. Cloud Training at Scale

Full-scale DAG execution and cloud spot GPU runs:

```bash
# Execute DVC DAG stages
uv run dvc repro

# Launch cloud spot GPU training via SkyPilot
make jobs-train
```

Cloud tasks use SkyPilot (`uv tool install 'skypilot[gcp]'`). Set `$PROJECT_ID` and ensure GCP storage bucket configurations match `deploy/gcp/train.sky.yaml`. For preemption and runtime profiling details, refer to [`performance.md`](performance.md) and [`gcp_setup.md`](gcp_setup.md).
