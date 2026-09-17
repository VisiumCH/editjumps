# Model Weights and Checkpoints

Specification of model weight storage, conversion workflows, and third-party dependencies.

## Contents

- [Artifact availability](#artifact-availability)
- [Checkpoint formats](#checkpoint-formats)
- [Restoring an editor from a checkpoint](#restoring-an-editor-from-a-checkpoint)
- [Loading a restored editor](#loading-a-restored-editor)
- [Third-party weights](#third-party-weights)
- [Local model training](#local-model-training)

## Artifact availability

Trained model checkpoints are stored in cloud object storage (`gs://$DVC_BUCKET/`) requiring Google Cloud authentication:

| Artifact | Location | Description |
|---|---|---|
| Editor training checkpoints | `gs://$DVC_BUCKET/checkpoints/experiments/<RUN_TAG>/checkpoint.pt` | Periodic optimizer and model state snapshots for preemption recovery |
| Loadable editor folders | `gs://$DVC_BUCKET/models/edit_flows/<RUN_TAG>/` | Target layout (`encoder/` + `evoflows_model.pt`) used by evaluation CLI |
| OAS-adapted ESM trunk | DVC-tracked `data/pretrain/esm2_oas` | ESM-2 continued pretraining on OAS heavy/joined sequences |
| Disjoint evotuned trunks | `gs://$DVC_BUCKET/models/esm2_evotuned/evotune-dj-{ty1,her2vh}/` | ESM-2 adapted on family members disjoint from editor training pairs |
| Experiment metrics & DBs | `gs://$DVC_BUCKET/repro-artifacts/` | Pipeline logs, MLflow SQLite DBs, and evaluation metrics |

Publicly accessible dependencies:
- **Base Models:** Standard ESM-2 trunks (`facebook/esm2_t12_35M_UR50D`, `facebook/esm2_t33_650M_UR50D`) and EvoDiff-MSA checkpoints are retrieved from public registries.
- **Data Pipeline:** Source datasets (OAS, OAS homolog clusters, seed families) are fetched from public repositories via `uv run dvc repro`.
- **Benchmark Metrics:** All reported evaluation numbers are committed under `metrics/` and documented in [`findings.md`](findings.md).

## Checkpoint formats

Training pipelines generate two distinct artifact representations:

| Artifact | Generation | Contents | Consumer |
|---|---|---|---|
| `checkpoint.pt` | Periodic snapshots during training | Model weights, optimizer state, and RNG state | Training loop resume handler |
| `encoder/` + `evoflows_model.pt` | Run completion export | Model architecture weights in evaluatable layout | `EvoFlowsModel.load_trained` / evaluation CLI |

If a training job terminates leaving only `checkpoint.pt`, use `restore-editor` to reconstruct the evaluatable folder format.

## Restoring an editor from a checkpoint

`editjumps restore-editor` (`editjumps/pipeline/train/restore_editor.py`) converts raw training checkpoints into the loadable directory format:

```bash
make restore-editor CKPT=gs://$DVC_BUCKET/checkpoints/experiments/<RUN_TAG>/checkpoint.pt
```

Explicit CLI options:

```bash
uv run --group train editjumps restore-editor \
  --checkpoint gs://$DVC_BUCKET/checkpoints/experiments/<RUN_TAG>/checkpoint.pt \
  --output-folder data/pretrain/edit_flows_restored \
  --model-name data/pretrain/esm2_oas \
  --rate-head mlp --q-head esm_lm_head
```

Configuration notes:
- `--model-name`: Base trunk path or HuggingFace identifier (e.g. `data/pretrain/esm2_oas` or `facebook/esm2_t12_35M_UR50D`).
- `--rate-head` and `--q-head`: Must match the training head architecture (`mlp` vs `linear`, `esm_lm_head` vs `fresh`).
- `--verify`: Enabled by default; executes a test load via `EvoFlowsModel.load_trained` prior to exiting.

## Loading a restored editor

Evaluations consume the exported model directory:

```bash
make generation-eval MODEL=data/pretrain/edit_flows_restored N_TEMPLATES=20 N_VARIANTS=20
```

`RATE_HEAD` and `Q_HEAD` environment variables must match the checkpoint architecture.

## Third-party weights

External foundation models are downloaded on demand:

| Model | Source | Fetch Method |
|---|---|---|
| ESM-2 (35M, 650M) | HuggingFace (`facebook/esm2_t12_35M_UR50D`, `facebook/esm2_t33_650M_UR50D`) | `editjumps fetch-base-checkpoint` or auto-downloaded by HuggingFace |
| EvoDiff-MSA | Zenodo record 8045076 (`msa-oaar-maxsub.tar`) | Handled via `install_evodiff.sh` smoke verification |

## Local model training

To train from scratch using public datasets:

```bash
uv sync --all-groups
bash editjumps/core/install_mmseqs.sh
uv run dvc repro
export DVC_BUCKET="gs://<your-bucket-name>"
make jobs-train
```

Configure `deploy/gcp/train.sky.yaml` and GCP project settings as outlined in [`gcp_setup.md`](gcp_setup.md). Refer to [`performance.md`](performance.md) for compute resource requirements.

## Where to go next

| Document | Description |
|---|---|
| [`performance.md`](performance.md) | Resource profiles and runtime benchmarks |
| [`reproducing.md`](reproducing.md) | Reproduction workflows and comparability controls |
| [`gcp_setup.md`](gcp_setup.md) | Cloud infrastructure and instance configuration |
| [`known_issues.md`](known_issues.md) | Diagnostic symptoms and operational caveats |

