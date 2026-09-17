# GCP Backing Store and Infrastructure

Specification and setup for cloud storage (DVC), artifact repositories, and experiment tracking (MLflow) on Google Cloud Platform, provisioned via Terraform under [`deploy/gcp/terraform/`](../deploy/gcp/terraform).

| Component | GCP Service | Terraform Resource |
|---|---|---|
| DVC remote (`data/**`) | GCS bucket | `google_storage_bucket.dvc` |
| MLflow artifacts | GCS bucket (proxied) | `google_storage_bucket.mlflow` |
| MLflow backend metadata | Cloud SQL PostgreSQL | `google_sql_database_instance.mlflow` |
| MLflow tracking server | Cloud Run (IAM-protected) | `google_cloud_run_v2_service.mlflow` |

Authentication uses Application Default Credentials (ADC). Access to the Cloud Run service is restricted to authenticated identities via `roles/run.invoker`.

## Initial Setup

```bash
gcloud auth application-default login                 # Configure ADC
cp deploy/gcp/config.env.example deploy/gcp/config.env # Define project, region, bucket variables
source deploy/gcp/config.env

# Create state storage bucket for Terraform
gcloud storage buckets create "gs://${PROJECT_ID}-profound-tfstate" \
  --location="$REGION" --uniform-bucket-level-access && \
  gcloud storage buckets update "gs://${PROJECT_ID}-profound-tfstate" --versioning

bash deploy/gcp/build_image.sh                         # Build and publish tracking server image

cd deploy/gcp/terraform
cp terraform.tfvars.example terraform.tfvars           # Set project, buckets, image
terraform init && terraform apply
```

`terraform output mlflow_url` reports the deployed service endpoint.

## MLflow Configuration

Tracking endpoints resolve through `editjumps/core/utils.py` in priority order:
1. `MLFLOW_TRACKING_URI` environment variable
2. `params.yaml` setting `mlflow.tracking_uri`
3. Local SQLite database (`sqlite:///mlflow.db`)

If the remote tracking server is unconfigured (`params.yaml: tracking_uri: null`), runs log locally without raising connection errors.

### Accessing the Web UI

To view the IAM-protected Cloud Run interface:

```bash
gcloud run services proxy profound-mlflow --region europe-west6 --port 5000
```

To view locally stored runs, run `make mlflow-ui`.

## DVC Storage

Bucket targets are defined in `.dvc/config`. Syncing operates via ADC:

```bash
make dvc-pull
make dvc-push
```

## Cloud Job Orchestration (SkyPilot)

Cloud execution requires SkyPilot (`uv tool install 'skypilot[gcp]'`):

```bash
sky check gcp
```

Configurations are located in `deploy/gcp/`:
- `deploy/gcp/train.sky.yaml`: Model training execution
- `deploy/gcp/repro.sky.yaml`: Pipeline DAG reproduction and remote push

Launch spot execution with automatic teardown:

```bash
sky launch -c editjumps-repro deploy/gcp/repro.sky.yaml \
  --retry-until-up --down --env STAGES=pretrain_esm
```

### Operational Requirements

1. **Service Account Permissions:** SkyPilot VM service accounts (`skypilot-v1@`) require `roles/run.invoker` to post metrics to the Cloud Run MLflow server. Ensure `invoker_members` is applied in Terraform.
2. **CUDA Driver Compatibility:** Ensure PyTorch CUDA wheel builds match the instance NVIDIA driver. SkyPilot recipes assert `torch.cuda.is_available()` during both setup and runtime phases to prevent fallback to CPU compute (see [`known_issues.md`](known_issues.md)).

### Managed Spot Execution

For multi-hour workloads, managed spot jobs provide automatic failure recovery and resume from the latest GCS checkpoint:

```bash
make jobs-repro     # Spot reproduction pipeline (STAGES=pretrain_esm by default)
make jobs-train     # Managed spot editor training
make jobs-status    # Status inspection
make jobs-logs NAME=editjumps-repro
make jobs-cancel NAME=editjumps-repro
```

Checkpoints synchronize to `gs://$DVC_BUCKET/checkpoints/<job>`. Upon preemption, SkyPilot reschedules the job and training resumes from the last completed checkpoint step.

## Infrastructure Notes

- **Autoscaling:** Cloud Run scales to zero when idle; the initial request after idle requires ~15–20 s container startup.
- **Worker Configuration:** Cloud Run gunicorn is configured with 2 workers on 2 vCPU / 2 GiB to match memory allocation constraints.

## Where to go next
| Document | Description |
|---|---|
| [`known_issues.md`](known_issues.md) | Diagnostic symptoms and operational caveats |
