# Terraform: MLflow backing store (GCP)

A single flat root — one live cloud, so no module indirection. Runbook and the one-time
state-bucket bootstrap: [`docs/gcp_setup.md`](../../../docs/gcp_setup.md).

```
main.tf       # all resources (GCS + Cloud SQL + Secret Manager + Cloud Run + Artifact Registry)
variables.tf  # inputs
versions.tf   # google provider + GCS backend
outputs.tf    # mlflow_url
terraform.tfvars(.example)   # per-env values (gitignored)
```

```bash
terraform init && terraform apply
```

## Multi-Cloud Considerations

This Terraform configuration provides a single-root deployment targeting Google Cloud Platform. Multi-cloud deployment patterns can be introduced by decomposing root resources into provider-specific modules (`modules/<cloud>/`).
