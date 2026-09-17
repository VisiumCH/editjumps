# GCP root inputs — passed straight through to modules/gcp.
variable "project_id" {
  type        = string
  description = "GCP project id."
}
variable "region" {
  type        = string
  default     = "europe-west6" # Zurich
  description = "GCP region."
}
variable "dvc_bucket_name" {
  type        = string
  description = "GCS bucket (bare name) backing the DVC remote."
}
variable "mlflow_bucket_name" {
  type        = string
  description = "GCS bucket (bare name) backing MLflow artifacts."
}
variable "sql_instance" {
  type    = string
  default = "profound-mlflow"
}
variable "sql_db" {
  type    = string
  default = "mlflow"
}
variable "sql_user" {
  type    = string
  default = "mlflow"
}
variable "image" {
  type        = string
  description = "Artifact Registry image for the MLflow Cloud Run service."
}
variable "invoker_members" {
  type = list(string)
  # Who may call the private MLflow service. `roles/editor` does NOT imply run.invoker, so every
  # identity that logs runs has to appear here.
  #
  #   domain:<your-org>   humans, via `gcloud run services proxy profound-mlflow --region <r>`
  #                       (docs/gcp_setup.md). NOT `make mlflow-ui`, which serves the LOCAL
  #                       sqlite file and never contacts this service.
  #   skypilot-v1         what the SkyPilot VMs ACTUALLY run as. Verified, not assumed:
  #                         gcloud compute instances describe <sky vm> \
  #                           --format='value(serviceAccounts[].email)'
  #                       returns skypilot-v1@… for both the jobs controller and the GPU worker.
  #                       Its absence is why no GPU run ever reached MLflow: the VM minted a valid
  #                       metadata identity token and /health answered 403, so every job silently
  #                       fell back to a sqlite file that died with the cluster. The previous
  #                       version of this comment asserted that the *compute* SA was the SkyPilot
  #                       identity, which is what kept the gap invisible - check with the command
  #                       above rather than trusting this list.
  #   compute SA          kept for plain-GCE paths that may rely on it; not the SkyPilot identity.
  # No default: the real identities name an organisation and a project number, so they live in
  # the gitignored `terraform.tfvars`, e.g.
  #     mlflow_invokers = [
  #       "domain:example.com",
  #       "serviceAccount:skypilot-v1@<project>.iam.gserviceaccount.com",
  #       "serviceAccount:<project-number>-compute@developer.gserviceaccount.com",
  #     ]
  description = "IAM members granted run.invoker on the (private) MLflow service."
}
