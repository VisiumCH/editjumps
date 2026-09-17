terraform {
  required_version = ">= 1.6"
  required_providers {
    google = { source = "hashicorp/google", version = "~> 6.0" }
  }
  # The state bucket is NOT named here: it identifies whoever owns it, and this repository is
  # meant to be publishable without naming them. Pass it at init time instead --
  #     terraform init -backend-config=backend.hcl
  # with a gitignored backend.hcl holding `bucket = "your-tfstate-bucket"`.
  backend "gcs" {
    prefix = "mlflow"
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}
