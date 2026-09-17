# editjumps GCP backing store — a single flat root (one live cloud). If multi-cloud
# ever becomes real, extract these resources into a module and add a per-cloud root;
# see git history (the modules/ split) or the datatonic terraform-google-mlflow module.
locals {
  connection_name = "${var.project_id}:${var.region}:${var.sql_instance}"
  run_sa_email    = google_service_account.run.email
}

# --- Storage: DVC data + MLflow artifacts -------------------------------------
resource "google_storage_bucket" "dvc" {
  name                        = var.dvc_bucket_name
  location                    = var.region
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"
}

resource "google_storage_bucket" "mlflow" {
  name                        = var.mlflow_bucket_name
  location                    = var.region
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"
}

# --- Secret: DB password (container only; the value is managed out-of-band) ---
resource "google_secret_manager_secret" "db" {
  secret_id = "mlflow-db-password"
  replication {
    auto {}
  }
}

data "google_secret_manager_secret_version" "db" {
  secret = google_secret_manager_secret.db.id
}

# --- MLflow backend: Cloud SQL Postgres ---------------------------------------
resource "google_sql_database_instance" "mlflow" {
  name             = var.sql_instance
  database_version = "POSTGRES_16"
  region           = var.region
  settings {
    tier                        = "db-f1-micro"
    edition                     = "ENTERPRISE"
    disk_autoresize             = true
    enable_dataplex_integration = true # matches the instance's current default
  }
}

resource "google_sql_database" "mlflow" {
  name     = var.sql_db
  instance = google_sql_database_instance.mlflow.name
}

resource "google_sql_user" "mlflow" {
  name     = var.sql_user
  instance = google_sql_database_instance.mlflow.name
  password = data.google_secret_manager_secret_version.db.secret_data
}

# --- Cloud Run runtime service account + least-privilege IAM ------------------
resource "google_service_account" "run" {
  account_id   = "profound-mlflow-run"
  display_name = "editjumps MLflow (Cloud Run)"
}

resource "google_storage_bucket_iam_member" "mlflow_obj" {
  bucket = google_storage_bucket.mlflow.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${local.run_sa_email}"
}

resource "google_project_iam_member" "run_sql" {
  project = var.project_id
  role    = "roles/cloudsql.client"
  member  = "serviceAccount:${local.run_sa_email}"
}

resource "google_project_iam_member" "run_secret" {
  project = var.project_id
  role    = "roles/secretmanager.secretAccessor"
  member  = "serviceAccount:${local.run_sa_email}"
}

# --- Image registry -----------------------------------------------------------
resource "google_artifact_registry_repository" "profound" {
  repository_id = "profound"
  location      = var.region
  format        = "DOCKER"
}

# --- MLflow tracking server (Cloud Run) ---------------------------------------
resource "google_cloud_run_v2_service" "mlflow" {
  name     = "profound-mlflow"
  location = var.region
  ingress  = "INGRESS_TRAFFIC_ALL"

  # Service-level scaling block Cloud Run populates in state; declared so plan is a no-op.
  scaling {
    min_instance_count = 0
  }

  template {
    service_account = google_service_account.run.email
    scaling {
      min_instance_count = 0
      max_instance_count = 2
    }
    containers {
      image = var.image
      ports { container_port = 8080 }
      resources {
        # 2 gunicorn workers (see entrypoint) need headroom; 1Gi OOM-killed them mid-request.
        limits            = { cpu = "2", memory = "2Gi" }
        startup_cpu_boost = true
      }
      env {
        name  = "INSTANCE_CONNECTION_NAME"
        value = local.connection_name
      }
      env {
        name  = "SQL_USER"
        value = var.sql_user
      }
      env {
        name  = "SQL_DB"
        value = var.sql_db
      }
      env {
        name  = "MLFLOW_ARTIFACT_ROOT"
        value = "gs://${var.mlflow_bucket_name}/mlflow"
      }
      # Skip MLflow's DNS-rebinding Host-header check: the *.run.app host isn't in the
      # default allowlist, and Cloud Run IAM (a bearer token) is already the real gate.
      env {
        name  = "MLFLOW_SERVER_ALLOWED_HOSTS"
        value = "*"
      }
      env {
        name = "MLFLOW_DB_PASSWORD"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.db.secret_id
            version = "latest"
          }
        }
      }
      # Cloud Run auto-mounts the cloud_sql_instance volume at /cloudsql; declared to match state.
      volume_mounts {
        name       = "cloudsql"
        mount_path = "/cloudsql"
      }
    }
    volumes {
      name = "cloudsql"
      cloud_sql_instance {
        instances = [local.connection_name]
      }
    }
  }

  depends_on = [
    google_project_iam_member.run_sql,
    google_project_iam_member.run_secret,
    google_sql_database.mlflow,
    google_sql_user.mlflow,
  ]
}

# Private service: access is IAM-gated (default: the whole org domain via `gcloud proxy`).
resource "google_cloud_run_v2_service_iam_member" "invoker" {
  for_each = toset(var.invoker_members)
  name     = google_cloud_run_v2_service.mlflow.name
  location = var.region
  role     = "roles/run.invoker"
  member   = each.value
}
