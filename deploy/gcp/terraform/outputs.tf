output "mlflow_url" {
  value       = google_cloud_run_v2_service.mlflow.uri
  description = "MLflow tracking server URL (set MLFLOW_TRACKING_URI to this)."
}
