#!/usr/bin/env bash
# Cloud Run entrypoint for the MLflow tracking server. Builds the Postgres backend
# URI from injected parts (password from Secret Manager, never baked in), then
# serves. Access is gated by Cloud Run IAM (run.invoker) — no in-app auth.
set -euo pipefail

: "${INSTANCE_CONNECTION_NAME:?}" "${SQL_USER:?}" "${SQL_DB:?}" "${MLFLOW_DB_PASSWORD:?}" "${MLFLOW_ARTIFACT_ROOT:?}"

# Cloud SQL is reached over the unix socket Cloud Run mounts at /cloudsql/<conn>.
BACKEND_URI="postgresql://${SQL_USER}:${MLFLOW_DB_PASSWORD}@/${SQL_DB}?host=/cloudsql/${INSTANCE_CONNECTION_NAME}"

# --workers is an mlflow flag (its default is 4, which OOMs a small instance); keep it
# low and matched to the CPU. --gunicorn-opts is only for the request timeout.
exec mlflow server \
  --host 0.0.0.0 --port "${PORT:-8080}" \
  --workers 2 \
  --backend-store-uri "$BACKEND_URI" \
  --artifacts-destination "$MLFLOW_ARTIFACT_ROOT" \
  --serve-artifacts \
  --gunicorn-opts "--timeout 300"
