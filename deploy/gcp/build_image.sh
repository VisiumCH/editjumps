#!/usr/bin/env bash
# Build + push the MLflow server image to Artifact Registry. Terraform deploys the
# service but does not build images, so run this whenever mlflow/ changes, then
# `terraform apply` (a new revision picks up the pushed :latest).
#   source deploy/gcp/config.env && bash deploy/gcp/build_image.sh
set -euo pipefail

: "${PROJECT_ID:?set it in deploy/gcp/config.env}" "${REGION:?}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/profound/mlflow:latest"

gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet
docker buildx build --platform linux/amd64 -t "$IMAGE" --push "${HERE}/mlflow"
echo "pushed $IMAGE"
