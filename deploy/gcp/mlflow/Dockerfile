# MLflow tracking server for editjumps — Cloud Run image.
# Postgres (Cloud SQL) backend + GCS artifact store (proxied). Access is gated by
# Cloud Run IAM (run.invoker), so no in-app auth is baked in.
FROM python:3.11-slim

RUN pip install --no-cache-dir \
      "mlflow==3.14.0" \
      "psycopg2-binary" \
      "google-cloud-storage"

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
