#!/usr/bin/env bash
# Launch `mlflow ui` against this repo's local SQLite tracking store.
#
# On Python 3.14, mlflow 3.14.0's server dies with "cannot import name 'Traversable'
# from 'importlib.abc'" (https://github.com/mlflow/mlflow/issues/24155). So
# editjumps/repo/_py314_shims/ goes on PYTHONPATH; its sitecustomize.py patches importlib.abc
# back in before mlflow imports it. Delete both once mlflow ships a fix.
#
# Usage: bash editjumps/repo/mlflow_ui.sh [extra mlflow ui args]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHONPATH="$REPO_ROOT/editjumps/repo/_py314_shims${PYTHONPATH:+:$PYTHONPATH}" \
  .venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db "$@"
