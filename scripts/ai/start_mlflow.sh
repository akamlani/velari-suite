#!/usr/bin/env bash
# Start the MLflow tracking server, configured from config/tracing/mlflow.yaml.
# Delegates to examples/ai/integrations/mlflow/serve_cli.py — see that file (and
# examples/ai/integrations/mlflow/README.md) for what's configurable and how.
#
# Usage:
#   ./scripts/ai/start_mlflow.sh
#   ./scripts/ai/start_mlflow.sh mlflow.connection.port=5099   # Hydra config override

set -euo pipefail

cd "$(dirname "$0")/../.."
uv run --isolated --with mlflow python examples/ai/integrations/mlflow/serve_cli.py "$@"
