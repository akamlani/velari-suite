#!/usr/bin/env bash
# Start the Arize Phoenix tracing server, configured from config/tracing/phoenix.yaml.
# Delegates to examples/ai/integrations/arize/serve_cli.py — see that file (and
# examples/ai/integrations/arize/README.md) for what's configurable and how.
#
# Usage:
#   ./scripts/ai/start_phoenix.sh
#   ./scripts/ai/start_phoenix.sh phoenix.connection.port=6007   # Hydra config override

set -euo pipefail

cd "$(dirname "$0")/../.."
uv run --extra evals python examples/ai/integrations/arize/serve_cli.py "$@"
