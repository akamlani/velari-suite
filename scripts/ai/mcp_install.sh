#!/usr/bin/env bash
# Install the MCP Inspector (https://github.com/modelcontextprotocol/inspector) globally via npm.
#
# Usage:
#   ./scripts/ai/mcp_install.sh

set -euo pipefail

PACKAGE="@modelcontextprotocol/inspector"

if ! command -v npm >/dev/null 2>&1; then
    echo "npm is required to install the MCP Inspector — install Node.js first (https://nodejs.org)." >&2
    exit 1
fi

echo "Installing ${PACKAGE}..."
npm install -g "${PACKAGE}"

if ! command -v mcp-inspector >/dev/null 2>&1; then
    echo "Install finished, but mcp-inspector is not on PATH — check your npm global bin directory." >&2
    exit 1
fi

echo
echo "Installed: $(npm list -g "${PACKAGE}" --depth=0 | tail -n 1)"
echo "Run it with: mcp-inspector          # web UI (default)"
echo "         or: mcp-inspector --cli    # CLI mode"
echo "         or: mcp-inspector --tui    # TUI mode"
