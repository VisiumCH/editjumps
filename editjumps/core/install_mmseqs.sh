#!/usr/bin/env bash
# Install the MMseqs2 CLI binary into this project's uv-managed .venv/bin.
#
# MMseqs2 (used by editjumps/core/cluster_split.py for the leakage-aware OAS split) is a
# compiled C++ tool with no legitimate PyPI wheel, so `uv add` cannot install it. So:
# download the project's official static binary (same channel as Homebrew/conda/Docker)
# into .venv/bin, where it resolves on the same PATH as `uv run`, with no env to manage.
#
# Usage: bash editjumps/core/install_mmseqs.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_BIN="$REPO_ROOT/.venv/bin"

if [ ! -d "$VENV_BIN" ]; then
    echo "error: $VENV_BIN not found - run 'uv sync' first" >&2
    exit 1
fi

if "$VENV_BIN/python" -c "import platform,sys; sys.exit(0 if platform.system()=='Darwin' else 1)"; then
    URL="https://mmseqs.com/latest/mmseqs-osx-universal.tar.gz"
else
    ARCH=$(uname -m)
    if [ "$ARCH" = "aarch64" ]; then
        URL="https://mmseqs.com/latest/mmseqs-linux-arm64.tar.gz"
    else
        URL="https://mmseqs.com/latest/mmseqs-linux-avx2.tar.gz"
    fi
fi

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

echo "downloading $URL ..."
curl -sL -o "$TMP/mmseqs.tar.gz" "$URL"
tar xzf "$TMP/mmseqs.tar.gz" -C "$TMP"
cp "$TMP/mmseqs/bin/mmseqs" "$VENV_BIN/mmseqs"
chmod +x "$VENV_BIN/mmseqs"

echo "installed: $("$VENV_BIN/mmseqs" version)"
echo "location:  $VENV_BIN/mmseqs"
