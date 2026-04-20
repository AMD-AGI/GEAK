#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Install FlyDSL dependency for the flydsl_preshuffle_gemm example.
# Usage: bash setup.sh
# Idempotent — skips install if flydsl is already available.

set -euo pipefail

if python3 -c "import flydsl" 2>/dev/null; then
    INSTALLED_VER="$(python3 -c 'import flydsl; print(flydsl.__version__)' 2>/dev/null || echo 'unknown')"
    echo "flydsl ${INSTALLED_VER} already installed."
    exit 0
fi

echo "Installing flydsl ..."
pip install flydsl --quiet
echo "Done. flydsl is ready."
