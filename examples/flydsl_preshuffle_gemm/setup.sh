#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Install pinned FlyDSL dependency for the flydsl_preshuffle_gemm example.
# Usage: bash setup.sh
# Idempotent — skips install if flydsl is already at the correct version.

set -euo pipefail

FLYDSL_VERSION="0.1.2"

INSTALLED_VER="$(python3 -c "import flydsl; print(flydsl.__version__)" 2>/dev/null || echo "")"

# Accept exact match or dev builds of the same minor (e.g. 0.1.2.dev463)
if [[ "${INSTALLED_VER}" == "${FLYDSL_VERSION}" || "${INSTALLED_VER}" == "${FLYDSL_VERSION}."* ]]; then
    echo "flydsl ${INSTALLED_VER} already installed (compatible with ${FLYDSL_VERSION})"
    exit 0
fi

echo "Installing flydsl==${FLYDSL_VERSION} ..."
pip install "flydsl==${FLYDSL_VERSION}" --quiet
echo "Done. flydsl ${FLYDSL_VERSION} is ready."
