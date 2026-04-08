#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Clone the ROCm/aiter dependency at the pinned commit for the mla_decode example.
# Usage: bash setup.sh
# Idempotent — skips clone if .aiter_repo/ already exists at the correct commit.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AITER_DIR="${SCRIPT_DIR}/.aiter_repo"
AITER_REPO="https://github.com/ROCm/aiter.git"
AITER_COMMIT="22122345c03991cb8026947b8df05e02f50d1f88"

# Check if already cloned at the right commit
if [ -d "${AITER_DIR}/.git" ]; then
    CURRENT_COMMIT="$(git -C "${AITER_DIR}" rev-parse HEAD 2>/dev/null || echo "")"
    if [ "${CURRENT_COMMIT}" = "${AITER_COMMIT}" ]; then
        echo "aiter already cloned at ${AITER_COMMIT:0:12} in ${AITER_DIR}"
        exit 0
    fi
    echo "aiter commit mismatch (have ${CURRENT_COMMIT:0:12}, want ${AITER_COMMIT:0:12}), re-cloning..."
    rm -rf "${AITER_DIR}"
fi

echo "Cloning ROCm/aiter at ${AITER_COMMIT:0:12} into ${AITER_DIR} ..."
git clone --quiet "${AITER_REPO}" "${AITER_DIR}"
git -C "${AITER_DIR}" checkout --quiet "${AITER_COMMIT}"
echo "Done. aiter is ready at ${AITER_DIR}"
