#!/bin/bash
# Copyright (c) [2026] Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Save an engineer's candidate patch atomically.
#
# best_patch.diff is read by the lane while the engineer may still be running (a timed-out engineer is
# stopped and its patch frozen by verify). `git diff > best_patch.diff` truncates the file first and then
# writes it, so a reader -- or a kill -- in between sees a partial patch. This script writes the diff to a
# temporary file in the same directory and renames it over best_patch.diff: rename(2) on one filesystem is
# atomic, so best_patch.diff is always either the previous complete patch or the new complete one.
# It also refuses to write once the lane has stopped the engineer (<output_dir>/STOP exists).
#
# Usage: save_patch.sh <kernel_path> <output_dir> [git-diff args, e.g. -- kernel_src]
#   exit 0 saved, 2 bad usage, 3 refused (engineer stopped), 1 git diff failed (best_patch.diff untouched)
set -uo pipefail

if [ $# -lt 2 ] || [ ! -d "$1" ] || [ ! -d "$2" ]; then
    echo "usage: save_patch.sh <kernel_path> <output_dir> [git-diff args]  (both existing directories)" >&2
    exit 2
fi
KERNEL_PATH="$1"
OUT_DIR="$2"
shift 2

if [ -e "$OUT_DIR/STOP" ]; then
    echo "GEAK_ENGINEER_STOPPED: $OUT_DIR/STOP exists; the lane has frozen this engineer's patch, not saving." >&2
    exit 3
fi

TMP="$OUT_DIR/.best_patch.diff.tmp.$$"
if ! (cd "$KERNEL_PATH" && git --no-pager diff "$@") > "$TMP"; then
    rm -f "$TMP"
    echo "save_patch.sh: git diff failed in $KERNEL_PATH; best_patch.diff left unchanged" >&2
    exit 1
fi
mv -f "$TMP" "$OUT_DIR/best_patch.diff"
echo "[SAVE-PATCH] $OUT_DIR/best_patch.diff ($(wc -c < "$OUT_DIR/best_patch.diff") bytes)"
