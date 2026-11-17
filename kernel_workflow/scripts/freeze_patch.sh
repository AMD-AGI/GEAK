#!/bin/bash
# Copyright (c) [2026] Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Freeze a candidate patch for verify: the file verify measures, and the lane integrates and commits.
#
# The source is the engineer's best_patch.diff, which an engineer that is still running could be
# rewriting. The copy is accepted only when the source is stable: its hash is the same before and after
# a settle interval and equals the copy's. Otherwise the copy is retried, and after the last attempt the
# freeze FAILS (exit 3) rather than hand verify a patch that may still be changing. The copy is written
# to a temporary file and renamed into place, so the frozen file is never partial.
#
# An existing frozen file is kept as is: a retried verify must measure the same bytes as the first one.
# A missing source freezes as an empty patch (verify then reports apply_failed, as for no patch at all).
#
# Usage: freeze_patch.sh <source_patch> <frozen_patch>
#   exit 0 frozen (or already frozen), 2 bad usage, 3 source never stable / copy failed
# Env: FREEZE_SETTLE_SEC (default 2), FREEZE_ATTEMPTS (default 3)
set -uo pipefail

if [ $# -ne 2 ] || [ -z "$1" ] || [ -z "$2" ]; then
    echo "usage: freeze_patch.sh <source_patch> <frozen_patch>" >&2
    exit 2
fi
SRC="$1"
DST="$2"
SETTLE="${FREEZE_SETTLE_SEC:-2}"
ATTEMPTS="${FREEZE_ATTEMPTS:-3}"

mkdir -p "$(dirname "$DST")" || { echo "FREEZE_FAILED: cannot create $(dirname "$DST")" >&2; exit 3; }
if [ -e "$DST" ]; then
    echo "[FREEZE] $DST already frozen ($(sha256sum < "$DST" | cut -c1-16)); keeping it"
    exit 0
fi

hash_of() { if [ -e "$1" ]; then sha256sum < "$1" | cut -d' ' -f1; else echo "absent"; fi; }

TMP="$DST.tmp.$$"
for attempt in $(seq 1 "$ATTEMPTS"); do
    before="$(hash_of "$SRC")"
    if [ "$before" = "absent" ]; then
        : > "$TMP" || break
    else
        cp "$SRC" "$TMP" 2>/dev/null || { rm -f "$TMP"; sleep "$SETTLE"; continue; }
    fi
    sleep "$SETTLE"
    after="$(hash_of "$SRC")"
    copied="$(hash_of "$TMP")"
    if [ "$before" = "$after" ] && { [ "$before" = "absent" ] || [ "$copied" = "$before" ]; }; then
        mv -f "$TMP" "$DST" || break
        echo "[FREEZE] $SRC -> $DST ($(wc -c < "$DST") bytes, sha256 $(hash_of "$DST" | cut -c1-16))"
        [ "$before" = "absent" ] && echo "[FREEZE] source was absent: frozen as an empty patch"
        exit 0
    fi
    echo "[FREEZE] attempt $attempt/$ATTEMPTS: $SRC changed during the freeze ($before -> $after); retrying" >&2
    rm -f "$TMP"
done
rm -f "$TMP"
echo "FREEZE_FAILED: $SRC did not stay stable for ${SETTLE}s in $ATTEMPTS attempts; not freezing a changing patch" >&2
exit 3
