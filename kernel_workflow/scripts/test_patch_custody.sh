#!/bin/bash
# Copyright (c) [2026] Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Contract test for the two ends of a candidate patch's custody (no GPU, no model):
#   save_patch.sh   - the engineer's write: atomic (temp file + rename, never truncate-in-place),
#                     refused once the lane has stopped the engineer, git-diff args passed through;
#   freeze_patch.sh - verify's copy: only a patch that holds still is frozen (else exit 3), the frozen
#                     file is never partial, a retried verify keeps the first frozen bytes, and a
#                     missing patch freezes as an empty one.
#
# Run:  bash kernel_workflow/scripts/test_patch_custody.sh
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAVE="$HERE/save_patch.sh"
FREEZE="$HERE/freeze_patch.sh"
T="$(mktemp -d)"
WRITER=""
cleanup() { [ -n "$WRITER" ] && kill -KILL "$WRITER" 2>/dev/null; rm -rf "$T"; }
trap cleanup EXIT

FAIL=0
ok()  { echo "  ok: $1"; }
bad() { echo "  FAIL: $1" >&2; FAIL=$((FAIL + 1)); }

# A workspace with one committed file, then an edit: the candidate.
WS="$T/eng/workspace"; OUT="$T/eng"
mkdir -p "$WS/kernel_src" "$WS/other"
(cd "$WS" && git init -q && printf 'a\n' > kernel_src/k.py && printf 'x\n' > other/o.py && git add -A \
    && git -c user.email=t@t -c user.name=t commit -qm base)
printf 'b\n' > "$WS/kernel_src/k.py"; printf 'y\n' > "$WS/other/o.py"

echo "save_patch.sh:"
bash "$SAVE" >/dev/null 2>&1; [ $? -eq 2 ] && ok "bad usage -> exit 2" || bad "bad usage should exit 2"
bash "$SAVE" "$WS" "$OUT" >/dev/null; rc=$?
[ $rc -eq 0 ] && cmp -s "$OUT/best_patch.diff" <(cd "$WS" && git --no-pager diff) \
    && ok "saves exactly git diff" || bad "saved patch differs from git diff (rc=$rc)"
ls "$OUT"/.best_patch.diff.tmp.* >/dev/null 2>&1 && bad "temp file left behind" || ok "no temp file left behind"

# Atomic replace: a new inode is renamed in, so a reader holding the old file still sees it whole.
exec 7< "$OUT/best_patch.diff"
ino1=$(stat -c %i "$OUT/best_patch.diff")
printf 'c\n' > "$WS/kernel_src/k.py"
bash "$SAVE" "$WS" "$OUT" >/dev/null
ino2=$(stat -c %i "$OUT/best_patch.diff")
[ "$ino1" != "$ino2" ] && ok "replaced by rename (new inode), not rewritten in place" || bad "best_patch.diff was rewritten in place"
grep -q '^+b$' <&7 && ok "a reader of the previous patch still reads it complete" || bad "previous patch changed under its reader"
exec 7<&-
grep -q '^+c$' "$OUT/best_patch.diff" && ok "the new patch is in place" || bad "new content missing"

bash "$SAVE" "$WS" "$OUT" -- kernel_src >/dev/null
grep -q 'other/o.py' "$OUT/best_patch.diff" && bad "pathspec ignored" || ok "git-diff args (pathspec) are passed through"

before=$(sha256sum < "$OUT/best_patch.diff")
touch "$OUT/STOP"
printf 'd\n' > "$WS/kernel_src/k.py"
err="$(bash "$SAVE" "$WS" "$OUT" 2>&1)"; rc=$?
[ $rc -eq 3 ] && echo "$err" | grep -q GEAK_ENGINEER_STOPPED && ok "refused after STOP (exit 3)" || bad "expected refusal after STOP (rc=$rc)"
[ "$(sha256sum < "$OUT/best_patch.diff")" = "$before" ] && ok "best_patch.diff untouched after STOP" || bad "best_patch.diff changed after STOP"
rm -f "$OUT/STOP"

echo "freeze_patch.sh:"
bash "$FREEZE" >/dev/null 2>&1; [ $? -eq 2 ] && ok "bad usage -> exit 2" || bad "bad usage should exit 2"
FZ="$T/eng/verify/candidate.diff"
FREEZE_SETTLE_SEC=1 bash "$FREEZE" "$OUT/best_patch.diff" "$FZ" >/dev/null; rc=$?
[ $rc -eq 0 ] && cmp -s "$OUT/best_patch.diff" "$FZ" && ok "a stable patch is frozen byte for byte" || bad "freeze of a stable patch failed (rc=$rc)"
ls "$FZ".tmp.* >/dev/null 2>&1 && bad "temp file left behind" || ok "no temp file left behind"

# A retried verify keeps the first frozen bytes, whatever the source says now.
first=$(sha256sum < "$FZ")
printf 'e\n' > "$WS/kernel_src/k.py"; bash "$SAVE" "$WS" "$OUT" >/dev/null
FREEZE_SETTLE_SEC=1 bash "$FREEZE" "$OUT/best_patch.diff" "$FZ" >/dev/null; rc=$?
[ $rc -eq 0 ] && [ "$(sha256sum < "$FZ")" = "$first" ] && ok "an existing frozen patch is kept" || bad "frozen patch was replaced on retry"

# A patch that keeps changing is never frozen.
FZ2="$T/eng/verify2/candidate.diff"
( i=0; while :; do i=$((i + 1)); printf 'line %s\n' "$i" > "$OUT/moving.diff.tmp"; mv -f "$OUT/moving.diff.tmp" "$OUT/moving.diff"; sleep 0.2; done ) &
WRITER=$!
sleep 0.5
err="$(FREEZE_SETTLE_SEC=1 FREEZE_ATTEMPTS=2 bash "$FREEZE" "$OUT/moving.diff" "$FZ2" 2>&1)"; rc=$?
kill -KILL "$WRITER" 2>/dev/null; wait "$WRITER" 2>/dev/null; WRITER=""
[ $rc -eq 3 ] && echo "$err" | grep -q FREEZE_FAILED && ok "a changing patch -> FREEZE_FAILED (exit 3)" || bad "changing patch should fail the freeze (rc=$rc: $err)"
[ -e "$FZ2" ] && bad "a frozen file was left for a changing patch" || ok "nothing frozen for a changing patch"
ls "$(dirname "$FZ2")"/candidate.diff.tmp.* >/dev/null 2>&1 && bad "temp file left behind" || ok "no temp file left behind"

# No patch at all freezes as an empty one (verify then reports apply_failed, as before).
FZ3="$T/eng/verify3/candidate.diff"
FREEZE_SETTLE_SEC=1 bash "$FREEZE" "$T/eng/no_such.diff" "$FZ3" >/dev/null; rc=$?
[ $rc -eq 0 ] && [ -e "$FZ3" ] && [ ! -s "$FZ3" ] && ok "missing patch -> empty frozen patch" || bad "missing patch handling (rc=$rc)"

if [ "$FAIL" -ne 0 ]; then
    echo "test_patch_custody: $FAIL failure(s)" >&2
    exit 1
fi
echo "test_patch_custody: all checks passed"
