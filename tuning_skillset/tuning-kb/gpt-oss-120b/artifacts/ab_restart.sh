#!/usr/bin/env bash
# Interleaved A/B across server restarts.
#
# measurement.md Rule 6b: back-to-back timing on gfx950 both manufactures and
# hides wins. The patches under test are Python/Triton source, so they only take
# effect at process start (Triton JIT + HIP graph capture) -- which means the arms
# must alternate at *restart* granularity rather than be run as two blocks.
#
# Round position is also a confound: throughput climbs ~0.6% over the first rounds
# against a fresh server before it settles. So this runs exactly ONE bench round
# per server start. Every base sample and every candidate sample is then a
# round-1-against-a-fresh-server sample and they are directly comparable. The
# governing noise floor is the restart-to-restart spread (0.501%), not the
# within-instance one.
#
# usage:  [PREFIX=tag] [BASE_PATCHES=a.patch,b.patch] ab_restart.sh <cand_patches|none> [pairs]
#   BASE_PATCHES  applied to BOTH arms (the already-banked stack)
#   $1            applied to the candidate arm ONLY, on top of BASE_PATCHES
#
# Each patch declares its own repo in a `# Repo: <path>` header line, so a single
# A/B can span the sglang and aiter checkouts.
set -u
PATCHES="$1"; shift
PAIRS="${1:-3}"
PREFIX="${PREFIX:-ab}"
BASE_PATCHES="${BASE_PATCHES:-none}"
ORDER="${ORDER:-fwd}"        # fwd = base first in each pair; rev = candidate first

repo_of() {  # $1 = patch path -> repo root
  local r
  r=$(grep -m1 '^# Repo:' "$1" | sed 's/^# Repo:[[:space:]]*//' | awk '{print $1}')
  echo "${r:-/sgl-workspace/sglang}"
}

file_of() {  # $1 = patch path -> repo-relative file
  grep -m1 '^# File:' "$1" | sed 's/^# File:[[:space:]]*//' | awk '{print $1}'
}

all_patches() {
  { [ "$BASE_PATCHES" != none ] && echo "$BASE_PATCHES" | tr ',' '\n'
    [ "$PATCHES" != none ] && echo "$PATCHES" | tr ',' '\n'; } | sed '/^$/d'
}

restore() {  # revert every file any patch touches, in its own repo
  local p
  for p in $(all_patches); do
    ( cd "$(repo_of "$p")" && git checkout -- "$(file_of "$p")" 2>/dev/null )
  done
}

apply() {  # $1 = on|off
  local list p
  restore
  list="$([ "$BASE_PATCHES" != none ] && echo "$BASE_PATCHES" | tr ',' ' ')"
  if [ "$1" = on ] && [ "$PATCHES" != none ]; then
    list="$list $(echo "$PATCHES" | tr ',' ' ')"
  fi
  for p in $list; do
    ( cd "$(repo_of "$p")" && git apply --whitespace=nowarn "$p" ) \
      || { echo "APPLY FAILED $p"; exit 1; }
  done
}

run() {  # $1 = tag
  cd /work || exit 1
  ./scripts/launch_server.sh --stop >/dev/null 2>&1
  sleep 8
  # foreground: the script exits only after it has verified the live server
  if ! ./scripts/launch_server.sh > "/tmp/ab_$1.log" 2>&1; then
    echo "$1: LAUNCH FAILED"; tail -3 "/tmp/ab_$1.log"; return 1
  fi
  if ! grep -q "config verified" "/tmp/ab_$1.log"; then
    echo "$1: CONFIG NOT VERIFIED"; tail -3 "/tmp/ab_$1.log"; return 1
  fi
  TAG="$1" ./scripts/run_bench.sh >/dev/null 2>&1
  d=$(ls -td /work/results/"$1"_* | head -1)
  python3 -c "
import json;j=json.load(open('$d/inferencex_result.json'))
print('$1  %.3f tok/s  ttft %.1f  tpot %.3f'%(j['output_throughput'],j['mean_ttft_ms'],j['mean_tpot_ms']))"
}

echo "base arm : ${BASE_PATCHES}"
echo "cand arm : ${BASE_PATCHES} + ${PATCHES}"
echo "order    : ${ORDER}"
# ORDER=fwd runs base-then-candidate in every pair. That is fine for a large
# effect, but it is a confound for a small one: base is always the earlier of the
# two runs, so any machine-level warm-up trend across the sequence lands on the
# candidate. ORDER=rev runs candidate-first; running one batch each way makes the
# order effect directly estimable instead of merely hoped-away.
for i in $(seq 1 "$PAIRS"); do
  if [ "$ORDER" = rev ]; then
    apply on;  run "${PREFIX}cand_$i"
    apply off; run "${PREFIX}base_$i"
  else
    apply off; run "${PREFIX}base_$i"
    apply on;  run "${PREFIX}cand_$i"
  fi
done
restore
./scripts/launch_server.sh --stop >/dev/null 2>&1
echo "AB DONE (trees restored)"
