#!/usr/bin/env bash
# How many hardware counters does rocprofv3 accept in ONE pass on this part?
#
# Ships with amd_rdna.md section 7 item 5. That item reports this cumulative set
# getting three deep and aborting on the fourth with "error code 38: Request
# exceeds the capabilities of the hardware to collect"; this script has to
# actually demonstrate that, so it sweeps 1..5 rather than bisecting.
#
# Two things it deliberately does NOT claim:
#   * that three of ANYTHING is safe. The sweep is cumulative over ONE ordered
#     set, so the result describes that set. A metric named in a `pmc:` line is
#     not necessarily one hardware counter -- some expand into several -- so a
#     different trio can exceed the same budget. Probe the set you want.
#   * that the ~85 s per invocation is a rocprofv3 cost. It is a fresh-process
#     cost: every iteration starts a new python3 that imports torch and
#     initialises HIP. Inside a warm harness the same passes cost ~5 s each.
#
# Exit status: 0 if the sweep ran, 1 if it could not run at all (no rocprofv3,
# no GPU, missing workload). A counter set that fails is a RESULT, not an error.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

command -v rocprofv3 >/dev/null 2>&1 || { echo "error: rocprofv3 not on PATH" >&2; exit 1; }
[ -f "$SCRIPT_DIR/pmc_work.py" ] || { echo "error: pmc_work.py not next to this script" >&2; exit 1; }

PY="${PYTHON:-python3}"
"$PY" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null \
  || { echo "error: no usable GPU for ${PY}" >&2; exit 1; }

# One cumulative set, cheapest first. Not a claim that these are atomic.
ALL="SQ_WAVES GRBM_GUI_ACTIVE FETCH_SIZE WRITE_SIZE GL2C_HIT"
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT

GFX="$(rocminfo 2>/dev/null | grep -oE 'gfx[0-9a-f]+' | head -1)"
echo "counter-budget sweep on ${GFX:-unknown gfx}"
for n in 1 2 3 4 5; do
  C="$(echo "$ALL" | cut -d' ' -f"1-$n")"
  rm -rf "$OUT/run"; mkdir -p "$OUT/run"
  printf 'pmc: %s\n' "$C" > "$OUT/in.txt"
  log="$(timeout 300 rocprofv3 -i "$OUT/in.txt" -d "$OUT/run" -- "$PY" pmc_work.py 2>&1)"
  if printf '%s' "$log" | grep -qiE 'exceeds the capabilities|error code 38|aborted'; then
    printf '  n=%d  FAIL (counter budget exceeded)  [%s]\n' "$n" "$C"
  elif printf '%s' "$log" | grep -q '^ok'; then
    printf '  n=%d  OK                              [%s]\n' "$n" "$C"
  else
    printf '  n=%d  ????  %s\n' "$n" "$(printf '%s' "$log" | tail -1 | cut -c1-70)"
  fi
done
echo "note: a FAIL above is a measurement, not a script error."
