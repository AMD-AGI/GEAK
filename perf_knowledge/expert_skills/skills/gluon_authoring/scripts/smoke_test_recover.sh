#!/usr/bin/env bash
# Smoke test for the closed-loop Gluon recovery toolchain
# (ttgir_to_gluon.py + recover_gluon.py + dump_ir.sh --emit-gluon).
#
# Two layers:
#   OFFLINE (always, no GPU): parser self-test + recover/verify against the bundled
#           gfx950-gluon-tutorials .ttgir dumps -- proves parsing, layout-equivalence,
#           anchor assembly, and that the emitted Gluon is valid Python.
#   GPU (optional): if triton + torch import, run the full compile -> recover ->
#           recompile -> verify + correctness loop via smoke_recover_gpu.py.
#
# Usage: bash smoke_test_recover.sh [path-to-gfx950-gluon-tutorials]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"   # so the inline/companion python can import the modules
TUT="${1:-$HOME/gfx950-gluon-tutorials}"
DUMPS="$TUT/kernels/gemm/a16w16"
V5="$DUMPS/v5_local_prefetch/ir_dump_K4096_fp16/v5_local_prefetch.ttgir"
V3SW="$DUMPS/v3_lds/ir_dump_K4096_fp16/swizzling_8-2-8/v3_lds_swizzling.ttgir"
V3NO="$DUMPS/v3_lds/ir_dump_K4096_fp16/no_swizzling/v3_lds_swizzling.ttgir"

echo "== 1. parser self-test =="
python3 "$SCRIPT_DIR/ttgir_to_gluon.py" --selftest

echo "== 1b. layout-equivalence self-test =="
# Runs even without the tutorials checkout, which is what makes it the guard for the
# unroll-skew regression: same layouts, different occurrence counts, must still PASS.
python3 "$SCRIPT_DIR/recover_gluon.py" --selftest

echo "== 1c. pipeline-recovery self-tests =="
# Also offline. The candidacy rule and the two version-dependent splice points are the
# things here that can be wrong without failing loudly, so they are pinned rather than
# left to the on-box run. gluon_swp reports itself skipped where no AMD backend imports.
python3 "$SCRIPT_DIR/pipeline_survey.py" --selftest
python3 "$SCRIPT_DIR/patch_reinject.py" --selftest
python3 "$SCRIPT_DIR/patch_async_reinject.py" --selftest
python3 "$SCRIPT_DIR/gluon_swp.py" --selftest

echo "== 1d. layout-bridge, occupancy and capability self-tests =="
# All four were reachable only by hand, which is how a wrong LDS/CU divisor and a wrong
# pipeline verdict both shipped. The bridge's cases include the attention-shaped false
# positive (a loop-carry count must not overrule tt.num_stages) and probe's include the
# per-arch divisor, so both are guarded here rather than on the box.
python3 "$SCRIPT_DIR/ttgir_bridge.py" --selftest
python3 "$SCRIPT_DIR/probe.py" --selftest
# amd_occupancy is where probe DELEGATES the divisor, and its self-test cross-checks
# hw_constants.json -- i.e. it is the one that catches a bad per-arch figure at the source
# rather than in the report. probe_levers degrades to available=None without triton rather
# than failing, so it is offline-safe here too.
python3 "$SCRIPT_DIR/amd_occupancy.py" --selftest
python3 "$SCRIPT_DIR/probe_levers.py" --selftest

if [ -f "$V5" ] && [ -f "$V3SW" ] && [ -f "$V3NO" ]; then
  echo "== 2. recover anchor (--with-pipeline) from a real plain .ttgir =="
  python3 "$SCRIPT_DIR/recover_gluon.py" --ttgir "$V5" --with-pipeline --out /tmp/smoke_anchor.py >/dev/null
  python3 - "$V5" /tmp/smoke_anchor.py <<'PY'
import ast, sys
import ttgir_to_gluon as t2g
ttgir = open(sys.argv[1]).read()
factory = t2g.emit_layout_factory(t2g.parse_layouts(ttgir))
# the recovered layouts must match the hand-written v5 kernel
for needle in ["gl.amd.AMDMFMALayout(version=4, instr_shape=[16, 16, 32]",
               "gl.PaddedSharedLayout([[512, 16]]",
               "gl.DotOperandLayout(operand_index=0, parent=mma, k_width=8)",
               "shape=[256, 64]"]:
    assert needle in factory, f"missing recovered layout: {needle}"
ast.parse(factory)                       # layouts are valid Python
ast.parse(open(sys.argv[2]).read())      # the full anchor is valid Python
print("   recovered layouts OK + anchor parses")
PY

  echo "== 3. verify: plain vs itself must PASS =="
  python3 "$SCRIPT_DIR/recover_gluon.py" --verify --ttgir "$V5" --anchor-ttgir "$V5" >/dev/null

  echo "== 4. verify: swizzle vs no-swizzle must be DETECTED (FAIL) =="
  if python3 "$SCRIPT_DIR/recover_gluon.py" --verify --ttgir "$V3SW" --anchor-ttgir "$V3NO" >/dev/null 2>&1; then
    echo "SMOKE FAIL: layout mismatch was not detected"; exit 1
  fi
  echo "   mismatch correctly detected"
  echo "OFFLINE SMOKE CHECKS PASS"
else
  echo "SKIP offline dump checks (set arg 1 to your gfx950-gluon-tutorials checkout; looked at $DUMPS)"
fi

echo "== 5. GPU end-to-end (optional) =="
if python3 -c "import triton, torch" 2>/dev/null; then
  python3 "$SCRIPT_DIR/smoke_recover_gpu.py"
else
  echo "   SKIP (no triton/torch import); run on the gfx950 box for the full loop"
fi

echo "SMOKE TEST PASS"
