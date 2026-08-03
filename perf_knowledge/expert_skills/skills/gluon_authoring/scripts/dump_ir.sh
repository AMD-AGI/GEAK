#!/usr/bin/env bash
# tile-programming-triton-gluon IR dump helper.
# Compiles a kernel via a per-variant TRITON_CACHE_DIR, then copies the
# .ttgir / .llir / .amdgcn artifacts out and strips .amdgcn into a stable .s.
#
# Use for: (a) recovering plain-Triton inferred layouts before transcription
# (the .ttgir shows #blocked/#mma/#shared + num_stages), and (b) verifying each
# Gluon layer landed (compiler-contract.md acceptance signals).
#
# Usage:
#   bash dump_ir.sh <compile_cmd ...> --variant <name> --out <ir_dir> [--knobs "LLIR_SCHED AMDGCN_AS RA_HINTS"]
#       [--emit-gluon layouts|anchor|pipeline] [--kernel module.path:object] [--arch gfx950]
# Example:
#   bash dump_ir.sh python bench.py --version plain --variant plain --out ir/
#   # auto-recover the inferred layouts into Gluon (closes the transcribe loop):
#   bash dump_ir.sh python bench.py --version plain --variant plain --out ir/ --emit-gluon layouts

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash dump_ir.sh <compile_cmd ...> --variant <name> --out <ir_dir>
      [--knobs "LLIR_SCHED AMDGCN_AS RA_HINTS"] [--emit-gluon layouts|anchor|pipeline]
      [--kernel module.path:object] [--arch gfx950]

Any token that is not one of the flags above is part of <compile_cmd>, so the
compile command may appear before, after or around the flags.

  bash dump_ir.sh python bench.py --version plain --variant plain --out ir/
  bash dump_ir.sh python bench.py --version plain --variant plain --out ir/ --emit-gluon layouts
EOF
}

VARIANT="variant"; OUT_DIR="ir"; KNOBS=""; EMIT_GLUON=""; KERNEL=""; ARCH=""
CMD=()
while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help)     usage; exit 0;;
    --variant)     VARIANT="$2"; shift 2;;
    --out)         OUT_DIR="$2"; shift 2;;
    --knobs)       KNOBS="$2"; shift 2;;
    --emit-gluon)  EMIT_GLUON="$2"; shift 2;;
    --kernel)      KERNEL="$2"; shift 2;;
    --arch)        ARCH="$2"; shift 2;;
    *)             CMD+=("$1"); shift;;
  esac
done
[ ${#CMD[@]} -gt 0 ] || { echo "ERROR: no compile command given" >&2; exit 1; }

# Strip CORRECTNESS/BENCH-only flags from the compile CMD: the IR dump only needs the kernel to
# COMPILE, not to run its oracle. A stray `--check` (a correctness flag) that a kernel's driver does
# not recognize used to CRASH the dump (v6: `prof_driver.py: error: unrecognized arguments: --check`)
# -> no asm_audit -> the static op-mix went empty -> the reduction router silently dropped the
# structural T0 directions (defuse / two-pass / packed-atomic). Removing these from a real compile
# command is harmless.
_CLEAN=()
_skip_next=""
for _tok in "${CMD[@]}"; do
  if [ -n "$_skip_next" ]; then _skip_next=""; continue; fi
  case "$_tok" in
    --check|--correctness) continue;;                 # correctness-only: drop
    --backends) _skip_next=1; continue;;              # bench-selector `--backends <sel>`: drop flag+value
    --backends=*) continue;;                          # `--backends=<sel>` form: drop
    *) _CLEAN+=("$_tok");;
  esac
done
[ ${#_CLEAN[@]} -gt 0 ] && CMD=("${_CLEAN[@]}")

# Auto-detect arch from the device when --arch was not given (gfx942 vs gfx950 matters);
# fall back to gfx950 if rocminfo is unavailable.
if [ -z "$ARCH" ]; then
  ARCH="$(rocminfo 2>/dev/null | grep -oE 'gfx9[0-9]{2}|gfx1[0-9]{3}' | head -1)"
  ARCH="${ARCH:-gfx950}"
  echo "[dump_ir] auto-detected arch=$ARCH (override with --arch)" >&2
fi

DEST="$OUT_DIR/$VARIANT"
mkdir -p "$DEST"
CACHE="$(mktemp -d "/tmp/perf_gluon_tile_cache_${VARIANT}.XXXX")"
export TRITON_CACHE_DIR="$CACHE"
rm -rf "${TRITON_CACHE_DIR:?}/"* 2>/dev/null || true

# Optional compiler-contract knobs.
for k in $KNOBS; do
  case "$k" in
    LLIR_SCHED) export TRITON_ENABLE_LLIR_SCHED=1;;
    AMDGCN_AS)  export TRITON_ENABLE_AMDGCN_AS=1;;
    RA_HINTS)   export TRITON_ENABLE_AMDGPU_RA_HINTS=1;;
  esac
done

echo "=== dump_ir variant=$VARIANT knobs='${KNOBS:-none}' cache=$CACHE ==="
"${CMD[@]}"

# Collect the freshest IR artifacts from the cache. Both @triton.jit AND @gluon.jit
# populate TRITON_CACHE_DIR, but the cache nesting depth can differ (gluon.jit may nest
# deeper than one level) -> try the one-level glob first, then a recursive fallback so a
# gluon.jit kernel is not silently missed (feedback: dump was Triton-biased).
for ext in ttgir llir amdgcn; do
  f="$(ls -t "$CACHE"/*/*."$ext" 2>/dev/null | head -1 || true)"
  [ -n "$f" ] || f="$(find "$CACHE" -name "*.$ext" -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2- || true)"
  [ -n "$f" ] && cp "$f" "$DEST/$VARIANT.$ext" && echo "  + $DEST/$VARIANT.$ext"
done
if ! ls "$DEST/$VARIANT".{ttgir,llir,amdgcn} >/dev/null 2>&1; then
  echo "  ! no ttgir/llir/amdgcn in the cache for variant=$VARIANT. For a @gluon.jit kernel make sure it actually compiled (not a cache hit from a prior run); TRITON_ALWAYS_COMPILE=1 forces a fresh compile." >&2
fi

# Kernel METADATA (<kernel>.json) carries `shared` = LDS bytes/workgroup -- the ONLY correct LDS
# source for a Triton kernel (the KD's group_segment_fixed_size and rocprof-compute 7.1.8 are
# structurally 0, because Triton sizes shared memory dynamically at launch). Copy EVERY kernel's
# metadata (not just the freshest): a multi-kernel dump needs the max to drive occupancy.
# `__grp__*.json` is launcher group metadata with no `shared` -- skipped.
_n_meta=0
while IFS= read -r m; do
  [ -n "$m" ] || continue
  cp "$m" "$DEST/meta_$(basename "$m")" && _n_meta=$((_n_meta+1))
done < <(find "$CACHE" -name '*.json' ! -name '__grp__*' 2>/dev/null)
[ "$_n_meta" -gt 0 ] && echo "  + $DEST/meta_*.json ($_n_meta kernel metadata -> LDS bytes/WG)" \
  || echo "  ! no kernel metadata json in the cache -> asm_loop_audit will report LDS UNAVAILABLE (it will NOT substitute a 0)." >&2

# Echo the config the dumped IR was compiled with, so it can be confirmed to match
# the pinned autotune-winning config (transcribe.md ## Procedure step 0).
if [ -f "$DEST/$VARIANT.ttgir" ]; then
  NW="$(grep -oE '"ttg.num-warps"[^,}]*' "$DEST/$VARIANT.ttgir" | grep -oE '[0-9]+' | head -1 || true)"
  NS="$(grep -oE 'num_stages[^,}]*[0-9]+' "$DEST/$VARIANT.ttgir" | grep -oE '[0-9]+' | head -1 || true)"
  echo "  config(from .ttgir): num_warps=${NW:-?} num_stages=${NS:-?}  (confirm == pinned best config)"
fi

# Strip .loc and .Ltmp labels for stable line anchors.
if [ -f "$DEST/$VARIANT.amdgcn" ]; then
  sed -e '/^[[:space:]]*\.loc[[:space:]]/d' -e '/^\.Ltmp[0-9]*:/d' \
      "$DEST/$VARIANT.amdgcn" > "$DEST/$VARIANT.s"
  echo "  + $DEST/$VARIANT.s (stripped)"
fi

# Optional: auto-recover Gluon from the dumped .ttgir (closes the transcribe loop).
if [ -n "$EMIT_GLUON" ]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  TTGIR="$DEST/$VARIANT.ttgir"
  if [ -f "$TTGIR" ]; then
    RG_ARGS=(--ttgir "$TTGIR" --out "$DEST/$VARIANT.gluon.py" --arch "$ARCH")
    case "$EMIT_GLUON" in
      layouts)  ;;                                   # layouts-only (default, methodology-preserving)
      anchor)   RG_ARGS+=(--with-skeleton);  [ -n "$KERNEL" ] && RG_ARGS+=(--kernel "$KERNEL");;
      pipeline) RG_ARGS+=(--with-skeleton --with-pipeline); [ -n "$KERNEL" ] && RG_ARGS+=(--kernel "$KERNEL");;
      *) echo "  ! unknown --emit-gluon mode '$EMIT_GLUON' (use layouts|anchor|pipeline)" >&2;;
    esac
    python3 "$SCRIPT_DIR/recover_gluon.py" "${RG_ARGS[@]}"
    # Auto-fill the experiment-records transcribe record.
    python3 "$SCRIPT_DIR/recover_gluon.py" --record --ttgir "$TTGIR" \
        > "$DEST/$VARIANT.transcribe_record.txt" 2>/dev/null \
        && echo "  + $DEST/$VARIANT.transcribe_record.txt"
    echo "  hint: verify layout-equivalence after recompiling the anchor:"
    echo "        python3 $SCRIPT_DIR/recover_gluon.py --verify --ttgir $TTGIR --anchor-ttgir <anchor>.ttgir [--harness '<cmd> --correctness']"
  else
    echo "  ! --emit-gluon set but $TTGIR not found (no .ttgir dumped)" >&2
  fi
fi
echo "=== done -> $DEST ==="
