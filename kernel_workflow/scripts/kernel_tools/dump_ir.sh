#!/usr/bin/env bash
# Triton-family IR dump helper.
# Compiles a kernel via a per-variant TRITON_CACHE_DIR, then copies the
# .ttgir / .llir / .amdgcn artifacts out and strips .amdgcn into a stable .s.
#
# Use for: (a) recovering plain-Triton inferred layouts before transcription
# (the .ttgir shows #blocked/#mma/#shared + num_stages), and (b) verifying each
# Gluon layer landed (compiler-contract.md acceptance signals).
#
# Lives in kernel_workflow/scripts/kernel_tools/ (GEAK shared kernel tool); the Gluon pack keeps a
# shim at perf_knowledge/expert_skills/skills/gluon_authoring/scripts/dump_ir.sh.
#
# Usage:
#   bash dump_ir.sh <compile_cmd ...> --variant <name> --out <ir_dir> [--knobs "COEXEC AGPR PLUGIN"]
#       [--emit-gluon layouts|anchor|pipeline] [--kernel module.path:object]
#       [--kernel-name <substring>] [--arch gfx950]
#   bash dump_ir.sh --selftest            # offline (fake compile command, no GPU, no triton)
# The compile command may also be ONE quoted string (run via `bash -c`):
#   bash dump_ir.sh "cd ws && python bench.py --version plain" --variant plain --out ir/
# Example:
#   bash dump_ir.sh python bench.py --version plain --variant plain --out ir/
#   # auto-recover the inferred layouts into Gluon (closes the transcribe loop) -- needs the
#   # gluon_authoring pack's recover_gluon.py ($GEAK_GLUON_PACK_DIR overrides where it is found):
#   bash dump_ir.sh python bench.py --version plain --variant plain --out ir/ --emit-gluon layouts --arch gfx950
#
# --arch: only --emit-gluon consumes it. Not given -> read from rocminfo (printed); still unknown ->
# --emit-gluon REFUSES (exit 4) instead of assuming one (gfx942 vs gfx950 changes the recovery).
# No `rm`: the per-variant compile cache is a fresh mktemp dir; a non-empty one is MOVED ASIDE.
#
# Writes: <ir_dir>/<variant>/<variant>.{ttgir,llir,amdgcn,s} + meta_*.json (LDS bytes/workgroup).
# Exit: 0 dumped | 1 no compile command / bare --kernel | 2 unknown option before the compile
#       command | 3 --emit-gluon requested without recover_gluon.py present
#       | 4 --emit-gluon with no arch given or detectable.

set -euo pipefail

SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"

usage() {
  cat <<'EOF'
Usage:
  bash dump_ir.sh <compile_cmd ...> --variant <name> --out <ir_dir>
      [--knobs "COEXEC EXPERT AGPR PLUGIN"] [--emit-gluon layouts|anchor|pipeline]
      [--kernel module.path:object] [--kernel-name <substring>] [--arch gfx950]
  bash dump_ir.sh --selftest

Any token that is not one of the flags above is part of <compile_cmd>, so the
compile command may appear before, after or around the flags -- but it must not
START with '--' (an unknown flag before the command is refused as a typo).

  --kernel       module.path:object, handed to recover_gluon.py for --emit-gluon anchor|pipeline
  --kernel-name  substring that PINS which compiled kernel's IR is copied (multi-kernel ops)

  bash dump_ir.sh python bench.py --version plain --variant plain --out ir/
  bash dump_ir.sh python bench.py --version plain --variant plain --out ir/ --emit-gluon layouts --arch gfx950
EOF
}

# ---- selftest (offline): a fake "compile" writes cache artifacts; no GPU, no triton ----------
selftest() {
  local f=0 T out rc
  set +e   # the probes below EXPECT non-zero exits; the script body runs under `set -e`
  T="$(mktemp -d "${TMPDIR:-/tmp}/dump_ir_selftest.XXXX")"
  local fake='d="$TRITON_CACHE_DIR/h1"; mkdir -p "$d"; printf "#blocked = #ttg.blocked<{}>\n\"ttg.num-warps\" = 4\n" > "$d/k.ttgir"; printf "define void @k()\n" > "$d/k.llir"; printf "\t.loc 1 2 3\nk:\n.Ltmp0:\n\ts_endpgm\n" > "$d/k.amdgcn"; printf "{\"name\": \"k\", \"shared\": 4096, \"num_warps\": 4}\n" > "$d/k.json"'
  out="$(TILE_HW_CACHE="$T/cache" bash "$SELF" bash -c "$fake" --variant v --out "$T/ir" 2>&1)"; rc=$?
  if [ $rc -eq 0 ] && [ -f "$T/ir/v/v.ttgir" ] && [ -f "$T/ir/v/v.s" ] && [ -f "$T/ir/v/meta_k.json" ]; then
    echo "  ok   dump: ttgir + stripped .s + metadata copied"
  else
    echo "  FAIL dump (rc=$rc): $out"; f=1
  fi
  if grep -qE '\.loc|\.Ltmp' "$T/ir/v/v.s" 2>/dev/null; then echo "  FAIL .s not stripped"; f=1; fi
  # the same compile command as ONE string (bash -c form)
  out="$(TILE_HW_CACHE="$T/cache" bash "$SELF" "$fake" --variant w --out "$T/ir" 2>&1)"; rc=$?
  if [ $rc -eq 0 ] && [ -f "$T/ir/w/w.ttgir" ]; then echo "  ok   single-string compile command"
  else echo "  FAIL single-string compile command (rc=$rc): $out"; f=1; fi
  TILE_HW_CACHE="$T/cache" bash "$SELF" --ariant x true >/dev/null 2>&1; rc=$?
  if [ $rc -eq 2 ]; then echo "  ok   unknown leading flag refused (2)"; else echo "  FAIL unknown flag rc=$rc"; f=1; fi
  TILE_HW_CACHE="$T/cache" bash "$SELF" true --kernel bare_name --out "$T/ir" >/dev/null 2>&1; rc=$?
  if [ $rc -eq 1 ]; then echo "  ok   bare --kernel refused (1)"; else echo "  FAIL bare --kernel rc=$rc"; f=1; fi
  # --emit-gluon with no arch anywhere must refuse BEFORE compiling, never assume one
  out="$(DUMP_IR_NO_ROCMINFO=1 TILE_HW_CACHE="$T/cache" \
         bash "$SELF" true --emit-gluon layouts --variant z --out "$T/ir" 2>&1)"; rc=$?
  if [ $rc -eq 4 ]; then echo "  ok   --emit-gluon without an arch refused (4)"
  else echo "  FAIL --emit-gluon without arch rc=$rc: $out"; f=1; fi
  if grep -nE '(^|[;&|]|then|do)[[:space:]]*rm[[:space:]]+-' "$SELF" >/dev/null; then
    echo "  FAIL an rm call is present (GEAK roles run this; move aside instead)"; f=1
  fi
  echo "  (scratch left at $T -- this script never deletes)"
  if [ $f -eq 0 ]; then echo "[dump_ir] SELFTEST PASS"; else echo "[dump_ir] SELFTEST FAIL"; fi
  return $f
}
if [ "${1:-}" = "--selftest" ]; then selftest; exit $?; fi

VARIANT="variant"; OUT_DIR="ir"; KNOBS=""; EMIT_GLUON=""; KERNEL=""; ARCH=""
KERNEL_NAME=""
CMD=()
while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help)     usage; exit 0;;
    --variant)     VARIANT="$2"; shift 2;;
    --out)         OUT_DIR="$2"; shift 2;;
    --knobs)       KNOBS="$2"; shift 2;;
    --emit-gluon)  EMIT_GLUON="$2"; shift 2;;
    --kernel)      KERNEL="$2"; shift 2;;
    --kernel-name) KERNEL_NAME="$2"; shift 2;;
    --arch)        ARCH="$2"; shift 2;;
    # A wrapper puts unknown tokens into the command it will RUN. That makes a mistyped wrapper flag
    # (`--ariant`, `--help` before this branch existed) something this script executes, after already
    # creating its scratch tree. Refuse anything that looks like a flag but is not ours, before any
    # directory is made; a real compile command never STARTS with `--`. Only the leading token is
    # tested: once the command has started, `--version plain` / `--check` are the command's own
    # arguments (the usage example passes one), and refusing them would break every documented call.
    --*)           if [ ${#CMD[@]} -eq 0 ]; then
                     echo "ERROR: unknown option '$1' (not a dump_ir flag; a compile command must not" \
                          "start with '--'). Known: --variant --out --knobs --emit-gluon --kernel" \
                          "--kernel-name --arch" >&2
                     exit 2
                   fi
                   CMD+=("$1"); shift;;
    *)             CMD+=("$1"); shift;;
  esac
done
[ ${#CMD[@]} -gt 0 ] || { echo "ERROR: no compile command given" >&2; exit 1; }

# `--kernel` is `module.path:object` for the translator; `--kernel-name` is the substring that PINS
# which compiled kernel gets copied. Passing a bare name to `--kernel` is the one mistake that
# defeats the multi-kernel guard below WITHOUT any symptom: nothing pins, the freshest artifact
# wins, and every layout recovered from it is confidently wrong for the body you meant. Refuse it
# here rather than let it through -- a doc that says "pass --kernel <substring> to pin it" has
# already sent one run into exactly that. (Overloading `--kernel` as the pin instead, as upstream
# does, breaks the other half: `--kernel mod.path:obj` for --emit-gluon anchor|pipeline then greps
# the cache for "mod.path:obj", matches nothing, and copies NO IR at all.)
case "$KERNEL" in
  ""|*:*) ;;
  *) echo "ERROR: --kernel takes module.path:object (for --emit-gluon). To PIN which compiled" >&2
     echo "       kernel is dumped, you want:  --kernel-name $KERNEL" >&2
     exit 1;;
esac

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

# Arch: only --emit-gluon consumes it (recover_gluon.py --arch). Not given -> read it off the
# device (printed, so it is never silent); still unknown -> --emit-gluon refuses BEFORE the
# compile. There is no default: gfx942 vs gfx950 changes what the recovery emits.
if [ -z "$ARCH" ] && [ -z "${DUMP_IR_NO_ROCMINFO:-}" ]; then
  ARCH="$(rocminfo 2>/dev/null | grep -oE 'gfx9[0-9]{2}|gfx1[0-9]{3}' | head -1 || true)"
  if [ -n "$ARCH" ]; then echo "[dump_ir] arch=$ARCH detected via rocminfo (override with --arch)" >&2; fi
fi
if [ -n "$EMIT_GLUON" ] && [ -z "$ARCH" ]; then
  echo "ERROR: --emit-gluon needs the target arch and none was given or detectable (rocminfo)." >&2
  echo "       Pass --arch gfx950 (or gfx942); no default is assumed." >&2
  exit 4
fi

DEST="$OUT_DIR/$VARIANT"
mkdir -p "$DEST"
CACHE_PARENT="${TILE_HW_CACHE:-$OUT_DIR/.tile-runtime/tile-hw}"
mkdir -p "$CACHE_PARENT"
CACHE="$(mktemp -d "$CACHE_PARENT/dump_ir_${VARIANT}.XXXX")"
# mktemp -d is fresh, so this never fires in practice; if it ever does, MOVE the stale cache
# aside (no `rm` in a script GEAK roles run) and start from an empty one.
if [ -n "$(ls -A "$CACHE" 2>/dev/null)" ]; then
  mv "$CACHE" "$CACHE.stale.$(date +%s).$$" && mkdir -p "$CACHE"
fi
export TRITON_CACHE_DIR="$CACHE"

# Optional compiler knobs, upstream-only. Everything here is a real upstream 3.8.0 control;
# see compiler-contract.md ## What upstream 3.8.0 actually gives you.
#   COEXEC   -- the stock matrix/VALU co-execution scheduler strategy
#   AGPR     -- pin matrix accumulators in AGPR (LLVM function attributes, 3.8.0 only)
#   PLUGIN   -- load an LLVM pass plugin you built; set LLVM_PLUGIN=/abs/path/lib*.so
# A plugin path is the caller's to supply; this script does not guess one.
for k in $KNOBS; do
  case "$k" in
    COEXEC) export TRITON_HIP_USE_COEXEC_SCHEDULER=1;;
    EXPERT) export TRITON_HIP_USE_EXPERT_SCHEDULING=1;;
    AGPR)
      # a compile OPTION, not an env var: pass it through to the launcher, which must set
      #   llvm_fn_attrs="amdgpu-agpr-alloc=256,amdgpu-mfma-vgpr-form=0"
      export DUMP_IR_WANT_AGPR_FN_ATTRS=1
      echo "note: AGPR is a per-compile option -- set llvm_fn_attrs on the kernel, not an env var" >&2
      ;;
    PLUGIN)
      if [ -n "${LLVM_PLUGIN:-}" ]; then
        export LLVM_PASS_PLUGIN_PATH="$LLVM_PLUGIN"
      else
        echo "note: PLUGIN requested but LLVM_PLUGIN=/abs/path/lib*.so is unset; not armed" >&2
      fi
      ;;
  esac
done

echo "=== dump_ir variant=$VARIANT knobs='${KNOBS:-none}' cache=$CACHE ==="
# ONE token with whitespace is a command STRING (`bash -c` form); anything else is an argv.
if [ ${#CMD[@]} -eq 1 ] && [[ "${CMD[0]}" =~ [[:space:]] ]]; then
  bash -c "${CMD[0]}"
else
  "${CMD[@]}"
fi

# Collect the freshest IR artifacts from the cache. Both @triton.jit AND @gluon.jit
# populate TRITON_CACHE_DIR, but the cache nesting depth can differ (gluon.jit may nest
# deeper than one level) -> try the one-level glob first, then a recursive fallback so a
# gluon.jit kernel is not silently missed (feedback: dump was Triton-biased).
#
# "Freshest" is only safe when the op compiles ONE kernel. A multi-kernel op -- e.g. an
# attention decode that compiles the attention body AND a split-K reduce, or an MLA op
# whose reduce kernel compiles LAST -- has the last-compiled kernel win, and every layout,
# warp count and tile extent recovered from it is then confidently wrong for the body that
# was meant. Silently: the dump looks fine. `--kernel-name` pins the selection (distinct
# from `--kernel`, which is `module.path:object` for the translator), and when more than
# one kernel is present and nothing was pinned this says so with the candidate list.
for ext in ttgir llir amdgcn; do
  _cands="$(find "$CACHE" -name "*.$ext" -printf '%T@ %p\n' 2>/dev/null | sort -rn | cut -d' ' -f2-)"
  [ -n "$_cands" ] || continue
  if [ -n "$KERNEL_NAME" ]; then
    _hit="$(printf '%s\n' "$_cands" | grep -F "$KERNEL_NAME" | head -1 || true)"
    if [ -z "$_hit" ]; then
      echo "  ! --kernel-name '$KERNEL_NAME' matched none of the .$ext in the cache:" >&2
      printf '      %s\n' $(printf '%s\n' "$_cands" | xargs -r -n1 basename) >&2
      continue
    fi
    f="$_hit"
  else
    f="$(printf '%s\n' "$_cands" | head -1)"
    _n="$(printf '%s\n' "$_cands" | xargs -r -n1 basename | sort -u | wc -l)"
    if [ "$_n" -gt 1 ]; then
      echo "  ! $_n DISTINCT kernels compiled; took the FRESHEST: $(basename "$f")" >&2
      echo "    This op is multi-kernel. If that is not the body you meant, every layout" >&2
      echo "    recovered from it is wrong for your body -- and it will still look fine." >&2
      echo "    Re-run with --kernel-name <substring>. Candidates:" >&2
      printf '      %s\n' $(printf '%s\n' "$_cands" | xargs -r -n1 basename | sort -u) >&2
    fi
  fi
  cp "$f" "$DEST/$VARIANT.$ext" && echo "  + $DEST/$VARIANT.$ext ($(basename "$f"))"
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
# the pinned autotune-winning config (gluon_authoring references/method/transcribe.md, step 0).
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
  _KT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  # This script is a GEAK shared kernel tool, but recover_gluon.py ships only in the Gluon pack
  # (transcription is the deep-dig tier's job). Resolve it there: $GEAK_GLUON_PACK_DIR, else this
  # checkout's gluon_authoring, else beside this script (copied out together). Missing -> say so
  # and exit non-zero instead of letting python3 die on a missing file, which reads like the dump
  # itself failed. The IR above is already written, so the caller loses nothing but this step.
  SCRIPT_DIR=""
  for _d in "${GEAK_GLUON_PACK_DIR:+$GEAK_GLUON_PACK_DIR/scripts}" \
            "$_KT_DIR/../../../perf_knowledge/expert_skills/skills/gluon_authoring/scripts" \
            "$_KT_DIR"; do
    if [ -n "$_d" ] && [ -f "$_d/recover_gluon.py" ]; then SCRIPT_DIR="$(cd "$_d" && pwd)"; break; fi
  done
  if [ -z "$SCRIPT_DIR" ]; then
    echo "  ! --emit-gluon needs recover_gluon.py from the gluon_authoring pack, and it was not found" >&2
    echo "    (\$GEAK_GLUON_PACK_DIR/scripts, perf_knowledge/expert_skills/skills/gluon_authoring/scripts)." >&2
    echo "    The IR dump above SUCCEEDED ($DEST). Transcription happens on the gluon side: hand the" >&2
    echo "    champion over per gluon_authoring references/method/entry.md and run --emit-gluon there." >&2
    exit 3
  fi
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
