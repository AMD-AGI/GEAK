#!/usr/bin/env bash
# Triton-family IR dump helper.
# Compiles a kernel via a per-variant TRITON_CACHE_DIR, then copies the
# .ttgir / .llir / .amdgcn artifacts out and strips .amdgcn into a stable .s.
#
# Use for: (a) recovering plain-Triton inferred layouts before transcription
# (the .ttgir shows #blocked/#mma/#shared + num_stages), and (b) verifying each
# Gluon layer landed (compiler-contract.md acceptance signals).
#
# Lives in kernel_workflow/scripts/kernel_tools/ (GEAK shared kernel tool).
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
# No `rm`: the per-variant compile cache is a fresh mktemp dir under ${TMPDIR:-/tmp}/geak-dump-ir
# (TILE_HW_CACHE overrides); a non-empty one is MOVED ASIDE.
#
# Writes: <ir_dir>/<variant>/<variant>.{ttgir,llir,amdgcn,s} + meta_*.json (LDS bytes/workgroup).
# Exit: 0 dumped | 1 no compile command / bare --kernel | 2 unknown option before the compile
#       command | 3 --emit-gluon requested without recover_gluon.py present
#       | 4 --emit-gluon with no arch given or detectable. 3 also covers no single compilation to
#       collect: E_IR_NOTHING_COMPILED (the command never launched the kernel, or overrode
#       TRITON_CACHE_DIR) or E_IR_SELECTION (several kernels/configurations).

set -euo pipefail

SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"

usage() {
  cat <<'EOF'
Usage:
  bash dump_ir.sh --variant <name> --out <ir_dir> [options] -- <compile_cmd ...>
  bash dump_ir.sh <compile_cmd ...> --variant <name> --out <ir_dir>
      [--knobs "COEXEC EXPERT AGPR PLUGIN"] [--emit-gluon layouts|anchor|pipeline]
      [--kernel module.path:object] [--kernel-name <substring>] [--all] [--arch gfx950]
  bash dump_ir.sh --selftest

Any token that is not one of the flags above is part of <compile_cmd>, so the
compile command may appear before, after or around the flags -- but it must not
START with '--' (an unknown flag before the command is refused as a typo).
Use '--' before the compile command to preserve all its arguments, including
names shared with this wrapper, such as --arch, --out, --variant or --help.

  --kernel       module.path:object, handed to recover_gluon.py for --emit-gluon anchor|pipeline
  --kernel-name  substring that PINS which compiled kernel's IR is copied (multi-kernel ops)
  --all          multi-case benchmark: dump EVERY compilation to <out>/<variant>_c<i>/ (index in
                 <out>/<variant>/ir_index.json, with each one's constexprs) instead of refusing with E_IR_SELECTION

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
  # a command that never launches the kernel compiles nothing: named, not an empty selection
  out="$(TILE_HW_CACHE="$T/cache" bash "$SELF" true --variant n --out "$T/ir" 2>&1)"; rc=$?
  if [ $rc -eq 3 ] && grep -q "E_IR_NOTHING_COMPILED" <<<"$out"; then echo "  ok   nothing compiled -> named (3)"
  else echo "  FAIL nothing-compiled rc=$rc: $out"; f=1; fi
  # --all: two configurations of one kernel, each indexed with the constexprs the hook logged (the
  # fake command checks the hook dir is on PYTHONPATH and writes the rows the hook would)
  local fake2='case ":$PYTHONPATH:" in *.hook:*) ;; *) exit 9;; esac; for c in 1 2; do d="$TRITON_CACHE_DIR/h$c"; mkdir -p "$d"; printf "x\n" > "$d/k.ttgir"; printf "x\n" > "$d/k.llir"; printf "k:\n\ts_endpgm\n" > "$d/k.amdgcn"; printf "{\"name\": \"k\", \"num_warps\": $((c*4)), \"num_stages\": 2}\n" > "$d/k.json"; printf "{\"kernel\": \"k\", \"num_warps\": $((c*4)), \"num_stages\": 2, \"constexprs\": {\"BLOCK\": \"$((c*64))\"}}\n" >> "$TRITON_CACHE_DIR/_constexprs.jsonl"; done'
  out="$(TILE_HW_CACHE="$T/cache" bash "$SELF" bash -c "$fake2" --all --variant a --out "$T/ir" 2>&1)"; rc=$?
  if [ $rc -eq 0 ] && python3 -c "import json,sys; ix=json.load(open(sys.argv[1])); assert sorted(r['constexprs']['BLOCK'] for r in ix)==['128','64'], ix" "$T/ir/a/ir_index.json" 2>/dev/null; then
    echo "  ok   --all: ir_index.json records each compilation's constexprs"
  else echo "  FAIL --all constexprs (rc=$rc): $out"; f=1; fi
  # --all, configs sharing num_warps/num_stages: told apart by the logged compile hash
  local fake3='for c in 1 2; do d="$TRITON_CACHE_DIR/s$c"; mkdir -p "$d"; printf "x\n" > "$d/k.ttgir"; printf "x\n" > "$d/k.llir"; printf "k:\n\ts_endpgm\n" > "$d/k.amdgcn"; printf "{\"name\": \"k\", \"num_warps\": 4, \"num_stages\": 2, \"hash\": \"h$c\"}\n" > "$d/k.json"; printf "{\"kernel\": \"k\", \"num_warps\": 4, \"num_stages\": 2, \"hash\": \"h$c\", \"constexprs\": {\"BLOCK\": \"$((c*64))\"}}\n" >> "$TRITON_CACHE_DIR/_constexprs.jsonl"; done'
  out="$(TILE_HW_CACHE="$T/cache" bash "$SELF" bash -c "$fake3" --all --variant s --out "$T/ir" 2>&1)"; rc=$?
  if [ $rc -eq 0 ] && python3 -c "import json,sys; ix={r['compilation'].split('/')[0]: r['constexprs']['BLOCK'] for r in json.load(open(sys.argv[1]))}; assert ix=={'s1':'64','s2':'128'}, ix" "$T/ir/s/ir_index.json" 2>/dev/null; then
    echo "  ok   --all: shared num_warps/num_stages told apart by the compile hash"
  else echo "  FAIL --all shared-warps constexprs (rc=$rc): $out"; f=1; fi
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
ALL=0
CMD=()
while [ $# -gt 0 ]; do
  case "$1" in
    --)            shift; CMD+=("$@"); break;;
    -h|--help|--help-full) usage; exit 0;;
    --variant)     VARIANT="$2"; shift 2;;
    --out)         OUT_DIR="$2"; shift 2;;
    --knobs)       KNOBS="$2"; shift 2;;
    --emit-gluon)  EMIT_GLUON="$2"; shift 2;;
    --kernel)      KERNEL="$2"; shift 2;;
    --kernel-name) KERNEL_NAME="$2"; shift 2;;
    --all)         ALL=1; shift;;
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
# used to leave multi-kernel selection ambiguous. Refuse it
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

# The compile command belongs to its caller; preserve every argument verbatim.
# In particular, --backends/--check may choose the kernel or configuration being inspected.

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

case "$EMIT_GLUON" in
  ""|layouts|anchor|pipeline) ;;
  *) echo "ERROR: invalid --emit-gluon mode '$EMIT_GLUON'" >&2; exit 2;;
esac

case "$VARIANT" in
  ""|.|..|*/*) echo 'ERROR: --variant must be a single directory name' >&2; exit 2;;
esac
DEST="$OUT_DIR/$VARIANT"
if [ -e "$DEST" ]; then
  mv "$DEST" "$DEST.previous.$(date +%s).$$"
fi
mkdir -p "$DEST"
# The compile cache lives outside the workspace (TMPDIR), so it can never travel in a candidate patch.
CACHE_PARENT="${TILE_HW_CACHE:-${TMPDIR:-/tmp}/geak-dump-ir}"
mkdir -p "$CACHE_PARENT"
CACHE_PARENT="$(cd "$CACHE_PARENT" && pwd)"
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
    COEXEC) export TRITON_HIP_USE_COEXEC_SCHEDULER=1   # read nowhere on stock 3.8.0 (inert); coexec is gfx1250-only
            echo "note: COEXEC env is inert on stock Triton 3.8.0; coexec is gfx1250-only (LLVM warns on gfx950/gfx942)" >&2;;
    EXPERT) export TRITON_HIP_USE_EXPERT_SCHEDULING=1;;
    AGPR)
      # a compile OPTION, not an env var: pass it through to the launcher, which must set
      #   llvm_fn_attrs=[["amdgpu-agpr-alloc", "256"]]   (amdgpu-mfma-vgpr-form is an llc option, inert as a fn attr)
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

# --all: log each compilation's constexprs (Triton's cache metadata does not record them) through
# Triton's jit_post_compile_hook, installed by a sitecustomize.py on PYTHONPATH for this command
# only. It chains to any sitecustomize it shadows and to an already-set hook; without Triton, or on
# any error, it does nothing. Each row carries the compiled kernel's hash (the cache metadata's
# "hash"; read from the JIT's device cache, checked against Triton 3.8.0) when it can be read.
# collect_ir.py matches rows by that hash, else by kernel + num_warps + num_stages.
if [ "$ALL" = 1 ]; then
  HOOK_DIR="$CACHE.hook"
  mkdir -p "$HOOK_DIR"
  cat > "$HOOK_DIR/sitecustomize.py" <<'PY'
import importlib, importlib.util, json, os, sys
_here = os.path.dirname(os.path.abspath(__file__))
_saved, _me = list(sys.path), sys.modules.get("sitecustomize")
try:  # chain to the sitecustomize this file shadows, if any
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _here]
    del sys.modules["sitecustomize"]
    importlib.import_module("sitecustomize")
except Exception:
    pass
finally:
    sys.path[:] = _saved
    if sys.modules.get("sitecustomize") is None:
        sys.modules["sitecustomize"] = _me
try:
    if importlib.util.find_spec("triton") is not None:
        from triton import knobs as _knobs
        _log = os.path.join(os.environ.get("TRITON_CACHE_DIR", ""), "_constexprs.jsonl")
        _prev = _knobs.runtime.jit_post_compile_hook

        def _dump_ir_hook(*a, **kw):
            try:
                fn, c = kw.get("fn"), kw.get("compile") or {}
                params = [p.name for p in getattr(getattr(fn, "jit_function", None), "params", [])]
                ce = {}
                for path, v in (c.get("constants") or {}).items():
                    i = path[0] if isinstance(path, tuple) and path else path
                    n = params[i] if isinstance(i, int) and i < len(params) else str(path)
                    ce[n if not isinstance(path, tuple) or len(path) == 1 else f"{n}{list(path[1:])}"] = repr(v)
                # The compiled kernel sits in the JIT's per-device cache under the same key; its
                # hash is the "hash" field of the cache metadata, which ties this row to one cache
                # dir even when configs share num_warps/num_stages. Absent -> collect_ir falls back.
                h = None
                try:
                    jf = getattr(fn, "jit_function", None)
                    h = getattr(jf.device_caches[c.get("device")][0].get(c.get("key")), "hash", None)
                except Exception:
                    h = None
                with open(_log, "a") as f:
                    f.write(json.dumps({"kernel": getattr(fn, "name", None), "num_warps": c.get("num_warps"),
                                        "num_stages": c.get("num_stages"), "hash": h,
                                        "constexprs": ce}) + "\n")
            except Exception:
                pass
            return _prev(*a, **kw) if _prev else None

        _knobs.runtime.jit_post_compile_hook = _dump_ir_hook
except Exception:
    pass
PY
  export PYTHONPATH="$HOOK_DIR${PYTHONPATH:+:$PYTHONPATH}"
fi

echo "=== dump_ir variant=$VARIANT knobs='${KNOBS:-none}' cache=$CACHE ==="
# ONE token with whitespace is a command STRING (`bash -c` form); anything else is an argv.
if [ ${#CMD[@]} -eq 1 ] && [[ "${CMD[0]}" =~ [[:space:]] ]]; then
  bash -c "${CMD[0]}"
else
  "${CMD[@]}"
fi

# A JIT kernel compiles at its first launch: a command that only imports or parses the source
# (a "compile" step that never launches) leaves this cache empty. Say that, instead of an
# empty selection that reads like a cache-location problem.
if [ -z "$(find "$CACHE" -type f \( -name '*.ttgir' -o -name '*.amdgcn' \) -print -quit 2>/dev/null)" ]; then
  echo "E_IR_NOTHING_COMPILED: the command wrote no IR into $CACHE." >&2
  echo "  Triton/Gluon compile a kernel at its FIRST LAUNCH; give dump_ir.sh a command that launches it" >&2
  echo "  once (a correctness run or a one-launch benchmark), not one that only imports/parses the source." >&2
  echo "  If it did launch, the command (or a script it runs) overrides TRITON_CACHE_DIR: drop that override." >&2
  exit 3
fi
# Select ONE compilation group (cache directory + kernel stem), then copy its IR and
# matching metadata together. Multiple kernels/configurations require an explicit selection;
# missing outputs are failures, never a reason to reuse a previous dump.
if [ "$ALL" = 1 ]; then
  # multi-case benchmark: one dump per compilation (<out>/<variant>_c<i>/), indexed in <out>/<variant>/ir_index.json
  python3 "$(dirname "$SELF")/collect_ir.py" "$CACHE" "$DEST" "$VARIANT" --kernel-name "$KERNEL_NAME" --all
  for d in "$DEST"_c*; do
    [ -f "$d/$VARIANT.amdgcn" ] && sed -e '/^[[:space:]]*\.loc[[:space:]]/d' -e '/^\.Ltmp[0-9]*:/d' \
        "$d/$VARIANT.amdgcn" > "$d/$VARIANT.s"
  done
  exit 0
fi
python3 "$(dirname "$SELF")/collect_ir.py" "$CACHE" "$DEST" "$VARIANT" --kernel-name "$KERNEL_NAME"

# Echo the config the dumped IR was compiled with, so it can be confirmed to match
# the pinned autotune-winning config (see isa_tools.md: pin one compilation configuration).
if [ -f "$DEST/$VARIANT.ttgir" ]; then
  NW="$(grep -oE '"ttg.num-warps"[^,}]*' "$DEST/$VARIANT.ttgir" | grep -oE '[0-9]+' | head -1 || true)"
  NS="$(grep -oE 'num_stages[^,}]*[0-9]+' "$DEST/$VARIANT.ttgir" | grep -oE '[0-9]+' | head -1 || true)"
  NS_SRC=".ttgir"
  # the pipeliner consumes tt.num_stages, so a pipelined dump usually has none: fall back to the
  # compile metadata written beside it
  if [ -z "$NS" ]; then
    NS="$(cat "$DEST"/meta_*.json 2>/dev/null | grep -oE '"num_stages": *[0-9]+' | grep -oE '[0-9]+$' | head -1 || true)"
    NS_SRC="compile metadata"
  fi
  # LDS ring depth = leading dim of a 3-D local_alloc that the loop indexes (memdesc_index)
  RING=""
  if grep -q 'ttg.memdesc_index' "$DEST/$VARIANT.ttgir"; then
    RING="$(grep -oE 'ttg\.local_alloc *: *\(\) *-> *!ttg\.memdesc<[0-9]+x[0-9]+x[0-9]+' "$DEST/$VARIANT.ttgir" \
            | sed -E 's/.*memdesc<([0-9]+)x.*/\1/' | sort -n | tail -1 || true)"
  fi
  echo "  config: num_warps=${NW:-?} num_stages=${NS:-?} (${NS_SRC})${RING:+, LDS ring depth=$RING (memdesc<${RING}x...>)}  (confirm == pinned best config)"
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
    echo "    champion into a Gluon-enabled environment and run --emit-gluon there." >&2
    exit 3
  fi
  # Only the optional language adapter owns transcription and Gluon mode semantics.
  [ -f "$SCRIPT_DIR/emit_gluon.py" ] || { echo 'ERROR: Gluon transcription adapter unavailable' >&2; exit 3; }
  python3 "$SCRIPT_DIR/emit_gluon.py" --mode "$EMIT_GLUON" --dir "$DEST" \
    --variant "$VARIANT" --arch "$ARCH" --kernel "$KERNEL"

fi
echo "=== done -> $DEST ==="
