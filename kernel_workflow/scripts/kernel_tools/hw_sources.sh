#!/usr/bin/env bash
# hw_sources.sh — acquire/generate AMD hardware primary-source FACTS on demand (three tiers;
# see perf_knowledge/expert_skills/skills/gluon_authoring/references/hardware/primary-sources.md). Machine-readable-first: generate layouts from the
# Matrix Instruction Calculator, decode encodings from the ISA XML; the PDF is a CDNA4-only
# fallback. Nothing large is committed — everything lands in a cache. Fail-safe: on any fetch/build
# failure, print a scoped tool-gap and exit 3 (caller falls back to distilled perf_knowledge/hardware
# + gluon_authoring/references/hardware values); never fabricate.
# Lives in kernel_workflow/scripts/kernel_tools/ (GEAK shared kernel tool); the in-repo databases
# are read from perf_knowledge/hardware/data/ via _hwdata.py ($GEAK_HW_DATA_DIR overrides).
#
# Usage:
#   hw_sources.sh layout <arch> <instr> [calc-flags...]   # Tier 2: generate the operand layout table
#                                                          #   arch: cdna1|cdna2|cdna3|rdna3|rdna4
#                                                          #   flags: -A -B -C -D -k --cbsz # --blgp # --opsel # --transpose
#   hw_sources.sh layout gfx950 <instr> [flags...]        # Tier 2: -> gfx950_isa.py, reads the committed
#                                                          #   CDNA4 layout DB. No fetch, no cache, no PDF.
#                                                          #   Richer queries (locate/encoding/sparse/hazard/
#                                                          #   errata) live on gfx950_isa.py directly.
#   hw_sources.sh xml <arch>                               # Tier 1: fetch+extract amdgpu_isa_<arch>.xml
#   hw_sources.sh decode <arch> <hex-bytes>               # Tier 1: build isa_spec_manager, decode an instruction
#   hw_sources.sh pdf <doc> [firstpage lastpage]          # Tier 3: fetch PDF, pdftoppm-render pages (CDNA4 layout / concepts)
#                                                          #   doc: cdna4-isa|cdna3-isa|cdna3-wp|cdna4-wp|rdna3-isa|rdna4-isa
#   hw_sources.sh --selftest                               # offline: data dir resolves, gfx950 layout answers
# No `rm` anywhere: GEAK roles run this, and the shared cache is never deleted from.
set -uo pipefail
HERE_SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Directory of the in-repo hardware databases, via the single locator (_hwdata.py).
hw_data_dir(){ python3 "$HERE_SELF/_hwdata.py" 2>/dev/null | grep -v '^None$' || true; }
gap(){ echo "SCOPED TOOL-GAP: $*" >&2; echo "  -> fall back to distilled perf_knowledge/hardware/ + gluon_authoring/references/hardware/ values (probe-then-trust)" >&2; exit 3; }

# gfx950 layout is dispatched BEFORE the cache gate below, deliberately: the CDNA4 layout database
# is committed inside the pack, so this path fetches nothing, builds nothing and writes nothing.
# Gating it on TILE_HW_CACHE would deny an agent a file it already has on disk. Every path past
# this point does touch the network or the tool cache, and stays gated.
if [ "${1:-}" = "layout" ]; then
  case "${2:-}" in
    cdna4|gfx950)
      instr="${3:?instr}"; shift 3
      g="$HERE_SELF/gfx950_isa.py"
      [ -f "$g" ] || gap "Matrix Calculator has NO CDNA4/gfx950 support and gfx950_isa.py is missing from this pack -> use: hw_sources.sh pdf cdna4-isa (render the layout table)"
      python3 "$g" facts "$instr" || exit $?
      python3 "$g" layout "$instr" -A "$@" || exit $?
      echo "# provenance: gfx950-mfma-layout.json (CDNA4 ISA ch.7, verified) — more: gfx950_isa.py --help" >&2
      exit 0 ;;
  esac
fi

usage(){
  echo "usage: hw_sources.sh {preflight | layout <arch> <instr> [flags] | xml <arch> | decode <arch> <bytes> | pdf <doc> [p0 p1]}"
  echo "  --help          this message"
  echo "  preflight       which arch x question class is answerable here, and how to repair a gap"
  echo "  layout          Tier 2 operand layout. gfx950 -> in-repo DB; cdna1/2/3 + rdna3/4 -> Matrix Calculator"
  echo "  xml             Tier 1 machine-readable ISA XML for <arch>"
  echo "  decode          Tier 1 decode raw instruction bytes (builds isa_spec_manager)"
  echo "  pdf             Tier 3 fetch + render pages (prose and anything not in a database)"
  echo "  --selftest      offline check (no cache, no network)"
  echo "see perf_knowledge/expert_skills/skills/gluon_authoring/references/hardware/primary-sources.md (tiers; gfx950 + cdna3/rdna3/rdna4 answer from"
  echo "    perf_knowledge/hardware/data/, no fetch)"
}
# --help is answered BEFORE the cache gate: asking a tool what it does must not require a
# configured cache, and it used to exit 2 through the unknown-command branch, which is what made
# the catalog record its help contract as unverified.
case "${1:-}" in --help|-h) usage; exit 0;; esac

# ---- selftest (offline) ------------------------------------------------------------------------
if [ "${1:-}" = "--selftest" ]; then
  f=0
  D="$(hw_data_dir)"
  if [ -n "$D" ] && [ -f "$D/gfx950-mfma-layout.json" ] && [ -f "$D/gfx950-encoding.json.gz" ]; then
    echo "  ok   data dir: $D"
  else
    echo "  FAIL data dir not resolved via _hwdata.py (got '${D}')"; f=1
  fi
  if bash "$HERE_SELF/hw_sources.sh" layout gfx950 v_mfma_f32_32x32x16_bf16 >/dev/null 2>&1; then
    echo "  ok   layout gfx950 answers offline from the committed DB"
  else
    echo "  FAIL layout gfx950 v_mfma_f32_32x32x16_bf16"; f=1
  fi
  # --help needs no cache; an unknown command is a usage error (2), not a tool-gap.
  bash "$HERE_SELF/hw_sources.sh" --help >/dev/null 2>&1 || { echo "  FAIL --help"; f=1; }
  if grep -nE '(^|[;&|]|then|do)[[:space:]]*rm[[:space:]]+-' "$HERE_SELF/hw_sources.sh" >/dev/null; then
    echo "  FAIL an rm call is present (GEAK roles run this script; move aside instead)"; f=1
  fi
  [ $f -eq 0 ] && echo "[hw_sources] SELFTEST PASS" || echo "[hw_sources] SELFTEST FAIL"
  exit $f
fi

# ---- preflight ------------------------------------------------------------------------------
# Dispatched BEFORE the cache gate on purpose: its whole job is to REPORT what is reachable,
# including "TILE_HW_CACHE is unset", and a gate that exits 3 on that can never say so.
#
# Why this exists. The cheap environment gate checks disk, scratch and the profiler, and said
# nothing about the hardware-source tier -- so an agent discovered that the Matrix Calculator
# was missing at the moment it needed an operand layout, which is the most expensive moment to
# find out. The layer is also no longer uniform: gfx950 answers everything from in-pack
# databases, cdna3/rdna3/rdna4 answer the instruction chapters from in-pack databases but their
# layout from the calculator, and cdna1/cdna2 need the calculator for everything. A single
# "hardware sources: ok" line cannot express that, so this prints the matrix.
if [ "${1:-}" = "preflight" ]; then
  HW_P="$(hw_data_dir)"
  rc=0; note=()
  printf 'hardware-source preflight  (arch x question class -> who answers)\n\n'
  # calculator: the same search order clone_calc uses, but reporting instead of gapping
  calc=""; calcwhy=""
  for d in "${TILE_HW_SEED:+$TILE_HW_SEED/amd_matrix_instruction_calculator}" \
           "${TILE_HW_TOOLS:+$TILE_HW_TOOLS/amd_matrix_instruction_calculator}" \
           "${TILE_HW_CACHE:+$TILE_HW_CACHE/amd_matrix_instruction_calculator}"; do
    [ -n "$d" ] && [ -f "$d/matrix_calculator.py" ] && { calc="$d"; break; }
  done
  if [ -z "$calc" ]; then
    calcwhy="absent"
  elif ! python3 -c 'import tabulate,numpy' 2>/dev/null; then
    calcwhy="present but its Python deps (tabulate, numpy) are not importable"
  fi
  # in-pack databases
  have(){ [ -f "$HW_P/$1" ] && echo yes || echo no; }
  L950=$(have gfx950-mfma-layout.json); E950=$(have gfx950-encoding.json.gz)
  F950=$(have gfx950-isa-facts.json.gz)
  printf '  %-9s %-22s %-22s %s\n' arch layout "encoding (ch.12-13)" "state/memory (ch.1-6,8-11)"
  printf '  %-9s %-22s %-22s %s\n' gfx950 \
    "$([ "$L950" = yes ] && echo 'in-repo DB' || echo 'MISSING')" \
    "$([ "$E950" = yes ] && echo 'in-repo DB' || echo 'MISSING')" \
    "$([ "$F950" = yes ] && echo 'in-repo DB' || echo 'MISSING')"
  for a in cdna3 rdna3 rdna4; do
    e=$(have "$a-encoding.json.gz")
    printf '  %-9s %-22s %-22s %s\n' "$a" \
      "$([ -z "$calcwhy" ] && echo 'calculator' || echo "calculator: $calcwhy")" \
      "$([ "$e" = yes ] && echo 'in-repo DB' || echo 'XML fetch')" \
      'not extracted -> PDF'
  done
  for a in cdna1 cdna2; do
    printf '  %-9s %-22s %-22s %s\n' "$a" \
      "$([ -z "$calcwhy" ] && echo 'calculator' || echo "calculator: $calcwhy")" \
      'XML fetch' 'not extracted -> PDF'
  done
  printf '\n'
  # the two things that can actually be broken, each with its repair
  if [ -n "$calcwhy" ]; then
    rc=1
    printf '  [!] Matrix Instruction Calculator %s\n' "$calcwhy"
    printf '      It is the ONLY layout source for cdna1/2/3 and rdna3/4 (it has no cdna4\n'
    printf '      support, which is why gfx950 ships a DB instead).\n'
    # WHO repairs this is not the same as HOW, and printing only the how is how a captain ends
    # up installing into a shared container it was told not to touch. The project rule is
    # explicit: captains consume the fleet environment snapshot and must not bootstrap,
    # install, repair, restart or chmod it.
    printf '      IF YOU ARE A CAPTAIN OR A WORKER: this is not yours to fix. Return\n'
    printf '      `blocked_environment` naming this line. Do not install into a shared container.\n'
    printf '      Meanwhile gfx950 still answers every chapter, and cdna3/rdna3/rdna4 still\n'
    printf '      answer chapters 12-13, from in-pack databases that need none of this.\n'
    printf '      FOR WHOEVER OWNS THE ENVIRONMENT (fleet / operator), in the repo checkout:\n'
    printf '        git submodule update --init third_party/amd_matrix_instruction_calculator\n'
    printf '        bash bootstrap.sh                    # + tabulate/numpy + the pinned ISA XML\n'
    printf '        python3 -m pip install tabulate numpy   # deps only, if the checkout is there\n'
    printf '      then export TILE_HW_TOOLS=<repo>/third_party so runtime finds it.\n'
    printf '      Note: a pack is distributed self-contained, so the repo is often NOT in the\n'
    printf '      container -- "bootstrap.sh is absent from this checkout" is the expected\n'
    printf '      answer there, not a second fault. Fix it where the repo actually is.\n'
  else
    printf '  [ok] Matrix Instruction Calculator: %s\n' "$calc"
  fi
  if [ -z "${TILE_HW_CACHE:-}" ]; then
    rc=1
    printf '  [!] TILE_HW_CACHE is unset -- every fetching tier (xml/decode/pdf) will gap.\n'
    printf '      The fleet runtime contract normally sets it; standalone: export TILE_HW_CACHE=~/.cache/tile-hw\n'
  elif [ -d "$TILE_HW_CACHE" ] && ! [ -w "$TILE_HW_CACHE" ]; then
    rc=1
    printf '  [!] TILE_HW_CACHE is not writable: %s\n' "$TILE_HW_CACHE"
  elif ! [ -d "$TILE_HW_CACHE" ] && ! [ -w "$(dirname "$TILE_HW_CACHE")" ]; then
    rc=1
    printf '  [!] TILE_HW_CACHE does not exist and its parent is not writable: %s\n' "$TILE_HW_CACHE"
  else
    printf '  [ok] TILE_HW_CACHE writable: %s\n' "$TILE_HW_CACHE"
  fi
  # in-repo DBs need nothing, so say so plainly -- it is the answer to "do I need the network"
  printf '  [ok] in-repo databases (%s) need no network, no cache and no calculator:\n' "${HW_P:-perf_knowledge/hardware/data: NOT FOUND}"
  printf '       gfx950 all chapters; cdna3/rdna3/rdna4 chapters 12-13 (gfx950_isa.py --arch <a>)\n'
  printf '\n  A "not extracted -> PDF" cell is missing KNOWLEDGE, not an absent hardware feature.\n'
  exit $rc
fi

# A fleet-created runtime contract must select a writable cache. Falling back
# to HOME would reintroduce the image's stale campaign path and can write
# outside the approved campaign tree.
CACHE="${TILE_HW_CACHE:-}"
[ -n "$CACHE" ] || gap "TILE_HW_CACHE is unset; consume the fleet runtime environment contract"
mkdir -p "$CACHE" 2>/dev/null || gap "TILE_HW_CACHE is not writable: $CACHE"
# TILE_HW_TOOLS points at the read-only, pinned tool assets. Runtime never
# installs packages or clones tools: a missing seed is an image contract error.
TOOLS="${TILE_HW_TOOLS:-}"
SEED="${TILE_HW_SEED:-}"
XML_ZIP="AMD_GPU_MR_ISA_XML_2026_03_05.zip"                       # pinned release (reproducible)
XML_URL="https://gpuopen.com/download/${XML_ZIP}"
MI_CALC="https://github.com/ROCm/amd_matrix_instruction_calculator"
ISA_SPEC="https://github.com/GPUOpen-Tools/isa_spec_manager"
# amd.com is fronted by a CDN whose bot filter DROPS curl's default User-Agent -- and it drops it
# *after* the TLS handshake, so the failure is not a 403: over HTTP/2 curl reports
# "stream 0 was not closed cleanly: INTERNAL_ERROR" (exit 92) and over HTTP/1.1 it simply hangs to
# timeout. Both read as "the network is blocked" and neither is. A browser UA is the whole fix
# (verified: same URL returns 206 with it, over h2 or h1.1 alike). gpuopen.com (the ISA XML host)
# does not need this today; kept scoped to the PDF fetch that does.
PDF_UA="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
declare -A PDF_URL=(
  [cdna4-isa]="https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf"
  [cdna3-isa]="https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf"
  [cdna3-wp]="https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-3-white-paper.pdf"
  [cdna4-wp]="https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-4-architecture-whitepaper.pdf"
  [rdna3-isa]="https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna3-shader-instruction-set-architecture-feb-2023.pdf"
  [rdna4-isa]="https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna4-instruction-set-architecture.pdf"
)

fetch_xml(){  # $1=arch -> echoes path to amdgpu_isa_<arch>.xml
  local a="$1" z="$CACHE/$XML_ZIP" x="$CACHE/amdgpu_isa_${1}.xml" seed_x="${SEED:+$SEED/amdgpu_isa_${1}.xml}"
  [ -n "$seed_x" ] && [ -f "$seed_x" ] && { echo "$seed_x"; return; }
  [ -f "$x" ] && { echo "$x"; return; }
  [ "${TILE_HW_ALLOW_FETCH:-0}" = "1" ] \
    || gap "ISA XML ${a} absent from the image seed and runtime cache"
  [ -f "$z" ] || curl -sL --max-time 120 -o "$z" "$XML_URL" || gap "fetch ISA XML zip ($XML_URL)"
  unzip -o -q "$z" "amdgpu_isa_${a}.xml" -d "$CACHE" 2>/dev/null || gap "extract amdgpu_isa_${a}.xml (arch name?)"
  [ -f "$x" ] && echo "$x" || gap "amdgpu_isa_${a}.xml absent after extract"
}
clone_calc(){
  local d="${SEED:+$SEED/amd_matrix_instruction_calculator}"             # immutable image seed
  [ -n "$d" ] && [ -f "$d/matrix_calculator.py" ] || d="${TOOLS:+$TOOLS/amd_matrix_instruction_calculator}"
  [ -n "$d" ] && [ -f "$d/matrix_calculator.py" ] || d="$CACHE/amd_matrix_instruction_calculator"
  [ -f "$d/matrix_calculator.py" ] \
    || gap "Matrix Calculator absent from the image seed and runtime cache"
  python3 -c 'import tabulate,numpy' 2>/dev/null \
    || gap "matrix calculator Python dependencies are absent from the prebuilt image"
  echo "$d"
}

case "${1:-}" in
  layout)
    arch="${2:?arch}"; instr="${3:?instr}"; shift 3
    # cdna4/gfx950 never reaches here — it is dispatched to gfx950_isa.py above the cache gate.
    d="$(clone_calc)" || exit 3
    # calculator needs a MODE flag; default to --register-layout (the VGPR/lane table = the PDF layout)
    case " $* " in *" -R "*|*register-layout*|*" -M "*|*matrix-layout*|*" -r "*|*get-register*|*matrix-entry*|*detail-instruction*) ;; *) set -- "$@" --register-layout;; esac
    python3 "$d/matrix_calculator.py" -a "$arch" -i "$instr" --markdown "$@" \
      || gap "calculator failed for $arch/$instr (list: matrix_calculator.py -a $arch -L)"
    echo "# provenance: MI-calc $arch $instr $* — distill into perf_knowledge/expert_skills/skills/gluon_authoring/references/hardware/isa-mechanisms.md, cite this line" >&2 ;;
  xml)
    x="$(fetch_xml "${2:?arch}")" || exit 3; echo "$x"
    echo "# ISA XML (encoding). For instruction DECODE build isa_spec_manager: hw_sources.sh decode $2 <bytes>" >&2 ;;
  decode)
    arch="${2:?arch}"; bytes="${3:?hex}"; x="$(fetch_xml "$arch")" || exit 3
    s="$CACHE/isa_spec_manager"; cli="$s/build/linux/output/isa_spec_cli"
    seed_cli="${SEED:+$SEED/isa_spec_manager/build/linux/output/isa_spec_cli}"
    if [ -n "$seed_cli" ] && [ -x "$seed_cli" ]; then
      "$seed_cli" -x "$x" -d "$bytes"
      exit $?
    fi
    [ -d "$s" ] || gap "isa_spec_manager absent from the image seed and runtime cache"
    # prebuild_linux.sh assumes CWD=<repo>/build (it does `mkdir linux; cd linux; cmake ../../`
    # relative to the CALLER's cwd) -> must cd there first, NOT into $s itself (that resolves
    # `../../` one level too high and cmake fails to find CMakeLists.txt). Binary lands at
    # build/linux/output/isa_spec_cli, not linux/isa_decoder.
    [ -x "$cli" ] || gap "isa_spec_cli is absent from the image seed and runtime cache"
    "$cli" -x "$x" -d "$bytes" ;;
  pdf)
    doc="${2:?doc}"; url="${PDF_URL[$doc]:-}"; [ -n "$url" ] || gap "unknown doc '$doc' (one of: ${!PDF_URL[*]})"
    p="$CACHE/${doc}.pdf"
    if [ ! -f "$p" ]; then
      [ "${TILE_HW_ALLOW_FETCH:-0}" = "1" ] || gap "PDF ${doc} absent from image seed/runtime cache"
      curl -sL -A "$PDF_UA" --max-time 180 -o "$p" "$url" || gap "fetch PDF $url"
    fi
    if [ -n "${3:-}" ]; then
      command -v pdftoppm >/dev/null || gap "pdftoppm missing (poppler-utils)"
      pdftoppm -png -r 200 -f "${3}" -l "${4:-$3}" "$p" "$CACHE/${doc}_p" && \
        echo "rendered $CACHE/${doc}_p-*.png (open + read the layout table by eye; pdftotext MANGLES tables)"
    else echo "$p (pdftotext '$p' - for PROSE only; use 'hw_sources.sh pdf $doc <p0> <p1>' to render layout tables)"; fi ;;
  *)
    usage >&2; exit 2 ;;
esac
