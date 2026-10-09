"""
GPU Kernel Hotspot Analyzer
Reads a decoded rocprofv3 ATT dispatch dir (ui_output_agent_*_dispatch_*: code.json [+ snapshots.json])
and identifies top-K stall hotspots plus the occupancy limiter.

GEAK shared kernel tool (kernel_workflow/scripts/kernel_tools/); the Gluon pack's
scripts/hotspot_analyzer.py is a shim to this file. Hardware constants (LDS pool sizes) are read from
perf_knowledge/hardware/data/hw_constants.json through the sibling `_hwdata` locator / amd_occupancy,
never from literals. The ATT input needs the rocprof-trace-decoder (not vendored; see
ROCPROF_ATT_LIBRARY_PATH in capture.sh / profile_kernel.sh --att).

Usage:
    python hotspot_analyzer.py <dispatch_dir> [--topk N] [--mode {asm,src,both}]
    python hotspot_analyzer.py <dispatch_dir> --topk 5 --mode src --detail --context 4
    python hotspot_analyzer.py --selftest
"""

import argparse
import csv
import glob
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


@dataclass
class Instruction:
    asm: str
    pc_index: int
    source_loc: str
    pc_addr: int
    exec_count: int
    total_cycles: int
    stall_cycles: int
    issue_cycles: int

    @property
    def stall_pct(self):
        return 100.0 * self.stall_cycles / self.total_cycles if self.total_cycles else 0.0

    @property
    def stall_type(self):
        asm = self.asm.lower()
        if "s_waitcnt" in asm:
            if "vmcnt" in asm:
                return "VMEM-wait"
            if "lgkmcnt" in asm:
                return "LDS/SMEM-wait"
            if "expcnt" in asm:
                return "EXP-wait"
            return "waitcnt"
        if "s_barrier" in asm or "s_wait_idle" in asm:
            return "barrier"
        if "buffer_load" in asm or "global_load" in asm or "flat_load" in asm:
            return "VMEM-load"
        if "buffer_store" in asm or "global_store" in asm:
            return "VMEM-store"
        if "ds_read" in asm or "ds_write" in asm:
            return "LDS"
        if "s_load" in asm or "s_store" in asm:
            return "SMEM"
        if "v_mfma" in asm or "v_fma" in asm:
            return "MFMA/FMA"
        return "other"


@dataclass
class SourceLineHotspot:
    source_loc: str
    total_stall_cycles: int = 0
    total_cycles: int = 0
    instructions: list = field(default_factory=list)

    @property
    def stall_pct(self):
        return 100.0 * self.total_stall_cycles / self.total_cycles if self.total_cycles else 0.0

    @property
    def dominant_stall_type(self):
        by_type = defaultdict(int)
        for inst in self.instructions:
            by_type[inst.stall_type] += inst.stall_cycles
        return max(by_type, key=by_type.get) if by_type else "other"


def load_source_map(dispatch_dir):
    """Parse snapshots.json nested tree -> {virtual_path: [source_lines]}."""
    snap_path = os.path.join(dispatch_dir, "snapshots.json")
    if not os.path.exists(snap_path):
        return {}
    with open(snap_path) as f:
        tree = json.load(f)

    path_map = {}

    def _walk(node, prefix):
        for key, val in node.items():
            segment = "" if key == "/" else key
            path = prefix.rstrip("/") + "/" + segment if segment else prefix
            if isinstance(val, dict):
                _walk(val, path)
            else:
                path_map[path] = val

    _walk(tree, "")

    source_cache = {}
    for vpath, local_name in path_map.items():
        local_path = os.path.join(dispatch_dir, local_name)
        if os.path.exists(local_path):
            with open(local_path) as f:
                source_cache[vpath] = f.readlines()
    return source_cache


def get_source_snippet(source_cache, source_loc, context=3):
    if ":" not in source_loc:
        return []
    path, lineno_str = source_loc.rsplit(":", 1)
    try:
        lineno = int(lineno_str)
    except ValueError:
        return []
    lines = source_cache.get(path)
    if not lines:
        return []
    start = max(0, lineno - context - 1)
    end = min(len(lines), lineno + context)
    return [(i + 1, lines[i].rstrip(), i + 1 == lineno) for i in range(start, end)]


def load_instructions(dispatch_dir):
    with open(os.path.join(dispatch_dir, "code.json")) as f:
        data = json.load(f)
    rows = data.get("code") or []  # rocprofv3 emits None when the traced CU caught no waves
    hdr = data.get("header")
    if isinstance(hdr, str):  # rocprofv3 v3.0.0: header is a comma-joined column string
        hdr = [h.strip() for h in hdr.split(",")]
    instructions = []
    if isinstance(hdr, list) and "Hit" in hdr:
        # v3.0.0 named columns: ISA,_,LineNumber,Source,Codeobj,Vaddr,Hit,Latency,Stall,Idle
        col = {n: i for i, n in enumerate(hdr)}

        def _gi(row, name, default=0):
            i = col.get(name)
            try:
                return int(row[i])
            except (TypeError, ValueError, IndexError):
                return default
        i_asm, i_src = col.get("ISA", 0), col.get("Source")
        for row in rows:
            asm = row[i_asm] if len(row) > i_asm else ""
            if not asm or not asm.strip() or asm.strip().startswith(";"):
                continue
            src = row[i_src] if (i_src is not None and len(row) > i_src and row[i_src]) else "<unknown>"
            instructions.append(Instruction(
                asm=asm, pc_index=_gi(row, "LineNumber"), source_loc=src,
                pc_addr=_gi(row, "Vaddr"), exec_count=_gi(row, "Hit"),
                total_cycles=_gi(row, "Latency"), stall_cycles=_gi(row, "Stall"),
                issue_cycles=_gi(row, "Idle")))
        return instructions
    # legacy positional format (pre-v3.0.0)
    for row in rows:
        if not isinstance(row[2], int) or row[2] == 0:
            continue
        instructions.append(
            Instruction(
                asm=row[0],
                pc_index=row[2],
                source_loc=row[3] if row[3] else "<unknown>",
                pc_addr=row[5],
                exec_count=row[6] if isinstance(row[6], int) else 0,
                total_cycles=row[7] if isinstance(row[7], int) else 0,
                stall_cycles=row[8] if isinstance(row[8], int) else 0,
                issue_cycles=row[9] if isinstance(row[9], int) else 0,
            )
        )
    return instructions


def aggregate_by_source(instructions):
    by_src = {}
    for inst in instructions:
        loc = inst.source_loc
        if loc not in by_src:
            by_src[loc] = SourceLineHotspot(source_loc=loc)
        hs = by_src[loc]
        hs.total_stall_cycles += inst.stall_cycles
        hs.total_cycles += inst.total_cycles
        if inst.stall_cycles > 0:
            hs.instructions.append(inst)
    return sorted(by_src.values(), key=lambda x: x.total_stall_cycles, reverse=True)


BAR_WIDTH = 30


def stall_bar(pct):
    filled = int(pct / 100 * BAR_WIDTH)
    return f"[{'█' * filled}{'░' * (BAR_WIDTH - filled)}] {pct:5.1f}%"


def fmt_cycles(n):
    if n >= 1_000_000:
        return f"{n/1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}K"
    return str(n)


def print_header(title):
    print(f"\n{'═' * 90}\n  {title}\n{'═' * 90}")


def print_stall_type_summary(instructions, total_stall):
    print_header("Stall Breakdown by Type")
    by_type = defaultdict(int)
    for inst in instructions:
        if inst.stall_cycles > 0:
            by_type[inst.stall_type] += inst.stall_cycles
    print(f"  {'Type':<14}  {'Stall':>8}  Bar")
    print(f"  {'-'*14}  {'-'*8}  {'-'*38}")
    for stype, cycles in sorted(by_type.items(), key=lambda x: x[1], reverse=True):
        pct = 100.0 * cycles / total_stall if total_stall else 0
        print(f"  {stype:<14}  {fmt_cycles(cycles):>8}  {stall_bar(pct)}")


def print_source_hotspots(hotspots, topk, total_stall):
    print_header(f"Top-{topk} Hotspot Source Lines  (stall cycles aggregated)")
    print(f"  {'#':>3}  {'Stall':>8}  {'%Total':>7}  {'StallBar':<38}  {'DomType':<12}  Source")
    print(f"  {'-'*3}  {'-'*8}  {'-'*7}  {'-'*38}  {'-'*12}  {'-'*40}")
    for rank, hs in enumerate(hotspots[:topk], 1):
        if hs.total_stall_cycles == 0:
            break
        pct = 100.0 * hs.total_stall_cycles / total_stall if total_stall else 0
        src_short = hs.source_loc[-48:] if len(hs.source_loc) > 48 else hs.source_loc
        print(
            f"  {rank:>3}  {fmt_cycles(hs.total_stall_cycles):>8}  {pct:>6.2f}%  "
            f"{stall_bar(hs.stall_pct):<38}  {hs.dominant_stall_type:<12}  {src_short}"
        )


def print_asm_hotspots(instructions, topk, total_stall):
    print_header(f"Top-{topk} Hotspot Instructions  (by stall cycles)")
    print(f"  {'#':>3}  {'Stall':>8}  {'%Total':>7}  {'Type':<12}  {'ASM':<48}  Source")
    print(f"  {'-'*3}  {'-'*8}  {'-'*7}  {'-'*12}  {'-'*48}  {'-'*30}")
    ranked = sorted([i for i in instructions if i.stall_cycles > 0], key=lambda x: x.stall_cycles, reverse=True)[:topk]
    for rank, inst in enumerate(ranked, 1):
        pct = 100.0 * inst.stall_cycles / total_stall if total_stall else 0
        asm_short = inst.asm[:47] + "…" if len(inst.asm) > 48 else inst.asm
        src_short = inst.source_loc[-38:] if len(inst.source_loc) > 38 else inst.source_loc
        print(
            f"  {rank:>3}  {fmt_cycles(inst.stall_cycles):>8}  {pct:>6.2f}%  "
            f"{inst.stall_type:<12}  {asm_short:<48}  {src_short}"
        )


def print_source_detail(hotspot, source_cache, context=3):
    print(
        f"\n    ── {hotspot.source_loc}  "
        f"(stall={fmt_cycles(hotspot.total_stall_cycles)}, {hotspot.stall_pct:.0f}% stall rate)"
    )
    snippet = get_source_snippet(source_cache, hotspot.source_loc, context=context)
    if snippet:
        print("    Source:")
        for lineno, text, is_hot in snippet:
            marker = ">>>" if is_hot else "   "
            print(f"      {marker} {lineno:4d} │ {text}")
    print("    Stalling instructions:")
    for inst in sorted(hotspot.instructions, key=lambda x: x.stall_cycles, reverse=True)[:6]:
        print(f"      stall={fmt_cycles(inst.stall_cycles):>7}  type={inst.stall_type:<12}  {inst.asm}")


def read_kernel_metadata(dispatch_dir, kernel_filter=""):
    """Read authoritative resource counts from ``out_kernel_trace.csv`` if present.

    The ATT ``code.json`` only contains the (possibly single-CU, possibly
    vgpr-form) disassembly, so it cannot reveal accum_vgpr / SGPR / LDS /
    workgroup size.  The kernel-trace CSV carries the real launch metadata.
    Searches the dispatch dir and its parent (staging often copies the CSV
    next to the ui_output_agent_* dir).  Returns {} if not found.

    Row selection priority:
      1. ``kernel_filter`` substring matched against Kernel_Name, optionally
         narrowed by Dispatch_Id when the dir name encodes ``dispatch_<id>``
         (rocprofv3 ``ui_output_agent_*_dispatch_<id>`` layout).  Dispatch_Id
         matching avoids false matches when a PyTorch reference kernel shares
         the same name substring.
      2. Bidirectional name heuristic against the directory basename (legacy
         path for timestamped dirs like ``20240101_120000_pa_decode_kernel``).
    """
    candidates = []
    for base in (dispatch_dir, os.path.dirname(os.path.abspath(dispatch_dir))):
        candidates += glob.glob(os.path.join(base, "*kernel_trace*.csv"))

    dir_name = os.path.basename(os.path.abspath(dispatch_dir))
    # Extract the dispatch id from rocprofv3's ui_output_agent_<N>_dispatch_<id> layout.
    _dispatch_id_m = re.search(r"dispatch_(\d+)$", dir_name)
    dispatch_id = _dispatch_id_m.group(1) if _dispatch_id_m else None

    for path in candidates:
        try:
            with open(path) as f:
                rows = list(csv.DictReader(f))
        except OSError:
            continue
        if not rows or "Accum_VGPR_Count" not in rows[0]:
            continue

        has_dispatch_col = "Dispatch_Id" in rows[0]

        chosen = None
        if kernel_filter:
            # Explicit filter: kernel name substring, narrowed by Dispatch_Id when available.
            can_disambiguate = bool(dispatch_id and has_dispatch_col)
            matches = [r for r in rows if kernel_filter in r.get("Kernel_Name", "")]
            if can_disambiguate:
                matches = [r for r in matches if str(r.get("Dispatch_Id", "")).strip() == dispatch_id]
            if matches:
                chosen = matches[0]
                if not can_disambiguate and len(matches) > 1:
                    # First-substring-wins: no dispatch id available to pick between same-named rows.
                    print(
                        f"  warning: --kernel '{kernel_filter}' matched {len(matches)} rows in "
                        f"{os.path.basename(path)} with no dispatch id to disambiguate; using the "
                        "first match (pass a more specific --kernel)"
                    )
        else:
            # Legacy heuristic: bidirectional substring match against the dir basename.
            # Works for timestamped dirs like ``20240101_120000_pa_decode_kernel``.
            short = re.sub(r"^\d{8}_\d{6}_", "", dir_name)  # strip YYYYMMDD_HHMMSS_

            def _matches(kn):
                if not kn:
                    return False
                return kn in dir_name or short in kn or kn.startswith(short) or short.startswith(kn)

            for r in rows:
                if _matches(r.get("Kernel_Name", "")):
                    chosen = r
                    break

        if chosen is None:
            continue  # no matching row in this CSV — try the next candidate

        def _int(key):
            try:
                return int(chosen.get(key, "") or 0)
            except (ValueError, TypeError):
                return 0

        return {
            "csv_path": path,
            "csv_vgpr": _int("VGPR_Count"),
            "csv_accum_vgpr": _int("Accum_VGPR_Count"),
            "csv_sgpr": _int("SGPR_Count"),
            "csv_lds": _int("LDS_Block_Size"),
            "csv_wg": _int("Workgroup_Size_X") * max(1, _int("Workgroup_Size_Y")) * max(1, _int("Workgroup_Size_Z")),
        }
    return {}


def _lds_per_cu_or_none(arch):
    """LDS bytes/CU for `arch` from hw_constants.json, or None. Never a literal.

    Deliberately has no default parameter. The whole failure this guards against is a hardware
    figure that is reachable without having established which part it belongs to -- and 65536
    is the most dangerous possible default, because it is the CORRECT answer on CDNA3. A wrong
    constant that is right somewhere else never looks wrong when you read the code.
    """
    try:
        import amd_occupancy as _o   # sibling in kernel_tools; reads hw_constants via _hwdata
        return _o.lds_per_cu(arch)
    except Exception:
        return None


def _rdna_lds_pool_or_none():
    """RDNA LDS pool in BYTES, agreed across every RDNA row of hw_constants.json, or None.

    This path knows the FAMILY -- a dump containing ``v_wmma_*``/``v_swmmac_*`` is RDNA -- but
    not the ARCH: nothing in the ISA separates gfx1100 from gfx1201. It used to answer that
    with the literal 131072, which is the same shape as the ``65536`` bug in the CDNA branch
    below: a hardware figure produced without having established which part it belongs to.
    One thing made it look safer, and it is worth naming because it is the trap -- every RDNA
    row that records a pool currently records the SAME 128 KiB, so the literal happened to be
    right. "Happened to be right" is not a mechanism, and it stops holding the day a row
    differs, silently, in the direction of a too-large budget.

    So read the rows instead. If every RDNA row that records ``lds_per_wgp_kib`` agrees, that
    one value is a fact about the family and may be used without narrowing the arch; the code
    is then correct BECAUSE of the agreement rather than by coincidence. If they ever disagree
    the quantity is arch-dependent, this path cannot have it, and None makes the caller drop
    the LDS limiter with a stated reason. Rows that record NO pool (RDNA, but the pool was
    never sourced for that arch) cast no vote either way -- they are the reason the caller must
    still report that the arch was not pinned.
    """
    try:
        import _hwdata               # sibling locator: perf_knowledge/hardware/data/hw_constants.json
        table = _hwdata.load_json("hw_constants.json").get("arch", {})
    except Exception:
        return None
    pools = {v["lds_per_wgp_kib"] for v in table.values()
             if (v.get("family") or "").upper().startswith("RDNA")
             and v.get("lds_per_wgp_kib")}
    return int(pools.pop() * 1024) if len(pools) == 1 else None


def detect_arch_and_reg_pressure(instructions, meta=None):
    """Detect GPU architecture from ISA and estimate occupancy.

    VGPR model (CDNA2/CDNA3/CDNA4 unified register file): arch_vgpr (256) and
    accum_vgpr (256) share ONE combined 512-entry budget per SIMD.  Occupancy
    from VGPR is ``512 // (arch_vgpr_alloc + accum_vgpr_alloc)``.  This is the
    same form on gfx942 and gfx950 — gfx942 is NOT a separate-pool
    ``256 / max(...)`` machine (that was gfx908/CDNA1).

    Occupancy (waves/SIMD) is the min across every resource limiter:
        occ = min(vgpr_limit, lds_limit, sgpr_limit, hw_max=8)
    where
        vgpr_limit = 512 // (arch_alloc + accum_alloc)                    [per SIMD]
        lds_limit  = (LDS_total // lds_per_wg) * waves_per_wg // 4_SIMDs   [per SIMD]
        sgpr_limit = (sgpr_total // sgpr_per_wave)                        [per SIMD]

    ``meta`` (from read_kernel_metadata) supplies accum_vgpr / LDS / SGPR /
    workgroup size, which the ISA scan alone cannot.  ISA-scanned arch_vgpr is
    combined via max() with the CSV value so a bogus/low CSV field can't
    under-report.

    ARCH GATE: every constant above is CDNA.  An RDNA kernel (``v_wmma_*``, wave32)
    divides a 1536-entry file per SIMD in granules of 24, caps at 16 waves, has no
    AGPR file, and its SGPRs are not an occupancy limiter at all -- so the model
    switches wholesale rather than reusing the CDNA numbers with a different label.
    """
    meta = meta or {}
    asms = [inst.asm for inst in instructions]

    # Detect the ISA family from the matrix instruction that only exists there, then the
    # CDNA4-specific MFMA shapes within CDNA.
    is_wmma = any("v_wmma_" in a or "v_swmmac_" in a for a in asms)
    is_gfx950 = (not is_wmma) and any(
        "v_mfma_scale_f32" in a or "v_mfma_f32_16x16x128" in a or "v_mfma_f32_32x32x64" in a
        for a in asms)
    if is_wmma:
        arch = "RDNA (WMMA, wave32; exact arch NOT determined)"
        wave_size, vgpr_total, vgpr_gran, max_occupancy = 32, 1536, 24, 16
        simds_per_lds_pool, sgpr_per_simd = 4, 0     # 4 SIMD32 per WGP; SGPRs are not a limiter
        # The pool is the per-WGP one (a WGP is 2 CUs) and it is READ, not written here -- see
        # _rdna_lds_pool_or_none. Note the scope: it is NOT per-CU, and the per-workgroup
        # allocation limit (64 KiB) is a third quantity that does not belong in this slot.
        lds_total = _rdna_lds_pool_or_none()
        lds_scope = "WGP"
    else:
        # THE LDS FIGURE IS NOT A CDNA CONSTANT, and this fork is a POSITIVE test for CDNA4
        # only: `is_gfx950` is False both for a real gfx942 AND for any CDNA kernel whose dump
        # happens to contain no CDNA4-only MFMA shape -- a gfx950 kernel that never issues one
        # lands in the `else` and is labelled "gfx942 (CDNA3)" with 64 KiB. That is 2.5x low on
        # a 160 KiB part, and too small an LDS budget understates wg/CU, so `occ` comes out
        # LOWER than the truth and the reader is sent hunting a limiter that is not binding.
        # It never looks wrong on inspection, because 65536 is the right answer somewhere.
        #
        # So the register model (genuinely shared across CDNA3/4 -- one 512-entry file, granule
        # 8, cap 8) stays, and the LDS term is taken from hw_constants.json keyed on the arch
        # we actually detected. Detected CDNA4 -> that row. NOT detected -> None, and the LDS
        # limiter is dropped from the min() with a stated reason, rather than being guessed.
        arch = "gfx950 (CDNA4)" if is_gfx950 else "CDNA (gfx9xx, exact arch NOT determined)"
        wave_size, vgpr_total, vgpr_gran, max_occupancy = 64, 512, 8, 8
        simds_per_lds_pool, sgpr_per_simd = 4, 800
        lds_total = _lds_per_cu_or_none("gfx950") if is_gfx950 else None
        lds_scope = "CU"

    # Scan for max VGPR/AccVGPR indices
    max_vgpr = 0
    max_agpr = 0
    for a in asms:
        for m in re.finditer(r"\bv(\d+)\b", a):
            max_vgpr = max(max_vgpr, int(m.group(1)))
        for m in re.finditer(r"\bv\[(\d+)", a):
            max_vgpr = max(max_vgpr, int(m.group(1)))
        for m in re.finditer(r"\ba(\d+)\b", a):
            max_agpr = max(max_agpr, int(m.group(1)))
        for m in re.finditer(r"\ba\[(\d+)", a):
            max_agpr = max(max_agpr, int(m.group(1)))

    # Total VGPR budget consumed (what occupancy divides into 512).
    #
    # IMPORTANT: the CSV's VGPR_Count and Accum_VGPR_Count are sub-counts of
    # the SAME .amdhsa_next_free_vgpr total — they SUM to the real allocation,
    # they are not two independent pools to add a third time.  For vgpr-form
    # MFMA (no a-registers in the disassembly) the "accum" portion lives in
    # the arch VGPR file and the ISA v-register scan already includes it.
    #
    #   combined = CSV.VGPR_Count + CSV.Accum_VGPR_Count   (preferred)
    #   fallback (no CSV): ISA arch scan, plus a separate AGPR scan ONLY if
    #                      a-registers were actually referenced (true agpr-form).
    isa_arch = max_vgpr + 1
    isa_accum = max_agpr + 1 if max_agpr > 0 else 0
    csv_vgpr = meta.get("csv_vgpr", 0)
    csv_accum = meta.get("csv_accum_vgpr", 0)

    # vgpr-form MFMA writes the accumulator into the arch VGPR file (the
    # disassembly references v-registers, no a-registers).  agpr-form uses a
    # real separate AGPR file (a-registers present).
    is_vgpr_form = max_agpr == 0

    if csv_vgpr or csv_accum:
        if is_vgpr_form:
            # No physical AGPR; the whole total lives in the arch file.  Guard
            # against a bogus-low CSV total with the ISA v-register scan.
            arch_vgpr_count = max(isa_arch, csv_vgpr + csv_accum)
            accum_vgpr_count = 0
            combined_count = arch_vgpr_count
        else:
            # agpr-form: arch and accum live in separate files; guard each
            # sub-count with its ISA scan so a low CSV field can't under-report.
            arch_vgpr_count = max(isa_arch, csv_vgpr)
            accum_vgpr_count = max(isa_accum, csv_accum)
            combined_count = arch_vgpr_count + accum_vgpr_count
    else:
        arch_vgpr_count = isa_arch
        accum_vgpr_count = isa_accum
        combined_count = isa_arch + isa_accum  # isa_accum=0 unless real a-regs seen

    # Round the TOTAL up to the allocation granule (granularity applies to
    # next_free_vgpr, not to each sub-count separately).
    def _alloc(n):
        return ((n + vgpr_gran - 1) // vgpr_gran) * vgpr_gran

    arch_vgpr_alloc = _alloc(arch_vgpr_count)
    accum_vgpr_alloc = _alloc(accum_vgpr_count) if accum_vgpr_count > 0 else 0
    combined_alloc = _alloc(combined_count)

    vgpr_limit = min(vgpr_total // combined_alloc, max_occupancy) if combined_alloc > 0 else max_occupancy

    # LDS limiter (waves/SIMD).  LDS is a per-CU (CDNA) / per-WGP (RDNA) resource shared by
    # all resident workgroups; convert workgroups to waves/SIMD via waves_per_wg / 4 SIMDs.
    lds_per_wg = meta.get("csv_lds", 0)
    wg_size = meta.get("csv_wg", 0)
    waves_per_wg = max(1, (wg_size + wave_size - 1) // wave_size) if wg_size else 0
    if lds_per_wg > 0 and waves_per_wg > 0 and lds_total:
        wg_per_cu_lds = lds_total // lds_per_wg
        lds_limit = max(1, (wg_per_cu_lds * waves_per_wg) // simds_per_lds_pool)
        lds_limit = min(lds_limit, max_occupancy)
        lds_limit_why = (f"{lds_total} B/{lds_scope} // {lds_per_wg} B/wg = {wg_per_cu_lds} "
                         f"wg, x {waves_per_wg} waves/wg / {simds_per_lds_pool} SIMD")
    elif lds_per_wg > 0 and waves_per_wg > 0 and not lds_total:
        # The kernel DOES use LDS and we cannot price it, because the pool size for this part
        # was never established. Dropping the term (max_occupancy) is the only safe direction
        # -- occupancy is a veto criterion, and a limiter we cannot compute must not be allowed
        # to veto -- but it is NOT the same as "LDS does not bind", so it is labelled, not
        # quietly folded into the min().
        lds_limit = max_occupancy
        lds_limit_why = (f"NOT COMPUTED: this kernel uses {lds_per_wg} B/wg of LDS, but the "
                         f"per-{lds_scope} pool size is unknown for the arch this dump was "
                         f"detected as ({arch}). The LDS term is DROPPED from the min() rather "
                         f"than guessed -- a wrong pool size is 2.5x between CDNA3 and CDNA4 "
                         f"and would read as a verdict. Name the arch to price it.")
    else:
        lds_limit = max_occupancy
        lds_limit_why = "no LDS used by this kernel (or no workgroup size in the metadata)"

    # SGPR limiter (waves/SIMD).  gfx9/CDNA: 800 SGPRs per SIMD, alloc gran 16. On RDNA the
    # SGPR file is fixed per wave and never binds occupancy, so it is not a limiter there.
    sgpr_count = meta.get("csv_sgpr", 0)
    if sgpr_count > 0 and sgpr_per_simd:
        sgpr_alloc = ((sgpr_count + 15) // 16) * 16
        sgpr_limit = min(sgpr_per_simd // sgpr_alloc, max_occupancy)
    else:
        sgpr_limit = max_occupancy

    occupancy = min(vgpr_limit, lds_limit, sgpr_limit)

    # Which resource binds?
    limiters = {"VGPR": vgpr_limit, "LDS": lds_limit, "SGPR": sgpr_limit}
    bound_by = min(limiters, key=limiters.get)

    # Target combined VGPR for next occupancy level (only meaningful if VGPR-bound)
    next_occ = occupancy + 1
    target_total = (vgpr_total // next_occ) if next_occ <= max_occupancy else None

    # Instruction mix counts
    mfma_count = sum(1 for a in asms if "v_mfma_" in a)
    buf_load = sum(1 for a in asms if "buffer_load" in a)
    buf_store = sum(1 for a in asms if "buffer_store" in a)
    ds_read = sum(1 for a in asms if "ds_read" in a or "ds_load" in a)
    ds_write = sum(1 for a in asms if "ds_write" in a or "ds_store" in a)

    return {
        "arch": arch,
        "is_gfx950": is_gfx950,
        "is_wmma": is_wmma,
        "wave_size": wave_size,
        "vgpr_file_per_simd": vgpr_total,
        "vgpr_alloc_granule": vgpr_gran,
        "max_waves_per_simd": max_occupancy,
        "lds_scope": lds_scope,
        "arch_vgpr": arch_vgpr_count,
        "arch_vgpr_alloc": arch_vgpr_alloc,
        "accum_vgpr": accum_vgpr_count,
        "accum_vgpr_alloc": accum_vgpr_alloc,
        "combined_vgpr_alloc": combined_alloc,
        "is_vgpr_form": is_vgpr_form,
        "vgpr_limit": vgpr_limit,
        "lds_per_wg": lds_per_wg,
        "lds_total": lds_total,
        "lds_limit": lds_limit,
        "lds_limit_why": lds_limit_why,
        "sgpr": sgpr_count,
        "sgpr_limit": sgpr_limit,
        "waves_per_wg": waves_per_wg,
        "occupancy": occupancy,
        "bound_by": bound_by,
        "target_for_next_occ": target_total,
        "next_occ": next_occ if next_occ <= max_occupancy else None,
        "has_meta": bool(meta),
        "mfma_count": mfma_count,
        "buffer_load": buf_load,
        "buffer_store": buf_store,
        "ds_read": ds_read,
        "ds_write": ds_write,
    }


def print_reg_pressure(reg_info):
    print_header("Register Pressure & Occupancy")
    print(f"  Architecture:   {reg_info['arch']}")
    if not reg_info["has_meta"]:
        print(
            "  (kernel_trace CSV not matched — accum/LDS/SGPR estimated from ISA only; "
            "pass --kernel <name_substr> to enable CSV metadata lookup)"
        )
    if reg_info.get("is_wmma"):
        print(f"  arch_vgpr:      {reg_info['arch_vgpr']}  (RDNA WMMA: no AGPR file)")
    elif reg_info["is_vgpr_form"]:
        print(f"  arch_vgpr:      {reg_info['arch_vgpr']}  (MFMA vgpr-form: accumulators in arch file, no AGPR)")
    else:
        print(f"  arch_vgpr:      {reg_info['arch_vgpr']}")
        print(f"  accum_vgpr:     {reg_info['accum_vgpr']}  (AGPR file)")
    print(f"  total VGPR:     {reg_info['combined_vgpr_alloc']} / {reg_info['vgpr_file_per_simd']}"
          f"  -> {reg_info['vgpr_limit']} waves/SIMD"
          f"  (wave{reg_info['wave_size']}, granule {reg_info['vgpr_alloc_granule']},"
          f" cap {reg_info['max_waves_per_simd']})")
    if reg_info["lds_per_wg"] > 0:
        if reg_info["lds_total"]:
            print(
                f"  LDS:            {reg_info['lds_per_wg']} B/wg  ({reg_info['waves_per_wg']} waves/wg,"
                f" {reg_info['lds_total']} B/{reg_info['lds_scope']})"
                f"  -> {reg_info['lds_limit']} waves/SIMD"
            )
        else:
            print(f"  LDS:            {reg_info['lds_per_wg']} B/wg  "
                  f"({reg_info['waves_per_wg']} waves/wg)  -> NOT PRICED")
            print(f"                  {reg_info['lds_limit_why']}")
    if reg_info["sgpr"] > 0:
        if reg_info.get("is_wmma"):
            print(f"  SGPR:           {reg_info['sgpr']}  (fixed per wave on RDNA -- not an "
                  f"occupancy limiter)")
        else:
            print(f"  SGPR:           {reg_info['sgpr']}  -> {reg_info['sgpr_limit']} waves/SIMD")

    print(f"\n  occupancy:      {reg_info['occupancy']} waves/SIMD  (bound by {reg_info['bound_by']})")
    if reg_info["next_occ"] is not None and reg_info["bound_by"] == "VGPR":
        what = "VGPR" if reg_info.get("is_wmma") else "combined VGPR (arch+accum)"
        print(
            f"  -> {reg_info['next_occ']} waves requires {what} "
            f"<= {reg_info['target_for_next_occ']}"
        )

    print("\n  Instruction mix:")
    print(
        f"    MFMA: {reg_info['mfma_count']},  buffer_load: {reg_info['buffer_load']},"
        f"  buffer_store: {reg_info['buffer_store']}"
    )
    print(f"    ds_read: {reg_info['ds_read']},  ds_write: {reg_info['ds_write']}")


def main():
    parser = argparse.ArgumentParser(description="GPU kernel hotspot analyzer")
    parser.add_argument("dispatch_dir", help="Path to ATT dispatch output directory")
    parser.add_argument("--topk", type=int, default=15)
    parser.add_argument("--mode", choices=["asm", "src", "both"], default="both")
    parser.add_argument(
        "--detail", action="store_true", help="Show source snippet + instruction breakdown under each source hotspot"
    )
    parser.add_argument("--context", type=int, default=3, help="Source lines of context around hotspot (default: 3)")
    parser.add_argument(
        "--kernel",
        default="",
        metavar="SUBSTR",
        help="Kernel name substring for CSV metadata lookup "
        "(e.g. 'pa_mqa_logits_fp4_kernel_0'). "
        "Required when the dispatch dir name does not encode the kernel name, "
        "as with rocprofv3 ui_output_agent_*_dispatch_<id> directories. "
        "Combined with the dispatch id from the dir name when a Dispatch_Id "
        "column is present in the CSV.",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.dispatch_dir):
        print(f"Error: directory not found: {args.dispatch_dir}")
        return 1

    print(f"\nLoading: {args.dispatch_dir}")
    instructions = load_instructions(args.dispatch_dir)
    source_hotspots = aggregate_by_source(instructions)
    source_cache = load_source_map(args.dispatch_dir)

    total_stall = sum(i.stall_cycles for i in instructions)
    total_cycles = sum(i.total_cycles for i in instructions)

    print(f"\n  Kernel:        {os.path.basename(args.dispatch_dir)}")
    print(f"  Instructions:  {len(instructions):,}")
    print(f"  Total cycles:  {fmt_cycles(total_cycles)}")
    print(f"  Total stalls:  {fmt_cycles(total_stall)}  ({100*total_stall/total_cycles:.1f}% of total cycles)")

    meta = read_kernel_metadata(args.dispatch_dir, kernel_filter=args.kernel)
    reg_info = detect_arch_and_reg_pressure(instructions, meta)
    print_reg_pressure(reg_info)

    print_stall_type_summary(instructions, total_stall)

    if args.mode in ("src", "both"):
        print_source_hotspots(source_hotspots, args.topk, total_stall)
        if args.detail:
            for hs in source_hotspots[: min(5, args.topk)]:
                if hs.total_stall_cycles > 0:
                    print_source_detail(hs, source_cache, context=args.context)

    if args.mode in ("asm", "both"):
        print_asm_hotspots(instructions, args.topk, total_stall)

    print()
    return 0


def _selftest():
    """Offline: a synthetic v3.0.0 code.json through the loader, the aggregation and the occupancy
    model (CDNA4 detected from a gfx950-only MFMA shape; LDS pool read from hw_constants.json)."""
    import tempfile
    fails = []
    with tempfile.TemporaryDirectory() as td:
        hdr = "ISA,_,LineNumber,Source,Codeobj,Vaddr,Hit,Latency,Stall,Idle"
        code = [
            ["s_waitcnt vmcnt(0)", 0, 10, "k.py:5", 0, 4096, 4, 400, 300, 0],
            ["v_mfma_f32_16x16x128_f8f6f4 a[0:3], v[0:7], v[8:15], a[0:3]", 0, 11, "k.py:7", 0, 4100, 4, 64, 8, 0],
            ["ds_read_b128 v[40:43], v1", 0, 12, "k.py:6", 0, 4104, 4, 80, 40, 0],
            ["; comment", 0, 13, "", 0, 4108, 0, 0, 0, 0],
        ]
        with open(os.path.join(td, "code.json"), "w") as fh:
            json.dump({"header": hdr, "code": code}, fh)
        ins = load_instructions(td)
        if len(ins) != 3:
            fails.append(f"loader kept {len(ins)} instructions, want 3 (comment rows dropped)")
        hs = aggregate_by_source(ins)
        if not hs or hs[0].source_loc != "k.py:5" or hs[0].dominant_stall_type != "VMEM-wait":
            fails.append("top source hotspot should be k.py:5 / VMEM-wait")
        info = detect_arch_and_reg_pressure(ins, {"csv_vgpr": 128, "csv_accum_vgpr": 128,
                                                  "csv_lds": 65536, "csv_wg": 256})
        if not info.get("is_gfx950"):
            fails.append("a gfx950-only MFMA shape must detect CDNA4")
        if info.get("vgpr_file_per_simd") != 512 or info.get("vgpr_alloc_granule") != 8 \
                or info.get("max_waves_per_simd") != 8:
            fails.append("CDNA register model must be 512/SIMD, granule 8, cap 8")
        if info.get("vgpr_limit") != 2:
            fails.append(f"128 arch + 128 accum VGPRs -> 512//256 = 2 waves/SIMD, got {info.get('vgpr_limit')}")
        if info.get("lds_total") != 160 * 1024:
            fails.append(f"gfx950 LDS pool must come from hw_constants (160 KiB), got {info.get('lds_total')}")
    for f in fails:
        print("FAIL", f)
    print("[hotspot_analyzer] SELFTEST " + ("FAIL" if fails else "PASS"))
    return 1 if fails else 0


if __name__ == "__main__":
    if "--selftest" in sys.argv[1:]:
        raise SystemExit(_selftest())
    raise SystemExit(main())
