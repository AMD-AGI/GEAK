"""
GPU Kernel Hotspot Analyzer
Reads a decoded rocprofv3 ATT dispatch dir (ui_output_agent_*_dispatch_*: code.json [+ snapshots.json])
and identifies top-K stall hotspots plus the occupancy limiter.

GEAK shared kernel tool (kernel_workflow/scripts/kernel_tools/); the Gluon pack's
scripts/hotspot_analyzer.py is a shim to this file. Hardware constants (LDS pool sizes) are read from
perf_knowledge/hardware/data/hw_constants.json through the sibling `_hwdata` locator / amd_occupancy,
never from literals. The ATT input needs the rocprof-trace-decoder (not vendored; see
ROCPROF_ATT_LIBRARY_PATH in capture.sh / rocprofv3_safe.sh --att).

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
            if len(matches) == 1:
                chosen = matches[0]
            elif matches:
                print(f"  warning: --kernel {kernel_filter!r} matched {len(matches)} rows in "
                      f"{os.path.basename(path)}; resource metadata withheld (pin a unique dispatch)")

        else:
            # Legacy heuristic: bidirectional substring match against the dir basename.
            # Works for timestamped dirs like ``20240101_120000_pa_decode_kernel``.
            short = re.sub(r"^\d{8}_\d{6}_", "", dir_name)  # strip YYYYMMDD_HHMMSS_

            def _matches(kn):
                if not kn:
                    return False
                return kn in dir_name or short in kn or kn.startswith(short) or short.startswith(kn)

            matches = [r for r in rows if _matches(r.get("Kernel_Name", ""))]
            if len(matches) == 1:
                chosen = matches[0]

        if chosen is None:
            continue  # no matching row in this CSV — try the next candidate

        def _int(key):
            try:
                return int(chosen[key])
            except (KeyError, ValueError, TypeError):
                return None

        dimensions = [_int("Workgroup_Size_" + axis) for axis in "XYZ"]
        workgroup = dimensions[0] * dimensions[1] * dimensions[2] if all(dimensions) else None
        return {
            "csv_path": path,
            "csv_vgpr": _int("VGPR_Count"),
            "csv_accum_vgpr": _int("Accum_VGPR_Count"),
            "csv_sgpr": _int("SGPR_Count"),
            "csv_lds": _int("LDS_Block_Size"),
            "csv_wg": workgroup,
        }
    return {}


def detect_arch_and_reg_pressure(instructions, meta=None):
    """Render shared static resource bounds for an ATT dispatch.

    ATT instruction excerpts are not register allocation records. Require an
    explicit architecture and matched CSV resource fields; never infer allocation
    from the largest register referenced in a sampled instruction stream.
    """
    from amd_resources import analyze
    from amd_occupancy import model_for, lds_per_cu
    meta = meta or {}
    asms = [inst.asm for inst in instructions]
    arch = meta.get("arch")
    vgpr, agpr = meta.get("csv_vgpr"), meta.get("csv_accum_vgpr")
    combined = vgpr + agpr if vgpr is not None and agpr is not None else None
    wave = meta.get("wave_size")
    text = f".wavefront_size: {wave}\n" if wave is not None else ""
    resources = analyze(text, arch=arch, vgpr=combined,
                        lds_bytes=meta.get("csv_lds"), workgroup_size=meta.get("csv_wg"))
    model = model_for(arch) or {}
    granule = model.get("vgpr_alloc_granule")
    def allocated(n):
        return ((n + granule - 1) // granule) * granule if n is not None and granule else None
    reg, lds = resources["waves_by_vgpr_term"], resources["waves_by_lds_term"]
    bounds = {key: value for key, value in (("VGPR", reg), ("LDS", lds)) if value is not None}
    waves = resources["waves_per_simd"]
    next_occ = waves + 1 if waves is not None and waves < model.get("max_waves_per_simd", 0) else None
    return {
        "arch": arch or "unknown (pass --arch from the execution record)",
        "is_gfx950": arch == "gfx950", "is_wmma": any("v_wmma_" in a for a in asms),
        "wave_size": resources["wave_size"], "vgpr_file_per_simd": model.get("vgpr_file_per_simd"),
        "vgpr_alloc_granule": granule, "max_waves_per_simd": model.get("max_waves_per_simd"),
        "lds_scope": resources["lds_capacity_scope"],
        "arch_vgpr": vgpr, "arch_vgpr_alloc": allocated(vgpr),
        "accum_vgpr": agpr, "accum_vgpr_alloc": allocated(agpr),
        "combined_vgpr_alloc": allocated(combined), "is_vgpr_form": agpr == 0,
        "vgpr_limit": reg, "lds_per_wg": resources["lds_bytes_per_wg"],
        "lds_total": lds_per_cu(arch), "lds_limit": lds,
        "lds_limit_why": resources["lds_detail"], "sgpr": meta.get("csv_sgpr"),
        "sgpr_limit": None, "waves_per_wg": ((meta['csv_wg'] + resources['wave_size'] - 1) // resources['wave_size']
            if meta.get('csv_wg') and resources['wave_size'] else None),
        "occupancy": waves, "bound_by": min(bounds, key=bounds.get) if waves is not None else None,
        "target_for_next_occ": (model['vgpr_file_per_simd'] // next_occ // granule * granule
            if next_occ else None), "next_occ": next_occ, "has_meta": combined is not None,
        "resource_warning": resources["waves_warning"],
        "register_source": "matched rocprof CSV VGPR_Count + Accum_VGPR_Count" if combined is not None else "missing",
        "mfma_count": sum("v_mfma_" in a for a in asms),
        "buffer_load": sum("buffer_load" in a for a in asms),
        "buffer_store": sum("buffer_store" in a for a in asms),
        "ds_read": sum("ds_read" in a or "ds_load" in a for a in asms),
        "ds_write": sum("ds_write" in a or "ds_store" in a for a in asms),
    }


def print_reg_pressure(reg_info):
    print_header("Register Pressure & Occupancy")
    print(f"  Architecture:   {reg_info['arch']}")
    if not reg_info["has_meta"]:
        print(
            "  (matched register metadata unavailable; allocation is not inferred from ATT; "
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
    if reg_info["lds_per_wg"] is not None and reg_info["lds_per_wg"] > 0:
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
    if reg_info["sgpr"] is not None and reg_info["sgpr"] > 0:
        if reg_info.get("is_wmma"):
            print(f"  SGPR:           {reg_info['sgpr']}  (fixed per wave on RDNA -- not an "
                  f"occupancy limiter)")
        else:
            print(f"  SGPR:           {reg_info['sgpr']}  -> {reg_info['sgpr_limit']} waves/SIMD")

    if reg_info["resource_warning"]:
        print("  Static bound warning:", reg_info["resource_warning"])
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
    parser.add_argument("--arch", help="Explicit target from the execution record, e.g. gfx942")
    parser.add_argument("--wave-size", type=int, choices=(32, 64), help="Actual compiled wave mode")
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
    meta.update(arch=args.arch, wave_size=args.wave_size)
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
    model (explicit gfx950 target; LDS pool read from hw_constants.json)."""
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
                                                  "csv_lds": 65536, "csv_wg": 256, "arch": "gfx950"})
        if not info.get("is_gfx950"):
            fails.append("the explicit gfx950 target must select CDNA4")
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
