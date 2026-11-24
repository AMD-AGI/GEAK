"""Regression tests for the tool-audit fixes F1, F9, F10, F14 (ISA / occupancy / audit / ab_bench).

CPU-only. Ground truth comes from gfx950 measurements (occupancy-by-LDS sweep
and MFMA co-issue microbenchmarks, one MI350X node, 2026-11) and LLVM metadata of Triton 3.8 kernels compiled for gfx950.
"""
import gzip
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
KT = REPO / "kernel_workflow" / "scripts" / "kernel_tools"
HW = REPO / "perf_knowledge" / "hardware" / "data"
sys.path.insert(0, str(KT))

import amd_occupancy  # noqa: E402

# (metadata.shared bytes, measured resident WG/CU) on MI350X, num_warps=4 (1 wave/SIMD per WG).
# Occupancy sweep: largest size fitting k WG/CU is a multiple of 1280 B; 32 KiB fits 4, not 5.
OCC_LDS_GFX950 = [
    (256, 8), (20480, 8), (22016, 7), (22784, 7), (23040, 7), (23296, 6), (25344, 6), (26368, 6),
    (26880, 6), (27136, 5), (29952, 5), (31488, 5), (31744, 5), (32000, 5), (32256, 4), (32768, 4),
    (36864, 4), (38912, 4), (39936, 4), (40448, 4), (40704, 4), (40960, 4), (47872, 3), (51200, 3),
    (52992, 3), (53760, 3), (54016, 2), (54272, 2), (54528, 2), (68352, 2), (75264, 2), (78592, 2),
    (80384, 2), (81152, 2), (81664, 2), (81920, 2), (163840, 1),
]


def run(*args, cwd=None):
    return subprocess.run([str(a) for a in args], text=True, capture_output=True, check=False,
                          cwd=cwd or "/tmp", timeout=120)


# ---------------------------------------------------------------- F1: LDS allocation granule
@pytest.mark.parametrize("shared,wg", OCC_LDS_GFX950)
def test_f1_lds_term_matches_measured_gfx950_occupancy(shared, wg):
    waves, detail = amd_occupancy.waves_by_lds(shared, "gfx950", 256)
    # num_warps=4 on a 4-SIMD CU: waves/SIMD == WG/CU; the probe kernel is capped at 8.
    assert min(waves, 8) == wg, (shared, waves, detail)


def test_f1_gfx942_rounds_to_512_byte_granule():
    # 65536 // roundup(16400, 512)=16896 -> 3 WG/CU (no rounding would give 3 too); pick a case
    # where rounding matters: 13000 -> 13312 B -> 4 WG/CU (65536 // 13000 = 5).
    waves, _ = amd_occupancy.waves_by_lds(13000, "gfx942", 256)
    assert waves == 4


def test_f1_hw_constants_name_the_granule():
    arch = json.loads((HW / "hw_constants.json").read_text())["arch"]
    assert arch["gfx950"]["lds_alloc_granule_bytes"] == 1280
    assert arch["gfx942"]["lds_alloc_granule_bytes"] == 512


def _probe_dir(tmp_path, vgpr, llvm_occ, shared, num_warps):
    d = tmp_path / "k"
    d.mkdir()
    (d / "k.amdgcn").write_text(
        '.amdgcn_target "amdgcn-amd-amdhsa--gfx950"\n.amdhsa_kernel k\n'
        f".amdhsa_next_free_vgpr {vgpr}\n; Occupancy: {llvm_occ}\n"
        f".amdhsa_accum_offset {vgpr}\n.end_amdhsa_kernel\n")
    (d / "k.json").write_text(json.dumps({"name": "k", "shared": shared, "num_warps": num_warps}))
    return d


def test_f1_probe_header_compares_whole_workgroups(tmp_path):
    # validity C3.4: 256x128x64 nw=8, 132 VGPR (LLVM 3 waves/SIMD), 101296 B LDS.
    # Registers: 3 waves/SIMD x 4 SIMD / 8 waves per WG = 1 WG/CU; LDS: 1 WG/CU -> both bind,
    # 2 waves/SIMD. The header must not call it LDS-bound.
    d = _probe_dir(tmp_path, 132, 3, 101296, 8)
    r = run(sys.executable, KT / "probe.py", "measure", "--dir", d, "--json", tmp_path / "p.json")
    assert r.returncode == 0, r.stderr
    assert "LDS-BOUND" not in r.stdout, r.stdout
    k = json.loads((tmp_path / "p.json").read_text())["kernels"][0]
    assert k["waves_per_simd"] == 2
    assert "both bind" in r.stdout


# ---------------------------------------------------------------- F9: asm_loop_audit
_PREAMBLE = [
    '\t.text', '\t.amdgcn_target "amdgcn-amd-amdhsa--gfx950"', '\t.amdhsa_code_object_version 5',
    '\t.globl\tk', '\t.p2align\t8', '\t.type\tk,@function', 'k:',
    '\ts_load_dwordx2 s[0:1], s[0:1], 0x0',
]


def _asm(tmp_path, body, name="k.s", private=0):
    lines = _PREAMBLE + ['.LBB0_1:'] + body + ['\ts_cbranch_scc1 .LBB0_1', '\ts_endpgm',
             '.Lfunc_end0:', '\t.size\tk, .Lfunc_end0-k',
             '\t.amdhsa_kernel k', f'\t\t.amdhsa_private_segment_fixed_size {private}',
             '\t\t.amdhsa_next_free_vgpr 64', '\t\t.amdhsa_next_free_sgpr 16',
             '\t\t.amdhsa_accum_offset 64', '\t.end_amdhsa_kernel']
    p = tmp_path / name
    p.write_text("\n".join(lines) + "\n")
    return p, lines


def _audit(path, *extra):
    r = run(sys.executable, KT / "asm_loop_audit.py", path, "--arch", "gfx950", *extra)
    assert r.returncode == 0, r.stderr
    return r.stdout


def test_f9_lgkmcnt0_with_nonzero_vmcnt_is_an_lds_drain(tmp_path):
    # validity C5.2: every wait was `s_waitcnt vmcnt(7) lgkmcnt(0)` and the tool said 0% drains.
    body = ['\tv_mfma_f32_32x32x16_bf16 v[0:15], v[16:19], v[20:23], v[0:15]',
            '\ts_waitcnt vmcnt(7) lgkmcnt(0)', '\tds_read_b128 v[24:27], v1',
            '\ts_waitcnt vmcnt(8) lgkmcnt(0)', '\tds_read_b128 v[28:31], v1',
            '\ts_waitcnt lgkmcnt(2)']
    p, _ = _asm(tmp_path, body)
    out = _audit(p)
    assert "lgkmcnt(0)=2" in out, out
    assert "67% of memory waits are full drains" in out, out


def test_f9_scratch_ops_are_a_spill_class_and_private_segment_is_printed(tmp_path):
    body = ['\tscratch_load_dwordx4 v[0:3], off, s0', '\tbuffer_load_dwordx4 v[4:7], v0, s[4:7], 0 offen',
            '\tscratch_store_dwordx4 off, v[0:3], s0', '\tglobal_store_dwordx4 v[8:9], v[4:7], off']
    p, _ = _asm(tmp_path, body, private=140)
    out = _audit(p)
    hist = out.split("histogram:")[1].split("per-opcode")[0]
    assert "scratch" in hist and " 2 " in hist.split("scratch")[1][:20], hist
    assert "g gld            1" in hist and "G gst            1" in hist, hist
    assert "private_segment_fixed_size = 140" in out, out


def test_f9_line_numbers_are_file_line_numbers(tmp_path):
    body = ['\tv_mfma_f32_32x32x16_bf16 v[0:15], v[16:19], v[20:23], v[0:15]', '\ts_waitcnt lgkmcnt(1)']
    p, lines = _asm(tmp_path, body)
    label = lines.index('.LBB0_1:') + 1                    # 1-based file line of the label
    branch = lines.index('\ts_cbranch_scc1 .LBB0_1') + 1
    out = _audit(p)
    assert f"hot loop: lines [{label + 1}, {branch}]" in out, out


def test_f11_no_back_edge_is_labelled_kernel_body(tmp_path):
    body = ['\tscratch_load_dwordx4 v[0:3], off, s0', '\tv_mfma_f32_32x32x16_bf16 v[0:15], v[16:19], v[20:23], v[0:15]',
            '\ts_barrier']
    p, lines = _asm(tmp_path, body)
    lines = [ln for ln in lines if ln != '\ts_cbranch_scc1 .LBB0_1']
    p.write_text("\n".join(lines) + "\n")
    out = _audit(p)
    assert "kernel body: lines [" in out and "per-opcode histogram (kernel body" in out, out
    assert "in the kernel body: register" in out and "barriers in kernel body" in out, out
    assert "hot loop" not in out.replace("no back-edge loop found", ""), out


def test_f9_diff_covers_class_histogram_and_drain_share(tmp_path):
    a, _ = _asm(tmp_path, ['\tv_mfma_f32_32x32x16_bf16 v[0:15], v[16:19], v[20:23], v[0:15]',
                           '\ts_waitcnt lgkmcnt(1)'], name="a.s")
    prev = tmp_path / "a.txt"
    prev.write_text(_audit(a))
    b, _ = _asm(tmp_path, ['\tv_mfma_f32_32x32x16_bf16 v[0:15], v[16:19], v[20:23], v[0:15]',
                           '\tv_mfma_f32_32x32x16_bf16 v[0:15], v[16:19], v[20:23], v[0:15]',
                           '\ts_waitcnt lgkmcnt(0)'], name="b.s")
    out = _audit(b, "--diff", prev)
    hist = out.split("histogram:")[1].split("per-opcode")[0]
    assert "+1 vs prev" in hist, hist
    assert "drain share 0% -> 100% vs prev" in out, out


# ---------------------------------------------------------------- F10: hw_constants gfx950
def _gfx950():
    return json.loads((HW / "hw_constants.json").read_text())["arch"]["gfx950"]


def test_f10_permlane_swap_costs_like_trans():
    # Co-issue microbenchmark: v_permlane16/32_swap: 3 fit in a 32x32x16 window, +8 cyc each -- not 20.
    c = _gfx950()["coexec_issue_cost_cyc"]
    assert c["cross_lane"] == c["trans"] == 8


def test_f10_mfma_cadence_is_keyed_by_instruction_shape():
    # MFMA interval microbenchmark: 32x32x16 bf16 32.6 cyc, 16x16x32 bf16 16.7, 32x32x64 f8 64.3, 16x16x128 f8 32.6.
    cad = _gfx950()["mfma_cadence_cyc"]
    assert cad == {"32x32x16_bf16": 32, "16x16x32_bf16": 16, "32x32x64_f8": 64, "16x16x128_f8": 32}


def test_f10_16x16x32_window_is_recorded():
    g = _gfx950()
    assert g["mfma_coexec_window_cyc"]["32x32x16"] == 24
    assert g["mfma_coexec_valu_fit"]["16x16x32"] == 2


def test_f10_notes_state_measured_scope():
    g = _gfx950()
    assert "L2 hit" in g["_l2_fabric_latency_note"] and "880" in g["_l2_fabric_latency_note"]
    assert "lane-to-lane" in g["_ds_read_stride_note"]
    assert g["ds_read_b128_2way_stride_bytes"] == 32
    assert "4 SIMDs" in g["_sched_interval_note"] and "gfx942" in g["_sched_interval_note"]


def test_f10_lds_request_fifo_is_not_presented_as_eight():
    # ATT trace: one wave queues ~16 ds_read_b128 before issue stalls; LDS-DMA does not block.
    g = _gfx950()
    assert g["lds_sp_fifo_entries_per_simd_pair"] is None
    assert g["ds_read_b128_queued_per_wave_before_stall"] == 16
    assert "ATT trace" in g["_lds_sp_fifo_note"]


# ---------------------------------------------------------------- F14: amd_isa measured facts
def _isa(*args):
    r = run(sys.executable, KT / "amd_isa.py", "--arch", "gfx950", *args)
    assert r.returncode == 0, r.stderr
    return r.stdout


@pytest.mark.parametrize("instr,cyc", [("V_MFMA_F32_32X32X64_F8F6F4", "64.3"),
                                       ("V_MFMA_F32_16X16X128_F8F6F4", "32.6")])
def test_f14_f8f6f4_facts_carry_measured_issue_interval(instr, cyc):
    out = _isa("facts", instr)
    assert f"issue_interval_measured {cyc} cyc" in out, out
    assert "measured gfx950" in out


def test_f14_lds_and_limits_state_the_measured_granule_and_broadcast():
    lds = _isa("lds")
    assert "1280" in lds and "measured gfx950" in lds
    assert "broadcast" in lds.lower() and "conflict-free" in lds
    assert "1280" in _isa("limits", "lds")


def test_f14_measured_lds_records_match_a_substring_query():
    assert "1280" in _isa("lds", "granule")


@pytest.mark.parametrize("tool", ["probe.py", "amd_occupancy.py", "asm_loop_audit.py", "amd_isa.py"])
def test_f14_help_full_is_accepted(tool):
    r = run(sys.executable, KT / tool, "--help-full")
    assert r.returncode == 0 and "Traceback" not in r.stderr, r.stderr


# ---------------------------------------------------------------- F14: ab_bench
def test_f14_ab_bench_default_cells_complete_the_rotation():
    import ab_bench
    # validity C6.2: 3 variants + auto control = 4 arms; the old default 6 was ROTATION INCOMPLETE.
    assert ab_bench.default_cells(4) == 8
    assert ab_bench.default_cells(3) == 6 and ab_bench.default_cells(2) == 6
    assert ab_bench.default_cells(7) == 7


def test_f14_ab_bench_text_has_no_dangling_tool_or_unreproduced_magnitude():
    src = (KT / "ab_bench.py").read_text()
    assert "round_record.py" not in src
    assert "10.7 us" not in src and "10.7us" not in src and "6.06" not in src
    r = run(sys.executable, KT / "ab_bench.py", "--help-full")
    assert r.returncode == 0 and "round_record" not in r.stdout and "6.06" not in r.stdout


def test_f14_dump_ir_accepts_help_full():
    r = run("bash", KT / "dump_ir.sh", "--help-full")
    assert r.returncode == 0, r.stderr


@pytest.mark.parametrize("extra", [["--diff", "{missing}"], ["--lds-per-cu", "-5"]])
def test_asm_loop_audit_refuses_missing_diff_and_bad_lds(tmp_path, extra):
    # an unreadable --diff used to print a report with no deltas ("nothing moved"), exit 0
    asm = tmp_path / "k.s"
    asm.write_text("  .vgpr_count: 128\n  .agpr_count: 0\n")
    extra = [str(tmp_path / "missing.txt") if x == "{missing}" else x for x in extra]
    result = run(sys.executable, KT / "asm_loop_audit.py", asm, "--arch", "gfx950", *extra)
    assert result.returncode == 2 and "error:" in result.stderr, (extra, result.returncode, result.stderr)
