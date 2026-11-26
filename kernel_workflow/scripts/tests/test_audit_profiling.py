"""Regression tests for the profiling-tool audit fixes (CPU-only; fake rocprofv3, no GPU).

F6  parse_pmc tcc_ea0 byte route sizes requests by their 128/64/32 B sub-counters.
F7  parse_pmc memory counters are split so no TCC derived byte counter shares a pass.
F8  no stale gfx95x FETCH_SIZE under-count caveat; TCC_HIT write half-lines noted.
F14 rocprofv3_safe.sh maps a KILL after the timeout (rc 137) to the timed-out state.
ATT audit: LLVM dual-load abort detected early; code:null on the target CU after the retry is
named; GEAK_GPU_LOCK_HELD misuse explained; mfma_efficiency takes several wave files;
hotspot_analyzer resolves the dispatch row without --kernel; profiler_deps flags duplicate comgr.
"""
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
from _fake_rocprofv3 import SCRIPTS, _decoder, _env

KT = SCRIPTS / "kernel_tools"


def _load(name):
    spec = importlib.util.spec_from_file_location(f"{name}_audit", KT / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(KT))
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path.pop(0)
    return mod


# 1 GiB copy (validity C7.3): 1 GiB read in 128 B requests + 1 GiB written in 64 B requests.
COPY_VALS = {
    "TCC_HIT_sum": [8391839.0], "TCC_MISS_sum": [16777529.0],
    "FETCH_SIZE": [1048587.5], "WRITE_SIZE": [1048576.0],
    "TCC_EA0_RDREQ_sum": [8388792.0], "TCC_EA0_WRREQ_sum": [16777216.0],
    "TCC_EA0_RDREQ_128B_sum": [8388608.0], "TCC_EA0_RDREQ_64B_sum": [390.0],
    "TCC_EA0_RDREQ_32B_sum": [0.0], "TCC_EA0_WRREQ_64B_sum": [16777216.0],
}


def test_f6_tcc_ea0_route_uses_request_sizes():
    pp = _load("parse_pmc")
    md = pp.memory_derived({"vals": COPY_VALS, "meta": {"duration_ns": 464622}}, arch="gfx950")
    ea0 = md["dram_bytes_by_route"]["tcc_ea0"]
    assert abs(ea0 / 1e6 - 2147.5) < 1.0, md["dram_bytes_by_route"]
    assert md["_crosscheck_ok"], md


def test_f6_tcc_ea0_route_dropped_without_size_subcounters():
    pp = _load("parse_pmc")
    vals = {k: v for k, v in COPY_VALS.items() if not k.endswith(("_128B_sum", "_64B_sum", "_32B_sum"))}
    md = pp.memory_derived({"vals": vals, "meta": {"duration_ns": 464622}}, arch="gfx950")
    assert "tcc_ea0" not in md["dram_bytes_by_route"], md["dram_bytes_by_route"]
    assert md["_crosscheck_ok"], md


def test_f7_tcc_byte_counters_do_not_share_a_pass():
    pp = _load("parse_pmc")
    groups = list(pp.PMC_GROUPS.values())
    flat = [c for g in groups for c in g]
    for c in ("FETCH_SIZE", "WRITE_SIZE", "TCC_HIT_sum", "TCC_MISS_sum"):
        assert c in flat, c
    for g in groups:
        assert not ({"FETCH_SIZE", "WRITE_SIZE"} <= set(g)), g
        if "FETCH_SIZE" in g or "WRITE_SIZE" in g:
            assert len(g) == 1, g
    rd = [g for g in groups if "TCC_EA0_RDREQ_128B_sum" in g]
    wr = [g for g in groups if "TCC_EA0_WRREQ_64B_sum" in g]
    assert rd and wr, groups


def test_f8_no_stale_fetch_size_caveat_and_write_hits_noted():
    pp = _load("parse_pmc")
    md = pp.memory_derived({"vals": COPY_VALS, "meta": {"duration_ns": 464622}}, arch="gfx950")
    assert "_fetch_size_caveat" not in md, md
    assert "write" in md.get("_l2_hit_note", ""), md
    text = (KT / "parse_pmc.py").read_text()
    assert "UNDER-count" not in text and "under-count on gfx95x" not in text


def _safe(tmp_path, fake_body, *args, extra_env=None):
    env = _env(tmp_path, KT_ALLOW_UNLOCKED="1")
    (tmp_path / "bin/rocprofv3").write_text(fake_body)
    env.update(extra_env or {})
    t0 = time.monotonic()
    r = subprocess.run(["bash", str(KT / "rocprofv3_safe.sh"), "--kernel", "k", "--out",
                        str(tmp_path / "o"), *args, "--", "true"],
                       env=env, cwd=tmp_path, capture_output=True, text=True, timeout=120)
    return r, time.monotonic() - t0, tmp_path / "o" / "rocprofv3_safe.json"


ABORT_THEN_HANG = """#!/usr/bin/env bash
case "${1:-}" in --version) echo fixture; exit 0;; --help) echo '--att --kernel-trace --pmc'; exit 0;; esac
echo ": CommandLine Error: Option 'spirv-expand-step' registered more than once!" >&2
echo "LLVM ERROR: inconsistency in registered CommandLine options" >&2
trap '' TERM
sleep 40
"""


def test_llvm_dual_load_abort_stops_early_with_its_own_state(tmp_path):
    r, elapsed, state = _safe(tmp_path, ABORT_THEN_HANG, "--pmc", "SQ_WAVES", "--timeout", "8",
                              extra_env={"KT_TIMEOUT_KILL_AFTER": "1"})
    data = json.loads(state.read_text())
    assert r.returncode == 7, (r.returncode, data, r.stderr[-800:])
    assert data["state"] == "llvm_dual_load" and data["reason"] == "llvm_dual_load", data
    assert elapsed < 7, elapsed
    assert "q.py llvm_dual_load" in r.stderr  # the fix pointer: the fact store card for this abort


TERM_IGNORED = """#!/usr/bin/env bash
case "${1:-}" in --version) echo fixture; exit 0;; --help) echo '--att --kernel-trace --pmc'; exit 0;; esac
trap '' TERM
sleep 40
"""


def test_f14_kill_after_timeout_reads_as_timed_out(tmp_path):
    r, _elapsed, state = _safe(tmp_path, TERM_IGNORED, "--pmc", "SQ_WAVES", "--timeout", "1",
                               extra_env={"KT_TIMEOUT_KILL_AFTER": "1"})
    data = json.loads(state.read_text())
    assert r.returncode == 124 and data["state"] == "timeout", (r.returncode, data)


def test_code_null_after_retry_names_the_empty_target_cu(tmp_path):
    env = _env(tmp_path, KT_ALLOW_UNLOCKED="1")
    decoder = tmp_path / "decoder"; decoder.mkdir(); _decoder(decoder)
    env["ROCPROF_ATT_LIBRARY_PATH"] = str(decoder)
    fake = tmp_path / "bin/rocprofv3"
    fake.write_text(fake.read_text().replace(
        '"$KT_DIR_FOR_FAKE" "$out/ui_output_agent_1_dispatch_1";;',
        '"$KT_DIR_FOR_FAKE" "$out/ui_output_agent_1_dispatch_1"\n'
        '       echo "$*" >> "$ATT_ARGS_LOG"\n'
        '       echo \'{"code":null}\' > "$out/ui_output_agent_1_dispatch_1/code.json";;'))
    env["ATT_ARGS_LOG"] = str(tmp_path / "att_args.log")
    r = subprocess.run(["bash", str(KT / "rocprofv3_safe.sh"), "--kernel", "_gemm", "--out",
                        str(tmp_path / "o"), "--att", "--", "true"],
                       env=env, cwd=tmp_path, capture_output=True, text=True, timeout=120)
    data = json.loads((tmp_path / "o" / "rocprofv3_safe.json").read_text())
    assert data["reason"] == "no_wave_on_target_cu", data
    assert "WG per CU" in data.get("hint", ""), data


def test_lock_held_must_be_a_gpu_id(tmp_path):
    env = _env(tmp_path, GEAK_GPU_LOCK_HELD="1", HIP_VISIBLE_DEVICES="7")
    r = subprocess.run(["bash", str(KT / "rocprofv3_safe.sh"), "--kernel", "k", "--out",
                        str(tmp_path / "o"), "--kernel-trace", "--", "true"],
                       env=env, cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert r.returncode == 2
    assert "GEAK_GPU_LOCK_HELD must be the GPU id" in r.stderr, r.stderr


def test_relative_out_gives_rocprofv3_an_absolute_dir(tmp_path):
    env = _env(tmp_path, KT_ALLOW_UNLOCKED="1")
    fake = tmp_path / "bin/rocprofv3"
    fake.write_text(fake.read_text().replace('mkdir -p "$out"', 'echo "$out" > "$DIR_LOG"; mkdir -p "$out"'))
    env["DIR_LOG"] = str(tmp_path / "dir.log")
    subprocess.run(["bash", str(KT / "rocprofv3_safe.sh"), "--kernel", "_gemm", "--out", "rel_out",
                    "--kernel-trace", "--", "true"], env=env, cwd=tmp_path,
                   capture_output=True, text=True, timeout=60)
    assert Path((tmp_path / "dir.log").read_text().strip()).is_absolute()


def _wave_files(tmp_path):
    sys.path.insert(0, str(KT))
    try:
        import _att_fixture
    finally:
        sys.path.pop(0)
    d = tmp_path / "ui_output_agent_1_dispatch_1"
    d.mkdir()
    _att_fixture.write(str(d))
    waves = sorted(d.glob("se*_wv*.json"))
    if len(waves) < 2:
        waves.append(d / "se1_sm0_sl0_wv0.json")
        waves[-1].write_text(waves[0].read_text())
    return d, waves


def test_mfma_efficiency_accepts_several_wave_files(tmp_path):
    d, waves = _wave_files(tmp_path)
    r = subprocess.run([sys.executable, str(KT / "mfma_efficiency.py"), *map(str, waves),
                        str(d / "code.json")], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert r.stdout.count("=== ") >= len(waves), r.stdout


def test_hotspot_resolves_dispatch_row_and_arch_without_flags(tmp_path):
    d, _ = _wave_files(tmp_path)
    (tmp_path / "r_kernel_trace.csv").write_text(
        "Kernel_Name,Dispatch_Id,VGPR_Count,Accum_VGPR_Count,SGPR_Count,LDS_Block_Size,"
        "Workgroup_Size_X,Workgroup_Size_Y,Workgroup_Size_Z\n"
        "_gemm_fwd_kernel,1,128,0,32,65536,256,1,1\nother,2,8,0,8,0,64,1,1\n")
    (tmp_path / "r_agent_info.csv").write_text('"Node_Id","Agent_Type","Name"\n0,"CPU","AMD EPYC"\n1,"GPU","gfx950"\n')
    r = subprocess.run([sys.executable, str(KT / "hotspot_analyzer.py"), str(d), "--mode", "asm"],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert "gfx950" in r.stdout and "unknown (pass --arch" not in r.stdout, r.stdout
    assert "128" in r.stdout, r.stdout


def test_profiler_deps_flags_duplicate_comgr(tmp_path, monkeypatch):
    site = tmp_path / "site"
    for part in ("_rocm_sdk_core/lib", "_rocm_sdk_devel/lib"):
        (site / part).mkdir(parents=True)
        (site / part / "libamd_comgr.so.3").write_bytes(b"same")
    monkeypatch.setenv("KT_ROCM_SDK_SITE", str(site))
    deps = _load("profiler_deps")
    assert deps.llvm_dual_load()["status"] == "duplicate_inode"
    (site / "_rocm_sdk_devel/lib/libamd_comgr.so.3").unlink()
    (site / "_rocm_sdk_devel/lib/libamd_comgr.so.3").symlink_to(site / "_rocm_sdk_core/lib/libamd_comgr.so.3")
    assert deps.llvm_dual_load()["status"] == "single_inode"


def test_capture_reports_llvm_dual_load_instead_of_a_clean_run(tmp_path):
    env = _env(tmp_path, KT_ALLOW_UNLOCKED="1", PROFILE_PMC_TIMEOUT="8", KT_TIMEOUT_KILL_AFTER="1")
    (tmp_path / "bin/rocprofv3").write_text(ABORT_THEN_HANG)
    source = tmp_path / "kernel.py"; source.write_text("# fixture\n")
    out = tmp_path / "cap"
    r = subprocess.run(["bash", str(KT / "capture.sh"), "case", str(out), "--source", str(source),
                        "--kernel", "_gemm", "--", "true"], env=env, cwd=tmp_path,
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 7, (r.returncode, r.stdout[-1500:], r.stderr[-1500:])
    assert json.loads((out / "capture.json").read_text())["trace_state"] == "llvm_dual_load"


@pytest.mark.parametrize("body", ["[]", '{"code": [[1]]}', '{"code": [1]}'])
def test_hotspot_rejects_a_wrong_shape_code_json(tmp_path, body):
    (tmp_path / "code.json").write_text(body)
    r = subprocess.run([sys.executable, str(KT / "hotspot_analyzer.py"), str(tmp_path)],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 1 and "cannot read" in r.stderr, (r.returncode, r.stderr)
    assert "Traceback" not in r.stderr


def test_hotspot_says_when_the_dir_has_no_code_json(tmp_path):
    # a directory with no code.json (e.g. the att/ parent, not a dispatch dir) used to traceback
    r = subprocess.run([sys.executable, str(KT / "hotspot_analyzer.py"), str(tmp_path)],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 1 and "no code.json" in r.stderr, (r.returncode, r.stderr)
    assert "Traceback" not in r.stderr
    (tmp_path / "code.json").write_text('{"code": null}')   # no waves caught: report zeros, no crash
    r = subprocess.run([sys.executable, str(KT / "hotspot_analyzer.py"), str(tmp_path),
                        "--arch", "gfx950", "--wave-size", "64"],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0 and "Traceback" not in r.stderr, r.stderr


def test_hotspot_reads_csvs_from_the_per_host_subdir(tmp_path):
    d, _ = _wave_files(tmp_path)
    host = tmp_path / "hostname0"; host.mkdir()
    (host / "1522_kernel_trace.csv").write_text(
        "Kernel_Name,Dispatch_Id,VGPR_Count,Accum_VGPR_Count,SGPR_Count,LDS_Block_Size,"
        "Workgroup_Size_X,Workgroup_Size_Y,Workgroup_Size_Z\n_gemm_fwd_kernel,1,128,0,32,65536,256,1,1\n")
    (host / "1522_agent_info.csv").write_text('"Node_Id","Agent_Type","Name"\n1,"GPU","gfx950"\n')
    r = subprocess.run([sys.executable, str(KT / "hotspot_analyzer.py"), str(d), "--mode", "asm"],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert "Architecture:   gfx950" in r.stdout and "128" in r.stdout, r.stdout
