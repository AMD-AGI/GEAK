#!/usr/bin/env python3
"""One profiling entry for the gluon_authoring method: the four dials (profile.md ## 3.1) per call.

The pack's own entry, used instead of kernel_workflow/scripts/profile_kernel.sh inside the Gluon
workflow. It only orchestrates the shared GEAK kernel tools (kernel_workflow/scripts/kernel_tools)
and records what each layer actually produced:

  trace  rocprofv3_safe.sh --kernel-trace           -> D: per-dispatch device time
  pmc    rocprofv3_safe.sh --pmc, one pass per       -> C: MfmaUtil / VALUBusy / MemUnitStalled /
         parse_pmc.PMC_GROUPS group, bisected on       wait-over-busy, fabric bytes and L2 hit
         abort/timeout (collect_pmc_group); the split
         that worked is remembered per box (--pmc-plan)
  att    capture.sh (ATT + IR dump + static audit)   -> B: cycles per MFMA and the inter-MFMA
         + mfma_efficiency over every decoded wave,      intervals, stall hotspots by instruction
         hotspot_analyzer, att_opclass                   and source line
  ir     probe.py measure, asm_loop_audit,           -> A: both occupancy limiters, spill; hot-loop
         deep_mfma_analysis                              mix, full-drain waits, MFMA stream breaks

Every profiler call has a kernel filter and a timeout, and the workload runs from a generated
cmd.sh so no profiler re-splits a quoted command. A pass that gpu_lock.sh refuses because another
tenant is on the GPU is retried after a wait (--busy-wait), never run on the busy GPU. A layer that fails is recorded with its state and
reason (never as zeros); the exit code says whether every requested layer is usable.

GPU: pass --gpu <id> (each tool takes the GEAK lock itself) or run the whole call under
`bash kernel_workflow/scripts/gpu_lock.sh <id>` (the tools reuse the held lock).

Usage:
  gluon_profile.py <out_dir> --kernel <substr> --source <kernel.py> [--gpu N] [--arch gfx942]
                   [--layers trace,pmc,att] [--timeout 300] [--compare <prev_out_dir>]
                   [--flops <per dispatch>] -- <cmd...>
  gluon_profile.py --compare-only <before_dir> <after_dir> [--out <dir>]
  gluon_profile.py --selftest

--flops (the operation count of ONE dispatch of the profiled case) adds the achieved rate to D. It is
the number to set against a peak (hw_budget.py --measured-tflops / --measured-ms); MfmaUtil and MFMA
busy cycles say how often the matrix pipe is occupied, which is not the fraction of peak throughput.

Outputs in <out_dir>: profile.json (layers + dials), report.md, compare.md (with --compare),
cmd.sh, trace/, pmc/, capture/ (att/, ir/), analysis/.
Exit: 0 every requested layer ok | 5 some layer partial/unavailable (recorded) | 4 nothing usable
      | 2 usage.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shlex
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
KT = REPO / "kernel_workflow" / "scripts" / "kernel_tools"
sys.path.insert(0, str(KT))

LAYERS = ("trace", "pmc", "att")
# Always collected (the IR / A dial rides the capture pass); accepted in --layers so a caller that
# names every layer profile.json reports is not refused.
ALWAYS = ("ir", "capture")
SCHEMA = "gluon_profile/1"


# --------------------------------------------------------------------------- helpers
def _run(cmd, log_path, timeout=None, env=None):
    """Run a tool, tee its output to log_path; return the exit code (124 on our own timeout)."""
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as log:
        log.write("$ " + shlex.join(str(c) for c in cmd) + "\n")
        log.flush()
        try:
            return subprocess.run([str(c) for c in cmd], stdout=log, stderr=subprocess.STDOUT,
                                  timeout=timeout, env=env).returncode
        except subprocess.TimeoutExpired:
            log.write(f"\n[gluon_profile] timed out after {timeout}s\n")
            return 124


BUSY_MARK = "still shows foreign work"   # gpu_lock.sh refusing a GPU another tenant is using
BUSY_LOG = []


def _run_gpu(a, cmd, log_path, timeout=None):
    """A GPU pass: when gpu_lock.sh refuses the GPU because another tenant is on it, wait and retry
    (up to --busy-wait seconds in total), never measure on it. Each wait is recorded."""
    waited, step = 0, 60
    while True:
        rc = _run(cmd, log_path, timeout=timeout)
        try:
            busy = rc != 0 and BUSY_MARK in Path(log_path).read_text()
        except OSError:
            busy = False
        if not busy or waited >= int(a.busy_wait):
            if busy:
                BUSY_LOG.append({"log": str(log_path), "waited_s": waited, "gave_up": True})
            return rc
        time.sleep(step)
        waited += step
        BUSY_LOG.append({"log": str(log_path), "waited_s": waited})


def _read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def _move_aside(path: Path):
    if path.exists():
        path.rename(path.with_name(f"{path.name}.old_{int(time.time())}_{os.getpid()}"))


def _gpu_args(flag, gpu):
    return [flag, str(gpu)] if gpu is not None else []


def _write_cmd(out: Path, cmd):
    script = out / "cmd.sh"
    script.write_text("#!/bin/bash\n# generated by gluon_profile.py: the profiled workload\n"
                      f"cd {shlex.quote(os.getcwd())}\nexec {shlex.join(cmd)}\n")
    script.chmod(0o755)
    return script


def _toolchain():
    info = {}
    try:
        info["rocm"] = Path("/opt/rocm/.info/version").read_text().strip()
    except OSError:
        info["rocm"] = None
    try:
        import triton  # noqa: F401  (the interpreter that runs this tool is the task's)
        info["triton"] = triton.__version__
    except Exception:  # noqa: BLE001 -- identity is best effort, recorded as unknown
        info["triton"] = None
    # No TRITON_HIP_USE_COEXEC_SCHEDULER here: 3.8.0 reads it nowhere (inert), and recording
    # it would suggest it shaped the run. coexec is gfx1250-only (LLVM warns on gfx950/gfx942).
    for k in ("TRITON_CACHE_DIR", "ROCPROF_ATT_LIBRARY_PATH"):
        info[k] = os.environ.get(k)
    return info


# --------------------------------------------------------------------------- layers
def layer_trace(a, out, cmd_sh):
    d = out / "trace"
    rc = _run_gpu(a, ["bash", KT / "rocprofv3_safe.sh", *_gpu_args("--gpu", a.gpu), "--kernel", a.kernel_regex,
                      "--out", d, "--kernel-trace", "--timeout", a.timeout, "--", cmd_sh], out / "logs" / "trace.log")
    st = _read_json(d / "rocprofv3_safe.json") or {"state": "error", "rc": rc}
    durs = []
    for f in glob.glob(str(d / "**" / "*kernel_trace.csv"), recursive=True):
        import csv
        with open(f) as fh:
            for row in csv.DictReader(fh):
                if re.search(a.kernel_regex, row.get("Kernel_Name", "")):
                    try:
                        durs.append((int(row["End_Timestamp"]) - int(row["Start_Timestamp"])) / 1e3)
                    except (KeyError, ValueError):
                        pass
    dial = None
    if durs:
        s = sorted(durs)
        dial = {"dispatches": len(s), "median_us": round(statistics.median(s), 2),
                "p10_us": round(s[len(s) // 10], 2), "p90_us": round(s[(len(s) * 9) // 10], 2),
                "basis": "rocprofv3 kernel trace under the profiler (screening only; GEAK verify decides)"}
    return {"state": st.get("state"), "rc": st.get("rc", rc), "dir": str(d)}, dial


def _pmc_plan_path(a):
    """Where the box's working PMC passes are remembered: the slot budget is box- and ROCm-dependent,
    and an over-budget group costs a full timeout before bisection splits it."""
    if a.pmc_plan:
        return Path(a.pmc_plan)
    import socket
    tc = _toolchain()
    key = re.sub(r"[^A-Za-z0-9_.-]", "_", f"{socket.gethostname()}_{a.arch or 'unknown'}_{tc.get('rocm') or 'unknown'}")
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "geak" / f"gluon_pmc_plan_{key}.json"


def layer_pmc(a, out, cmd_sh):
    import parse_pmc as P
    d = out / "pmc"
    d.mkdir(parents=True, exist_ok=True)
    plan_path = _pmc_plan_path(a)
    plan = (_read_json(plan_path) or {}).get("groups", {})
    groups = []
    for name, counters in P.PMC_GROUPS.items():
        n = [0]

        def run_pass(cs, name=name, n=n):
            n[0] += 1
            pd = d / f"pmc_{name}" if n[0] == 1 else d / f"pmc_{name}_{n[0]}"
            rc = _run_gpu(a, ["bash", KT / "rocprofv3_safe.sh", *_gpu_args("--gpu", a.gpu), "--kernel", a.kernel_regex,
                              "--out", pd, "--pmc", " ".join(cs), "--timeout", a.timeout, "--", cmd_sh],
                          out / "logs" / f"pmc_{name}_{n[0]}.log")
            st = _read_json(pd / "rocprofv3_safe.json") or {}
            return {"rc": st.get("rc", rc), "state": st.get("state", "error")}

        known = plan.get(name)
        res = None
        if known and sorted(c for p_ in known["passes"] for c in p_) + sorted(known["dropped"]) == sorted(counters):
            # The remembered split: run its passes as they are; any failure falls back to bisection.
            res = {"passes": [], "dropped": list(known["dropped"]), "failed": [], "partial": [], "plan": str(plan_path)}
            for cs in known["passes"]:
                o = run_pass(cs)
                if o["rc"] != 0:
                    res = None
                    break
                res["passes"].append(list(cs))
        if res is None:
            res = P.collect_pmc_group(counters, run_pass)
        res["group"] = name
        groups.append(res)
        if not res["failed"] and not res["partial"] and res["passes"]:
            plan[name] = {"passes": res["passes"], "dropped": res["dropped"]}
    state = P.collection_state(groups)
    try:
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        plan_path.write_text(json.dumps({"schema": "gluon_pmc_plan/1", "groups": plan}, indent=1))
    except OSError:
        pass
    parsed = P.parse_pmc(str(d), re.compile(a.kernel_regex))
    der = P.derived(parsed, arch=a.arch)
    with open(out / "analysis" / "pmc.txt", "w") as fh:
        subprocess.run([sys.executable, KT / "parse_pmc.py", d, a.kernel_regex, "--regex",
                        *(["--arch", a.arch] if a.arch else [])], stdout=fh, stderr=subprocess.STDOUT)
    v = parsed["vals"]
    wait, busy = P.avg(v, "SQ_WAIT_ANY"), P.avg(v, "SQ_BUSY_CYCLES")
    mem = der.get("memory") or {}
    dial = {"MfmaUtil": der.get("MfmaUtil"), "VALUBusy": der.get("VALUBusy"),
            "MemUnitStalled": der.get("MemUnitStalled"), "LDSBankConflict": der.get("LDSBankConflict"),
            "occupancy": der.get("occupancy"),
            "wait_over_busy": round(wait / busy, 3) if wait and busy else None,
            "memory": mem or None,
            "dropped_counters": sorted({c for g in groups for c in g["dropped"]})}
    info = {"state": "ok" if state == "collected" else state, "groups": [
        {"group": g["group"], "passes": g["passes"], "dropped": g["dropped"],
         "failed": [f.get("state") for f in g.get("failed", [])],
         "partial": [p.get("state") for p in g.get("partial", [])], "from_plan": bool(g.get("plan"))}
        for g in groups], "dir": str(d), "pmc_plan": str(plan_path)}
    return info, dial


def _att_dispatch(cap: Path):
    st = _read_json(cap / "att" / "rocprofv3_safe.json") or {}
    disp = st.get("att_dispatch")
    if not disp or not Path(disp).is_dir():
        found = sorted(glob.glob(str(cap / "att" / "ui_output_agent_*")))
        disp = found[0] if found else None
    return st, disp


def mfma_dial(dispatch):
    import mfma_efficiency as ME
    code = os.path.join(dispatch, "code.json")
    per_wave, meds, cls = [], [], Counter()
    nm = []
    for w in sorted(glob.glob(os.path.join(dispatch, "se*_wv*.json"))):
        try:
            r = ME.analyze(ME.load(w, code))
        except (ValueError, KeyError, OSError):
            continue
        if not r["n_mfma"] or not r["span"]:
            continue
        nm.append(r["n_mfma"])
        per_wave.append(r["span"] / r["n_mfma"])
        meds.append(r["cadence_median"])
        cls.update(r["cyc_by_cls"])
    if not per_wave:
        return None
    tot = sum(cls.values()) or 1
    share = {k: round(100 * c / tot, 1) for k, c in cls.most_common()}
    feed = {"lds": share.get("lds", 0.0), "gmem": share.get("gmem", 0.0)}
    return {"waves": len(per_wave), "mfma_per_wave": statistics.median(nm),
            "cycles_per_mfma": round(statistics.median(per_wave), 1),
            "cadence_median": statistics.median(meds), "inter_mfma_share_pct": share, "feed_split_pct": feed,
            "basis": "ATT, one traced CU; wave span / MFMA count, median over decoded waves"}


def hotspot_dial(dispatch, topk=6):
    import hotspot_analyzer as H
    ins = H.load_instructions(dispatch)
    total_stall = sum(i.stall_cycles for i in ins) or 1
    total = sum(i.total_cycles for i in ins) or 1
    by_type = Counter()
    for i in ins:
        by_type[i.stall_type] += i.stall_cycles
    top = sorted(ins, key=lambda i: i.stall_cycles, reverse=True)[:topk]
    src = H.aggregate_by_source(ins)[:topk]
    return {"stall_pct_of_cycles": round(100 * total_stall / total, 1),
            "stall_by_type_pct": {k: round(100 * c / total_stall, 1) for k, c in by_type.most_common()},
            "top_instructions": [{"asm": i.asm.strip()[:60], "type": i.stall_type, "source": i.source_loc,
                                  "stall_pct": round(100 * i.stall_cycles / total_stall, 2)} for i in top],
            "top_source_lines": [{"source": h.source_loc, "type": h.dominant_stall_type,
                                  "stall_pct": round(100 * h.total_stall_cycles / total_stall, 2)} for h in src]}


def _asm_file(cap: Path):
    for pat in ("**/*.s", "**/*.amdgcn"):
        found = sorted(p for p in glob.glob(str(cap / "ir" / pat), recursive=True) if ".old_" not in p)
        if found:
            return Path(found[0])
    return None


def static_dial(asm: Path, a, ana: Path):
    txt = (ana / "asm_audit.txt").read_text() if (ana / "asm_audit.txt").exists() else ""
    m = re.search(r"hot loop: lines \[\d+, \d+\]\s+\((\d+) instructions\)", txt)
    fd = re.search(r"->\s+(\d+)% of memory waits are full drains", txt)
    mf = re.search(r"M mfma\s+(\d+)", txt)
    dm = (ana / "deep_mfma.txt").read_text() if (ana / "deep_mfma.txt").exists() else ""
    br = re.search(r"what breaks the MFMA stream \((\d+) inter-MFMA instr\)", dm)
    pj = _read_json(ana / "probe.json") or {}
    k = next((x for x in pj.get("kernels", []) if str(x.get("artifact_path", "")).endswith(asm.suffix)),
             (pj.get("kernels") or [None])[0])
    occ = None
    if k:
        wpw = (k.get("wg_size") or 0) // (k.get("wave_size") or 64) or None
        simds = 4
        to_wg = (lambda w: (w * simds) // wpw if (w and wpw) else None)
        occ = {"vgpr": k.get("vgpr_combined"), "agpr": k.get("agpr"), "spill_bytes": k.get("spill_bytes"),
               "lds_bytes_per_wg": k.get("lds_bytes_per_wg"), "waves_per_simd": k.get("waves_per_simd"),
               "vgpr_wg_per_cu": to_wg(k.get("waves_by_vgpr_term")),
               "lds_wg_per_cu": to_wg(k.get("waves_by_lds_term")),
               "source": k.get("waves_source"), "warning": k.get("waves_warning")}
        if occ["vgpr_wg_per_cu"] is not None and occ["lds_wg_per_cu"] is not None:
            v, l_ = occ["vgpr_wg_per_cu"], occ["lds_wg_per_cu"]
            occ["binding"] = "both" if v == l_ else ("registers" if v < l_ else "lds")
    return {"occupancy": occ,
            "hot_loop": {"instructions": int(m.group(1)) if m else None, "mfma": int(mf.group(1)) if mf else None,
                         "full_drain_wait_pct": int(fd.group(1)) if fd else None,
                         "instructions_between_mfma": int(br.group(1)) if br else None}}


def layer_att(a, out, cmd_sh, want_att):
    cap = out / "capture"
    ana = out / "analysis"
    rc = _run_gpu(a, ["bash", KT / "capture.sh", a.name, cap, "--kernel", a.kernel, "--source", a.source,
               *_gpu_args("--dev", a.gpu), *([] if want_att else ["--no-att"]), "--", cmd_sh],
              out / "logs" / "capture.log", timeout=int(a.timeout) * 4)
    cj = _read_json(cap / "capture.json") or {}
    info = {"capture_rc": rc, "capture_json": str(cap / "capture.json"), "branch": cj.get("branch")}
    att_info, b_dial = {"state": "not_requested"}, None
    if want_att:
        st, disp = _att_dispatch(cap)
        att_info = {"state": st.get("state") or ("unavailable" if not disp else "unknown"),
                    "reason": st.get("reason") or cj.get("att_state"), "dispatch": disp,
                    "decoded_waves": st.get("decoded_waves")}
        if disp and Path(disp, "code.json").exists():
            b_dial = {"mfma": mfma_dial(disp)}
            try:
                b_dial["hotspots"] = hotspot_dial(disp)
            except (OSError, ValueError, KeyError) as exc:
                b_dial["hotspots"] = {"error": str(exc)}
            _run([sys.executable, KT / "hotspot_analyzer.py", disp, "--topk", "12", "--mode", "both",
                  "--kernel", a.kernel, *(["--arch", a.arch] if a.arch else [])], ana / "hotspot.txt")
            _run([sys.executable, KT / "att_opclass.py", os.path.join(disp, "code.json"),
                  "--json", ana / "opclass.json"], ana / "opclass.txt")
    asm = _asm_file(cap)
    ir_info, a_dial = {"state": "unavailable", "reason": "no .s/.amdgcn in capture/ir"}, None
    if asm:
        ir_info = {"state": "ok", "asm": str(asm)}
        arch = ["--arch", a.arch] if a.arch else []
        _run([sys.executable, KT / "probe.py", "measure", "--dir", asm.parent, *arch,
              "--json", ana / "probe.json"], ana / "probe.txt")
        _run([sys.executable, KT / "asm_loop_audit.py", asm, "--kernel", a.kernel, "--opcodes", "20",
              "--meta", asm.parent, *arch], ana / "asm_audit.txt")
        _run([sys.executable, KT / "deep_mfma_analysis.py", asm, "--kernel", a.kernel], ana / "deep_mfma.txt")
        a_dial = static_dial(asm, a, ana)
    return info, att_info, ir_info, a_dial, b_dial


# --------------------------------------------------------------------------- compare / report
def _flatten(d, prefix=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(_flatten(v, f"{prefix}{k}."))
    elif isinstance(d, (int, float)) and not isinstance(d, bool):
        out[prefix[:-1]] = d
    return out


def compare(before: dict, after: dict):
    fb, fa = _flatten(before.get("dials", {})), _flatten(after.get("dials", {}))
    rows = []
    for k in sorted(set(fb) | set(fa)):
        b, a_ = fb.get(k), fa.get(k)
        delta = (a_ - b) if (b is not None and a_ is not None) else None
        pct = (100.0 * delta / b) if (delta is not None and b) else None
        rows.append({"metric": k, "before": b, "after": a_, "delta": delta,
                     "delta_pct": round(pct, 1) if pct is not None else None})
    return rows


def _fmt(v):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.3g}" if abs(v) < 100 else f"{v:,.1f}"
    return str(v)


def write_compare(rows, path, before_dir, after_dir):
    moved = [r for r in rows if r["delta"] not in (None, 0)]
    lines = [f"# Profile delta: {before_dir} -> {after_dir}", "",
             "| dial | before | after | delta | delta % |", "| --- | --- | --- | --- | --- |"]
    lines += [f"| {r['metric']} | {_fmt(r['before'])} | {_fmt(r['after'])} | {_fmt(r['delta'])} | "
              f"{_fmt(r['delta_pct'])} |" for r in moved]
    missing = [r["metric"] for r in rows if r["before"] is None or r["after"] is None]
    if missing:
        lines += ["", "Present on one side only (not a zero): " + ", ".join(missing)]
    Path(path).write_text("\n".join(lines) + "\n")


def write_report(prof, path):
    L = [f"# gluon_profile: {prof['name']}", "",
         f"kernel `{prof['kernel']}` · source `{prof['source']}` (sha256 {prof.get('source_sha256', '?')[:12]}) · "
         f"arch {prof.get('arch') or '?'} · toolchain {json.dumps(prof['toolchain'])}", "",
         "## Layers", "", "| layer | state | detail |", "| --- | --- | --- |"]
    for k, v in prof["layers"].items():
        L.append(f"| {k} | {v.get('state')} | {v.get('reason') or v.get('dir') or v.get('asm') or ''} |")
    d = prof["dials"]
    occ = (d.get("A") or {}).get("occupancy") or {}
    hl = (d.get("A") or {}).get("hot_loop") or {}
    L += ["", "## A - occupancy and registers", "",
          f"VGPR {_fmt(occ.get('vgpr'))} (AGPR {_fmt(occ.get('agpr'))}, spill {_fmt(occ.get('spill_bytes'))} B) · "
          f"LDS/WG {_fmt(occ.get('lds_bytes_per_wg'))} B · WG/CU by registers {_fmt(occ.get('vgpr_wg_per_cu'))}, "
          f"by LDS {_fmt(occ.get('lds_wg_per_cu'))} -> binding {occ.get('binding', 'n/a')}", "",
          f"hot loop: {_fmt(hl.get('instructions'))} instructions, {_fmt(hl.get('mfma'))} MFMA, "
          f"{_fmt(hl.get('instructions_between_mfma'))} between MFMAs, "
          f"{_fmt(hl.get('full_drain_wait_pct'))}% of memory waits are full drains "
          "(analysis/asm_audit.txt, deep_mfma.txt)"]
    mf = ((d.get("B") or {}).get("mfma")) or {}
    hs = ((d.get("B") or {}).get("hotspots")) or {}
    L += ["", "## B - instruction stream (ATT)", ""]
    if mf:
        L += [f"cycles per MFMA {_fmt(mf.get('cycles_per_mfma'))} · median MFMA-to-MFMA {_fmt(mf.get('cadence_median'))} "
              f"· {mf.get('waves')} waves", "",
              "inter-MFMA cycles by preceding class: " +
              ", ".join(f"{k} {v}%" for k, v in mf.get("inter_mfma_share_pct", {}).items())]
    if hs.get("top_instructions"):
        L += ["", f"stalls = {hs['stall_pct_of_cycles']}% of cycles; by type: " +
              ", ".join(f"{k} {v}%" for k, v in hs["stall_by_type_pct"].items()), "",
              "| # | stall % | type | instruction | source |", "| --- | --- | --- | --- | --- |"]
        L += [f"| {i + 1} | {t['stall_pct']} | {t['type']} | `{t['asm']}` | {t['source']} |"
              for i, t in enumerate(hs["top_instructions"])]
    if not mf and not hs:
        L.append("ATT not available: see layers.")
    c = d.get("C") or {}
    L += ["", "## C - counters", ""]
    if c:
        L.append(f"MfmaUtil {_fmt(c.get('MfmaUtil'))} · VALUBusy {_fmt(c.get('VALUBusy'))} · MemUnitStalled "
                 f"{_fmt(c.get('MemUnitStalled'))} · LDSBankConflict {_fmt(c.get('LDSBankConflict'))} · "
                 f"wait/busy {_fmt(c.get('wait_over_busy'))} · dropped {c.get('dropped_counters') or 'none'} "
                 "(memory routes: analysis/pmc.txt)")
    else:
        L.append("PMC not available: see layers.")
    t = d.get("D") or {}
    L += ["", "## D - device time under the profiler (screening only)", "",
          f"{_fmt(t.get('median_us'))} us median (p10 {_fmt(t.get('p10_us'))}, p90 {_fmt(t.get('p90_us'))}), "
          f"{_fmt(t.get('dispatches'))} dispatches" if t else "trace not available: see layers.",
          *([f"achieved {_fmt(t.get('achieved_tflops'))} TFLOP/s at {t.get('flops_per_dispatch'):.4g} op/dispatch: set "
             "it against the peak with hw_budget.py --measured-tflops; MfmaUtil (C) is pipe occupancy, not % of peak"]
            if t.get("achieved_tflops") else []),
          "", "Full tool outputs: analysis/ (hotspot.txt, opclass.txt, asm_audit.txt, deep_mfma.txt, "
          "probe.txt, pmc.txt), logs/."]
    Path(path).write_text("\n".join(L) + "\n")


# --------------------------------------------------------------------------- main
def profile(a, cmd):
    out = Path(a.out).resolve()
    _move_aside(out)
    (out / "analysis").mkdir(parents=True)
    (out / "logs").mkdir()
    a.kernel_regex = re.escape(a.kernel)
    a.name = a.name or re.sub(r"[^A-Za-z0-9_]", "_", out.name)
    cmd_sh = _write_cmd(out, cmd)
    import hashlib
    prof = {"schema": SCHEMA, "name": a.name, "kernel": a.kernel, "source": str(Path(a.source).resolve()),
            "source_sha256": hashlib.sha256(Path(a.source).read_bytes()).hexdigest(),
            "arch": a.arch, "command": cmd, "toolchain": _toolchain(), "layers": {}, "dials": {}}
    want = [x for x in a.layers.split(",") if x and x not in ALWAYS]
    if "trace" in want:
        prof["layers"]["trace"], prof["dials"]["D"] = layer_trace(a, out, cmd_sh)
        dd = prof["dials"]["D"]
        if dd and getattr(a, "flops", None) and dd.get("median_us"):
            dd["flops_per_dispatch"] = a.flops
            dd["achieved_tflops"] = round(a.flops / (dd["median_us"] * 1e-6) / 1e12, 2)
    if "pmc" in want:
        prof["layers"]["pmc"], prof["dials"]["C"] = layer_pmc(a, out, cmd_sh)
    cap, att, ir, a_dial, b_dial = layer_att(a, out, cmd_sh, "att" in want)
    prof["layers"]["capture"] = {"state": "ok" if cap["capture_rc"] == 0 else "error", **cap}
    if "att" in want:
        prof["layers"]["att"] = att
    prof["layers"]["ir"] = ir
    prof["dials"]["A"], prof["dials"]["B"] = a_dial, b_dial
    states = [v.get("state") for k, v in prof["layers"].items() if k in want or k == "ir"]
    prof["complete"] = all(s == "ok" for s in states)
    prof["gpu_busy_waits"] = BUSY_LOG
    (out / "profile.json").write_text(json.dumps(prof, indent=1, default=str))
    write_report(prof, out / "report.md")
    if a.compare:
        before = _read_json(Path(a.compare) / "profile.json")
        if before:
            write_compare(compare(before, prof), out / "compare.md", a.compare, out)
        else:
            (out / "compare.md").write_text(f"No profile.json in {a.compare}; nothing to compare.\n")
    print(f"[gluon_profile] {out}/report.md  layers: " +
          ", ".join(f"{k}={v.get('state')}" for k, v in prof["layers"].items()))
    if prof["complete"]:
        return 0
    return 5 if any(s in ("ok", "partial", "collected") for s in states) else 4


def _selftest():
    fail = 0

    def chk(cond, msg):
        nonlocal fail
        if not cond:
            print("FAIL:", msg)
            fail = 1

    b = {"dials": {"A": {"occupancy": {"vgpr": 204, "lds_wg_per_cu": 2}}, "B": {"mfma": {"cycles_per_mfma": 222.5}},
                   "C": {"MfmaUtil": 23.6, "memory": None}, "D": {"median_us": 564.7}}}
    a = {"dials": {"A": {"occupancy": {"vgpr": 204, "lds_wg_per_cu": 2}}, "B": {"mfma": {"cycles_per_mfma": 184.0}},
                   "C": {"MfmaUtil": None}, "D": {"median_us": 459.4}}}
    rows = {r["metric"]: r for r in compare(b, a)}
    chk(rows["B.mfma.cycles_per_mfma"]["delta"] == -38.5, "cycles per MFMA delta")
    chk(rows["D.median_us"]["delta_pct"] == -18.6, "device time delta %")
    chk(rows["C.MfmaUtil"]["after"] is None and rows["C.MfmaUtil"]["delta"] is None,
        "a missing value stays missing, never zero")
    chk(rows["A.occupancy.vgpr"]["delta"] == 0, "unchanged dial")
    import tempfile
    with tempfile.TemporaryDirectory() as t:
        write_compare(list(rows.values()), Path(t) / "c.md", "a", "b")
        txt = (Path(t) / "c.md").read_text()
        chk("B.mfma.cycles_per_mfma" in txt and "A.occupancy.vgpr" not in txt, "compare lists moved dials only")
        chk("C.MfmaUtil" in txt.split("one side only")[-1], "one-sided dial reported as missing")
        out = Path(t) / "o"
        out.mkdir()
        sh = _write_cmd(out, ["python3", "x.py", "a b", "--flag=1"])
        chk("exec python3 x.py 'a b' --flag=1" in sh.read_text(), "cmd.sh keeps argument boundaries")
        prof = {"name": "n", "kernel": "k", "source": "k.py", "toolchain": {}, "layers": {}, "dials": {"D": {
            "median_us": 100.0, "p10_us": 99.0, "p90_us": 101.0, "dispatches": 5,
            "flops_per_dispatch": 2e12, "achieved_tflops": 20.0}}}
        write_report(prof, Path(t) / "r.md")
        chk("achieved 20 TFLOP/s" in (Path(t) / "r.md").read_text(), "report carries the achieved rate")
    import contextlib
    import io
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        try:
            main(["/nonexistent/o", "--kernel", "k", "--source", "/nonexistent/k.py", "--layers", "trace,ir", "--", "true"])
        except SystemExit:
            pass
    chk("unknown layer" not in err.getvalue() and "--source not found" in err.getvalue(),
        "ir is accepted in --layers (it is always collected)")
    print("gluon_profile selftest:", "FAIL" if fail else "OK")
    return fail


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["--selftest"]:
        return _selftest()
    if argv[:1] == ["--compare-only"]:
        if len(argv) < 3:
            print("usage: gluon_profile.py --compare-only <before_dir> <after_dir> [--out <dir>]", file=sys.stderr)
            return 2
        before, after = _read_json(Path(argv[1]) / "profile.json"), _read_json(Path(argv[2]) / "profile.json")
        if not before or not after:
            print("both dirs need a profile.json", file=sys.stderr)
            return 2
        dst = Path(argv[argv.index("--out") + 1]) if "--out" in argv else Path(argv[2])
        write_compare(compare(before, after), dst / "compare.md", argv[1], argv[2])
        print(dst / "compare.md")
        return 0
    cmd = []
    if "--" in argv:
        i = argv.index("--")
        argv, cmd = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--kernel", required=True, help="device kernel name substring")
    ap.add_argument("--source", required=True, help="the kernel source file being profiled")
    ap.add_argument("--gpu", help="GPU id; each tool takes the GEAK lock itself (else run under gpu_lock.sh)")
    ap.add_argument("--arch", help="gfx target from the execution record, e.g. gfx942")
    ap.add_argument("--layers", default=",".join(LAYERS),
                    help="comma list of trace,pmc,att (ir/capture are always taken and may be named)")
    ap.add_argument("--flops", type=float,
                    help="operations of ONE dispatch of the profiled case: D gains the achieved TFLOP/s")
    ap.add_argument("--timeout", default="300", help="seconds per profiler pass")
    ap.add_argument("--pmc-plan", help="file remembering this box's working PMC passes "
                    "(default ~/.cache/geak/gluon_pmc_plan_<host>_<arch>_<rocm>.json)")
    ap.add_argument("--busy-wait", default="600",
                    help="seconds to keep retrying a pass gpu_lock.sh refuses because another tenant is on the GPU")
    ap.add_argument("--compare", help="a previous gluon_profile out dir: write compare.md")
    ap.add_argument("--name", help="capture name (default: out dir name)")
    a = ap.parse_args(argv)
    if not cmd:
        ap.error("the workload command goes after --")
    bad = [x for x in a.layers.split(",") if x and x not in LAYERS + ALWAYS]
    if bad:
        ap.error(f"unknown layer(s): {bad}")
    if not Path(a.source).is_file():
        ap.error(f"--source not found: {a.source}")
    if a.gpu is None and not os.environ.get("GEAK_GPU_LOCK_HELD"):
        import held_lock   # kernel_tools: a parent gpu_lock.sh holding the one visible GPU
        held = held_lock.held_gpu()
        if held is None:
            ap.error("pass --gpu <id> or run under kernel_workflow/scripts/gpu_lock.sh <id>")
        os.environ["GEAK_GPU_LOCK_HELD"] = held   # the tools below reuse the held lock
    return profile(a, cmd)


if __name__ == "__main__":
    sys.exit(main())
