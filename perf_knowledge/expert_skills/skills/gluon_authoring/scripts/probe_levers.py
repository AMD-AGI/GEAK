#!/usr/bin/env python3
"""probe_levers.py - per-skill (gluon) RUNTIME probes for version/arch-sensitive levers.

Cross-cutting rule (plan Part 5): probe-per-build, never bake. Lever availability, valid
LLVM strategy strings, and low-precision ISA exposure are resolved on the TARGET box at
runtime by these functions - the `probe` field of each `lever-cards.json` card names one.
The current container was used ONLY to prove the mechanisms EXIST; the VALUE is always
probed here on the box the agent is running on.

Each probe returns {available: bool|None, evidence: str, cmd: str} and NEVER raises.

    python3 probe_levers.py --all [--arch gfx942]     # live probe, JSON out
    python3 probe_levers.py --selftest                 # offline structural check
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys


def _run(cmd, timeout=25):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout \
            + subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stderr
    except Exception:  # noqa: BLE001
        return ""


def probe_llvm_knob(knob="amdgpu-sched-strategy"):
    """gluon attention intra-loop scheduling: is the LLVM knob present? (version-sensitive).
    DEFAULT-OFF: presence alone does NOT sanction use; valid strategy strings vary by build."""
    cmd = f"llc --help-hidden 2>&1 | grep -i {knob}"
    out = _run(["bash", "-c", cmd])
    present = knob in out
    # try to harvest the valid enum strings if present
    strings = []
    if present:
        for tok in ("max-occupancy", "max-ilp", "max-memory-clause", "iterative-ilp",
                    "iterative-minreg", "iterative-maxbb"):
            if tok in out:
                strings.append(tok)
    return {"available": present, "valid_strings": strings,
            "evidence": out.strip()[:200] or "knob not found in llc --help-hidden",
            "cmd": cmd, "policy": "DEFAULT-OFF; probe-per-build; do not conclude from one docker"}


def probe_python_symbol(module, attr_path):
    """Is a DSL symbol exposed by THIS build (e.g. gluon mfma_scaled)?"""
    try:
        obj = __import__(module, fromlist=["_"])
        for a in attr_path.split("."):
            obj = getattr(obj, a)
        return {"available": True, "evidence": f"{module}.{attr_path} present", "cmd": f"import {module}"}
    except Exception as e:  # noqa: BLE001
        return {"available": False, "evidence": f"{type(e).__name__}: {e}"[:160],
                "cmd": f"import {module}.{attr_path}"}


def probe_fp8_scaled_mfma(arch=None):
    """gluon fp8/fp4 scaled-MFMA: silicon requires gfx950+ AND the gluon build must expose it.
    Full probe = compile a tiny gl.amd.cdna4.mfma_scaled kernel; here we do the light symbol +
    arch gate (the heavy compile is the card's escalation)."""
    arch_ok = bool(arch and (arch >= "gfx950" or arch.startswith("gfx125")))
    sym = probe_python_symbol("triton.experimental.gluon.language.amd.cdna4", "mfma_scaled")
    return {"available": bool(arch_ok and sym["available"]),
            "arch_gate": f"{arch} scaled-MFMA silicon = {arch_ok}",
            "symbol": sym["available"], "evidence": sym["evidence"],
            "cmd": "compile tiny gl.amd.cdna4.mfma_scaled(a,None,'e4m3',b,None,'e4m3',acc) for arch"}


def probe_ds_read_tr():
    """gluon ds_read_b64_tr transpose-on-read: no Gluon SOURCE API -> Tier-B (compiler co-design)."""
    return {"available": False, "tier": "B",
            "evidence": "no gluon source API for ds_read_tr; sanctioned co-design only",
            "cmd": "check asm for ds_read_b64_tr after a sanctioned pass"}


def probe_reinject_pipeliner(arch=None):
    """Route-1 (reinject_ttgir_pipeliner): are plain's TTGIR software-pipeliner passes present in
    THIS libtriton.so? They are the passes make_ttgir already calls, so re-injecting them into
    gluon_to_ttgir (gated num_stages>1) needs NO rebuild - a Python-only compiler.py edit. Light
    probe = pass-symbol presence; full de-risk = opt_swp_test.py (run them on a dumped .ttgir)."""
    need = ["add_schedule_loops", "add_pipeline", "add_optimize_dot_operands"]
    try:
        from triton._C.libtriton import amd  # noqa: F401
        mod = amd.passes.ttgpuir
        missing = [p for p in need if not hasattr(mod, p)]
        ok = not missing
        return {"available": ok,
                "evidence": ("pipeliner passes present in libtriton.so (" + ", ".join(need) + ")")
                            if ok else f"missing passes: {missing}",
                "cmd": "opt_swp_test.py <dumped_gluon.ttgir> 2 -> expect local_alloc 0->2 / "
                       "local_store 0->4 / local_load 0->4 (no rebuild)",
                "note": "Route 1: splice these into gluon_to_ttgir after add_combine_tensor_select_and_if, "
                        "gated num_stages>1 (Python-only compiler.py; default-OFF; cache-key-safe)"}
    except Exception as e:  # noqa: BLE001
        return {"available": None,
                "evidence": f"cannot import triton AMD passes: {type(e).__name__}: {e}"[:160],
                "cmd": "opt_swp_test.py <dumped_gluon.ttgir> 2 (needs triton with the AMD backend)",
                "note": "Route 1 needs the AMD pipeliner passes in libtriton.so (they ship with make_ttgir)"}


def probe_gemm_compiler_stack(arch=None):
    """intra_wave GEMM compiler stack (gemm_compiler_stack card): are the base -> llir ->
    llir+ra -> llir+amdgcnas rungs present on THIS build? LLIR_SCHED / RA_HINTS are
    build-pinned (tutorial fork); amdgcnas is decoupled (pure-Python asm hook, stock Triton).
    Probe-per-build, NEVER bake: presence alone does not sanction use, and absence of a rung
    is a scoped toolchain ceiling for that rung, not for the whole card."""
    rungs = {}
    # llir rung: the LLIR schedule pass symbol in the AMD backend (fork-only)
    try:
        from triton._C.libtriton import amd  # noqa: F401
        mod = getattr(getattr(amd, "passes", None), "ttgpuir", None)
        names = dir(mod) if mod else []
        rungs["llir"] = any("llir" in n.lower() and "sched" in n.lower() for n in names)
    except Exception as e:  # noqa: BLE001
        rungs["llir"] = None
    # ra rung: the LLVM AGPR-form knobs the RA hint drives
    ra = probe_llvm_knob("amdgpu-mfma-vgpr-form")
    rungs["ra"] = ra["available"]
    # amdgcnas rung: the pure-Python post-asm tool (decoupled; stock Triton)
    rungs["amdgcnas"] = _has_module("triton.tools.amdgcnas")
    have = [k for k, v in rungs.items() if v]
    return {"available": (True if rungs["llir"] else (None if rungs["llir"] is None else False)),
            "rungs": rungs,
            "evidence": f"present rungs: {have or 'none conclusive'}; llir/ra are fork-pinned, "
                        "amdgcnas decoupled",
            "cmd": "toggle TRITON_ENABLE_LLIR_SCHED / TRITON_FORCE_MFMA_AGPR / "
                   "TRITON_AMDGCNAS_PLUGIN then dump_ir.sh + asm_loop_audit to confirm each landed",
            "policy": "DEFAULT-OFF; probe-per-build; GEMM-only (pure MFMA->MFMA); a missing rung "
                      "is a scoped ceiling -> fall back to non-scheduled latency hiding"}


def probe_warp_pipeline(arch=None):
    """inter_wave wave-ping-pong (warp_pipeline_schedule card): is warp_pipeline_stage exposed by
    THIS gluon build? Runs on stock Triton (no plugins/env), so this is a plain symbol probe."""
    for m in ("triton.experimental.gluon.language.amd",
              "triton.experimental.gluon.language",
              "triton.experimental.gluon"):
        r = probe_python_symbol(m, "warp_pipeline_stage")
        if r["available"]:
            return {"available": True, "evidence": f"{m}.warp_pipeline_stage present",
                    "cmd": r["cmd"], "policy": "stock Triton; needs 2 waves/SIMD to interleave"}
    return {"available": False,
            "evidence": "warp_pipeline_stage not found in the probed gluon modules",
            "cmd": "import triton.experimental.gluon...warp_pipeline_stage",
            "policy": "stock Triton; if absent, use the manual multi-buffer pipeline instead"}


def _has_module(name):
    try:
        __import__(name)
        return True
    except Exception:  # noqa: BLE001
        return False


PROBES = {
    "reinject_ttgir_pipeliner": probe_reinject_pipeliner,
    "attn_intra_loop_schedule": lambda arch: probe_llvm_knob("amdgpu-sched-strategy"),
    "scaled_mfma_lowprec": probe_fp8_scaled_mfma,
    "ds_read_tr_transpose": lambda arch: probe_ds_read_tr(),
    "gemm_compiler_stack": probe_gemm_compiler_stack,
    "warp_pipeline_schedule": probe_warp_pipeline,
}


def run_all(arch=None):
    out = {}
    for name, fn in PROBES.items():
        try:
            out[name] = fn(arch)
        except Exception as e:  # noqa: BLE001
            out[name] = {"available": None, "evidence": f"probe error {e}"[:120]}
    return out


def _selftest() -> int:
    # offline: every probe returns a dict with 'available'; scaled-MFMA arch gate works
    r = run_all("gfx942")
    for k, v in r.items():
        assert "available" in v, k
    # gfx942 has no scaled-MFMA silicon -> must be False regardless of symbol
    assert r["scaled_mfma_lowprec"]["available"] is False, r["scaled_mfma_lowprec"]
    # gfx950 arch gate opens (symbol may still be absent in a given build -> that's fine)
    r950 = probe_fp8_scaled_mfma("gfx950")
    assert "gfx950" in r950["arch_gate"]
    assert probe_ds_read_tr()["tier"] == "B"
    # Route-1 pipeliner probe: structural (returns available + names opt_swp_test.py de-risk)
    rp = r["reinject_ttgir_pipeliner"]
    assert "opt_swp_test.py" in rp["cmd"], rp
    print(f"[selftest] probes: {list(r)}")
    print(f"[selftest] reinject_ttgir_pipeliner available={rp['available']} ({rp['evidence'][:60]})")
    print(f"[selftest] scaled-MFMA gfx942={r['scaled_mfma_lowprec']['available']} "
          f"gfx950-archgate-open (symbol={r950['symbol']}); ds_read_tr Tier-B")
    print("PROBE_LEVERS SELFTEST PASS")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--arch", default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(_selftest())
    print(json.dumps(run_all(a.arch), indent=2))


if __name__ == "__main__":
    main()
