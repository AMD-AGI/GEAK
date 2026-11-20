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
import os
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
    the Gluon path needs NO rebuild and NO edit to an installed file - both mechanisms wrap the
    FUNCTION and run a second pass manager over its result. Light probe = pass-symbol presence;
    full de-risk = arm the shim and diff ttg.memdesc_index in the dumped TTGIR."""
    need = ["add_schedule_loops", "add_pipeline", "add_optimize_dot_operands"]
    try:
        from triton._C.libtriton import amd  # noqa: F401
        mod = amd.passes.ttgpuir
        missing = [p for p in need if not hasattr(mod, p)]
        ok = not missing
        return {"available": ok,
                "evidence": ("pipeliner passes present in libtriton.so (" + ", ".join(need) + ")")
                            if ok else f"missing passes: {missing}",
                "cmd": "gluon_swp.py --selftest, then arm it and diff the dumped TTGIR: expect "
                       "ttg.memdesc_index to appear and local_alloc/store/load counts to rise",
                "note": "parity-recovery route: run these over the module gluon_to_ttgir returns "
                        "(upstream add_stages_inspection_hook, or the pack's in-process shim). No "
                        "rebuild, no installed-file edit. Ceiling is plain parity, so it repays a "
                        "lost_pipeline debt rather than opening a climb."}
    except Exception as e:  # noqa: BLE001
        return {"available": None,
                "evidence": f"cannot import triton AMD passes: {type(e).__name__}: {e}"[:160],
                "cmd": "gluon_swp.py --selftest (needs triton with the AMD backend to say more)",
                "note": "re-injection needs the AMD pipeliner passes in libtriton.so (they ship with "
                        "make_ttgir); probe on the TARGET box"}


def probe_gemm_compiler_stack(arch=None):
    """intra_wave GEMM compiler stack (gemm_compiler_stack card): which rungs are reachable on
    THIS build? Upstream 3.8.0 reaches the AGPR rung per compile (llvm_fn_attrs carries the
    LLVM function attributes) and ships a `coexec` scheduler strategy for the matrix/VALU
    region class; it ships NO post-assembly stage and no toggleable LLIR scheduler.
    Probe-per-build, NEVER bake: presence alone does not sanction use, and absence of a rung
    is a scoped toolchain ceiling for that rung, not for the whole card."""
    rungs = {}
    # scheduling rung: the stock co-execution strategy, and the open fn-attr pass-through
    rungs["coexec_strategy"] = probe_llvm_knob("amdgpu-sched-strategy")["available"]
    rungs["llvm_fn_attrs"] = probe_version_knobs(arch).get("knobs", {}) \
        .get("llvm_fn_attrs", {}).get("state") == "live"
    # agpr rung: the LLVM AGPR-form attributes, reachable through llvm_fn_attrs on 3.8.0
    rungs["agpr_hint"] = probe_llvm_knob("amdgpu-mfma-vgpr-form")["available"]
    # post-assembly rung: nothing upstream implements this stage
    rungs["post_asm_peephole"] = False
    have = [k for k, v in rungs.items() if v]
    return {"available": (True if rungs["llvm_fn_attrs"] else None),
            "rungs": rungs,
            "evidence": f"present rungs: {have or 'none conclusive'}; agpr is an llvm_fn_attrs "
                        "pass-through (3.8.0); no upstream post-assembly stage exists",
            "cmd": "set TRITON_HIP_USE_COEXEC_SCHEDULER and/or compile with "
                   "llvm_fn_attrs=\"amdgpu-agpr-alloc=256,amdgpu-mfma-vgpr-form=0\", then "
                   "dump_ir.sh + asm_loop_audit to confirm each landed",
            "policy": "DEFAULT-OFF; probe-per-build; the AGPR rung is GEMM-only (pure MFMA->MFMA); "
                      "a missing rung is a scoped ceiling -> fall back to non-scheduled latency hiding"}


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


def _installed_triton():
    """(version, package dir) of the Triton this box will actually compile with, or (None, None)."""
    try:
        import triton
        return getattr(triton, "__version__", None), os.path.dirname(os.path.abspath(triton.__file__))
    except Exception:  # noqa: BLE001
        return None, None


def _read_tree(root, suffixes=(".py", ".cpp", ".cc", ".h", ".td")):
    """Concatenate a source tree. Returns None when the tree is absent, which is a different
    answer from 'present and says nothing' and must not collapse into it."""
    if not root or not os.path.isdir(root):
        return None
    chunks = []
    # followlinks=True: an installed Triton tree is routinely a symlink farm, and without this
    # the walk silently returns an empty tree, which reads as "present and says nothing".
    for base, _dirs, names in os.walk(root, followlinks=True):
        for name in names:
            if name.endswith(suffixes):
                try:
                    with open(os.path.join(base, name), encoding="utf-8", errors="ignore") as fh:
                        chunks.append(fh.read())
                except OSError:
                    pass
    return "\n".join(chunks) if chunks else None


# Four knobs that are version-DISJOINT on the AMD backend, and each one fails SILENTLY outside its
# range: the option is accepted, it hashes into the compile key, and it changes no IR. That is why
# probing beats reading a table -- a sweep run on the wrong build comes back FLAT, and a flat result
# reads as "this kernel does not want the lever" when it actually means "this build has no lever".
#
# `reader` is the symbol that proves something CONSUMES the knob. Declared-without-reader is the
# state worth naming separately: it is the one that looks supported from the caller's side.
_VERSION_KNOBS = {
    "llvm_fn_attrs": {
        "what": "per-compile LLVM function attributes; reaches amdgpu-sched-strategy, and is an "
                "open name=value passthrough with no allow-list",
        "where": "backends/amd",
        "reader": "_parse_llvm_fn_attrs",
        "flat_sweep_means": "this build has no such option (it is rejected as an unrecognized "
                            "compile option, not tuned to neutral) -- NOT 'this kernel is not "
                            "ILP-starved'",
    },
    "schedule_hint": {
        "what": "curated AMD scheduler presets (iglp / sched.barrier / sink-to-avoid-spills)",
        "where": "backends/amd",
        "reader": "InstructionSchedHint",
        "flat_sweep_means": "the field is a dead declaration on this build -- accepted, no readers "
                            "left, no IR change. Read the version before blaming the kernel",
    },
    "loop_unroll_factor": {
        "what": "tl.range(..., loop_unroll_factor=N) on a Gluon body",
        "where": None,          # frontend pass list, not an AMD backend option
        "reader": "add_loop_unroll",
        "gluon_pipeline": True,
        "flat_sweep_means": "the pass is not in this build's Gluon pass list, so the attribute is "
                            "set on the loop and nothing lowers it",
    },
    "num_stages": {
        "what": "tl.range(..., num_stages=N) on a Gluon body -- the compiler auto-pipeline",
        "where": None,
        # NOT `add_loop_scheduling`: that symbol has ZERO occurrences on 3.8.0. The real
        # consumers are amd.passes.ttgpuir.add_schedule_loops / add_pipeline, and neither is in
        # the Gluon pass list -- which is exactly what makes num_stages inert on that path.
        "reader": "add_schedule_loops",
        "gluon_pipeline": True,
        "flat_sweep_means": "inert on the Gluon path: the pipeliner is not injected, so the value "
                            "is carried and never acted on. Author the staging or re-inject",
    },
}


def probe_version_knobs(arch=None):
    """Which version-sensitive compile knobs are LIVE on the installed build.

    The four in `_VERSION_KNOBS` are version-disjoint and all four fail silently outside their
    range, so the only honest source is the build in front of you. Three states are distinguished
    because they need different responses:

      live               -- declared AND something reads it. Sweep it.
      dead-declaration   -- declared, no reader. Accepted, hashes into the compile key, no IR
                            change. A flat sweep here is a property of the build.
      absent             -- not declared. An unrecognized compile option, which raises rather
                            than measuring flat.

    Offline and GPU-free: it reads the installed package's own source. `unknown` when Triton is not
    importable here, which is a fact about the probe host and not about the knob."""
    version, pkg = _installed_triton()
    if not pkg:
        return {"available": None, "triton_version": None, "knobs": {},
                "evidence": "triton is not importable on this host -- probe on the TARGET box; a "
                            "capability table is not a substitute for the build",
                "cmd": "python3 probe_levers.py --all",
                "policy": "probe-per-build; every one of these four fails silently out of range"}
    amd = _read_tree(os.path.join(pkg, "backends", "amd"))
    gluon_src = _read_tree(os.path.join(pkg, "experimental", "gluon"), suffixes=(".py",))
    # The Gluon lowering registers its own pass list. Search the whole package for it rather than
    # guessing the module path, which has moved between versions.
    pkg_py = _read_tree(pkg, suffixes=(".py",))
    knobs, decided = {}, True
    for name, spec in _VERSION_KNOBS.items():
        if spec.get("gluon_pipeline"):
            haystack, scope = pkg_py, "installed triton (.py) -- NOT a source tag"
            declared = None if haystack is None else (name in haystack)
        else:
            haystack, scope = amd, "backends/amd of the INSTALLED package"
            declared = None if haystack is None else (name in haystack)
        readers = None if haystack is None else haystack.count(spec["reader"])
        if declared is None:
            state = "unknown"
            decided = False
        elif not declared:
            state = "absent"
        elif readers:
            state = "live"
        else:
            state = "dead-declaration"
        knobs[name] = {"state": state, "declared": declared, "reader": spec["reader"],
                       "reader_hits": readers, "scope": scope, "what": spec["what"],
                       "flat_sweep_means": spec["flat_sweep_means"]}
    live = sorted(n for n, k in knobs.items() if k["state"] == "live")
    inert = sorted(n for n, k in knobs.items() if k["state"] in ("dead-declaration", "absent"))
    return {
        "available": True if decided else None,
        "triton_version": version,
        "knobs": knobs,
        "live": live,
        "inert": inert,
        "evidence": f"triton {version}: live={live or 'none'} inert={inert or 'none'} "
                    f"(gluon surface {'seen' if gluon_src else 'not found'})",
        "cmd": "python3 probe_levers.py --all",
        "policy": "sweep only a LIVE knob. A flat result from an inert one is a fact about the "
                  "build, and reading it as a fact about the kernel spends the next round wrong",
    }


def probe_llir_declarative_schedule(arch=None):
    """llir_declarative_schedule card: can this host LOAD an out-of-tree LLVM pass plugin?

    Four gates, three of which fail QUIETLY (llir-codesign.md ## The plugin tier). This probes
    the host-side ones structurally, with no GPU and no plugin binary. It CANNOT tell you the
    pass bites -- that is the IR's job -- and it deliberately reports `available: None` unless
    every gate it can see is green, because a partially-satisfied host is the state that
    produces a silent regression rather than an error."""
    gates, notes = {}, []

    # Gate 2 (loud): does the host export compiler symbols? A -fvisibility=hidden build has
    # none for a plugin to bind against. Read the built library's dynamic symbol table.
    libtriton = None
    try:
        import triton._C.libtriton as _lt
        libtriton = getattr(_lt, "__file__", None)
    except Exception as e:  # noqa: BLE001
        notes.append(f"cannot import libtriton: {type(e).__name__}")
    if libtriton:
        out = _run(["bash", "-c", f"nm -D --defined-only {libtriton!r} 2>/dev/null | grep -c ' T .*llvm' || true"])
        n = next((int(t) for t in out.split() if t.isdigit()), 0)
        gates["exports_llvm_symbols"] = (n > 0) if out.strip() else None
        notes.append(f"exported llvm text symbols in libtriton: {n}")
    else:
        gates["exports_llvm_symbols"] = None

    # Gate 1 (quiet, and the one most often mis-read): keeping the TargetMachine for plugins is
    # a SOURCE property of the host build. Setting the env var does not create the support, so
    # probe for the support, never for the variable.
    src = _run(["bash", "-c",
                "python3 -c \"import triton,inspect,pathlib;"
                "p=pathlib.Path(inspect.getfile(triton)).parent;"
                "print('\\n'.join(str(f) for f in p.rglob('compiler*.py')))\" 2>/dev/null"])
    # Upstream builds the TargetMachine only when NO plugin is set -- `if (!arch.empty() &&
    # pluginFile.empty())` in python/src/llvm.cc -- so on a stock host this is FALSE by design,
    # not by defect. A host can be patched to keep it; that is a source property of the build and
    # no environment variable supplies it. We look for any such patch and default to "stock".
    keeps = None
    for f in [l for l in src.splitlines() if l.strip().endswith(".py")]:
        hit = _run(["bash", "-c", f"grep -l 'keep_target_machine\\|KeepTargetMachine' {f!r} "
                                  "2>/dev/null || true"])
        if hit.strip():
            keeps = True
            break
    if keeps is not True:
        keeps = False
    gates["host_keeps_target_machine"] = keeps
    if keeps is not True:
        notes.append("host does NOT keep the TargetMachine for plugins -- this is UPSTREAM'S "
                     "DESIGNED behaviour (the TargetMachine is built only when no plugin is set), "
                     "so plugin-on runs an UNTARGETED optimizer and plugin-on vs plugin-off is "
                     "not a clean A/B. Separate the two effects before attributing anything")

    # Gate 3: ABI lock. Report the host's LLVM revision so it can be compared against the
    # plugin binary's build target; we cannot read the .so's target without being given one.
    llvm_rev = _run(["bash", "-c",
                     "python3 -c \"import triton,inspect,pathlib,json;"
                     "p=pathlib.Path(inspect.getfile(triton)).parent;"
                     "f=next(iter(p.rglob('llvm-info.json')),None);"
                     "print(json.load(open(f)).get('rev','') if f else '')\" 2>/dev/null"]).strip()
    gates["host_llvm_revision"] = llvm_rev or None
    notes.append("gate 3 (ABI): plugin .so must be built against this exact revision; "
                 "a mismatch SEGFAULTS rather than degrading")
    notes.append("gate 4 (cost model): confirm the loop's mfma shape is priced by your pass, "
                 "else the region is dropped silently and reads as a hardware ceiling")

    hard = [gates.get("exports_llvm_symbols"), gates.get("host_keeps_target_machine")]
    available = True if all(g is True for g in hard) else (False if False in hard else None)
    return {"available": available, "gates": gates,
            "evidence": "; ".join(notes)[:400],
            "cmd": "LLVM_PASS_PLUGIN_PATH=<abs>/<your_pass>.so <run>  # then confirm in the IR. "
                   "NOTE upstream drops the TargetMachine when a plugin is set, by design, so "
                   "plugin-on and plugin-off are not the same optimization environment",
            "policy": "DEFAULT-OFF; four gates, three silent; a green probe means the host CAN "
                      "load a plugin, NOT that the pass changed anything -- verify in the IR"}


PROBES = {
    "reinject_ttgir_pipeliner": probe_reinject_pipeliner,
    "attn_intra_loop_schedule": lambda arch: probe_llvm_knob("amdgpu-sched-strategy"),
    "scaled_mfma_lowprec": probe_fp8_scaled_mfma,
    "ds_read_tr_transpose": lambda arch: probe_ds_read_tr(),
    "gemm_compiler_stack": probe_gemm_compiler_stack,
    "warp_pipeline_schedule": probe_warp_pipeline,
    "llir_declarative_schedule": probe_llir_declarative_schedule,
    "version_disjoint_knobs": probe_version_knobs,
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
    # pipeliner-reinjection probe: structural, and it must name a script the pack actually SHIPS.
    # It used to name `opt_swp_test.py`, which exists nowhere in the pack -- an unrunnable de-risk
    # step reads as a step someone skipped rather than as a missing script.
    rp = r["reinject_ttgir_pipeliner"]
    assert "gluon_swp.py" in rp["cmd"], rp
    # plugin-tier probe: reports its gates and never claims True on a partially-green host
    pl = r["llir_declarative_schedule"]
    assert "gates" in pl and pl["available"] in (True, False, None), pl
    assert "host_keeps_target_machine" in pl["gates"], pl
    if pl["available"] is True:
        assert all(pl["gates"].get(g) is True
                   for g in ("exports_llvm_symbols", "host_keeps_target_machine")), pl
    # version-disjoint knobs: all four are always reported, and a host without triton says
    # `unknown` rather than inventing `absent` -- which would read as a settled negative.
    vk = r["version_disjoint_knobs"]
    assert set(vk["knobs"]) == set(_VERSION_KNOBS) or vk["available"] is None, vk
    for name, k in vk["knobs"].items():
        assert k["state"] in ("live", "dead-declaration", "absent", "unknown"), (name, k)
        assert k["flat_sweep_means"], name
    if vk["available"] is None:
        assert vk["triton_version"] is None or "unknown" in {
            k["state"] for k in vk["knobs"].values()}, vk
    print(f"[selftest] probes: {list(r)}")
    print(f"[selftest] reinject_ttgir_pipeliner available={rp['available']} ({rp['evidence'][:60]})")
    print(f"[selftest] scaled-MFMA gfx942={r['scaled_mfma_lowprec']['available']} "
          f"gfx950-archgate-open (symbol={r950['symbol']}); ds_read_tr Tier-B")
    print(f"[selftest] version knobs (triton={vk['triton_version']}): "
          + ", ".join(f"{n}={k['state']}" for n, k in vk["knobs"].items()) or "  none resolved")
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
