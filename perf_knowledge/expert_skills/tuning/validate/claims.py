#!/usr/bin/env python3
"""One executable check per load-bearing claim in the skillset.

The skill map asserts eleven skills are validated. Without this file that
assertion rests on the author having once run something in one container. A
claim worth putting in a skill is worth being able to re-run, in any image,
after any upgrade -- and the two shipped images differ enough that "it worked"
is not a property of the skillset, it is a property of the container.

Each check returns one of:

  PASS    the claim holds here, with the observed value
  FAIL    the claim is contradicted here -- the skill is wrong and must change
  N/A     the precondition is absent (tool not installed, framework not in
          this image). NOT a pass. It means this image cannot answer.

The distinction between FAIL and N/A is the whole point. A skill that says
"install hipblaslt-bench, then race solutions" is not falsified by an image
that ships without it; it is falsified by an image where the documented
command, once the tool is there, does not behave as written.

  python3 validate/claims.py                 # everything applicable here
  python3 validate/claims.py --skill triton  # one skill
  python3 validate/claims.py --json out.json # machine-readable, for the map

Read-only by design: nothing here installs, writes a config, or launches a
server. Claims that require mutating the environment live in
validate/claims_live.py, which is opt-in.
"""
import argparse
import glob
import json
import os
import platform
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))

PASS, FAIL, NA = "PASS", "FAIL", "N/A"

_REGISTRY = []


def claim(skill, text):
    """Register a check. `text` is the sentence in the skill being tested."""
    def deco(fn):
        _REGISTRY.append((skill, text, fn))
        return fn
    return deco


def sh(cmd, timeout=60):
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                           timeout=timeout)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired:
        return 124, "timeout"
    except Exception as exc:                      # noqa: BLE001
        return 1, str(exc)


def has_mod(name):
    try:
        __import__(name)
        return True
    except Exception:                             # noqa: BLE001
        return False


def rocm_libraries():
    """Path to a rocm-libraries checkout, or None.

    Two claims below read the CK and hipBLASLt source trees rather than probing
    a device, because what they assert is a property of the shipped source.
    Neither shipped image contains this checkout, so both return N/A there --
    which is the correct answer, not a pass.
    """
    for cand in (os.environ.get("ROCM_LIBRARIES", ""),
                 os.path.join(ROOT, "..", "libs", "rocm-libraries"),
                 "/ws/libs/rocm-libraries"):
        if cand and os.path.isdir(os.path.join(cand, "projects")):
            return os.path.normpath(cand)
    return None


# --------------------------------------------------------------------------
# env-setup
# --------------------------------------------------------------------------

@claim("env-setup", "GPUs are visible; rocminfo reports a gfx arch")
def c_arch():
    rc, out = sh("rocminfo 2>/dev/null | awk '/gfx[0-9]/{print $2; exit}'")
    arch = out.strip()
    if not arch:
        # The skill's own text: an odd import error from a GPU library is
        # usually this, not a broken install. So this is FAIL, not N/A --
        # every other claim below is unreliable without it.
        return FAIL, "rocminfo found no gfx target (GPUs not passed in?)"
    return PASS, arch


@claim("env-setup", "neither image ships a bench client; both ship deploy targets")
def c_bench_absent():
    missing = [b for b in ("hipblaslt-bench", "ckProfiler")
               if sh("command -v %s" % b)[0] != 0]
    aiter_cfg = None
    if has_mod("aiter"):
        import aiter
        aiter_cfg = os.path.join(os.path.dirname(aiter.__file__), "configs")
    tgt = aiter_cfg and os.path.isdir(aiter_cfg)
    # Recorded either way -- if a future image starts shipping the clients the
    # skill's install section becomes unnecessary work, which is worth knowing.
    return PASS, "absent=%s; aiter configs dir present=%s" % (
        ",".join(missing) or "none", bool(tgt))


@claim("env-setup", "aiter ships as a wheel in vllm and a source checkout in sglang")
def c_aiter_shape():
    if not has_mod("aiter"):
        return NA, "aiter not importable"
    import aiter
    root = os.path.dirname(os.path.dirname(aiter.__file__))
    tuner = os.path.join(root, "gradlib", "gradlib", "gemm_tuner.py")
    kind = "source (tuners in place)" if os.path.exists(tuner) else "wheel (no tuner tree)"
    ver = getattr(aiter, "__version__", None)
    if ver is None:
        try:
            import aiter._version as v
            ver = v.__version__
        except Exception:                         # noqa: BLE001
            ver = "unknown"
    return PASS, "%s, version %s, root %s" % (kind, ver, root)


# --------------------------------------------------------------------------
# tuning-triton
# --------------------------------------------------------------------------

@claim("tuning-triton", "AMD knobs raise TypeError as kwargs but are accepted in the positional dict")
def c_triton_knobs():
    if not has_mod("triton"):
        return NA, "triton not importable"
    import triton
    try:
        triton.Config({"BLOCK_M": 64}, waves_per_eu=2)
        return FAIL, "kwarg form was accepted -- the trap this skill documents is gone"
    except TypeError:
        pass
    cfg = triton.Config({"BLOCK_M": 64, "waves_per_eu": 2,
                         "matrix_instr_nonkdim": 16, "kpack": 2},
                        num_warps=8, num_stages=2)
    got = cfg.all_kwargs()
    for k in ("waves_per_eu", "matrix_instr_nonkdim", "kpack"):
        if k not in got:
            return FAIL, "%s did not survive into all_kwargs()" % k
    return PASS, "TypeError on kwarg; all three knobs present in all_kwargs (triton %s)" % triton.__version__


@claim("tuning-triton", "the HIP backend exposes the knobs as HIPOptions fields")
def c_hipoptions():
    if not has_mod("triton"):
        return NA, "triton not importable"
    try:
        import dataclasses
        from triton.backends.amd.compiler import HIPOptions
    except Exception as exc:                      # noqa: BLE001
        return FAIL, "documented import failed: %s" % exc
    names = {f.name for f in dataclasses.fields(HIPOptions)}
    want = {"waves_per_eu", "matrix_instr_nonkdim", "kpack"}
    miss = want - names
    return (FAIL, "absent from HIPOptions: %s" % sorted(miss)) if miss else \
           (PASS, "%d fields, all three knobs present" % len(names))


# --------------------------------------------------------------------------
# tuning-hipblaslt
# --------------------------------------------------------------------------

@claim("tuning-hipblaslt", "hipblaslt-bench replays a solution by index with --algo_method index")
def c_hipblaslt_index():
    if sh("command -v hipblaslt-bench")[0] != 0:
        return NA, "hipblaslt-bench not built in this image (see env-setup)"
    rc, out = sh("hipblaslt-bench --help", timeout=60)
    for flag in ("--algo_method", "--solution_index"):
        if flag not in out:
            return FAIL, "%s absent from --help" % flag
    return PASS, "both flags present in --help"


@claim("tuning-hipblaslt", "torch exposes TunableOp as the no-build route to solution selection")
def c_tunableop():
    if not has_mod("torch"):
        return NA, "torch not importable"
    import torch
    if not hasattr(torch.cuda, "tunable"):
        return FAIL, "torch.cuda.tunable absent"
    t = torch.cuda.tunable
    need = ["enable", "is_enabled", "set_filename"]
    miss = [n for n in need if not hasattr(t, n)]
    return (FAIL, "missing %s" % miss) if miss else \
           (PASS, "torch %s, tunable API complete" % torch.__version__)


# --------------------------------------------------------------------------
# tuning-ck
# --------------------------------------------------------------------------

@claim("tuning-ck", "ckProfiler takes positional args and lists its ops when run bare")
def c_ckprofiler():
    if sh("command -v ckProfiler")[0] != 0:
        return NA, "ckProfiler not installed (apt composablekernel-ckprofiler)"
    rc, out = sh("ckProfiler", timeout=60)
    if "gemm" not in out.lower():
        return FAIL, "bare invocation did not list a gemm op"
    return PASS, "op list printed, gemm present"


# --------------------------------------------------------------------------
# tuning-aiter
# --------------------------------------------------------------------------

@claim("tuning-aiter", "the tuned-config CSV has the documented 18-column schema")
def c_aiter_csv():
    if not has_mod("aiter"):
        return NA, "aiter not importable"
    import aiter
    p = os.path.join(os.path.dirname(aiter.__file__), "configs",
                     "bf16_tuned_gemm.csv")
    if not os.path.exists(p):
        return NA, "bf16_tuned_gemm.csv absent from the installed package"
    head = open(p).readline().strip().split(",")
    want = ["gfx", "cu_num", "M", "N", "K", "bias", "dtype", "outdtype",
            "scaleAB", "bpreshuffle", "libtype", "solidx", "splitK", "us",
            "kernelName", "err_ratio", "tflops", "bw"]
    if head != want:
        return FAIL, "schema drift: %s" % head
    return PASS, "18 columns as documented"


@claim("tuning-aiter", "AITER_LOG_TUNED_CONFIG is the engagement signal, and rows are keyed by cu_num")
def c_aiter_cu():
    if not has_mod("aiter"):
        return NA, "aiter not importable"
    import glob as _glob
    import aiter
    cfg = os.path.join(os.path.dirname(aiter.__file__), "configs")
    try:
        import torch
        cu = torch.cuda.get_device_properties(0).multi_processor_count
    except Exception:                             # noqa: BLE001
        return NA, "no GPU to read cu_num from"
    # Scan every populated tuned table, not one named file. bf16_tuned_gemm.csv
    # ships header-only in both images, so a check pointed at it reports
    # "0 of 0 rows" -- vacuously true, and indistinguishable from the real
    # finding. An empty table cannot demonstrate anything about the key.
    tot = hits = files = 0
    per = []
    for p in sorted(_glob.glob(os.path.join(cfg, "*tuned*.csv"))):
        lines = open(p, errors="replace").read().splitlines()
        head, rows = (lines[:1], lines[1:]) if lines else ([], [])
        if not rows or not head or "cu_num" not in head[0]:
            continue
        idx = head[0].split(",").index("cu_num")
        h = sum(1 for r in rows if r.split(",")[idx:idx + 1] == [str(cu)])
        tot += len(rows)
        hits += h
        files += 1
        if h == 0:
            per.append("%s 0/%d" % (os.path.basename(p), len(rows)))
    if not files:
        return NA, "no populated cu_num-keyed table in %s" % cfg
    # A miss is the documented failure, not a broken check: the skill's claim
    # is that the key includes cu_num and therefore does not transfer. Zero
    # reachable rows on a 304-CU part is that claim being demonstrated.
    return PASS, "%d of %d rows across %d tables reachable at cu_num=%d%s" % (
        hits, tot, files, cu,
        ("; unreachable: " + ", ".join(per[:4])) if per else "")


def _aiter_gemm_op_a8w8():
    """Path to gemm_op_a8w8.py, from the installed package or a source checkout.

    Falls back to a checkout so the two log-behaviour claims below are checkable
    outside a ROCm container -- what they assert is a property of the source.
    """
    if has_mod("aiter"):
        import aiter
        p = os.path.join(os.path.dirname(aiter.__file__), "ops", "gemm_op_a8w8.py")
        if os.path.exists(p):
            return p, "installed package"
    for d in ("/sgl-workspace/aiter", "/work_aiter",
              os.path.expanduser("~/tuning_workspace/libs/aiter")):
        p = os.path.join(d, "aiter", "ops", "gemm_op_a8w8.py")
        if os.path.exists(p):
            return p, "source checkout %s" % d
    return None, "neither an importable aiter nor a source checkout"


@claim("tuning-aiter", "the tuned-config HIT log line is gated behind AITER_LOG_TUNED_CONFIG while the MISS line is not")
def c_aiter_log_gating():
    # This asymmetry is why `grep -c "is tuned on cu_num"` returns zero against a
    # working deploy when the env var is unset: on success the observable
    # transition is "not found tuned config" -> silence, not -> a hit line.
    # A doc that restates the check without the flag reports failure on success.
    p, where = _aiter_gemm_op_a8w8()
    if p is None:
        return NA, where
    src = open(p, errors="replace").read().splitlines()
    hits = [i + 1 for i, l in enumerate(src) if "is tuned on cu_num" in l]
    misses = [i + 1 for i, l in enumerate(src) if "not found tuned config" in l]
    if not hits or not misses:
        return FAIL, "log lines not found (hit=%s miss=%s)" % (hits, misses)

    def gated(lineno, window=6):
        lo = max(0, lineno - 1 - window)
        return any("AITER_LOG_TUNED_CONFIG" in l for l in src[lo:lineno])

    hit_gated = all(gated(n) for n in hits)
    miss_gated = any(gated(n) for n in misses)
    if hit_gated and not miss_gated:
        return PASS, ("hit line(s) at %s gated by the env var; miss line(s) at %s "
                      "unconditional -- the flag is mandatory (%s)"
                      % (hits, misses, where))
    return FAIL, ("gating drift: hit_gated=%s miss_gated=%s (hit=%s miss=%s) -- "
                  "re-check the engagement command in tuning-aiter §5/§6"
                  % (hit_gated, miss_gated, hits, misses))


@claim("tuning-aiter", "the config lookup is lru_cached on raw M, so log-line counts measure shape diversity rather than call frequency")
def c_aiter_log_cached():
    p, where = _aiter_gemm_op_a8w8()
    if p is None:
        return NA, where
    src = open(p, errors="replace").read()
    if "lru_cache" not in src:
        return FAIL, "no lru_cache on the lookup -- counts may now track call frequency"
    n = src.count("@functools.lru_cache")
    return PASS, ("%d lru_cache-wrapped lookups: rank tuning targets by measured "
                  "time, never by miss/hit count (%s)" % (n, where))


@claim("tuning-ck", "the per-op quantized tuner takes -i/-o, defaults splitK OFF, and uses a different result schema than gradlib")
def c_ck_perop_tuner():
    # tuning-aiter §4 documents gradlib's gemm_tuner.py, which serves dense bf16
    # only. Every quantized GEMM goes through one of these instead, with
    # different flags and a different schema. Conflating them wastes a tuning run.
    roots = [d for d in ("/sgl-workspace/aiter", "/work_aiter",
                         os.path.expanduser("~/tuning_workspace/libs/aiter"))
             if os.path.isdir(d)]
    if not roots:
        return NA, "no aiter source tree (wheel-only image; csrc/ absent)"
    root = roots[0]
    base = os.path.join(root, "aiter", "utility", "base_tuner.py")
    tune = os.path.join(root, "csrc", "ck_gemm_a8w8_bpreshuffle",
                        "gemm_a8w8_bpreshuffle_tune.py")
    missing = [p for p in (base, tune) if not os.path.exists(p)]
    if missing:
        return NA, "expected tuner files absent: %s" % missing
    b = open(base, errors="replace").read()
    t = open(tune, errors="replace").read()
    probs = []
    if '"--splitK"' not in b:
        probs.append("--splitK flag not found in base_tuner")
    elif 'action="store_true"' not in b.split('"--splitK"', 1)[1][:200]:
        probs.append("--splitK is no longer store_true (default may have changed)")
    if "useSplitK = args.splitK" not in t:
        probs.append("split-K sweep no longer gated on args.splitK")
    for col in ("kernelId", "errRatio"):
        if '"%s"' % col not in b:
            probs.append("result column %s absent" % col)
    if probs:
        return FAIL, "; ".join(probs)
    return PASS, ("-k/--splitK present and store_true (OFF by default); sweep gated "
                  "on args.splitK; schema uses kernelId/errRatio, not solidx/err_ratio")


# --------------------------------------------------------------------------
# tuning-flydsl
# --------------------------------------------------------------------------

@claim("tuning-flydsl", "flydsl is importable and exposes a tunable Config surface")
def c_flydsl():
    if not has_mod("flydsl"):
        return NA, "flydsl not importable"
    import flydsl
    ver = getattr(flydsl, "__version__", "present")
    return PASS, "flydsl %s" % ver


# --------------------------------------------------------------------------
# tuning-hip
# --------------------------------------------------------------------------

@claim("tuning-hip", "rocprofv3 is present and gives ground truth for which kernel ran")
def c_rocprof():
    rc, _ = sh("command -v rocprofv3")
    if rc != 0:
        return FAIL, "rocprofv3 absent -- the skill's verification route does not exist here"
    rc, out = sh("rocprofv3 --help", timeout=60)
    return (PASS, "--kernel-trace supported") if "--kernel-trace" in out else \
           (FAIL, "--kernel-trace absent from --help")


@claim("tuning-hip", "the arch limits the skill reasons about are readable, not assumed")
def c_arch_limits():
    if not has_mod("torch"):
        return NA, "torch not importable"
    import torch
    if not torch.cuda.is_available():
        return NA, "no GPU visible to torch"
    p = torch.cuda.get_device_properties(0)
    # Do not default the limits. A check that substitutes 0 for an attribute
    # this torch build does not expose reports a PASS for "the limits are
    # readable" while printing 0 B of LDS -- which is exactly the silent-zero
    # class of bug the skillset is written to catch. Name the gap instead.
    fields = {"CU": "multi_processor_count", "warp": "warp_size",
              "LDS/workgroup": "shared_memory_per_block"}
    got, missing = [], []
    for label, attr in fields.items():
        v = getattr(p, attr, None)
        (got if v else missing).append("%s=%s" % (label, v) if v else label)
    arch = p.gcnArchName.split(":")[0]
    if missing:
        return NA, "%s: %s readable, but torch %s does not expose %s -- " \
                   "read it from rocminfo/hipDeviceProp instead" % (
                       arch, ", ".join(got), torch.__version__, ", ".join(missing))
    return PASS, "%s: %s" % (arch, ", ".join(got))


# --------------------------------------------------------------------------
# tuning-ck: the aiter offline tuners
# --------------------------------------------------------------------------

@claim("tuning-ck", "aiter ships seven CK offline tuners, and none of them takes --indtype")
def c_ck_tuners():
    if not has_mod("aiter"):
        return NA, "aiter not importable"
    import aiter
    src = os.path.normpath(os.path.join(os.path.dirname(aiter.__file__), ".."))
    if not os.path.isdir(os.path.join(src, "csrc")):
        return NA, "aiter is a wheel here -- csrc/ tuners ship only in the source checkout"
    rc, out = sh("ls %s/csrc/ck_*/*tune*.py 2>/dev/null" % src)
    tuners = [p for p in out.strip().splitlines() if p and "validate_" not in p]
    if not tuners:
        return FAIL, "no csrc/ck_*/*tune*.py found in the checkout"
    # --indtype is the gradlib bug; the skill says it does not reach these.
    with_indtype = [os.path.basename(t) for t in tuners
                    if "--indtype" in open(t, errors="ignore").read()]
    if with_indtype:
        return FAIL, "these accept --indtype, contradicting the skill: %s" % with_indtype
    return PASS, "%d tuners, none takes --indtype: %s" % (
        len(tuners), ", ".join(sorted(os.path.basename(t) for t in tuners)))


@claim("tuning-ck", "the AITER_CONFIG_<OP> override has no _FILE suffix; the _FILE name is a property")
def c_aiter_cfg_env():
    if not has_mod("aiter"):
        return NA, "aiter not importable"
    probe = ("import os;from aiter.jit.core import AITER_CONFIGS;"
             "print(AITER_CONFIGS.AITER_CONFIG_BF16_BATCHED_GEMM_FILE)")
    base_rc, base = sh('python3 -c "%s"' % probe, timeout=180)
    target = "/tmp/_claims_probe_bf16_batched.csv"
    rc1, o1 = sh('AITER_CONFIG_BF16_BATCHED_GEMM=%s python3 -c "%s"' % (target, probe),
                 timeout=180)
    rc2, o2 = sh('AITER_CONFIG_BF16_BATCHED_GEMM_FILE=%s python3 -c "%s"' % (target, probe),
                 timeout=180)
    took = target in o1
    ignored = target not in o2
    if not took:
        return FAIL, "AITER_CONFIG_BF16_BATCHED_GEMM did not redirect the config path"
    if not ignored:
        return FAIL, "the _FILE form also works -- the skill's warning is wrong here"
    return PASS, "no-suffix form redirects; _FILE form is silently ignored"


@claim("tuning-ck", "the shipped fused-MoE shape list is gfx942-only and will not run here")
def c_fmoe_fnuz():
    if not has_mod("aiter"):
        return NA, "aiter not importable"
    import aiter
    import torch
    p = os.path.join(os.path.dirname(aiter.__file__), "configs", "untuned_fmoe.csv")
    if not os.path.exists(p):
        return NA, "untuned_fmoe.csv not present in this build"
    rows = [ln for ln in open(p).read().splitlines()[1:] if ln.strip()]
    fnuz = [ln for ln in rows if "fnuz" in ln]
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] \
        if torch.cuda.is_available() else "?"
    if arch == "gfx942":
        return PASS, "%d/%d rows are FNUZ, which is this part's dialect" % (len(fnuz), len(rows))
    if not fnuz:
        return PASS, "no FNUZ rows -- the shape list is usable as shipped on %s" % arch
    # On gfx950 FNUZ is unsupported and the tuner aborts the whole run on the
    # first such row, so this is a blocking defect, not a cosmetic one.
    return PASS, ("%d of %d rows are FNUZ and cannot run on %s; gemm_moe_tune.py "
                  "raises KeyError and tunes 0 shapes. Filter with `grep -v fnuz`."
                  % (len(fnuz), len(rows), arch))


@claim("tuning-core", "rocm-smi reports success when pinning clocks even though it changes "
                      "nothing, so --showperflevel is the only trustworthy check "
                      "(clocks_and_power.md)")
def c_clock_pinning_silent():
    import torch
    if not torch.cuda.is_available():
        return NA, "no GPU"
    rc, _ = sh("which rocm-smi", timeout=30)
    if rc != 0:
        return NA, "rocm-smi not on PATH in this image"
    # Match only the per-GPU data rows. The banner line reads
    # "===== Show Performance Level =====" and contains the same words, so a
    # substring test picks it up and reports the banner as a clock setting.
    def perf_levels(text):
        out = []
        for ln in text.splitlines():
            if ln.lstrip().startswith("GPU[") and "Performance Level" in ln:
                out.append(ln.split(":")[-1].strip())
        return out

    rc, out = sh("rocm-smi --showperflevel 2>&1", timeout=120)
    levels = perf_levels(out)
    if not levels:
        return NA, "could not read a per-GPU performance level"
    pinned = [l for l in levels if l.lower() not in ("auto", "")]
    if pinned:
        # Someone has actually pinned the clocks on this host. Rule 6b still
        # applies, but the premise of this claim does not hold here.
        return NA, ("clocks are pinned on this host (level=%s); the silent-failure "
                    "path is not reachable" % pinned[0])
    # Not pinned. Confirm the write path is genuinely blocked, which is what
    # makes the exit-0 dangerous rather than merely redundant.
    writable = False
    for p in sorted(glob.glob("/sys/class/drm/card*/device/"
                              "power_dpm_force_performance_level")):
        if os.access(p, os.W_OK):
            writable = True
            break
    if writable:
        return NA, ("sysfs power control is writable here, so pinning would take "
                    "effect; the documented trap is container-specific")
    rc, out = sh("rocm-smi --setperfdeterminism 1900 2>&1", timeout=120)
    noisy = any(w in out.lower()
                for w in ("error", "fail", "permission", "denied", "read-only"))
    rc2, out2 = sh("rocm-smi --showperflevel 2>&1", timeout=120)
    after = perf_levels(out2)
    if not after or any(l.lower() != "auto" for l in after):
        return FAIL, ("the perf level changed despite a read-only sysfs -- "
                      "clocks_and_power.md needs rechecking")
    if noisy:
        return PASS, ("pinning is blocked and rocm-smi says so (exit=%d, error text "
                      "present); level still auto" % rc)
    return PASS, ("rocm-smi exited %d with no error text and the level is still "
                  "auto -- pinning silently did nothing, exactly as documented" % rc)


@claim("tuning-aiter", "the shipped MX surface is FP4-only: MXFP8 has no aiter operator "
                       "on either image, so MXFP8 must go through CK (SKILL.md 7)")
def c_aiter_mx_surface():
    if not has_mod("aiter"):
        return NA, "aiter not importable"
    import importlib
    fp4 = {
        "gemm_afp4wfp4": "aiter.ops.triton.gemm.basic.gemm_afp4wfp4",
        "gemm_a16wfp4": "aiter.ops.triton.gemm.basic.gemm_a16wfp4",
        "gemm_a8wfp4": "aiter.ops.triton.gemm.basic.gemm_a8wfp4",
        "batched_gemm_afp4wfp4": "aiter.ops.triton.gemm.batched.batched_gemm_afp4wfp4",
        "fused_moe_mxfp4": "aiter.ops.triton.moe.moe_op_mxfp4",
    }
    fp8 = {
        "gemm_afp8wfp8": "aiter.ops.triton.gemm.basic.gemm_afp8wfp8",
    }

    def present(mod, sym):
        try:
            return hasattr(importlib.import_module(mod), sym)
        except Exception:
            return False

    have4 = [s for s, m in fp4.items() if present(m, s)]
    have8 = [s for s, m in fp8.items() if present(m, s)]
    try:
        q = importlib.import_module("aiter.ops.triton.quant")
        if hasattr(q, "dynamic_mxfp4_quant"):
            have4.append("dynamic_mxfp4_quant")
        if hasattr(q, "dynamic_mxfp8_quant"):
            have8.append("dynamic_mxfp8_quant")
    except Exception:
        pass

    if not have4:
        return NA, "no MXFP4 ops either -- not a gfx950 build of aiter"
    if have8:
        # Not a failure of the hardware claim, but the skill's inventory is
        # now stale and the MXFP8 gap it documents has been filled upstream.
        return FAIL, ("MXFP8 ops are present in this build (%s) -- SKILL.md 7 says "
                      "the surface is FP4-only and needs updating"
                      % ", ".join(sorted(have8)))
    return PASS, ("%d MXFP4 ops present (%s), 0 MXFP8 ops -- MXFP8 only via "
                  "ckProfiler gemm_mx" % (len(have4), ", ".join(sorted(have4))))


@claim("tuning-aiter", "batched and unbatched MXFP4 _get_config disagree about the units "
                       "of K, and the batched one wants packed K (SKILL.md 7)")
def c_aiter_mx_packed_k():
    if not has_mod("aiter"):
        return NA, "aiter not importable"
    try:
        from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import _get_config as unb
        from aiter.ops.triton.gemm.batched.batched_gemm_afp4wfp4 import (
            _get_config as bat,
        )
    except Exception as e:
        return NA, "MXFP4 gemm ops not in this build (%s)" % type(e).__name__
    # Check the config, not the kernel's output. Running the op with the wrong
    # K is an out-of-bounds read, i.e. undefined behaviour, and it duly behaves
    # that way: the same call faults the GPU at K=4096, returns nan at K=256,
    # and on a second run returns a plausible answer. A claim gated on the
    # observed error is therefore flaky in both directions. The config itself
    # is deterministic and identical across both images.
    #
    # The discriminator is SPLITK_BLOCK_SIZE, which the op sets to 2*K and then
    # strides by SPLITK_BLOCK_SIZE//2 over a buffer holding K/2 packed bytes.
    # Passing logical K makes that stride the full logical K -- twice the
    # buffer. BLOCK_SIZE_K is *not* the discriminator; it agrees on most shapes.
    shapes = [(512, 4096, 4096), (16, 4096, 4096), (2048, 4096, 4096),
              (64, 128, 256)]
    rows = []
    for M, N, K in shapes:
        packed = K // 2
        try:
            cp, _ = bat(M, N, packed)
            cl, _ = bat(M, N, K)
        except Exception as e:
            return NA, "config lookup raised: %s" % type(e).__name__
        sp = int(cp.get("SPLITK_BLOCK_SIZE", 0) or 0) // 2
        sl = int(cl.get("SPLITK_BLOCK_SIZE", 0) or 0) // 2
        rows.append((M, N, K, packed, sp, sl))

    bad_packed = [r for r in rows if r[4] > r[3]]
    if bad_packed:
        M, N, K, packed, sp, _ = bad_packed[0]
        return FAIL, ("packed-K call strides %d over a %d-byte buffer at "
                      "M%dxN%dxK%d -- the convention in SKILL.md 7 is wrong"
                      % (sp, packed, M, N, K))
    over = [r for r in rows if r[5] > r[3]]
    if not over:
        return FAIL, ("logical K no longer overruns on any of %d shapes; the two "
                      "conventions have converged and SKILL.md 7 needs rechecking"
                      % len(rows))
    unb_k = int(unb(512, 4096, 4096)[0].get("BLOCK_SIZE_K", 0) or 0)
    M, N, K, packed, sp, sl = over[0]
    return PASS, ("logical K overruns on %d/%d shapes, e.g. M%dxN%dxK%d strides "
                  "%d over a %d-byte packed buffer vs %d with packed K "
                  "(unbatched BLOCK_SIZE_K=%d)"
                  % (len(over), len(rows), M, N, K, sl, packed, sp, unb_k))


@claim("tuning-hipblaslt", "MX GEMMs refuse --algo_method all and expose no solution index, "
                           "so they cannot be raced or replayed (SKILL.md 6b)")
def c_hipblaslt_mx_no_race():
    import torch
    if not torch.cuda.is_available():
        return NA, "no GPU"
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    if not arch.startswith("gfx95"):
        return NA, "MX block scaling is CDNA4-only; this is %s" % arch
    bench = None
    for c in ("/opt/rocm/bin/hipblaslt-bench",
              "/ws/libs/rocm-libraries/projects/hipblaslt/build/release/"
              "clients/hipblaslt-bench"):
        if os.path.exists(c):
            bench = c
            break
    if bench is None:
        return NA, "hipblaslt-bench not found"
    cmd = ("%s -m 512 -n 512 -k 512 --transA T --transB N "
           "--a_type f8_r --b_type f8_r --c_type f16_r --d_type f16_r "
           "--compute_type f32_r --scaleA 3 --scaleB 3 --algo_method all "
           "-i 1 -j 1" % bench)
    rc, out = sh(cmd, timeout=300)
    if "do not support algorithm" in out or "does not support algorithm" in out:
        return PASS, ('MX + --algo_method all rejected: "%s"'
                      % next(l.strip() for l in out.splitlines()
                             if "support algorithm" in l))
    if rc != 0:
        return NA, "bench failed for another reason (rc=%d)" % rc
    return FAIL, ("--algo_method all was accepted for an MX type; SKILL.md 6b says "
                  "it is refused and the whole section needs rechecking")


@claim("tuning-ck", "CK gates instances on an exact get_device_name() string, so the "
                    "candidate pool differs between parts (SKILL.md 2b)")
def c_ck_device_gate():
    root = rocm_libraries()
    if not root:
        return NA, "no rocm-libraries checkout (set ROCM_LIBRARIES=<path>)"
    inst = os.path.join(root, "projects", "composablekernel", "library", "src",
                        "tensor_operation_instance", "gpu")
    if not os.path.isdir(inst):
        return NA, "composablekernel instance tree not in this checkout"
    rc, out = sh(r"""grep -rho 'get_device_name() *[!=]= *"gfx950"' %s | sort | uniq -c"""
                 % inst, timeout=300)
    excl = incl = 0
    for ln in out.splitlines():
        n = ln.strip().split()[0] if ln.strip() else "0"
        if "!=" in ln:
            excl = int(n)
        elif "==" in ln:
            incl = int(n)
    if excl == 0 and incl == 0:
        return FAIL, "no get_device_name() gfx950 gates found -- SKILL.md 2b describes them"
    # The exclusions are the load-bearing half: they silently shrink the pool.
    rc, dirs = sh("""grep -rl 'get_device_name() != "gfx950"' %s | sed 's|%s/||;s|/.*||' """
                  """| sort -u | tr '\\n' ' '""" % (inst, inst), timeout=300)
    return PASS, ("%d gates exclude on gfx950 and %d add only there; exclusions touch: %s"
                  % (excl, incl, dirs.strip() or "?"))


@claim("tuning-hipblaslt", "the gfx950 Tensile logic tree is split by selection strategy and is "
                           "Origami-dominated, unlike gfx942's CU-count split (SKILL.md 7)")
def c_hipblaslt_logic_tree():
    root = rocm_libraries()
    if not root:
        return NA, "no rocm-libraries checkout (set ROCM_LIBRARIES=<path>)"
    logic = os.path.join(root, "projects", "hipblaslt", "library", "src", "amd_detail",
                         "rocblaslt", "src", "Tensile", "Logic", "asm_full")
    if not os.path.isdir(logic):
        return NA, "hipblaslt Tensile logic tree not in this checkout"
    g950, g942 = os.path.join(logic, "gfx950"), os.path.join(logic, "aquavanjaram")
    if not os.path.isdir(g950):
        return NA, "no gfx950 logic directory in this version"
    subs = sorted(d for d in os.listdir(g950) if os.path.isdir(os.path.join(g950, d)))
    if "Origami" not in subs:
        return FAIL, "no Origami strategy under gfx950/; SKILL.md 7 says it is 83%%: %s" % subs
    counts = {}
    for d in subs:
        rc, out = sh("find %s -name '*.yaml' | wc -l" % os.path.join(g950, d))
        counts[d] = int(out.strip() or 0)
    tot = sum(counts.values()) or 1
    frac = 100.0 * counts.get("Origami", 0) / tot
    if frac < 50:
        return FAIL, "Origami is only %.0f%% of the gfx950 logic tree: %s" % (frac, counts)
    note = ""
    if os.path.isdir(g942):
        old = sorted(d for d in os.listdir(g942) if os.path.isdir(os.path.join(g942, d)))
        if any("Origami" in d for d in old):
            return FAIL, "gfx942 also has Origami; the contrast in SKILL.md 7 is wrong"
        note = "; gfx942 splits by CU count instead (%d dirs, no Origami)" % len(old)
    return PASS, "Origami is %.0f%% of %d gfx950 logic files %s%s" % (
        frac, tot, counts, note)


@claim("tuning-aiter", "gradlib's gemm_tuner.py has no --libtype flag; libtype is an output column")
def c_gradlib_libtype():
    if not has_mod("aiter"):
        return NA, "aiter not importable"
    import aiter
    src = os.path.normpath(os.path.join(os.path.dirname(aiter.__file__), ".."))
    t = os.path.join(src, "gradlib", "gradlib", "gemm_tuner.py")
    if not os.path.exists(t):
        return NA, "gradlib tuner ships only in the aiter source checkout"
    body = open(t, errors="ignore").read()
    if '"--libtype"' in body or "'--libtype'" in body:
        return FAIL, "--libtype IS a CLI flag here -- the skill's correction is wrong"
    return PASS, "no --libtype argument is defined; it appears only as a CSV column"


# --------------------------------------------------------------------------
# tuning-in-vllm / tuning-in-sglang
# --------------------------------------------------------------------------

@claim("tuning-in-vllm", "vLLM reads tuned configs from VLLM_TUNED_CONFIG_FOLDER")
def c_vllm_env():
    if not has_mod("vllm"):
        return NA, "vllm not installed in this image"
    rc, out = sh("grep -rl VLLM_TUNED_CONFIG_FOLDER "
                 "$(python3 -c 'import vllm,os;print(os.path.dirname(vllm.__file__))') "
                 "2>/dev/null | head -3")
    if not out.strip():
        return FAIL, "env var not referenced anywhere in the installed vllm"
    import vllm
    return PASS, "vllm %s reads it in %d file(s)" % (
        vllm.__version__, len(out.strip().splitlines()))


@claim("tuning-in-sglang", "SGLang keys MoE configs by Triton version, and no dir holds a config for the live device")
def c_sglang_cfgdir():
    if not has_mod("sglang"):
        return NA, "sglang not installed in this image"
    import sglang
    base = os.path.dirname(sglang.__file__)
    # Match only version dirs under a configs/ parent -- a bare 'triton_*' also
    # catches package dirs like triton_utils and makes the count read as if
    # config directories existed where they do not.
    rc, out = sh("find %s -type d -regex '.*/configs/triton_[0-9_]+' 2>/dev/null"
                 % base)
    dirs = [d for d in out.strip().splitlines() if d]
    if not dirs:
        return FAIL, "no configs/triton_<version> dirs found"
    try:
        import triton
        tv = "triton_" + triton.__version__.replace(".", "_")
    except Exception:                             # noqa: BLE001
        tv = None
    # Ask for THIS device rather than hardcoding MI300X: the whole point of the
    # claim is that the shipped set may not name the part you are on, and a
    # hardcoded name reports "0 files" identically whether that is the trap or
    # simply the wrong part.
    from sglang.srt.utils import get_device_name
    dev = (get_device_name() or "").replace(" ", "_")
    counts = {os.path.basename(d):
              len([f for f in os.listdir(d) if dev and dev in f]) for d in dirs}
    live = counts.get(tv, 0) if tv else -1
    total = sum(counts.values())
    # Two distinct situations, both worth reporting rather than collapsing:
    # a populated older dir (stale, reached by fallback) or nothing anywhere.
    return PASS, "device=%s; installed=%s has %d; anywhere: %d; all dirs: %s" % (
        dev, tv, live, total, counts)


@claim("tuning-in-sglang", "SGLANG_MOE_CONFIG_DIR must name the parent of configs/, and naming the version dir raises")
def c_sglang_cfgdir_layout():
    if not has_mod("sglang"):
        return NA, "sglang not installed in this image"
    import json as _json
    import tempfile
    import triton
    try:
        from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_config import (
            get_config_file_name, get_moe_configs)
        from sglang.srt.server_args import (ServerArgs,
                                            set_global_server_args_for_scheduler)
    except Exception as exc:                      # noqa: BLE001
        return NA, "config lookup not importable here: %s" % exc
    try:
        from sglang.srt.server_args import get_global_server_args
        get_global_server_args()
    except Exception:                             # noqa: BLE001
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))

    tv = "triton_" + triton.__version__.replace(".", "_")
    fname = get_config_file_name(256, 256, "fp8_w8a8", [128, 128])
    payload = {"1": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 128,
                     "GROUP_SIZE_M": 1, "num_warps": 2, "num_stages": 2}}
    prev = os.environ.get("SGLANG_MOE_CONFIG_DIR")
    try:
        # correct layout -> hit
        root = tempfile.mkdtemp()
        d = os.path.join(root, "configs", tv)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, fname), "w") as fh:
            _json.dump(payload, fh)
        os.environ["SGLANG_MOE_CONFIG_DIR"] = root
        get_moe_configs.cache_clear()
        hit = get_moe_configs(256, 256, "fp8_w8a8", block_n=128, block_k=128)

        # env var pointed at the version dir -> the fallback scan listdirs a
        # configs/ that does not exist
        os.environ["SGLANG_MOE_CONFIG_DIR"] = d
        get_moe_configs.cache_clear()
        try:
            get_moe_configs(256, 256, "fp8_w8a8", block_n=128, block_k=128)
            raised = None
        except FileNotFoundError as exc:
            raised = type(exc).__name__
        except Exception as exc:                  # noqa: BLE001
            raised = type(exc).__name__
    finally:
        os.environ.pop("SGLANG_MOE_CONFIG_DIR", None)
        if prev is not None:
            os.environ["SGLANG_MOE_CONFIG_DIR"] = prev
        get_moe_configs.cache_clear()

    if hit is None:
        return FAIL, "correct layout $DIR/configs/%s/<file> did not load" % tv
    if raised != "FileNotFoundError":
        return FAIL, ("naming the version dir gave %s, not FileNotFoundError -- "
                      "the skill says this is a hard failure" % raised)
    return PASS, "parent layout loads; version-dir layout raises FileNotFoundError"


@claim("tuning-in-vllm", "the device name comes from the framework, not from torch or gcnArchName")
def c_vllm_device_name():
    if not has_mod("vllm"):
        return NA, "vllm not installed in this image"
    import torch
    from vllm.platforms import current_platform
    from vllm.model_executor.layers.fused_moe.fused_moe import get_config_file_name
    plat = current_platform.get_device_name()
    tname = torch.cuda.get_device_name()
    arch = torch.cuda.get_device_properties(0).gcnArchName
    fname = get_config_file_name(128, 1024, "fp8_w8a8")
    if not plat:
        return FAIL, "current_platform.get_device_name() is empty -- nothing to key on"
    if plat.replace(" ", "_") not in fname:
        return FAIL, "filename %s does not carry the platform name %s" % (fname, plat)
    if arch in fname:
        return FAIL, "filename carries gcnArchName, contradicting the skill"
    return PASS, "platform=%r torch=%r arch=%r -> %s" % (plat, tname, arch, fname)


@claim("tuning-in-sglang", "SGLANG_MOE_CONFIG_DIR is an override the installed code honors")
def c_sglang_env():
    if not has_mod("sglang"):
        return NA, "sglang not installed in this image"
    import sglang
    base = os.path.dirname(sglang.__file__)
    rc, out = sh("grep -rl SGLANG_MOE_CONFIG_DIR %s 2>/dev/null | head -3" % base)
    return (PASS, "referenced in %d file(s)" % len(out.strip().splitlines())) \
        if out.strip() else (FAIL, "env var not referenced in installed sglang")


# --------------------------------------------------------------------------
# tuning-core
# --------------------------------------------------------------------------

@claim("tuning-core", "an idle-GPU check exists, so 'pin your GPUs' is actionable")
def c_smi():
    rc, out = sh("rocm-smi --showuse 2>/dev/null | grep -c 'GPU use'")
    n = out.strip().splitlines()[-1] if out.strip() else "0"
    if not n.isdigit() or int(n) == 0:
        return FAIL, "rocm-smi --showuse returned no per-GPU utilization"
    return PASS, "%s GPUs reporting utilization" % n


@claim("tuning-core", "relative error is the gate; bf16 at K=4096 shows why absolute is not")
def c_relerr():
    if not has_mod("torch"):
        return NA, "torch not importable"
    import torch
    if not torch.cuda.is_available():
        return NA, "no GPU visible to torch"
    torch.manual_seed(0)
    m = k = n = 512
    K = 4096
    a = torch.randn(m, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(K, n, device="cuda", dtype=torch.bfloat16)
    got = (a @ b).float()
    ref = a.float() @ b.float()
    abs_err = (got - ref).abs().max().item()
    rel = abs_err / ref.abs().max().item()
    if rel >= 0.05:
        return FAIL, "a correct bf16 GEMM exceeded the documented 0.05 gate (rel=%.4f)" % rel
    return PASS, "abs=%.3f but rel=%.5f -- absolute error alone would look alarming" % (
        abs_err, rel)


# --------------------------------------------------------------------------
# arch-migration: the gfx942 <-> gfx950 claims
#
# These were prose for as long as there was only one part to run on, and prose
# is where an arch assumption hides. Each one below is a sentence from
# tuning-core/arch_migration.md or tuning-hip/SKILL.md turned into a run.
#
# They are written to be meaningful on *either* architecture rather than to
# assert gfx950 specifically: the useful output on gfx942 is "FNUZ computes,
# OCP does not", and on gfx950 the exact reverse. A check that only passes on
# one part cannot tell you the port worked.
# --------------------------------------------------------------------------

# Which fp8 dialect each architecture can actually compute in. The pair is the
# claim -- that exactly one of them works, and which one depends on the part.
_FP8_BY_ARCH = {
    "gfx942": ("float8_e4m3fnuz", "float8_e4m3fn"),   # (computes, does not)
    "gfx950": ("float8_e4m3fn", "float8_e4m3fnuz"),
}


def _arch_of(torch):
    return torch.cuda.get_device_properties(
        torch.cuda.current_device()).gcnArchName.split(":")[0]


@claim("tuning-core", "fp8 dialect is arch-specific: exactly one of FNUZ/OCP computes")
def c_fp8_dialect():
    if not has_mod("torch"):
        return NA, "torch not importable"
    import torch
    if not torch.cuda.is_available():
        return NA, "no GPU visible to torch"
    arch = _arch_of(torch)
    if arch not in _FP8_BY_ARCH:
        return NA, "no recorded fp8 dialect expectation for %s" % arch
    want_ok, want_bad = _FP8_BY_ARCH[arch]

    def works(name):
        dt = getattr(torch, name, None)
        if dt is None:
            return False, "no torch dtype"
        try:
            n = 256
            a = torch.randn(n, n, device="cuda").to(dt)
            b = torch.randn(n, n, device="cuda").to(dt).t()
            s = torch.ones((), device="cuda", dtype=torch.float32)
            torch._scaled_mm(a, b, scale_a=s, scale_b=s,
                             out_dtype=torch.bfloat16)
            return True, "scaled_mm OK"
        except Exception as exc:                  # noqa: BLE001
            return False, str(exc).split("\n")[0][:70]

    ok_ok, ok_msg = works(want_ok)
    bad_ok, bad_msg = works(want_bad)
    if not ok_ok:
        return FAIL, "%s should compute on %s but did not: %s" % (
            want_ok, arch, ok_msg)
    if bad_ok:
        # Worse than the reverse failure: it means an artifact carrying the
        # wrong dialect runs and returns numbers instead of refusing.
        return FAIL, ("%s computed on %s -- the dialects are not exclusive "
                      "here, so a wrong-dialect artifact will not announce "
                      "itself" % (want_bad, arch))
    return PASS, "%s computes; %s refused with '%s'" % (want_ok, want_bad,
                                                        bad_msg)


@claim("tuning-hip", "LDS per workgroup is read from the device, and any two sources agree")
def c_lds_two_sources():
    # sweep.py prunes the search space against this number and falls back to
    # 64 KB when it cannot be read. On gfx942 that fallback happened to equal
    # the truth, so a silent substitution was harmless; on gfx950 it is wrong
    # by 2.5x and still silent. Confirm at least one source answers, and that
    # any two that answer agree.
    vals = {}
    if has_mod("torch"):
        import torch
        if torch.cuda.is_available():
            try:
                vals["torch"] = int(torch.cuda.get_device_properties(
                    torch.cuda.current_device()).shared_memory_per_block)
            except Exception:                     # noqa: BLE001
                pass
    rc, out = sh("rocminfo 2>/dev/null")
    if rc == 0 and out:
        import re
        found = [int(kb) * 1024 for kb in re.findall(
            r"Segment:\s+GROUP\s*\n\s*Size:\s+(\d+)\(\S+\)\s*KB", out)]
        if found:
            vals["rocminfo"] = max(found)
    if not vals:
        return FAIL, ("neither torch nor rocminfo reports LDS per workgroup; "
                      "sweep.py would silently prune against a 64 KB guess")
    if len(set(vals.values())) > 1:
        return FAIL, "sources disagree: %s" % vals
    only = " (only source available)" if len(vals) == 1 else ""
    return PASS, "%d bytes from %s%s" % (
        list(vals.values())[0], "+".join(sorted(vals)), only)


@claim("tuning-triton", "Gluon exposes a cdna4 dialect, and it is the gfx950-only surface")
def c_gluon_cdna4():
    try:
        from triton.experimental.gluon.language import amd as gl_amd
    except Exception as exc:                      # noqa: BLE001
        return NA, "triton gluon amd dialect not importable: %s" % type(exc).__name__
    have = [d for d in ("cdna3", "cdna4") if hasattr(gl_amd, d)]
    if "cdna4" not in have:
        return FAIL, "no cdna4 namespace; Gluon MXFP4 kernels cannot be built"
    c4 = {s for s in dir(gl_amd.cdna4) if not s.startswith("_")}
    # mfma_scaled is the instruction the microscaled dtypes exist for; without
    # it the namespace is present but the reason to use it is not.
    want = {"mfma_scaled", "get_mfma_scale_layout"}
    missing = want - c4
    if missing:
        return FAIL, "cdna4 present but missing %s" % ", ".join(sorted(missing))
    return PASS, "cdna3+cdna4 present; cdna4 has mfma_scaled and scale layout"


@claim("tuning-core", "microscaled FP4 is reachable on gfx950, and torch.zeros is the wrong probe")
def c_mxfp4_reachable():
    if not has_mod("torch"):
        return NA, "torch not importable"
    import torch
    if not torch.cuda.is_available():
        return NA, "no GPU visible to torch"
    arch = _arch_of(torch)
    dt = getattr(torch, "float4_e2m1fn_x2", None)
    e8m0 = getattr(torch, "float8_e8m0fnu", None)
    if dt is None:
        return NA, "torch build has no float4_e2m1fn_x2"
    if arch != "gfx950":
        # On gfx942 the correct outcome is that nothing works, which is still
        # worth asserting -- an FP4 path that "works" there is emulating.
        try:
            torch.empty((64, 32), dtype=dt, device="cuda")
            return FAIL, "%s allocated an FP4 tensor; it has no FP4 path" % arch
        except Exception:                         # noqa: BLE001
            return PASS, "%s cannot allocate FP4, as expected" % arch

    zeros_ok = True
    try:
        torch.zeros((64, 32), dtype=dt, device="cuda")
    except Exception:                             # noqa: BLE001
        zeros_ok = False
    try:
        torch.empty((64, 32), dtype=dt, device="cuda")
    except Exception as exc:                      # noqa: BLE001
        return FAIL, "cannot allocate FP4 even with empty(): %s" % str(exc)[:60]
    if e8m0 is None:
        return FAIL, "no float8_e8m0fnu; MX block scales are unrepresentable"
    # Two independent routes to an MXFP4 GEMM, and they do not agree across
    # images. torch._scaled_mm accepts e8m0 block scales on the vllm image's
    # torch 2.10 and rejects them on sglang's 2.9.1 with "Invalid scaling
    # configuration" -- same GPU, same arch, same dtype. aiter's own kernels
    # work on both. Reporting only the torch route would call FP4
    # unreachable in an image where the corpus demonstrably runs two FP4
    # cases, so check both and say which one answered.
    torch_route, torch_msg = True, "ok"
    try:
        m = n = 256
        k_log = 256
        a = torch.empty((m, k_log // 2), dtype=dt, device="cuda")
        b = torch.empty((n, k_log // 2), dtype=dt, device="cuda")
        sc = torch.full((m, k_log // 32), 127, dtype=torch.uint8,
                        device="cuda").view(e8m0)
        torch._scaled_mm(a, b.t(), scale_a=sc, scale_b=sc,
                         out_dtype=torch.bfloat16)
    except Exception as exc:                      # noqa: BLE001
        torch_route = False
        torch_msg = str(exc).split("\n")[0][:60]

    aiter_route, aiter_msg = False, "not tried"
    try:
        from aiter.ops.triton.gemm_afp4wfp4 import gemm_afp4wfp4
        from aiter.ops.triton.quant import dynamic_mxfp4_quant
        xq, xs = dynamic_mxfp4_quant(
            torch.randn(256, 256, device="cuda", dtype=torch.bfloat16))
        wq, ws = dynamic_mxfp4_quant(
            torch.randn(256, 256, device="cuda", dtype=torch.bfloat16))
        gemm_afp4wfp4(xq, wq, xs, ws, dtype=torch.bfloat16)
        torch.cuda.synchronize()
        aiter_route, aiter_msg = True, "ok"
    except Exception as exc:                      # noqa: BLE001
        aiter_msg = "%s: %s" % (type(exc).__name__, str(exc).split("\n")[0][:50])

    zeros_note = ("torch.zeros also works" if zeros_ok else
                  "torch.zeros still raises, so probing with it would report "
                  "FP4 unsupported on the one arch that has it")
    if not torch_route and not aiter_route:
        return FAIL, "FP4 allocates but no GEMM route runs: torch=%s aiter=%s" % (
            torch_msg, aiter_msg)
    if not torch_route:
        # Not a FAIL: the hardware path exists and the corpus reaches it. It
        # is a fact about this image's torch, and the one that decides whether
        # a case may call _scaled_mm directly.
        return PASS, ("aiter mxfp4 GEMM runs but torch._scaled_mm does not "
                      "(%s) -- torch %s. Reach FP4 through aiter here. %s"
                      % (torch_msg, torch.__version__.split("+")[0],
                         zeros_note))
    if not aiter_route:
        return FAIL, "torch._scaled_mm mxfp4 runs but aiter's does not: %s" % aiter_msg
    return PASS, "mxfp4 runs via both torch._scaled_mm and aiter; %s" % zeros_note


# --------------------------------------------------------------------------

def run(selected=None):
    rows = []
    for skill, text, fn in _REGISTRY:
        if selected and not any(s in skill for s in selected):
            continue
        t0 = time.time()
        try:
            status, detail = fn()
        except Exception as exc:                  # noqa: BLE001
            status, detail = FAIL, "check itself raised: %s: %s" % (
                type(exc).__name__, exc)
        rows.append({"skill": skill, "claim": text, "status": status,
                     "detail": detail, "seconds": round(time.time() - t0, 2)})
    return rows


def image_id():
    """Best-effort identity of the container, for the report header."""
    for mod, label in (("vllm", "vllm"), ("sglang", "sglang")):
        if has_mod(mod):
            try:
                return "%s %s" % (label, __import__(mod).__version__)
            except Exception:                     # noqa: BLE001
                return label
    return platform.node()


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--skill", action="append", default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv[1:])

    rows = run(a.skill)
    print("image: %s   python %s" % (image_id(), platform.python_version()))
    print()
    cur = None
    for r in rows:
        if r["skill"] != cur:
            cur = r["skill"]
            print("## %s" % cur)
        print("  %-5s %s" % (r["status"], r["claim"]))
        print("        -> %s" % r["detail"])
    n = len(rows)
    p = sum(1 for r in rows if r["status"] == PASS)
    f = sum(1 for r in rows if r["status"] == FAIL)
    na = n - p - f
    print("\n%d claims: %d PASS, %d FAIL, %d N/A (precondition absent here)" % (n, p, f, na))
    if a.json:
        json.dump({"image": image_id(), "rows": rows}, open(a.json, "w"), indent=2)
        print("wrote %s" % a.json)
    return 1 if f else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
