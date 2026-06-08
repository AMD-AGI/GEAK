#!/usr/bin/env python3
"""Auto-generated task runner for vllm_fused_add_rms_norm (HIP).

Inputs are generated each run from the shape/dtype signatures in
test_cases.json — no .pt files are loaded.
"""
import sys, os, json, argparse, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _runtime as rt

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)
TASK_NAME = "vllm_fused_add_rms_norm"
NAMESPACE = "extracted_fused_add_rms_norm"          # extracted_<op>
OP_NAME = "fused_add_rms_norm"
REF_SOURCE = "vllm"        # "vllm" | "sglang" — used to pick the correct reference

SRC_DIR = os.path.join(TASK_DIR, "src")
BUILD_DIR = os.path.join(TASK_DIR, "build")
TEST_CASES = os.path.join(TASK_DIR, "test_cases.json")

# Lock the offload arch to the runtime device's gfx so torch.utils.cpp_extension
# does not try to compile for every ROCm target (RDNA gfx1100 etc. break on
# vllm's cub/bf16 templates). Override by setting PYTORCH_ROCM_ARCH externally.
def _detect_gfx():
    try:
        import torch
        if torch.cuda.is_available():
            arch = torch.cuda.get_device_properties(0).gcnArchName
            return arch.split(":")[0]  # e.g. "gfx942:sramecc+:xnack-" -> "gfx942"
    except Exception:
        pass
    return "gfx942"


def _build():
    import torch
    from torch.utils.cpp_extension import load
    # vLLM/SGLang csrc IS written for portable CUDA-style code that depends
    # on torch.cpp_extension's hipify pass to rewrite ``cudaStream_t`` →
    # ``hipStream_t`` etc. Leave hipify enabled here. (AITER tasks use a
    # different runner that disables hipify because its CK template trees
    # don't survive the rewrite.)
    # Lock the offload-arch list to the runtime device. Torch's default
    # populates PYTORCH_ROCM_ARCH with EVERY arch its build knows
    # (gfx90a;gfx942;...;gfx1100;gfx1101;...), and the RDNA targets fail on
    # vllm's cub bf16 templates. Override with the live device's gfx unless
    # the user has set a single non-default arch explicitly.
    cur = os.environ.get("PYTORCH_ROCM_ARCH", "")
    detected = _detect_gfx()
    if not cur or ";" in cur or " " in cur or any(
        a in cur for a in ("gfx10", "gfx11", "gfx12")
    ):
        os.environ["PYTORCH_ROCM_ARCH"] = detected
    os.makedirs(BUILD_DIR, exist_ok=True)
    # Strip any in-place hipify residue from prior builds. torch.cpp_extension
    # generates ``foo.hip`` and ``foo_hip.cuh`` next to the original ``foo.cu``
    # / ``foo.cuh``. If both linger between builds the linker gets duplicate
    # symbols. The .hip / *_hip.* files are regenerable from the .cu source.
    import shutil as _sh
    for stale in glob.glob(os.path.join(SRC_DIR, "**", "*.hip"), recursive=True):
        try: os.unlink(stale)
        except Exception: pass
    for stale in glob.glob(os.path.join(SRC_DIR, "**", "*_hip.*"), recursive=True):
        try: os.unlink(stale)
        except Exception: pass
    sources = []
    for ext in ("cu", "cpp"):
        sources.extend(sorted(glob.glob(os.path.join(SRC_DIR, "**", f"*.{ext}"), recursive=True)))
    if not sources:
        raise RuntimeError("no sources under src/")
    # is_python_module=False because bindings.cpp registers via TORCH_LIBRARY
    # rather than defining a PyInit_* symbol — the op is reached via
    # ``torch.ops.<NAMESPACE>.<OP_NAME>`` after load() returns.
    # ``torch.utils.cpp_extension`` defaults to passing
    # ``-D__HIP_NO_HALF_OPERATORS__=1 -D__HIP_NO_HALF_CONVERSIONS__=1`` to the
    # HIP compiler. PyTorch needs those to keep its own ``c10::Half`` happy,
    # but vLLM/SGLang csrc uses raw ``__half2 += __half2`` (in
    # ``type_convert.cuh`` and similar) which is precisely the operator that
    # flag disables. Their official setup.py build doesn't set the flag; we
    # have to undefine it here to make the same source compile through
    # ``cpp_extension.load``.
    cflags = [
        "-U__HIP_NO_HALF_OPERATORS__",
        "-U__HIP_NO_HALF_CONVERSIONS__",
        # vLLM/SGLang csrc gates large blocks of fp8 / bf16 helpers behind
        # ``ENABLE_FP8`` / ``ENABLE_BF16`` (see e.g.
        # quantization/w8a8/fp8/amd/quant_utils.cuh's ``namespace fp8``).
        # Their CMake/setup.py define these for the ROCm build; cpp_extension
        # doesn't, so the symbols vanish and the dependent .cu files fail with
        # "no member named 'scaled_vec_conversion' in namespace 'vllm::fp8'".
        "-DENABLE_FP8",
        "-DENABLE_BF16",
        # AITER fp8 sources also expect this to choose the FNUZ vs E4M3 layout.
        "-DHIP_FP8_TYPE_FNUZ",
    ]
    # vLLM's csrc references ``TORCH_HIP_VERSION`` (a macro vLLM's setup.py
    # would normally define from ``HIP_VERSION_MAJOR/MINOR``). Recreate it
    # here so the same source compiles under cpp_extension.load.
    try:
        import torch
        hv = getattr(torch.version, "hip", None)
        if hv:
            major, _, rest = hv.partition(".")
            minor = rest.split(".", 1)[0] if rest else "0"
            # Torch's TORCH_HIP_VERSION convention: HIP_VERSION_MAJOR*100 +
            # HIP_VERSION_MINOR (e.g. 702 for ROCm 7.2). NOT *10000 — that
            # extra factor broke ``#if TORCH_HIP_VERSION >= 12090`` style
            # gates inherited from CUDA-versioned source (after hipify
            # rewrites CUDA_VERSION → TORCH_HIP_VERSION) by making 70200
            # satisfy a CUDA 12.9+ comparison.
            torch_hip_version = int(major) * 100 + int(minor)
            cflags.append("-DTORCH_HIP_VERSION=" + str(torch_hip_version))
    except Exception:
        cflags.append("-DTORCH_HIP_VERSION=702")
    return load(
        name=NAMESPACE,
        sources=sources,
        extra_include_paths=[
            SRC_DIR,
            os.path.join(SRC_DIR, "core"),
            os.path.join(SRC_DIR, "include"),
        ],
        extra_cflags=cflags,
        extra_cuda_cflags=cflags,
        verbose=False,
        with_cuda=True,
        is_python_module=False,
        build_directory=BUILD_DIR,
    )


def _load_op():
    import torch
    _build()
    return getattr(getattr(torch.ops, NAMESPACE), OP_NAME)


def _test_cases():
    if not os.path.isfile(TEST_CASES):
        return []
    with open(TEST_CASES) as f:
        return json.load(f)


def run_compile():
    try:
        _build()
        return True, None
    except Exception as e:
        return False, str(e)


def run_correctness():
    import torch
    try:
        fn = _load_op()
    except Exception as e:
        return False, f"build failed: {e}"
    cases = _test_cases()
    if not cases:
        return True, "no recorded launch signatures (compile-only check)"
    # Drop empty signatures (kernel was registered but never called with args
    # in the captured run — eg. `_C` ops shadowed by AITER's wrappers).
    cases = [c for c in cases if c.get("args_sig") or c.get("kwargs_sig")]
    if not cases:
        return True, "all recorded launches had empty signatures (compile-only check)"
    ref = rt.reference_for(OP_NAME, REF_SOURCE)
    for tc in cases:
        try:
            args1, kwargs1 = rt.build_inputs(tc, seed=42)
            pre = rt.snapshot(args1)
            ret1 = fn(*args1, **kwargs1)
            out1 = rt.detect_output(pre, args1, ret1)
            if out1 is None:
                continue  # nothing observable to compare
            if ref is not None:
                args_r, kwargs_r = rt.build_inputs(tc, seed=42)
                expected = ref(args_r, kwargs_r)
                if isinstance(expected, dict):
                    # [BugA-fix] reference returns {arg_index: expected_tensor}
                    # for in-place / multi-output kernels; compare each mutated arg.
                    err = None
                    for _idx, _exp in expected.items():
                        err = rt.compare(args1[_idx], _exp)
                        if err:
                            err = f"arg{_idx}: {err}"
                            break
                else:
                    err = rt.compare(out1, expected)
                if err:
                    return False, f"{tc['test_case_id']}: vs reference: {err}"
            else:
                # determinism check: same seed, same input, byte-identical out
                args2, kwargs2 = rt.build_inputs(tc, seed=42)
                pre2 = rt.snapshot(args2)
                ret2 = fn(*args2, **kwargs2)
                out2 = rt.detect_output(pre2, args2, ret2)
                err = rt.compare(out1, out2)
                if err:
                    return False, f"{tc['test_case_id']}: non-deterministic: {err}"
        except Exception as e:
            return False, f"{tc['test_case_id']}: kernel raised {e}"
    return True, None


def run_performance():
    import torch
    try:
        fn = _load_op()
    except Exception:
        return []
    cases = [c for c in _test_cases() if c.get("args_sig") or c.get("kwargs_sig")]
    out = []
    for tc in cases:
        try:
            args, kwargs = rt.build_inputs(tc, seed=42)
            for _ in range(10):
                fn(*args, **kwargs)
            torch.cuda.synchronize()
            n_iter = 100
            starts = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
            ends = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
            for j in range(n_iter):
                starts[j].record(); fn(*args, **kwargs); ends[j].record()
            torch.cuda.synchronize()
            avg = sum(s.elapsed_time(e) for s, e in zip(starts, ends)) / n_iter
            out.append({"test_case_id": tc["test_case_id"], "execution_time_ms": avg, "params": tc.get("params_repr", {})})
        except Exception as e:
            out.append({"test_case_id": tc["test_case_id"], "execution_time_ms": -1.0, "params": {"error": str(e)[:120]}})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["compile", "correctness", "performance"])
    args = ap.parse_args()
    os.makedirs(BUILD_DIR, exist_ok=True)
    if args.mode == "compile":
        ok, err = run_compile()
        json.dump({"status": "ok" if ok else "fail", "error": err}, open(os.path.join(BUILD_DIR, "compile_report.json"), "w"))
        print(f"Compilation: {'PASS' if ok else 'FAIL'}")
        if err: print("Error:", err)
        sys.exit(0 if ok else 1)
    if args.mode == "correctness":
        ok, err = run_correctness()
        json.dump({"status": "ok" if ok else "fail", "error": err}, open(os.path.join(BUILD_DIR, "correctness_report.json"), "w"))
        print(f"Correctness: {'PASS' if ok else 'FAIL'}")
        if err: print("Error:", err)
        sys.exit(0 if ok else 1)
    cases = run_performance()
    json.dump({"test_cases": cases}, open(os.path.join(BUILD_DIR, "performance_report.json"), "w"), indent=2)
    for c in cases:
        print(f"Performance: {c['execution_time_ms']:.4f} ms ({c['test_case_id']})")
    sys.exit(0)


if __name__ == "__main__":
    main()
