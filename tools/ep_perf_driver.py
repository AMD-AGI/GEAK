#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""Device-time oracle for an ORT plugin-EP op (frozen: copied into each generated
task's harness/ by tools/gen_ep_task.py and never edited by the optimizer).

Each case is one ONNX graph of `nodes` independent ops that share the activation
but have distinct weights, sized so the weights of one inference exceed the GPU's
caches; every launch streams its weights from DRAM the way consecutive decoder
layers do. The graph runs through the EP library with HIPDNN_EP_TRACE_FILE set:
the EP's Chrome trace has one GPU span per op (the gap between consecutive
fenceless markers on the in-order stream), and the reported time is the median
span of the timed inferences. Every case's outputs are also checked against
ORT's CPU EP, so a fast-but-wrong kernel fails here even on a shape or
zero-point layout the numeric pytest suite does not cover.

Prints one `GEAK_EP_PERF <json>` line per case (`GEAK_EP_CHECK` with
--check-only) and exits non-zero if any case fails.
"""
import argparse
import contextlib
import gc
import glob
import json
import math
import os
import re
import statistics
import sys
import tempfile
import zlib

LINE_PREFIX = {"perf": "GEAK_EP_PERF", "check": "GEAK_EP_CHECK"}
_MIN_NODES, _MAX_NODES = 2, 32


# ----------------------------------------------------------------------------
# Case builders: case dict -> (model, feeds, trace name, trace shape, nodes)
# ----------------------------------------------------------------------------
def nodes_for(weight_bytes, rotate_bytes):
    """Independent copies needed so one inference's weights total rotate_bytes."""
    return max(_MIN_NODES, min(_MAX_NODES, math.ceil(rotate_bytes / max(1, weight_bytes))))


def matmul_nbits_case(case, rotate_bytes, seed):
    import numpy as np
    from onnx import TensorProto, helper, numpy_helper

    M, K, N = case["M"], case["K"], case["N"]
    bits, bs = case.get("bits", 4), case.get("block_size", 128)
    zp, scale = case.get("zp", "none"), case.get("scale", 0.01)
    if bits != 4:
        raise ValueError(f"case {case['id']}: only bits=4 is supported, got {bits}")
    if zp not in ("none", "u8", "fp16"):
        raise ValueError(f"case {case['id']}: zp must be none|u8|fp16, got {zp!r}")
    n_blocks = -(-K // bs)
    w_shape = (N, n_blocks, bs * bits // 8)
    zp_shape = {"u8": (N, -(-n_blocks * bits // 8)), "fp16": (N, n_blocks)}.get(zp)
    weight_bytes = math.prod(w_shape) + 2 * N * n_blocks
    if zp_shape:
        weight_bytes += math.prod(zp_shape) * (1 if zp == "u8" else 2)
    nodes = case.get("nodes") or nodes_for(weight_bytes, rotate_bytes)

    rng = np.random.default_rng([seed, zlib.crc32(case["id"].encode())])
    inits, graph_nodes, outputs = [], [], []
    for j in range(nodes):
        inputs = ["X", f"W{j}", f"S{j}"]
        inits.append(numpy_helper.from_array(
            rng.integers(0, 256, w_shape, dtype=np.uint8), f"W{j}"))
        inits.append(numpy_helper.from_array(
            rng.uniform(-scale, scale, (N, n_blocks)).astype(np.float16), f"S{j}"))
        if zp == "u8":
            inits.append(numpy_helper.from_array(
                rng.integers(0, 256, zp_shape, dtype=np.uint8), f"Z{j}"))
            inputs.append(f"Z{j}")
        elif zp == "fp16":
            inits.append(numpy_helper.from_array(
                rng.integers(0, 16, zp_shape).astype(np.float16), f"Z{j}"))
            inputs.append(f"Z{j}")
        graph_nodes.append(helper.make_node(
            "MatMulNBits", inputs, [f"Y{j}"], name=f"matmul_nbits_{j}",
            domain="com.microsoft", K=K, N=N, bits=bits, block_size=bs,
            accuracy_level=case.get("accuracy_level", 4)))
        outputs.append(helper.make_tensor_value_info(f"Y{j}", TensorProto.FLOAT16, [1, M, N]))
    graph = helper.make_graph(
        graph_nodes, f"geak_{case['id']}",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT16, [1, M, K])],
        outputs, initializer=inits)
    # Pinned so a newer onnx package never emits an IR version the installed ORT rejects.
    model = helper.make_model(graph, ir_version=10, opset_imports=[
        helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)])
    feeds = {"X": rng.uniform(-1, 1, (1, M, K)).astype(np.float16)}
    return model, feeds, "matmul_nbits", f"m={M},n={N},k={K}", nodes


CASE_BUILDERS = {"matmul_nbits": matmul_nbits_case}


# ----------------------------------------------------------------------------
# Trace parsing and statistics (pure; unit-tested on CPU)
# ----------------------------------------------------------------------------
def device_samples(traces, name, shape, runs, warmup):
    """Per-op GPU durations (us) of the timed inferences.

    `traces` holds one parsed Chrome trace per EP runtime state (a session that
    compiles several partitions writes several files). Each must contain the same
    number of `name`/`shape` GPU spans for each of the `runs` inferences, in
    enqueue order. The first `warmup` inferences are dropped, and so is the first
    op of every inference: its span starts at the per-inference epoch, so it also
    carries the wrapper's launch latency rather than only the kernel."""
    samples = []
    for t in traces:
        spans = sorted((e for e in t.get("traceEvents", [])
                        if e.get("cat") == "gpu" and e.get("name") == name
                        and e.get("args", {}).get("shape") == shape),
                       key=lambda e: e["ts"])
        if not spans:
            continue
        if len(spans) % runs:
            raise ValueError(f"{len(spans)} {name} GPU spans for {shape} is not a "
                             f"multiple of the {runs} inferences run")
        per = len(spans) // runs
        for i in range(warmup, runs):
            samples += [float(e["dur"]) for e in spans[i * per + 1:(i + 1) * per]]
    if not samples:
        raise ValueError(f"no {name} GPU spans for {shape} in the EP trace (is "
                         f"HIPDNN_EP_TRACE_FILE honoured, and did every node run on the EP?)")
    return samples


def summarize(samples):
    s = sorted(samples)

    def q(p):
        return s[min(len(s) - 1, round(p * (len(s) - 1)))]

    mean = statistics.fmean(s)
    return {"median_us": statistics.median(s), "min_us": s[0], "p10_us": q(0.1),
            "p90_us": q(0.9), "mean_us": mean,
            "cv": statistics.pstdev(s) / mean if mean > 0 else 0.0, "samples": len(s)}


def compare(actual, expected, atol, rtol, cos_min):
    """Return None when every output matches, else a short reason."""
    import numpy as np

    if len(actual) != len(expected):
        return f"{len(actual)} outputs != reference {len(expected)}"
    for i, (a, e) in enumerate(zip(actual, expected)):
        a = np.asarray(a, dtype=np.float64).ravel()
        e = np.asarray(e, dtype=np.float64).ravel()
        if a.shape != e.shape:
            return f"output {i}: shape {a.shape} != reference {e.shape}"
        if not np.isfinite(a).all():
            return f"output {i}: non-finite values"
        bad = np.abs(a - e) > atol + rtol * np.abs(e)
        if bad.any():
            j = int(np.argmax(np.abs(a - e)))
            return (f"output {i}: {int(bad.sum())}/{a.size} elements outside "
                    f"atol={atol} rtol={rtol} (worst {a[j]:.5g} vs {e[j]:.5g})")
        denom = float(np.linalg.norm(a) * np.linalg.norm(e))
        cos = float(a @ e) / denom if denom > 0 else (1.0 if not a.any() and not e.any() else 0.0)
        if cos < cos_min:
            return f"output {i}: cosine {cos:.6f} < {cos_min}"
    return None


# ----------------------------------------------------------------------------
# ORT plumbing
# ----------------------------------------------------------------------------
# What the HIP EP's profiler prints to stderr when tracing: a [PERF] table and a
# Total/H2D/Compute/D2H block per inference, and a [PERF SUMMARY] at teardown.
_PROFILER_NOISE = re.compile(r"^(\[PERF[\] ]|\s+(Total|H2D|D2H|Compute):)")


@contextlib.contextmanager
def _filtered_stderr():
    """Capture fd 2 and replay everything except the EP profiler's per-inference
    output once the block exits."""
    sys.stderr.flush()
    saved = os.dup(2)
    with tempfile.TemporaryFile() as tmp:
        os.dup2(tmp.fileno(), 2)
        try:
            yield
        finally:
            sys.stderr.flush()
            os.dup2(saved, 2)
            os.close(saved)
            tmp.seek(0)
            for line in tmp.read().decode(errors="replace").splitlines():
                if line.strip() and not _PROFILER_NOISE.match(line):
                    print(line, file=sys.stderr)


def _register_ep(ort, ep_name, ep_dll):
    capi = ort.capi._pybind_state
    capi.register_execution_provider_library(ep_name, ep_dll)
    for dev in capi.get_ep_devices():
        if dev.ep_name == ep_name:
            return dev
    raise RuntimeError(f"EP {ep_name!r} registered from {ep_dll} but exposes no device")


def _ep_session(ort, device, provider_options, model_path):
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    opts.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    opts.add_provider_for_devices([device], dict(provider_options))
    return ort.InferenceSession(model_path, sess_options=opts)


def run_case(ort, device, provider_options, case, args, trace_base):
    model, feeds, name, shape, nodes = CASE_BUILDERS[args.op](
        case, args.rotate_mib * 2**20, args.seed)
    result = {"id": case["id"], "params": {k: v for k, v in case.items() if k != "id"},
              "nodes": nodes, "shape": shape}
    with tempfile.TemporaryDirectory(prefix="geak_ep_perf_") as tmp:
        import onnx
        model_path = os.path.join(tmp, "model.onnx")
        onnx.save(model, model_path)
        del model
        cpu = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        expected = cpu.run(None, feeds)
        del cpu

        for stale in glob.glob(trace_base + ".*"):
            os.remove(stale)
        runs = 1 if args.check_only else args.warmup + args.iters
        with _filtered_stderr():
            sess = _ep_session(ort, device, provider_options, model_path)
            for _ in range(runs):
                actual = sess.run(None, feeds)
            del sess
            gc.collect()
    err = compare(actual, expected, case.get("atol", 5e-2), case.get("rtol", 1e-2),
                  case.get("cos", 0.999))
    if err:
        return {**result, "status": "fail", "error": f"mismatch vs CPU EP: {err}"}
    if args.check_only:
        return {**result, "status": "ok"}

    traces = []
    for path in sorted(glob.glob(trace_base + ".*")):
        with open(path) as f:
            traces.append(json.load(f))
        os.remove(path)
    samples = device_samples(traces, name, shape, runs, args.warmup)
    return {**result, "status": "ok", **summarize(samples)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ep-dll", required=True)
    ap.add_argument("--ep-name", default="AMDGPUExecutionProvider")
    ap.add_argument("--ep-option", action="append", default=[], help="KEY=VALUE, repeatable")
    ap.add_argument("--op", default="matmul_nbits", choices=sorted(CASE_BUILDERS))
    ap.add_argument("--cases", required=True, help="JSON list of case dicts, or @file")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--rotate-mib", type=int, default=128,
                    help="weight bytes per inference; sets each case's node count")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check-only", action="store_true",
                    help="one inference per case, CPU comparison only, no timing")
    args = ap.parse_args()
    if args.iters < 1 or args.warmup < 0:
        sys.exit("--iters must be >= 1 and --warmup >= 0")

    raw = args.cases
    if raw.startswith("@"):
        with open(raw[1:]) as f:
            raw = f.read()
    cases = json.loads(raw)
    options = {}
    for opt in args.ep_option:
        key, sep, value = opt.partition("=")
        if not sep or not key:
            sys.exit(f"--ep-option {opt!r}: expected KEY=VALUE")
        options[key] = value

    trace_dir = tempfile.mkdtemp(prefix="geak_ep_trace_")
    trace_base = os.path.join(trace_dir, "trace")
    if not args.check_only:
        # Read once per process by the EP, so it must be set before the library loads.
        os.environ["HIPDNN_EP_TRACE_FILE"] = trace_base + ".json"
    os.environ["PATH"] = os.path.dirname(os.path.abspath(args.ep_dll)) + os.pathsep + \
        os.environ.get("PATH", "")

    import onnxruntime as ort
    device = _register_ep(ort, args.ep_name, os.path.abspath(args.ep_dll))
    prefix = LINE_PREFIX["check" if args.check_only else "perf"]
    failed = 0
    for case in cases:
        try:
            res = run_case(ort, device, options, case, args, trace_base)
        except Exception as e:  # noqa: BLE001 -- a crash in one case must not hide the others
            res = {"id": case.get("id"), "status": "fail", "error": f"{type(e).__name__}: {e}"}
        failed += res["status"] != "ok"
        print(f"{prefix} {json.dumps(res, sort_keys=True)}", flush=True)
    for leftover in glob.glob(os.path.join(trace_dir, "*")):
        os.remove(leftover)
    os.rmdir(trace_dir)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
