#!/usr/bin/env python3
"""Launch the real GEAK Semantics 1.2 metadata+marker replay."""
import argparse
import base64
import glob
import json
import os
import subprocess
import time


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RUNTIME_CAPTURE = os.path.join(
    SCRIPT_DIR, "semantic_runtime_capture.py")
DEFAULT_MODEL_RUNNER = (
    "/sgl-workspace/sglang/python/sglang/srt/"
    "model_executor/model_runner.py")
DEFAULT_RUNTIME_MODULE = (
    "/sgl-workspace/sglang/python/sglang/srt/"
    "geak_semantic_runtime_capture.py")
SENTINEL = "# GEAK_SEMANTICS_CAPTURE_BOOTSTRAP_V1"


def _run(command, stdout=None):
    return subprocess.run(command, check=True, stdout=stdout,
                          stderr=subprocess.STDOUT)


def _docker(container, shell_command, stdout=None):
    return _run(
        ["docker", "exec", container, "bash", "-lc", shell_command],
        stdout=stdout)


def _stop_service(container, port):
    port = int(port)
    pattern = "[s]glang.launch_server.*--port(=| )%d" % port
    command = (
        "pkill -TERM -f '%s' || true; "
        "for _ in 1 2 3 4 5; do "
        "pgrep -f '%s' >/dev/null || exit 0; sleep 1; done; "
        "pkill -KILL -f '%s' || true" %
        (pattern, pattern, pattern))
    _docker(container, command)


def _deploy(container, model_runner, runtime_module):
    _run(["docker", "cp", RUNTIME_CAPTURE,
          "%s:%s" % (container, runtime_module)])
    bootstrap = """

# GEAK_SEMANTICS_CAPTURE_BOOTSTRAP_V1
import os as _geak_os
if _geak_os.environ.get("GEAK_SEMANTICS_CAPTURE", "0") in ("1", "true", "True"):
    import sys as _geak_sys
    try:
        from sglang.srt import geak_semantic_runtime_capture as _geak_capture
        _geak_original_load_model = ModelRunner.load_model
        def _geak_load_model(self, *args, **kwargs):
            result = _geak_original_load_model(self, *args, **kwargs)
            _geak_capture.install_on_model(self.model)
            return result
        ModelRunner.load_model = _geak_load_model
        _geak_sys.stderr.write("[GEAK_SEMANTICS] ModelRunner.load_model wrapped\\n")
    except Exception as _geak_error:
        _geak_sys.stderr.write("[GEAK_SEMANTICS] bootstrap failed: %s\\n" % _geak_error)
"""
    program = """
import os, shutil
path = %r
sentinel = %r
bootstrap = %s
with open(path) as fh:
    text = fh.read()
backup = path + ".geak_semantics_bak"
if not os.path.exists(backup):
    shutil.copyfile(path, backup)
if sentinel not in text:
    with open(path, "a") as fh:
        fh.write(bootstrap)
""" % (model_runner, SENTINEL, repr(bootstrap))
    encoded = base64.b64encode(program.encode("utf-8")).decode("ascii")
    _docker(container, "python3 -c \"import base64;exec(base64.b64decode('%s'))\"" %
            encoded)


def _latest_trace(trace_dir):
    candidates = glob.glob(
        os.path.join(trace_dir, "**", "*.trace.json*"), recursive=True)
    if not candidates:
        return ""
    rank_zero = [
        path for path in candidates if "-TP-0.trace.json" in path]
    if rank_zero:
        candidates = rank_zero
    return max(candidates, key=os.path.getmtime)


def _with_disable_cuda_graph(benchmark_text):
    needle = "--disable-radix-cache"
    if needle not in benchmark_text:
        raise RuntimeError(
            "official benchmark lacks expected server argument anchor")
    if "--disable-cuda-graph" in benchmark_text:
        return benchmark_text
    return benchmark_text.replace(
        needle, "--disable-radix-cache --disable-cuda-graph", 1)


def _required_phases(plan):
    return sorted({
        str(bucket.get("phase")).lower()
        for bucket in plan.get("target_buckets", [])
        if bucket.get("phase")
    })


def capture(setup_path, capture_plan_path, out_dir,
            disable_cuda_graph=False, phases=None,
            forwards_per_bucket=1):
    with open(setup_path) as fh:
        setup = json.load(fh)
    disable_cuda_graph = bool(
        disable_cuda_graph or setup.get("disable_cuda_graph", False))
    phases = list(phases or setup.get("capture_phases", []))
    if forwards_per_bucket == 1 and setup.get("forwards_per_bucket") is not None:
        forwards_per_bucket = int(setup["forwards_per_bucket"])
    with open(capture_plan_path) as fh:
        plan = json.load(fh)
    if not phases:
        phases = _required_phases(plan)
    required = ("container", "model", "benchmark", "port",
                "tensor_parallel_size", "workload")
    missing = [name for name in required if setup.get(name) is None]
    if missing:
        raise ValueError(
            "shape capture setup missing: %s" % ", ".join(missing))

    container = setup["container"]
    model_runner = setup.get("sglang_model_runner", DEFAULT_MODEL_RUNNER)
    runtime_module = setup.get(
        "geak_runtime_capture_module", DEFAULT_RUNTIME_MODULE)
    layers = sorted(set(
        int(target["representative_layer_id"])
        for target in plan.get("capture_targets", [])
        if target.get("representative_layer_id") is not None))
    if not layers:
        raise ValueError("capture plan has no representative layers")

    os.makedirs(out_dir, exist_ok=True)
    trace_dir = os.path.join(out_dir, "trace")
    os.makedirs(trace_dir, exist_ok=True)
    shape_log = os.path.join(out_dir, "shape.jsonl")
    benchmark_log = os.path.join(out_dir, "benchmark.log")
    for path in (shape_log,):
        if os.path.exists(path):
            os.remove(path)

    _stop_service(container, setup["port"])
    _deploy(container, model_runner, runtime_module)
    workload = setup["workload"]
    repository = setup.get(
        "benchmark_repository",
        "/mnt/raid0/zhaoan12/repo/InferenceX")
    benchmark = setup["benchmark"]
    if disable_cuda_graph:
        benchmark_source = os.path.join(repository, benchmark)
        with open(benchmark_source) as fh:
            benchmark_text = fh.read()
        benchmark_lib = os.path.join(
            repository, "benchmarks", "benchmark_lib.sh")
        benchmark_text = benchmark_text.replace(
            'source "$(dirname "$0")/../../benchmark_lib.sh"',
            'source "%s"' % benchmark_lib)
        benchmark_text = _with_disable_cuda_graph(benchmark_text)
        benchmark = os.path.join(out_dir, "benchmark_eager_decode.sh")
        with open(benchmark, "w") as fh:
            fh.write(benchmark_text)
    command = """
set -e
export GEAK_SEMANTICS_CAPTURE=1
export GEAK_SEMANTICS_SHAPE_LOG=%s
export GEAK_SEMANTICS_RANK=0
export GEAK_SEMANTICS_LAYERS=%s
export GEAK_SEMANTICS_PHASES=%s
export GEAK_SEMANTICS_FORWARDS_PER_BUCKET=%s
export GEAK_SEMANTICS_CALLABLE_TARGETS=%s
export GEAK_SEMANTICS_REQUIRE_PROFILER=1
export PROFILE=1
export SGLANG_TORCH_PROFILER_DIR=%s
export MODEL=%s
export TP=%s
export CONC=%s
export ISL=%s
export OSL=%s
export RANDOM_RANGE_RATIO=%s
export PORT=%s
export RESULT_FILENAME=geak_semantics_1_2_capture
export EVAL_ONLY=false
export RUN_EVAL=false
export ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
cd %s
bash %s
""" % (
        shape_log, ",".join(str(layer) for layer in layers),
        ",".join(phases or []), forwards_per_bucket,
        ",".join(setup.get("callable_targets", [])), trace_dir,
        setup["model"], setup["tensor_parallel_size"],
        workload["concurrency"], workload["input_length"],
        workload["output_length"], workload.get("random_range_ratio", 0.8),
        setup["port"], repository, benchmark)
    started = time.time()
    try:
        with open(benchmark_log, "w") as log:
            _docker(container, command, stdout=log)
    finally:
        # Stop only the service started on this replay's dedicated port.
        _stop_service(container, setup["port"])
    trace = _latest_trace(trace_dir)
    if not os.path.exists(shape_log) or os.path.getsize(shape_log) == 0:
        raise RuntimeError(
            "GEAK runtime capture produced no shape metadata: %s" %
            benchmark_log)
    if not trace:
        raise RuntimeError(
            "GEAK runtime capture produced no profiler trace: %s" %
            benchmark_log)
    result = {
        "schema_version": 1,
        "status": "pass",
        "capture_mode": "metadata_plus_runtime_markers",
        "disable_cuda_graph": bool(disable_cuda_graph),
        "capture_phases": list(phases or []),
        "forwards_per_bucket": int(forwards_per_bucket),
        "container": container,
        "representative_layers": layers,
        "callable_targets": list(setup.get("callable_targets", [])),
        "callable_kernel_map": list(
            setup.get("callable_kernel_map", [])),
        "source_wrapper_map": list(
            setup.get("source_wrapper_map", [])),
        "shape_log": shape_log,
        "capture_trace": trace,
        "benchmark_log": benchmark_log,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    result_path = os.path.join(out_dir, "CAPTURE_RESULT.json")
    result["result_json"] = result_path
    with open(result_path, "w") as fh:
        json.dump(result, fh, indent=2)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--setup", required=True)
    parser.add_argument("--capture-plan", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--disable-cuda-graph", action="store_true")
    parser.add_argument("--phase", action="append", default=[])
    parser.add_argument("--forwards-per-bucket", type=int, default=1)
    parser.add_argument("--result-json", default="")
    args = parser.parse_args()
    result = capture(
        args.setup, args.capture_plan, args.out_dir,
        args.disable_cuda_graph, args.phase,
        args.forwards_per_bucket)
    if args.result_json:
        with open(args.result_json, "w") as fh:
            json.dump(result, fh, indent=2)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
