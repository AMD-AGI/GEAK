#!/usr/bin/env python3
"""A CI-point leg's aiperf export -> the CI row, through InferenceX's own processor.

    ci_result.py <point_dir> <leg_out> <duration_s>

Runs `infx.results.agentic.process_agentic_result` (from $INFERENCEX_PATH, unmodified) with the
environment the CI job gave it, taken from the point's config.yaml benchmark env and its
baseline config, so the leg's row and the CI row come out of the same code with the same keys.
Then writes <leg_out>/e2e/bench_summary.json for the slot's finalize.py:

    throughput   request_metrics.throughput.output.tokens_per_second (the baseline's metric)
    metric_basis that key, so finalize can refuse a mismatched comparison
    secondary    p90 interactivity (InferenceX: 1000 / p90 TPOT ms) and per-GPU total tok/s
    reference    "ci": the denominator is the point's CI number (baseline/out/points/<pid>)
"""
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ci_config import point_config  # noqa: E402

KEY = "request_metrics.throughput.output.tokens_per_second"


def dig(doc, dotted):
    for part in dotted.split("."):
        if not isinstance(doc, dict):
            return None
        doc = doc.get(part)
    return doc


def main(point_dir, out, duration):
    c = point_config(point_dir)
    pc, be = c["point_config"], c["bench_env"]
    view = os.path.join(out, "e2e", "view")
    os.makedirs(view, exist_ok=True)
    link = os.path.join(view, "aiperf_artifacts")
    if not os.path.lexists(link):
        os.symlink(os.path.join(out, "aiperf_artifacts"), link)
    kv = pc.get("kv_offloading") or "none"
    backend = pc.get("kv_offload_backend") or ""
    env = {"PATH": os.environ.get("PATH", ""), "HOME": os.environ.get("HOME", ""),
           "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": os.environ["INFERENCEX_PATH"],
           "HF_HUB_CACHE": os.environ.get("CI_HF_HUB", ""), "HF_HUB_OFFLINE": "1",
           "HF_DATASETS_OFFLINE": "1", "RESULT_FILENAME": "leg", "RESULT_DIR": view,
           "AGENTIC_OUTPUT_DIR": os.path.join(out, "e2e"), "RUNNER_TYPE": "applyback-local",
           "RECIPE_FINGERPRINT": "", "CONC": str(c["conc"]), "IMAGE": c["image"],
           "MODEL": c["model"], "MODEL_PREFIX": be.get("MODEL_PREFIX", ""),
           "FRAMEWORK": be.get("FRAMEWORK", ""), "PRECISION": be.get("PRECISION", ""),
           "SPEC_DECODING": str(pc.get("spec_decoding") or "none"), "DISAGG": "false",
           "IS_MULTINODE": "false", "TP": str(pc.get("tp")), "EP_SIZE": str(pc.get("ep")),
           "DP_ATTENTION": str(pc.get("dp_attention") or "false"),
           "PP_SIZE": be.get("PP_SIZE", "1"), "DCP_SIZE": be.get("DCP_SIZE", "1"),
           "PCP_SIZE": be.get("PCP_SIZE", "1"), "KV_OFFLOADING": kv,
           "KV_OFFLOAD_BACKEND": backend,
           "KV_OFFLOAD_BACKEND_METADATA": json.dumps({"name": backend}) if backend else ""}
    log = open(os.path.join(out, "e2e", "process_result.log"), "w")
    rc = subprocess.call([sys.executable, "-m", "infx.results.agentic.process_agentic_result"],
                         env=env, stdout=log, stderr=subprocess.STDOUT)
    row_path = os.path.join(out, "e2e", "leg.json")
    row = json.load(open(row_path)) if os.path.exists(row_path) else {}
    tp = dig(row, KEY)
    per_gpu = dig(row, "request_metrics.throughput.per_gpu.total_tput_tps")
    p90 = dig(row, "request_metrics.latency.intvty.p90")
    # The window the client actually measured: a server that dies ends it early, and the
    # configured duration alone cannot show that.
    export = os.path.join(out, "aiperf_artifacts", "profile_export_aiperf.json")
    measured = dig(json.load(open(export)), "benchmark_duration.avg") if os.path.exists(export) else None
    summary = {"rc": rc if tp else (rc or 1), "reference": "ci", "throughput": tp,
               "metric_basis": KEY, "repeats": 1, "duration_s": int(duration),
               "measured_duration_s": measured,
               "ci_duration_s": c["duration"], "row": "e2e/leg.json",
               "secondary": {"p90_interactivity": p90, "per_gpu_total_tok_s": per_gpu},
               "errors": dig(row, "request_accounting.records_error_dropped")}
    with open(os.path.join(out, "e2e", "bench_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    print("%s: output %.3f tok/s, p90 interactivity %s, per-GPU total %s (rc=%d)" % (
        os.path.basename(point_dir.rstrip("/")), tp or float("nan"), p90, per_gpu, rc))
    return 0 if tp else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2], sys.argv[3]))
