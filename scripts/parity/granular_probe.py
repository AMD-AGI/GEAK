#!/usr/bin/env python3
"""Granular component-level parity probe: refactor-test vs origin/main.

Invokes ``run_preprocessor`` directly (no full CLI loop) on a single
kernel through EACH pipeline, instrumenting every major stage so we
can compare outputs side-by-side.

Instrumented components:

  1. Discovery phase           — codebase_context, ATD output
  2. Harness resolution        — harness_path, test_command, source
                                  layer (refactor-test 7-layer /
                                  origin-main 6-layer)
  3. Contract validation       — validate_harness result
  4. Baseline metrics          — duration_us, bottleneck, metrics
  5. Commandment               — sections, byte size
  6. Final output dict         — top-level keys, value types

Usage:
    python3 granular_probe.py <pipeline> <kernel_path> <output_dir>

``pipeline`` is one of ``refactor-test`` / ``origin-main``.

Output: ``<output_dir>/granular_probe.json`` with a dict of
per-stage captures + ``<output_dir>/stderr.log`` with the live
subprocess stderr stream.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOTS = {
    "refactor-test": "/data/sapmajum/GEAK",
    "origin-main": "/data/sapmajum/parity_test/GEAK-main",
}


def _build_driver_script(
    *,
    kernel_path: Path,
    output_dir: Path,
    pipeline_label: str,
    kernel_analysis_on: bool,
) -> str:
    """Return the Python script body that runs run_preprocessor in the container.

    The script:
      - Loads the LLM model via the pipeline's own get_model()
      - Calls run_preprocessor() with (kernel, output_dir, harness-only env)
      - Writes the returned dict to ``<output_dir>/preprocess_result.json``
      - Writes per-stage captures to ``<output_dir>/granular_probe.json``
    """
    kpath = str(kernel_path)
    odir = str(output_dir)
    return f"""
import json, logging, os, sys, time, traceback
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s %(message)s')
logger = logging.getLogger('granular_probe')

odir = Path({odir!r})
odir.mkdir(parents=True, exist_ok=True)

captures = {{
    "pipeline": {pipeline_label!r},
    "kernel": {kpath!r},
    "output_dir": str(odir),
    "stages": {{}},
    "errors": [],
}}

def record(stage, **kwargs):
    captures["stages"][stage] = {{"t": time.strftime('%Y-%m-%dT%H:%M:%S'), **kwargs}}
    logger.info("STAGE %s -> %s", stage, list(kwargs.keys()))

# Stage 0: bootstrap — import the pipeline's run_preprocessor
try:
    from minisweagent.run.preprocess.preprocessor import run_preprocessor
    from minisweagent.models import get_model
    record("bootstrap", ok=True, preprocessor_module=run_preprocessor.__module__)
except Exception as e:
    record("bootstrap", ok=False, error=str(e), traceback=traceback.format_exc())
    (odir / "granular_probe.json").write_text(json.dumps(captures, indent=2, default=str))
    sys.exit(2)

# Stage 1: model factory
try:
    model = get_model(
        "claude-sonnet-4.5",
        {{
            "model_class": "amd_llm",
            "model_name": "claude-sonnet-4.5",
            "model_kwargs": {{"temperature": 0.0, "max_tokens": 8000}},
        }},
    )
    record("model", ok=True, model_class=type(model).__name__)
except Exception as e:
    record("model", ok=False, error=str(e), traceback=traceback.format_exc())
    (odir / "granular_probe.json").write_text(json.dumps(captures, indent=2, default=str))
    sys.exit(3)

# Stage 2: run_preprocessor
t0 = time.monotonic()
try:
    result = run_preprocessor(
        kernel_url={kpath!r},
        output_dir=odir,
        gpu_id=0,
        model=model,
        benchmark_timeout=120,
    )
    record(
        "run_preprocessor",
        ok=True,
        elapsed_s=round(time.monotonic() - t0, 1),
        returned_keys=sorted(result.keys()) if isinstance(result, dict) else None,
    )
    (odir / "preprocess_result.json").write_text(
        json.dumps(result, indent=2, default=str)
    )
except Exception as e:
    record(
        "run_preprocessor",
        ok=False,
        elapsed_s=round(time.monotonic() - t0, 1),
        error=str(e),
        traceback=traceback.format_exc()[-4000:],
    )
    (odir / "granular_probe.json").write_text(json.dumps(captures, indent=2, default=str))
    sys.exit(4)

# Stage 3: discovery
disc = (result or {{}}).get("discovery") or {{}}
record(
    "discovery",
    tests_count=len(disc.get("tests") or []),
    benchmarks_count=len(disc.get("benchmarks") or []),
    kernel_type=(disc.get("kernel") or {{}}).get("type"),
    focused_test_present=bool(disc.get("focused_test")),
    codebase_context_path=result.get("codebase_context_path"),
)

# Stage 4: harness
harness_path = result.get("harness_path")
test_command = result.get("test_command")
harness_contents_ok = False
has_flags = has_markers = False
if harness_path and Path(harness_path).is_file():
    htext = Path(harness_path).read_text(errors='ignore')
    has_flags = all(f in htext for f in ('--correctness', '--benchmark', '--full-benchmark', '--profile'))
    has_markers = all(m in htext for m in ('GEAK_RESULT_LATENCY_MS', 'GEAK_RESULT_SPEEDUP'))
    harness_contents_ok = has_flags or has_markers

record(
    "harness",
    harness_path=harness_path,
    harness_size_bytes=Path(harness_path).stat().st_size if harness_path and Path(harness_path).is_file() else None,
    test_command=test_command,
    has_contract_flags=has_flags,
    has_contract_markers=has_markers,
    contents_pass=harness_contents_ok,
    selected_source=(result.get("testcase_selection") or {{}}).get("selected_source"),
)

# Stage 5: contract validation
try:
    from minisweagent.kernel_languages.contract import validate_harness
    validate_harness(Path(harness_path)) if harness_path else None
    record("contract_validate_harness", ok=True)
except ImportError:
    record("contract_validate_harness", ok=None, reason="module_not_present")
except Exception as e:
    record("contract_validate_harness", ok=False, error=str(e)[:300])

# Stage 6: baseline metrics
bm = result.get("baseline_metrics")
if isinstance(bm, dict):
    record(
        "baseline_metrics",
        duration_us=bm.get("duration_us"),
        bottleneck=bm.get("bottleneck"),
        has_metrics=bool(bm.get("metrics")),
        top_kernels_count=len(bm.get("top_kernels") or []),
    )
else:
    record("baseline_metrics", present=False, raw_type=type(bm).__name__)

# Stage 7: commandment
cm_path = result.get("commandment_path")
cm_contents = None
cm_sections = []
if cm_path and Path(cm_path).is_file():
    cm_contents = Path(cm_path).read_text(errors='ignore')
    cm_sections = [s for s in ('## Setup', '## Correctness', '## Benchmark', '## Full Benchmark', '## Profile') if s in cm_contents]
record(
    "commandment",
    commandment_path=cm_path,
    commandment_size_bytes=len(cm_contents) if cm_contents else 0,
    sections_present=cm_sections,
    sections_missing=[s for s in ('## Setup', '## Correctness', '## Benchmark', '## Full Benchmark', '## Profile') if s not in (cm_contents or '')],
)

# Stage 8: profile
prof = result.get("profiling")
profile_json = odir / "profile.json"
record(
    "profile",
    profile_present=bool(prof),
    profile_json_exists=profile_json.exists(),
    profile_json_size=profile_json.stat().st_size if profile_json.exists() else 0,
)

(odir / "granular_probe.json").write_text(json.dumps(captures, indent=2, default=str))
print("GRANULAR_PROBE_OK")
"""


def run_probe_in_container(
    *,
    pipeline: str,
    kernel_path: Path,
    output_dir: Path,
    container: str = "geak_agent",
    kernel_analysis_on: bool = False,
    harness_only: bool = True,
) -> dict:
    repo_root = REPO_ROOTS[pipeline]
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write the driver to the output dir so it's introspectable after.
    driver_path = output_dir / "driver.py"
    driver_path.write_text(
        _build_driver_script(
            kernel_path=kernel_path,
            output_dir=output_dir,
            pipeline_label=pipeline,
            kernel_analysis_on=kernel_analysis_on,
        )
    )

    env_flags = {
        "AMD_LLM_API_KEY": os.environ["AMD_LLM_API_KEY"],
        "GEAK_USE_KERNEL_ANALYSIS": "1" if kernel_analysis_on else "0",
        "GEAK_USE_KNOWLEDGE_BASE": "0",
        "GEAK_SAVE_TO_KNOWLEDGE_BASE": "0",
    }
    if harness_only:
        env_flags["GEAK_HARNESS_ONLY"] = "1"
    env_str = " ".join(f"{k}={v}" for k, v in env_flags.items())

    stderr_path = output_dir / "stderr.log"
    cmd = [
        "docker", "exec", container,
        "bash", "-c",
        f"cd {repo_root} && "
        f"pip install -q -e . 2>&1 | tail -3 && "
        f"{env_str} python {driver_path} 2>{stderr_path}",
    ]

    t0 = time.monotonic()
    completed = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    elapsed = time.monotonic() - t0

    run_record = {
        "pipeline": pipeline,
        "kernel": str(kernel_path),
        "output_dir": str(output_dir),
        "returncode": completed.returncode,
        "elapsed_s": round(elapsed, 1),
        "stdout_tail": (completed.stdout or "")[-2000:],
        "stderr_tail": (completed.stderr or "")[-2000:],
    }

    probe_path = output_dir / "granular_probe.json"
    if probe_path.exists():
        run_record["captures"] = json.loads(probe_path.read_text())
    else:
        run_record["captures"] = None

    return run_record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("pipeline", choices=list(REPO_ROOTS.keys()))
    parser.add_argument("kernel_path", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--kernel-analysis-on", action="store_true")
    parser.add_argument("--no-harness-only", action="store_true")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    if args.clean and args.output_dir.exists():
        shutil.rmtree(args.output_dir)

    record = run_probe_in_container(
        pipeline=args.pipeline,
        kernel_path=args.kernel_path,
        output_dir=args.output_dir,
        kernel_analysis_on=args.kernel_analysis_on,
        harness_only=not args.no_harness_only,
    )
    Path(args.output_dir / "run_record.json").write_text(
        json.dumps(record, indent=2, default=str)
    )
    print(json.dumps(record, indent=2, default=str))
    return 0 if record["returncode"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
