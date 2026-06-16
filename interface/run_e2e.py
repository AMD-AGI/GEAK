#!/usr/bin/env python3
"""PerfSkills/GEAK e2e runner — the ONLY entry point Hyperloom (or any external
orchestrator) calls.

Contract (stable, see interface/run_e2e.md):

    run_e2e.py <handoff.json> <result.json> [--dry-run]

* Reads ``handoff.json``  (external orchestrator -> e2e workflow).
* Maps the stable handoff fields onto ``e2e_workflow/e2e_workflow.js``
  args (this mapping is the ONLY thing that changes when the JS workflow's args
  evolve; the handoff/result JSON contract stays put).
* Invokes the JS workflow through the Claude Code ``Workflow`` tool (the JS
  workflow CANNOT be run with ``node`` directly — it needs the agent runtime's
  Workflow/agent/parallel/phase primitives, which are only exposed under
  ``--effort ultracode``). Prefers the Python ``claude_agent_sdk``; falls back
  to the ``claude -p`` CLI.
* Normalizes the workflow artifacts (``director_e2e_validation.json`` +
  ``baseline/bench_summary.json`` + ``final/``) into the stable ``result.json``.

All Claude-SDK / ``--effort`` / args-mapping detail lives HERE, inside this
repo, so the external caller only deals with two JSON files + one command
path. See interface/run_e2e.md for the full contract.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

SCHEMA_VERSION = 1

# interface/ is a sibling of e2e_workflow/ under the repo root.
INTERFACE_DIR = Path(__file__).resolve().parent
PERFSKILLS_ROOT = INTERFACE_DIR.parent
E2E_DIR = PERFSKILLS_ROOT / "e2e_workflow"
E2E_SCRIPT = E2E_DIR / "e2e_workflow.js"
BENCH_SCRIPT = E2E_DIR / "scripts" / "bench_e2e.sh"

# Workflow primitives are only available at this effort tier (see README).
CLAUDE_EFFORT = os.environ.get("PERFSKILLS_CLAUDE_EFFORT", "ultracode")
CLAUDE_MODEL = os.environ.get("PERFSKILLS_CLAUDE_MODEL", "claude-opus-4-8")
ALLOWED_TOOLS = ["Workflow", "Bash", "Read", "Write"]


# ---------------------------------------------------------------------------
# handoff (stable)  ->  e2e_workflow.js args (volatile, owned here)
# ---------------------------------------------------------------------------
def map_args(h: dict) -> dict:
    workload = h.get("workload") or {}
    tp = int(h.get("tp", 1) or 1)
    # gpu_ids is the optimization-parallelism pool AND the serving device set.
    # Default to 0..tp-1 so serving honours the requested tensor-parallel size.
    gpu_ids = h.get("gpu_ids") or ",".join(str(i) for i in range(max(tp, 1)))
    ps_args = {
        "model_path": h["model_path"],
        "workflow_dir": str(E2E_DIR),
        "backend": h.get("framework", "sglang"),
        "tp": tp,
        "gpu_ids": str(gpu_ids),
        "isl": int(workload.get("isl", 1024)),
        "osl": int(workload.get("osl", 1024)),
        "conc": int(workload.get("conc", 64)),
        # Seed the baseline with Hyperloom's accepted best config so the
        # baseline == Hyperloom best config (fair engagement start).
        "initial_extra_server_args": h.get("accepted_flags", "") or "",
        "initial_extra_env": h.get("accepted_env", "") or "",
        # Hyperloom already did config/param search in EXPLORE; do not double-run.
        "config_tune": "false",
        # Produce the final/ bundle (final_launch.sh + overlay) so the caller can
        # reuse it for a workload sweep.
        "apply_to_original": "true",
        "exp_root": h["exp_root"],
    }
    if h.get("launch_recipe"):
        ps_args["launch_script"] = h["launch_recipe"]
    return ps_args


def build_prompt(ps_args: dict) -> str:
    return (
        "Invoke the Workflow tool exactly once with:\n"
        f'  scriptPath: "{E2E_SCRIPT}"\n'
        f"  args: {json.dumps(ps_args)}\n"
        "Run the full e2e pipeline (Setup -> Profile -> Strategize -> "
        "HeadKernel -> Milestone -> Finalize -> Report -> Validate). When it "
        "finishes, print EXACTLY ONE final line of compact JSON that is the "
        "Workflow tool's full return value (it includes eval_dir, "
        "baseline_throughput_tok_s, final_throughput_tok_s, throughput_speedup, "
        "validation_status, output_parity, final_overlay, final_launch_script, "
        "report_path, accepted_kernels, accepted_config). Print nothing after it."
    )


# ---------------------------------------------------------------------------
# Bench-client口径 alignment.
# ---------------------------------------------------------------------------
def apply_bench_client(h: dict) -> str:
    """Decide + export the bench CLIENT so workflow bench_e2e.sh calls inherit it.

    handoff.bench_client: "auto" (default) | "inferencex" | "native".
    "auto" => use InferenceX's benchmark_serving.py (口径-identical to the
    caller's Magpie harness) when an InferenceX checkout is discoverable, else
    fall back to each backend's native client. The value is exported into the
    environment so every ``bench_e2e.sh`` invocation the agents make inherits it.
    """
    requested = str(h.get("bench_client", "auto") or "auto").strip().lower()
    ix_path = str(h.get("inferencex_path") or os.environ.get("INFERENCEX_PATH", "")).strip()
    if ix_path:
        os.environ["INFERENCEX_PATH"] = ix_path
    if requested == "auto":
        client = "inferencex" if ix_path else "native"
    else:
        client = requested
    if client == "inferencex" and not ix_path:
        sys.stderr.write(
            "bench_client=inferencex requested but no INFERENCEX_PATH; "
            "falling back to native client (口径 NOT aligned).\n"
        )
        client = "native"
    os.environ["BENCH_CLIENT"] = client
    return client


# ---------------------------------------------------------------------------
# Invocation: SDK preferred, CLI fallback.
# ---------------------------------------------------------------------------
def _invoke_via_sdk(prompt: str, timeout_s: int) -> str:
    import anyio
    from claude_agent_sdk import ClaudeAgentOptions, query

    async def _run() -> str:
        opts = ClaudeAgentOptions(
            model=CLAUDE_MODEL,
            allowed_tools=ALLOWED_TOOLS,
            permission_mode="bypassPermissions",
            extra_args={"effort": CLAUDE_EFFORT},
            cwd=str(E2E_DIR),
        )
        last = ""
        async for msg in query(prompt=prompt, options=opts):
            text = getattr(msg, "text", None)
            if getattr(msg, "type", "") == "assistant" and text:
                last = text
        return last

    return anyio.run(_run)


def _invoke_via_cli(prompt: str, timeout_s: int) -> str:
    claude = shutil.which("claude") or os.environ.get("CLAUDE_BIN", "claude")
    cmd = [
        claude, "-p", prompt,
        "--output-format", "json",
        "--effort", CLAUDE_EFFORT,
        "--model", CLAUDE_MODEL,
        "--allowed-tools", ",".join(ALLOWED_TOOLS),
        "--permission-mode", "auto",
    ]
    env = dict(os.environ, IS_SANDBOX="1")
    proc = subprocess.run(
        cmd, cwd=str(E2E_DIR), env=env, capture_output=True, text=True,
        timeout=timeout_s,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"claude CLI failed (rc={proc.returncode}): {proc.stderr[-2000:]}"
        )
    # claude -p --output-format json wraps the assistant text; try to unwrap.
    out = proc.stdout.strip()
    try:
        wrapped = json.loads(out)
        if isinstance(wrapped, dict):
            return str(wrapped.get("result") or wrapped.get("text") or out)
    except json.JSONDecodeError:
        pass
    return out


def invoke_workflow(prompt: str, timeout_s: int) -> dict:
    """Run the JS workflow and return its parsed JSON return value."""
    try:
        import claude_agent_sdk  # noqa: F401
        raw = _invoke_via_sdk(prompt, timeout_s)
    except ImportError:
        raw = _invoke_via_cli(prompt, timeout_s)
    return _parse_last_json_line(raw)


def _parse_last_json_line(raw: str) -> dict:
    for line in reversed([ln for ln in (raw or "").splitlines() if ln.strip()]):
        try:
            obj = json.loads(line.strip())
            if isinstance(obj, dict) and obj.get("eval_dir"):
                return obj
        except json.JSONDecodeError:
            continue
    raise RuntimeError(
        "Could not parse a JSON workflow return (with eval_dir) from the agent "
        f"output. Last 2000 chars:\n{(raw or '')[-2000:]}"
    )


# ---------------------------------------------------------------------------
# Normalize workflow artifacts -> stable result.json
# ---------------------------------------------------------------------------
def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def normalize_result(h: dict, wf: dict) -> dict:
    eval_dir = Path(wf["eval_dir"])
    validation = _read_json(eval_dir / "director_e2e_validation.json")
    baseline_summary = _read_json(eval_dir / "baseline" / "bench_summary.json")
    final_summary = _read_json(eval_dir / "validation" / "final" / "bench_summary.json")

    speedup = float(wf.get("throughput_speedup") or validation.get("throughput_speedup") or 1.0)
    status = "ok" if speedup > 1.0 else "no_gain"

    final_launch = (
        wf.get("final_launch_script")
        or validation.get("final_launch_script")
        or str(eval_dir / "final" / "final_launch.sh")
    )
    workload = h.get("workload") or {"isl": 1024, "osl": 1024, "conc": 64}

    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "eval_dir": str(eval_dir),
        "baseline_throughput_tok_s": float(
            wf.get("baseline_throughput_tok_s")
            or validation.get("baseline_throughput_tok_s")
            or 0.0
        ),
        "final_throughput_tok_s": float(
            wf.get("final_throughput_tok_s")
            or validation.get("director_verified_throughput_tok_s")
            or 0.0
        ),
        "throughput_speedup": speedup,
        "output_parity": wf.get("output_parity") or validation.get("output_parity") or "unknown",
        # Latency口径 (median ms), aligned field names with Hyperloom.
        "ttft_ms": final_summary.get("ttft_ms_median") or baseline_summary.get("ttft_ms_median"),
        "tpot_ms": final_summary.get("tpot_ms_median") or baseline_summary.get("tpot_ms_median"),
        # Sweep-reuse handles (see interface/run_e2e.md).
        "final_launch_script": final_launch,
        "bench_script": str(eval_dir / "bench_e2e.sh"),
        "final_patch": str(eval_dir / "final" / "final_patch.diff"),
        "final_overlay": wf.get("final_overlay") or str(eval_dir / "final" / "overlay"),
        # Measurement basis: reports aggregate output tok/s (not per-GPU),
        # matching Hyperloom's Magpie output_throughput. See run_e2e.md alignment table.
        "metric_basis": "aggregate_output_tok_s",
        # Which bench client measured these numbers. "inferencex" => identical
        # client to Hyperloom/Magpie (benchmark_serving.py); "native" => the
        # backend's own client (small cross-harness差异 may remain).
        "bench_client": os.environ.get("BENCH_CLIENT", "native"),
        # The kernels are only extracted/validated at this single workload point;
        # the caller must redo parity on out-of-regime sweep points.
        "validated_regimes": [workload],
        # What the kernel phase actually did (req: report must carry this).
        "accepted_kernels": wf.get("accepted_kernels") or [],
        "accepted_heads": wf.get("accepted_heads") or [],
        "accepted_config": wf.get("accepted_config") or {},
        "report_path": wf.get("report_path") or str(eval_dir / "final_report.md"),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: list[str]) -> int:
    args = [a for a in argv if not a.startswith("--")]
    flags = {a for a in argv if a.startswith("--")}
    if len(args) < 2:
        sys.stderr.write(
            "usage: run_e2e.py <handoff.json> <result.json> [--dry-run]\n"
        )
        return 2
    handoff_path, result_path = Path(args[0]), Path(args[1])
    timeout_s = int(os.environ.get("PERFSKILLS_E2E_TIMEOUT_S", "21600"))  # 6h

    h = _read_json(handoff_path)
    if not h:
        sys.stderr.write(f"empty/invalid handoff: {handoff_path}\n")
        return 2

    ps_args = map_args(h)
    bench_client = apply_bench_client(h)
    prompt = build_prompt(ps_args)

    if "--dry-run" in flags:
        print(json.dumps({"mapped_args": ps_args, "bench_client": bench_client,
                          "inferencex_path": os.environ.get("INFERENCEX_PATH", ""),
                          "prompt": prompt, "e2e_script": str(E2E_SCRIPT)}, indent=2))
        return 0

    try:
        wf = invoke_workflow(prompt, timeout_s)
        out = normalize_result(h, wf)
    except Exception as e:  # crash: status=error + nonzero exit (caller distinguishes)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(
            {"schema_version": SCHEMA_VERSION, "status": "error", "error": str(e)},
            indent=2), encoding="utf-8")
        sys.stderr.write(f"PerfSkills e2e failed: {e}\n")
        return 1

    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({"status": out["status"], "result_json": str(result_path),
                      "speedup": out["throughput_speedup"]}))
    return 0 if out["status"] != "error" else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
