#!/usr/bin/env python3
"""GEAK e2e runner — the ONLY entry point Hyperloom (or any external
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

import atexit
import glob
import json
import math
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 2
KERNEL_JOURNEY_SCHEMA_VERSION = 1

# result.json must never state a speedup its own baseline/final pair
# contradicts. Anything beyond this absolute gap on final/baseline means the
# ratio was computed against a different pair than the one we report.
SPEEDUP_SELF_CONSISTENCY_TOL = 1e-3

# Headroom over an op's theoretical Amdahl ceiling before a measured e2e delta
# is treated as corruption rather than a win. Mirrors
# IMPLAUSIBLE_SPEEDUP_MARGIN in e2e_workflow.js (1.0 => must exceed 2x the
# ceiling); both sides MUST agree or the recovery path banks what the live path
# refuses, so this is the SINGLE source of truth: map_args() forwards it to the
# JS workflow (A.implausible_speedup_margin) and the recovery path reads the same
# constant. Overridable via env for tuning; validated finite/non-negative (an
# accepted nan/inf/negative would silently disable the guard) exactly like
# SAME_CONFIG_DIVERGENCE_WARN_PCT below.
try:
    IMPLAUSIBLE_SPEEDUP_MARGIN = float(
        os.environ.get("GEAK_IMPLAUSIBLE_SPEEDUP_MARGIN", "1.0")
    )
    if (
        not math.isfinite(IMPLAUSIBLE_SPEEDUP_MARGIN)
        or IMPLAUSIBLE_SPEEDUP_MARGIN < 0.0
    ):
        raise ValueError("implausible-speedup margin must be finite and non-negative")
except (TypeError, ValueError):
    IMPLAUSIBLE_SPEEDUP_MARGIN = 1.0

try:
    SAME_CONFIG_DIVERGENCE_WARN_PCT = float(
        os.environ.get("GEAK_SAME_CONFIG_DIVERGENCE_WARN_PCT", "3.0")
    )
    if (
        not math.isfinite(SAME_CONFIG_DIVERGENCE_WARN_PCT)
        or SAME_CONFIG_DIVERGENCE_WARN_PCT < 0.0
    ):
        raise ValueError("warning threshold must be finite and non-negative")
except (TypeError, ValueError):
    SAME_CONFIG_DIVERGENCE_WARN_PCT = 3.0

BASELINE_ALIGNMENT_BEGIN = "<!-- GEAK_BASELINE_ALIGNMENT_BEGIN -->"
BASELINE_ALIGNMENT_END = "<!-- GEAK_BASELINE_ALIGNMENT_END -->"

# interface/ is a sibling of e2e_workflow/ under the repo root.
INTERFACE_DIR = Path(__file__).resolve().parent
GEAK_ROOT = INTERFACE_DIR.parent
E2E_DIR = GEAK_ROOT / "e2e_workflow"
E2E_SCRIPT = E2E_DIR / "e2e_workflow.js"
BENCH_SCRIPT = E2E_DIR / "scripts" / "bench_e2e.sh"

# Workflow primitives are only available at this effort tier (see README).
CLAUDE_EFFORT = os.environ.get("GEAK_CLAUDE_EFFORT", "ultracode")
CLAUDE_MODEL = os.environ.get("GEAK_CLAUDE_MODEL", "claude-opus-4-8")
# WebSearch/WebFetch are required by the Deep Research Agent (kernel_workflow's opt-in `Research`
# phase, args.dra_enabled=true): its per-question research agents do native web research. They are
# harmless when the DRA is off (nothing opts into them) — the reason v4 previously "had no websearch"
# is simply that no tool was on the allowlist. The allowlist is the union of tools any agent in the
# (possibly nested) Workflow session may call, so listing them here makes them available to the
# kernel_workflow research agents the e2e pipeline drives. (For a standalone `claude -p ... Workflow`
# invocation of kernel_workflow with dra_enabled, pass the same names via --allowed-tools.)
ALLOWED_TOOLS = ["Workflow", "Bash", "Read", "Write", "WebSearch", "WebFetch"]

# Public claude builds (>=2.1.x) REJECT "--effort ultracode". The Workflow /
# parallel / phase primitives that e2e_workflow.js needs are instead gated behind
# the `enableWorkflows` + `ultracode` settings keys (the highest-priority "flag
# settings" layer, == CLI `--settings`). Inject them so the Workflow tool truly
# executes the JS pipeline instead of the agent merely "backgrounding" it.
VALID_EFFORTS = {"low", "medium", "high", "xhigh", "max"}
WORKFLOW_SETTINGS = os.environ.get(
    "GEAK_CLAUDE_SETTINGS",
    json.dumps({"enableWorkflows": True, "ultracode": True}),
)
# Override which claude binary the SDK drives. The claude_agent_sdk otherwise
# prefers its OWN bundled CLI (claude_agent_sdk/_bundled/claude) over $PATH, so
# swapping the system claude alone has no effect on the SDK path. Set
# GEAK_CLAUDE_BIN to pin a specific build (e.g. an older native version).
CLAUDE_BIN = os.environ.get("GEAK_CLAUDE_BIN", "").strip()

# Background-task completion race (see _invoke_via_sdk completion gate):
# when the SDK turn "looks done" (a background task notified terminal + the
# main turn produced a ResultMessage) but the workflow has NOT yet written its
# authoritative on-disk terminal marker, the workflow may still be finishing a
# DETACHED leg (e.g. the integrate A/B reference/candidate bench). Tearing the
# runner down here orphans that leg and discards a still-completing measurement.
# Instead we keep the persistent SDK client open (which keeps the CLI + the
# backgrounded workflow alive) and poll the disk for the terminal marker for a
# BOUNDED grace window. The outer anyio.fail_after(timeout_s) is the ultimate
# backstop, so this can never exceed the run's hard budget.
DONE_GRACE_S = float(os.environ.get("GEAK_DONE_GRACE_S", "1800"))
DONE_POLL_S = float(os.environ.get("GEAK_DONE_POLL_S", "15"))


# ---------------------------------------------------------------------------
# Serving-launch FIDELITY: backend-agnostic knob -> per-adapter CLI flag map.
# ---------------------------------------------------------------------------
# Each serving adapter (scripts/adapters/<backend>.sh) names the same physical
# knob differently (max context window, GPU-memory headroom). This map lets ONE
# generic fold translate the handoff's structured fidelity knobs into whatever
# the CURRENT backend expects — so a new backend is a one-line map entry, never a
# case-by-case patch. A knob whose backend has no mapping is left to the adapter
# default (we never guess a flag name for an unknown stack).
_SERVING_FIDELITY_FLAGS: dict[str, dict[str, str]] = {
    "vllm": {"max_model_len": "--max-model-len", "mem_fraction": "--gpu-memory-utilization"},
    "sglang": {"max_model_len": "--context-length", "mem_fraction": "--mem-fraction-static"},
}


def _flag_present(server_args: str, flag: str) -> bool:
    """True when ``flag`` already appears in a server-args string.

    Matches both the ``--flag value`` and ``--flag=value`` forms so an explicit
    caller choice is never silently duplicated/overridden by the fidelity fold.

    Args:
        server_args: The server-args string to scan.
        flag: The flag to look for, INCLUDING leading dashes (e.g. ``--max-model-len``).

    Returns:
        Whether the flag is already present.
    """
    if not server_args or not flag:
        return False
    try:
        toks = shlex.split(server_args)
    except ValueError:
        toks = server_args.split()
    prefix = flag + "="
    return any(t == flag or t.startswith(prefix) for t in toks)


def _fold_serving_fidelity_flags(
    server_args: str,
    *,
    backend: str,
    max_model_len: int = 0,
    mem_fraction: float = 0.0,
) -> str:
    """Fold serving-fidelity knobs into a server-args string as backend flags.

    The e2e workflow applies ``initial_extra_server_args`` (JS ``INIT_FLAGS`` ->
    ``curFlags``) to EVERY serving launch — baseline, config sweep, integrate
    ref/cand, and validation — so folding the orchestrator's max-model-len /
    gpu-mem-util here makes GEAK launch the IDENTICAL vLLM/sglang engine that
    Hyperloom measured, WITHOUT the JS or the adapters needing a per-knob change
    (see #805: a slower default stack silently eats the kernel win e2e). Generic
    and non-destructive:

      * translates each knob to the CURRENT backend's flag via
        ``_SERVING_FIDELITY_FLAGS`` (unknown backend => returned untouched),
      * NEVER overrides a flag the caller already set (explicit config wins),
      * appends nothing when a knob is unset => byte-identical to the input.

    Args:
        server_args: The seed server-args string (Hyperloom accepted_flags).
        backend: The serving backend ("vllm" | "sglang" | ...).
        max_model_len: Resolved max-model-len (<=0 => omitted).
        mem_fraction: Resolved gpu-memory-utilization / mem-fraction (<=0 => omitted).

    Returns:
        The server-args string with the resolved, non-duplicate knobs appended.
    """
    fmap = _SERVING_FIDELITY_FLAGS.get(str(backend or "").strip().lower())
    if not fmap:
        return str(server_args or "")
    out = str(server_args or "").strip()

    pending: list[tuple[str, str]] = []
    try:
        mml = int(max_model_len or 0)
    except (TypeError, ValueError):
        mml = 0
    if mml > 0 and fmap.get("max_model_len"):
        pending.append((fmap["max_model_len"], str(mml)))
    try:
        mem = float(mem_fraction or 0.0)
    except (TypeError, ValueError):
        mem = 0.0
    if mem > 0 and fmap.get("mem_fraction"):
        pending.append((fmap["mem_fraction"], f"{mem:g}"))

    for flag, val in pending:
        if not _flag_present(out, flag):
            out = (out + " " + flag + " " + val).strip()
    return out


# ---------------------------------------------------------------------------
# handoff (stable)  ->  e2e_workflow.js args (volatile, owned here)
# ---------------------------------------------------------------------------
def map_args(h: dict, timeout_s: int | None = None) -> dict:
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
    # Forward the orchestrator's HARD wall-clock budget (the same timeout_s this
    # runner enforces via anyio.fail_after / subprocess timeout) so the JS
    # workflow can self-pace and FINISH (Finalize/Report/Validate + workflow_return
    # flush) BEFORE the SIGKILL, instead of being torn down mid-flight (the deep
    # 24h-budget-vs-real-kill failure). The workflow treats this as the single
    # source of its wall-clock budget and carves its own safety tail; we only
    # forward the truth. Omitted when timeout_s is unknown => workflow stays
    # budget-unaware (byte-identical to a direct, non-interface invocation).
    if timeout_s is not None and timeout_s > 0:
        ps_args["time_budget_s"] = int(timeout_s)
    # How that budget is split at the end. Defaults live in the JS (60min reserve
    # capped at 20% of the budget, 10min re-curate floor); these let an operator
    # widen the reserve per run without editing either file -- e.g.
    # GEAK_FINAL_RESERVE_S=5400 for 90min on a model known to write slowly.
    for env_key, arg_key in (("GEAK_FINAL_RESERVE_S", "final_reserve_s"),
                             ("GEAK_FINAL_RECURATE_MIN_S", "final_recurate_min_s")):
        v = _int_or_none(os.environ.get(env_key), env_key)
        if v is not None:
            ps_args[arg_key] = v
    if h.get("launch_recipe"):
        ps_args["launch_script"] = h["launch_recipe"]
    # Serving-launch fidelity (see Hyperloom handoff builder / #805): forward the
    # SAME max-model-len / gpu-mem-util Hyperloom's baseline served with so GEAK's
    # baseline launches the IDENTICAL vLLM engine (else it re-baselines a slower
    # default stack and kernel deltas do not reproduce e2e). Only forwarded when
    # the handoff carried them; absent => the vllm adapter keeps its own defaults.
    try:
        _mml = int(h.get("max_model_len") or 0)
    except (TypeError, ValueError):
        _mml = 0
    if _mml > 0:
        ps_args["max_model_len"] = _mml
    try:
        _mem = float(h.get("mem_fraction") or 0.0)
    except (TypeError, ValueError):
        _mem = 0.0
    if _mem > 0:
        ps_args["mem_fraction"] = _mem
    # Close the loop: also fold the SAME knobs into the seed server-args so the
    # workflow APPLIES them on every serving launch through its existing
    # INIT_FLAGS -> curFlags channel. The standalone keys above are advisory
    # metadata; these flags are what the adapters actually launch with. Backend
    # translation + dedup live in _fold_serving_fidelity_flags (generic; a new
    # backend is one map entry). No knobs / unknown backend => unchanged.
    ps_args["initial_extra_server_args"] = _fold_serving_fidelity_flags(
        ps_args["initial_extra_server_args"],
        backend=str(ps_args.get("backend") or ""),
        max_model_len=_mml,
        mem_fraction=_mem,
    )
    # Optional phase scoping / resume. Pass-through of the workflow's own
    # phase-by-phase driving (args.phases): e.g. "final" re-enters only the
    # Finalize gate against a pinned eval_dir, which (with the disk-reconstruct +
    # finish-all-pending logic) drives every incomplete A/B on disk to a complete
    # ref+cand measurement WITHOUT re-running Setup/Profile/Kernel. General: any
    # subset of {setup,profile,config,head,kernel,final} (default unset => "all").
    if h.get("phases"):
        ps_args["phases"] = str(h["phases"])
    # Optional A/B repeat count override (bounds the cost of a resume / finalize
    # A/B — e.g. 1 repeat per leg is enough to PROVE both legs ran). General.
    if h.get("e2e_repeats") is not None:
        ps_args["e2e_repeats"] = int(h["e2e_repeats"])
    # Carried cross-phase state (the prior workflow return's `state`), so a
    # resume continues from where a previous phase invocation left off.
    if h.get("state"):
        ps_args["state"] = h["state"]
    # Pin ONE EVAL_DIR for the whole run (workflow reads A.eval_dir ->
    # EVAL_DIR_OVERRIDE). Without it, every PHASE=setup invocation mints a fresh
    # timestamped dir, so a re-entered setup leaves an abandoned preflight-only
    # scaffold beside the authoritative run. Honor an explicit handoff/env
    # override first (resume); otherwise mint a single fresh dir here so BOTH
    # the preflight smoke and the real baseline/profile/kernel land under it.
    eval_dir = str(h.get("eval_dir") or os.environ.get("GEAK_EVAL_DIR", "")).strip()
    if not eval_dir:
        model_name = Path(h["model_path"]).name
        # Second-resolution names collide when two dry-runs/jobs start together,
        # causing their recipe_env.nul files and artifacts to overwrite each
        # other. Keep the readable UTC stamp but add microseconds + random run id.
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        run_id = uuid.uuid4().hex[:8]
        eval_dir = str(Path(h["exp_root"]) / f"e2e_{model_name}_{ts}_{run_id}Z")
    ps_args["eval_dir"] = eval_dir
    # Keep the JS live-path implausible-speedup guard and THIS runner's recovery
    # path on ONE margin: forward the (validated) Python value so the workflow's
    # A.implausible_speedup_margin can never silently drift from the constant the
    # recovery path applies. See IMPLAUSIBLE_SPEEDUP_MARGIN above. Forwarded
    # unconditionally (not gated on a handoff key) so both sides always agree.
    ps_args["implausible_speedup_margin"] = IMPLAUSIBLE_SPEEDUP_MARGIN
    # Bridge the upstream TraceLens / kernel-agent artifacts INTO the workflow args
    # (not just the driver prompt) so the JS Profile/Strategize/Extract phases can
    # use them as a prior. Only non-null paths are forwarded; when nothing is found
    # the key is omitted entirely, so a tracelens-less run is byte-identical.
    tl = resolve_tracelens_report(h.get("exp_root", ""))
    tl_paths = {k: v for k, v in tl.items() if k != "search_root" and v}
    if tl_paths:
        ps_args["tracelens"] = tl_paths
    return ps_args


# ---------------------------------------------------------------------------
# TraceLens / kernel-agent artifact discovery.
# ---------------------------------------------------------------------------
# The four artifacts live ABOVE the handoff's ``geak`` directory, under
# the experiment root (the parent of ``geak``). ``**`` denotes one or
# more randomly-named nested directories, so the lookup is glob based
# (recursive) and stays generic across runs.
_TRACELENS_ARTIFACT_PATTERNS = {
    "analysis_md": "kernel-agent/**/tracelens/analysis.md",
    "kernel_candidates_json": "kernel-agent/**/kernel_candidates.json",
    "tracelens_report_json": "kernel-agent/**/tracelens/tracelens_report.json",
    "trace_file": "runs/roofline/**/torch_trace",
}


def _experiment_root_from_exp_root(exp_root: str) -> str:
    """Return the experiment root (the directory that CONTAINS ``geak``).

    ``handoff.exp_root`` points at ``<experiment_root>/geak`` so the four
    TraceLens artifacts live one level up, beside ``geak``.
    """
    norm = str(exp_root or "").rstrip("/")
    if os.path.basename(norm) == "geak":
        return os.path.dirname(norm)
    return norm


def _find_latest_artifact(root: str, pattern: str) -> str | None:
    """Return the latest match for ``pattern`` under ``root`` (or None).

    Matches are sorted for determinism; the timestamps embedded in the run
    directory names sort chronologically, so the last entry is the most recent.
    """
    matches = sorted(glob.glob(os.path.join(root, pattern), recursive=True))
    return matches[-1] if matches else None


def resolve_tracelens_report(exp_root: str) -> dict:
    """Resolve the four TraceLens artifacts beside the handoff's ``geak``.

    Returns a dict with ``search_root`` plus the four artifact paths
    (``analysis_md``, ``kernel_candidates_json``, ``tracelens_report_json``,
    ``trace_file``); any artifact that cannot be located is ``None``.
    """
    root = _experiment_root_from_exp_root(exp_root)
    report: dict = {"search_root": root}
    for key, pattern in _TRACELENS_ARTIFACT_PATTERNS.items():
        report[key] = _find_latest_artifact(root, pattern) if root else None
    return report


# The e2e workflow prepends this to every role agent it spawns (PROCESS_SAFETY in
# e2e_workflow.js). The TOP-LEVEL driver agent we spawn here was the one agent that
# never saw it, yet it is the one closest to the caller: its Bash tool runs with
# bypassPermissions as a direct child of this process, so a single `pkill -f vllm`
# from it reproduces issue #397 (a pattern kill that reaches the caller's orchestrator)
# with nothing in between. Same rule, same wording, one level up.
PROCESS_SAFETY = (
    "## PROCESS SAFETY (a violation can kill the caller's orchestrator, failing the "
    "whole task)\n"
    "This container's PID 1 is the CALLER's orchestrator process, not yours, and it is "
    "NOT restartable. NEVER run global or pattern-matched process cleanup: no "
    "`pkill -f` / `pgrep -f ... | xargs kill` / `killall` / `ps aux | grep ... | xargs "
    "kill`, and never `kill -- -PGID` for a group you did not create. A pattern as "
    "innocent as `-f vllm` matches the orchestrator's own command line and TERMs it.\n"
    "Manage ONLY processes you started, by the pid you captured at launch. Freeing a "
    "GPU, unwedging a port, or recovering from a failed Workflow call is NOT an "
    "exception to this rule: leave the process alone and report it instead.\n\n"
)


def build_prompt(ps_args: dict) -> str:
    eval_dir = ps_args.get("eval_dir", "")
    # Locate the upstream TraceLens / kernel-agent artifacts (analysis.md,
    # kernel_candidates.json, tracelens_report.json) plus the roofline torch
    # trace, and surface them to the agent as a single tracelens_report block.
    tracelens_report = resolve_tracelens_report(ps_args.get("exp_root", ""))
    # The prompt only needs the four artifact paths, not the internal search_root.
    tracelens_prompt_payload = {
        k: v for k, v in tracelens_report.items() if k != "search_root"
    }
    tracelens_block = (
        "\n\ntracelens_report (upstream kernel-agent / roofline artifacts; "
        "any path is null when that artifact was not produced):\n"
        f"  {json.dumps(tracelens_prompt_payload)}\n"
    )
    # NOTE: the wall-clock budget is NOT surfaced in this prompt. The top driver
    # agent only invokes the Workflow tool once and waits, so it never acts on the
    # budget; enforcement lives entirely in the JS (the time_budget_s arg drives the
    # setTimeout deadlines), and the value is already passed via args.time_budget_s.
    return (
        PROCESS_SAFETY
        + "Invoke the Workflow tool exactly once with:\n"
        f'  scriptPath: "{E2E_SCRIPT}"\n'
        f"  args: {json.dumps(ps_args)}\n"
        "CRITICAL: pass `args` as a real JSON OBJECT (a mapping), NOT as a "
        "JSON-encoded string. Do not wrap it in quotes or call json.dumps on it. "
        "If args arrives as a string the workflow cannot read args.workflow_dir "
        "and aborts immediately.\n"
        "Run the full e2e pipeline (Setup -> Profile -> Strategize -> "
        "HeadKernel -> Milestone -> Finalize -> Report -> Validate). The workflow "
        f'persists its full return value to "{eval_dir}/workflow_return.json" as '
        "its final act; that file is the source of truth. When it finishes, print "
        "EXACTLY ONE final line of compact JSON that is the Workflow tool's full "
        "return value (it includes eval_dir, baseline_throughput_tok_s, "
        "final_throughput_tok_s, throughput_speedup, validation_status, "
        "output_parity, final_overlay, final_launch_script, report_path, "
        "accepted_kernels, accepted_config). If for ANY reason "
        f'"{eval_dir}/workflow_return.json" does not exist when the tool returns, '
        "write that exact return value there yourself with the Write tool before "
        "printing. Print nothing after the JSON line."
        + tracelens_block
    )


# ---------------------------------------------------------------------------
# Bench-client measurement-protocol alignment.
# ---------------------------------------------------------------------------
def apply_bench_client(h: dict) -> str:
    """Decide + export the bench CLIENT so workflow bench_e2e.sh calls inherit it.

    handoff.bench_client: "auto" (default) | "inferencex" | "native".
    "auto" => use InferenceX's benchmark_serving.py (measurement-protocol-identical to the
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
            "falling back to native client (measurement protocol NOT aligned).\n"
        )
        client = "native"
    os.environ["BENCH_CLIENT"] = client
    return client


# ---------------------------------------------------------------------------
# Server-launch RECIPE alignment (WHO launches the server, not the client).
# ---------------------------------------------------------------------------
# Backends for which Magpie ships a server-phase launch script (its scripts all
# share ONE contract, so a single backend-agnostic launcher adapter serves them
# all). Extend this set as Magpie adds backends — never add per-backend code.
_MAGPIE_BACKENDS = {"sglang", "vllm"}

# The flat scalars we need out of the orchestrator's launch recipe. Keep this
# lightweight scan separate from the BaseLoader parse used for the nested
# ``envs:`` mapping: these fields are optional launch-discovery hints, whereas
# malformed environment replay must follow the strict fail-closed path.
_RECIPE_KEYS = ("inferencex_path", "benchmark_script", "framework", "runner_type")

# Names the recipe may carry that GEAK must nevertheless own, because they
# address THIS run's resources rather than the served configuration. Replaying
# the orchestrator's values would not be fidelity -- it would point the server
# at a port the orchestrator released months ago, a GPU this run was not given,
# or another run's log file. Everything NOT listed here is replayed verbatim.
#
# The overlay is on the list for a different reason: applying it is the entire
# purpose of a GEAK run, so the orchestrator's PYTHONPATH must not displace it.
_RECIPE_ENV_GEAK_OWNED = frozenset({
    "PORT", "RESULT_DIR", "SERVER_LOG", "PROFILE", "PROFILE_DIR", "PYTHONPATH",
    "MAGPIE_RUN_PHASE", "MAGPIE_SERVER_PID_FILE", "BENCHMARK_BASE_URL",
    # What this run was asked to serve, not what the recipe happened to record.
    # The launcher passes both explicitly and so overrides them by ordering
    # anyway; naming them here is what keeps RECIPE_ENV_REPLAYED honest instead
    # of advertising a variable a later layer silently displaces.
    "MODEL", "TP",
    # GPU pinning: the recipe names whichever of these its own runner used, and
    # any of them left at the orchestrator's value would fight the device this
    # run was allocated. Filtering them from the RECIPE replay is independent of
    # how the launcher pins devices (ROCR-only on a bare box vs HIP-on-top when
    # an outer ROCR mask is already set — see magpie.sh).
    "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES",
    "VLLM_TORCH_PROFILER_DIR", "SGLANG_TORCH_PROFILER_DIR",
})


# A shell environment-variable name: a leading letter/underscore then word
# characters. Anything else (notably a leading-dash token like
# ``-SCUDA_VISIBLE_DEVICES`` that GNU ``env`` parses as the --split-string
# OPTION) is NOT an env var and must never reach the replay file.
_VALID_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class _RecipeYAMLError(Exception):
    """The launch recipe exists and PyYAML is available, but it does not parse.

    Distinct from "PyYAML absent": a genuine parse error must fail closed rather
    than be silently reinterpreted by the degraded line scanner.
    """


def _env_flag(name: str) -> bool:
    """Truthiness for an opt-in env switch, matching the spelling used elsewhere."""
    return str(os.environ.get(name, "")).strip().lower() in {"1", "true", "yes", "on"}


def _find_envs_map(node: Any) -> dict | None:
    """Locate the recipe's ``envs`` mapping in a parsed YAML document.

    The orchestrator emits ``envs:`` either nested under ``benchmark:`` (in the
    launch recipe) or at the document root (in the per-round effective config).
    Search depth-first and prefer the first NON-EMPTY mapping: an unrelated
    metadata ``envs: {}`` must not hide the later launch environment. Return an
    empty mapping only when at least one empty ``envs`` mapping exists and no
    non-empty mapping exists anywhere in the document.
    """
    if isinstance(node, dict):
        cand = node.get("envs")
        saw_empty = isinstance(cand, dict)
        if isinstance(cand, dict) and cand:
            return cand
        for value in node.values():
            found = _find_envs_map(value)
            if found:
                return found
            if found is not None:
                saw_empty = True
        return {} if saw_empty else None
    elif isinstance(node, list):
        saw_empty = False
        for item in node:
            found = _find_envs_map(item)
            if found:
                return found
            if found is not None:
                saw_empty = True
        return {} if saw_empty else None
    return None


def _stringify_env_value(value: Any) -> str | None:
    """Render one recipe env value as the string the shell must receive.

    Env vars are strings, but the recipe records typed YAML scalars (an int for
    ``MAX_MODEL_LEN``, a quoted string for a boolean). Non-scalars -- a nested
    map or a list -- are not environment variables and are dropped (``None``);
    ``None`` (a bare ``KEY:`` with no value under a typing loader such as
    ``safe_load``) is dropped for the same reason. Literal/folded block scalars
    arrive with a trailing newline the dumper added that carries no meaning once
    the value is word-split as args, so it is stripped.

    An EXPLICIT empty value (``KEY: ''``) is KEPT as the empty string: the
    orchestrator recorded ``KEY`` set-but-empty, which is distinct from unset,
    and dropping it would silently change the replayed environment. (Under the
    string-preserving ``BaseLoader`` the production path uses, a bare ``KEY:``
    also arrives as ``''`` rather than ``None``; keeping it errs toward fidelity
    -- replaying an empty var -- rather than dropping a value the recipe held.)
    """
    if value is None or isinstance(value, (dict, list)):
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return value.rstrip("\n")
    return str(value)


def _recipe_envs_from_yaml(text: str) -> dict[str, str] | None:
    """Parse the ``envs:`` map with a real YAML parser (PyYAML).

    Returns the extracted mapping on success (possibly empty when the recipe
    names no ``envs`` block), or ``None`` only when PyYAML is unavailable so a
    non-strict caller can fall back to the line scanner. Parse errors raise.

    A real parser is required because replay-warm recipes fold a multi-line
    ``EXTRA_<BE>_ARGS`` as an IMPLICIT plain scalar (no ``|``/``>`` indicator)
    whose continuation lines can carry JSON with colons, e.g.
    ``--speculative-config {"method":"ngram",...}``. A line-oriented scan cannot
    tell such a folded continuation from a new mapping key without
    reimplementing YAML, and silently truncates the kernel-dispatch flags.

    Returns ``None`` ONLY when PyYAML is unavailable (``ImportError``) so the
    caller degrades to the line scanner. When PyYAML IS present but the document
    does not parse, raises :class:`_RecipeYAMLError`: a malformed recipe must
    fail closed, not be handed to the degraded scanner (which would happily
    store a truncated flow scalar as if it were a valid environment).

    Parsed with ``BaseLoader`` so every scalar and key is read as its literal
    STRING, not YAML 1.1's implicit types -- ``yes``/``on`` stay ``"yes"``/``"on"``
    (not ``True``), ``0x10`` stays ``"0x10"`` (not ``16``), and a key ``ON`` stays
    ``"ON"`` (not the boolean ``True``). Environment variables are strings, so
    whether PyYAML is installed must not change the replayed value; BaseLoader is
    what keeps this path and the scanner fallback in agreement.
    """
    try:
        import yaml
    except ImportError:
        return None
    try:
        doc = yaml.load(text, Loader=yaml.BaseLoader)
    except yaml.YAMLError as exc:
        raise _RecipeYAMLError(str(exc)) from exc
    envs = _find_envs_map(doc)
    if envs is None:
        return {}
    out: dict[str, str] = {}
    for key, value in envs.items():
        rendered = _stringify_env_value(value)
        if rendered is not None:
            out[str(key)] = rendered
    return out


def _recipe_env_block_scan(text: str) -> dict[str, str]:
    """Indentation-scoped fallback used only when PyYAML is unavailable.

    Handles the same shapes as the YAML path -- single-line values, explicit
    ``|``/``>`` block scalars, nested-map skipping, and IMPLICIT plain-scalar
    continuation (more-indented lines folded into the value with spaces) -- but a
    hand scan cannot disambiguate every YAML edge, so it is strictly the degraded
    path taken only when the parser is missing.
    """
    envs: dict[str, str] = {}
    block_indent: int | None = None
    nested_indent: int | None = None
    lines = text.splitlines()
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        i += 1
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip())
        if block_indent is None:
            if line.strip() == "envs:":
                block_indent = indent
            continue
        if indent <= block_indent:
            break  # dedented out of the block
        if nested_indent is not None:
            if indent > nested_indent:
                continue  # still inside a nested map
            nested_indent = None
        key, sep, value = line.strip().partition(":")
        if not sep or not key:
            continue
        key = key.strip()
        value = value.strip()
        # Explicit YAML block scalar: `|` (literal) / `>` (folded), optionally
        # followed by chomping/indent indicators (+, -, digits). Consume the
        # more-indented block, joining folded scalars with spaces and literal
        # scalars with newlines (both word-split harmlessly as server args).
        if value and value[0] in "|>" and (
            len(value) == 1 or set(value[1:]) <= set("+-0123456789")
        ):
            folded = value[0] == ">"
            block_lines: list[str] = []
            content_indent: int | None = None
            while i < n:
                nxt = lines[i]
                if not nxt.strip():
                    block_lines.append("")
                    i += 1
                    continue
                nxt_indent = len(nxt) - len(nxt.lstrip())
                if content_indent is None:
                    if nxt_indent <= indent:
                        break  # nothing more-indented than the key => empty scalar
                    content_indent = nxt_indent
                elif nxt_indent < content_indent:
                    break  # dedented out of the scalar content
                block_lines.append(nxt[content_indent:])
                i += 1
            while block_lines and not block_lines[-1]:
                block_lines.pop()  # drop trailing padding the dumper may add
            joined = (" " if folded else "\n").join(block_lines)
            envs[key] = joined.strip() if folded else joined
            continue
        if not value:
            # BaseLoader represents a bare `KEY:` as an empty string unless the
            # following significant line is more indented (a nested map). Look
            # ahead so the fallback preserves that same set-but-empty semantic
            # while still skipping nested values and their children.
            j = i
            while j < n and (
                not lines[j].strip() or lines[j].lstrip().startswith("#")
            ):
                j += 1
            if j < n and (len(lines[j]) - len(lines[j].lstrip())) > indent:
                nested_indent = indent
            else:
                envs[key] = ""
            continue
        # Inline plain scalar. Fold any IMPLICIT continuation: lines indented
        # deeper than the key with no block indicator are part of this value
        # (YAML plain-scalar folding), e.g. a multi-line EXTRA_<BE>_ARGS. This is
        # unambiguous inside envs: a key with an inline value can never be
        # followed by a nested map, so a deeper line must be a continuation.
        while i < n:
            nxt = lines[i]
            if not nxt.strip():
                break  # a blank line ends the plain scalar
            if (len(nxt) - len(nxt.lstrip())) <= indent:
                break  # dedent -> next key or end of block
            value = f"{value} {nxt.strip()}"
            i += 1
        envs[key] = value.strip("'\"")
    return envs


def _recipe_env_block(recipe_path: str) -> dict[str, str]:
    """Read the orchestrator's recorded launch environment (the ``envs:`` map).

    This is the block :func:`_recipe_fields` skips. It is the only record of
    what the orchestrator's server actually inherited, and until it is replayed
    the two launches agree only where their DEFAULTS happen to agree -- which is
    agreement by coincidence, not by construction, and silently stops holding
    the moment the orchestrator sets something explicitly.

    Parsed with PyYAML (a declared dependency) so both shapes the orchestrator
    emits are handled -- ``envs:`` nested under ``benchmark:`` and ``envs:`` at
    column 0 -- and, critically, so an IMPLICIT plain-scalar continuation of a
    multi-line ``EXTRA_<BE>_ARGS`` survives intact (a line scan truncates it).
    Falls back to an indentation scan only if PyYAML is MISSING (ImportError).

    Fail-close semantics: a recipe that PyYAML rejects, or bytes that are not
    valid UTF-8, are NOT quietly handed to the degraded line scanner (which
    would store a truncated flow scalar as though it were a valid environment)
    -- they raise, and this function turns that into a hard stop under
    ``GEAK_STRICT_RECIPE_ENV`` or a warning + ``{}`` otherwise. Missing PyYAML is
    the ONLY condition that legitimately reaches the scanner.

    Returns ``{}`` when the file is unreadable or names no ``envs:`` block --
    callers distinguish "nothing recorded" from "could not read" via the recipe
    fields they already hold.
    """
    if not recipe_path:
        return {}
    # No errors="ignore": undecodable bytes mean a corrupt recipe, and silently
    # dropping them could split a flow scalar mid-token. Fail closed instead.
    try:
        text = Path(recipe_path).read_text(encoding="utf-8")
    except OSError:
        return {}
    except UnicodeDecodeError as exc:
        return _recipe_env_fail_closed(
            recipe_path, f"is not valid UTF-8 ({exc})"
        )
    try:
        parsed = _recipe_envs_from_yaml(text)
    except _RecipeYAMLError as exc:
        return _recipe_env_fail_closed(recipe_path, f"is not valid YAML ({exc})")
    if parsed is not None:
        return parsed
    # A strict aligned launch must be independently validated as YAML. The
    # scanner supports old/offline hosts in non-strict mode, but cannot prove a
    # malformed document safe, so strict mode refuses when PyYAML is unavailable.
    if _env_flag("GEAK_STRICT_RECIPE_ENV"):
        return _recipe_env_fail_closed(
            recipe_path,
            "cannot be validated because PyYAML is unavailable",
        )
    print(
        "[run_e2e] WARNING: PyYAML is unavailable; using the degraded recipe "
        "env scanner. Set GEAK_STRICT_RECIPE_ENV=1 to refuse.",
        file=sys.stderr,
    )
    return _recipe_env_block_scan(text)


def _recipe_env_fail_closed(recipe_path: str, why: str) -> dict[str, str]:
    """React to an unparseable recipe env block.

    Under ``GEAK_STRICT_RECIPE_ENV`` a malformed recipe is a hard stop: the
    launch env cannot be reconstructed, so serving a coincidentally-defaulted
    stack would be a silent fidelity break. Otherwise warn loudly and return an
    empty block so the caller proceeds on script defaults (the same posture as a
    recipe that records no ``envs:`` at all), never on partial garbage.
    """
    msg = f"recipe env block {recipe_path} {why}"
    if _env_flag("GEAK_STRICT_RECIPE_ENV"):
        raise SystemExit(
            f"[run_e2e] {msg}; refusing to launch under GEAK_STRICT_RECIPE_ENV.\n"
            "    Fix the recipe, or unset GEAK_STRICT_RECIPE_ENV to fall back to\n"
            "    script defaults with no recorded environment replayed."
        )
    print(
        f"[run_e2e] WARNING: {msg}; replaying NO recorded environment "
        "(script defaults only). Set GEAK_STRICT_RECIPE_ENV=1 to refuse.",
        file=sys.stderr,
    )
    return {}


def _sanitize_replay_path(path_value: str) -> tuple[str | None, list[str]]:
    """Keep only existing directories from a recorded ``PATH``.

    The orchestrator's recipe often records a host-local venv prefix
    (``/opt/venv/bin:...``). Replaying it verbatim on a box where that prefix
    is gone would put a dead entry first on ``PATH`` and silently select the
    wrong interpreter / miss the intended one. Drop missing components; if
    nothing remains, drop ``PATH`` from the replay entirely so the ambient
    process ``PATH`` stands.

    Returns ``(sanitized_or_None, dropped_components)``.
    """
    kept: list[str] = []
    dropped: list[str] = []
    for part in path_value.split(":"):
        if not part:
            continue
        if Path(part).is_dir():
            kept.append(part)
        else:
            dropped.append(part)
    return (":".join(kept) if kept else None), dropped


def _recipe_launch_env(h: dict) -> tuple[dict[str, str], list[str]]:
    """Partition the recipe's recorded environment into replay set + GEAK-owned.

    Returns ``(replay, owned)`` where ``replay`` is applied verbatim underneath
    the launch and ``owned`` names the variables GEAK deliberately overrode.
    Reporting ``owned`` rather than dropping it silently is what lets a reviewer
    see the complete list of ways this launch is allowed to differ.

    ``PATH`` is existence-checked before replay: missing directories are dropped
    (and the whole variable omitted when nothing remains).
    """
    recorded = _recipe_env_block(str(h.get("launch_recipe") or ""))
    # Every replayed key becomes an `env NAME=VALUE` operand at the shell. A name
    # that is not a valid POSIX identifier (spaces, `-`, a leading `-` that `env`
    # would read as an OPTION, an `=` in the name) cannot be a real environment
    # variable and could smuggle an option/command token onto the launch line.
    # Drop such names here so the shell only ever sees clean assignments; the
    # launcher's own allowlist is the second half of this defence in depth.
    bad = [k for k in recorded if not _VALID_ENV_NAME.match(str(k))]
    if bad:
        if _env_flag("GEAK_STRICT_RECIPE_ENV"):
            raise SystemExit(
                "[run_e2e] recipe env block records non-identifier name(s) "
                f"{bad!r}; refusing to launch under GEAK_STRICT_RECIPE_ENV."
            )
        print(
            ">>> recipe alignment: dropping recorded env name(s) that are not "
            f"valid identifiers: {bad!r}",
            file=sys.stderr,
        )
        recorded = {k: v for k, v in recorded.items() if _VALID_ENV_NAME.match(str(k))}
    replay = {k: v for k, v in recorded.items() if k not in _RECIPE_ENV_GEAK_OWNED}
    owned = sorted(k for k in recorded if k in _RECIPE_ENV_GEAK_OWNED)
    if "PATH" in replay:
        sanitized, dropped = _sanitize_replay_path(replay["PATH"])
        if dropped:
            print(
                ">>> recipe alignment: dropping missing PATH component(s) from "
                f"replay: [{':'.join(dropped)}]",
                file=sys.stderr,
            )
        if sanitized is None:
            print(
                ">>> recipe alignment: recorded PATH has no existing directories; "
                "leaving the ambient PATH in place.",
                file=sys.stderr,
            )
            del replay["PATH"]
        else:
            replay["PATH"] = sanitized
    return replay, owned


def _recipe_fields(recipe_path: str) -> dict[str, str]:
    """Read the flat scalars we need out of an orchestrator launch recipe.

    Returns ``{}`` for a missing/unreadable recipe so every caller degrades to
    the native launch instead of failing the run.
    """
    if not recipe_path:
        return {}
    try:
        text = Path(recipe_path).read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return {}
    fields: dict[str, str] = {}
    for line in text.splitlines():
        key, sep, value = line.strip().partition(":")
        # First occurrence wins; the nested env map uses UPPERCASE names and so
        # cannot collide with these keys.
        if sep and key in _RECIPE_KEYS and key not in fields:
            fields[key] = value.strip().strip("'\"")
    return fields


def _magpie_script_from_recipe(h: dict) -> str:
    """Rebuild the path of the launch script the orchestrator itself ran.

    This is the one piece of the launch recipe that never survived the handoff.
    The orchestrator transfers its accepted flags and env, but the script it
    launched with owns the platform kernel preset, ``--trust-remote-code`` and
    the gpu-mem-util default — none of which are flags, so none of which a
    flag-scraper can recover. Without the script GEAK re-baselines a different
    engine and the same configuration serves measurably slower.

    The handoff does name the recipe file, and the recipe names both the
    InferenceX checkout and the script inside it, so the path is derivable
    without asking the orchestrator to send anything new.

    The checkout the RECIPE names is host-local and content-hash addressed
    (``.../inferencex_local/<hash>``), so a container rebuild, a move to another
    box, or a post-mortem re-run invalidates that one path while the very same
    checkout is still on disk elsewhere. We therefore try, in order:

      1. the checkout the recipe names — the only one the orchestrator provably
         launched from, so whenever it resolves nothing else is consulted and
         the result is exactly what it was before this fallback existed;
      2. ``handoff.inferencex_path`` — same handoff, same writer, but usually
         on durable storage rather than in a container-local cache;
      3. ``$INFERENCEX_PATH``.

    2 and 3 are the same two sources :func:`apply_bench_client` already trusts
    to locate the bench client, so the launcher and the client now resolve
    against ONE checkout instead of disagreeing about which one exists. Every
    candidate passes the identical usability check below, and the first is
    still preferred, so a fallback can only turn "" into a working script — it
    can never displace a good one.

    Returns "" whenever no script can be confirmed usable on this box.
    """
    fields = _recipe_fields(str(h.get("launch_recipe") or ""))
    name = fields.get("benchmark_script", "")
    if not name:
        return ""
    seen: set[str] = set()
    for root in (
        fields.get("inferencex_path", ""),
        str(h.get("inferencex_path") or ""),
        os.environ.get("INFERENCEX_PATH", ""),
    ):
        root = root.strip()
        if not root or root in seen:
            continue
        seen.add(root)
        benchmarks = Path(root) / "benchmarks"
        # Mirror the orchestrator's own lookup: top level first, then
        # subdirectories (its checkouts keep some scripts under single_node/ /
        # multi_node/).
        candidates = [benchmarks / name]
        try:
            candidates.extend(sorted(benchmarks.rglob(name)))
        except OSError:
            pass
        for candidate in candidates:
            if not candidate.is_file():
                continue
            # The script sources benchmark_lib.sh from its own directory and
            # dies without it. Checking here turns a half-populated checkout
            # into a native-launch degrade instead of a hard failure at first
            # bench.
            if not (candidate.parent / "benchmark_lib.sh").is_file():
                continue
            return str(candidate)
    return ""


def apply_bench_launcher(h: dict) -> str:
    """Align the SERVER LAUNCH recipe with the external orchestrator (Magpie).

    A "completely-aligned" throughput number needs the SERVER launched the SAME
    way the orchestrator's baseline was: same mem-fraction / gpu-mem-util,
    ``--disable-radix-cache``, ``--trust-remote-code``, ``*_USE_AITER`` /
    firmware-gated envs. The backend adapter's built-in ``launch_server`` line
    diverges from Magpie's script, which is the single biggest baseline gap. When
    the caller points us at Magpie's script we export ``BENCH_LAUNCHER=magpie`` +
    ``MAGPIE_LAUNCH_SCRIPT`` so EVERY ``bench_e2e.sh`` launches the server through
    that script (with the authored-kernel overlay prepended by the launcher
    adapter — which Magpie itself cannot do), mirroring :func:`apply_bench_client`.

    BACKEND-AGNOSTIC (never model/case specific): the SAME ``magpie`` launcher and
    the SAME resolution logic serve sglang, vllm and any future Magpie backend —
    the launcher derives the per-backend flag/profiler var names from ``$BACKEND``.

    Resolution:
      * explicit ``handoff.bench_launcher`` / ``$BENCH_LAUNCHER`` wins
        (``BENCH_LAUNCHER=native`` is the escape hatch when the orchestrator's
        script cannot run on this box);
      * else enable ``magpie`` ONLY when a script is discoverable
        (``handoff.launch_server_script``, or generic ``$MAGPIE_LAUNCH_SCRIPT``,
        or per-backend ``$MAGPIE_<BACKEND>_SCRIPT`` e.g. ``$MAGPIE_VLLM_SCRIPT``,
        or derived from ``handoff.launch_recipe``)
        AND the backend is one Magpie supports; otherwise ``native``.

    When nothing is discoverable the native backend launch is kept, so the
    standalone / unaligned path is byte-identical to before.

    Returns the resolved launcher name (for --dry-run / logging).
    """
    requested = str(
        h.get("bench_launcher") or os.environ.get("BENCH_LAUNCHER", "") or ""
    ).strip().lower()
    backend = str(h.get("framework", "sglang") or "sglang").strip().lower()
    # Discover the Magpie launch script, most explicit source first: handoff,
    # generic env, per-backend env (MAGPIE_SGLANG_SCRIPT / MAGPIE_VLLM_SCRIPT),
    # then derived from the launch recipe the handoff points at. The recipe is
    # last because it is inferred rather than stated, but in practice it is the
    # only source that is ever populated: the orchestrator names its recipe on
    # every handoff and names the script on none of them.
    script = str(h.get("launch_server_script") or "").strip()
    source = "handoff"
    if not script:
        script = str(
            os.environ.get("MAGPIE_LAUNCH_SCRIPT", "")
            or os.environ.get(f"MAGPIE_{backend.upper()}_SCRIPT", "")
            or ""
        ).strip()
        source = "env"
    if not script:
        script = _magpie_script_from_recipe(h)
        source = "launch_recipe"
    if script:
        # Normalise onto the generic var the backend-agnostic launcher reads.
        os.environ["MAGPIE_LAUNCH_SCRIPT"] = script
        os.environ["MAGPIE_LAUNCH_SCRIPT_SOURCE"] = source

    if requested and requested != "auto":
        launcher = requested
    elif script and backend in _MAGPIE_BACKENDS:
        launcher = "magpie"
    else:
        launcher = "native"
    os.environ["BENCH_LAUNCHER"] = launcher

    # Magpie's script defaults max-model-len to a value of its own (4096) that
    # has nothing to do with this run, and the orchestrator overrode it via env
    # when it measured the reference. Forward the same env so the script bakes
    # in the right value, instead of leaving the correct number to arrive as a
    # duplicate --max-model-len in EXTRA_<BACKEND>_ARGS and win only because
    # argparse happens to take the last occurrence. Only forwarded when the
    # handoff carried it; absent => the script's own default stands, which is
    # what the orchestrator served with.
    #
    # gpu-mem-util is deliberately NOT forwarded the same way: no handoff has
    # ever carried mem_fraction, and the script's 0.95 default IS the recipe we
    # are trying to match.
    if launcher == "magpie":
        replay, owned = _recipe_launch_env(h)
        _export_recipe_env(h, replay, owned, source)

        try:
            max_model_len = int(h.get("max_model_len") or 0)
        except (TypeError, ValueError):
            max_model_len = 0
        # The recipe's own record outranks the handoff scalar: it is what the
        # reference server inherited, whereas max_model_len is a summary field
        # that travelled separately and can drift from it. Disagreement is not
        # fatal (a re-scoped run legitimately differs) but it is never silent --
        # an unnoticed drift here is precisely the class of bug that produced
        # the divergence this alignment exists to close.
        recorded_mml = replay.get("MAX_MODEL_LEN", "").strip()
        if recorded_mml:
            if max_model_len > 0 and recorded_mml != str(max_model_len):
                print(
                    f"!!! recipe records MAX_MODEL_LEN={recorded_mml} but the handoff "
                    f"says {max_model_len}; replaying the recipe's value.",
                    file=sys.stderr,
                )
            # Cleared so the launcher's own MAX_MODEL_LEN pass-through cannot
            # land on top of the replayed value.
            os.environ.pop("MAX_MODEL_LEN", None)
        elif max_model_len > 0:
            os.environ["MAX_MODEL_LEN"] = str(max_model_len)
    return launcher


def _export_recipe_env(
    h: dict, replay: dict[str, str], owned: list[str], source: str
) -> str:
    """Materialise the replay set for the launcher, or refuse to launch.

    Written as a NUL-delimited ``NAME=VALUE`` file rather than passed as a
    string: the recipe records PATH, and any value containing a space would be
    word-split into two bogus assignments by the unquoted ``env $VAR`` idiom the
    launcher uses for the flags it already had.

    WARNS when the recipe carries no replayable ``envs:`` values, but proceeds
    on defaults so GEAK runs smoothly even against older orchestrator output
    that omits the block. A block containing only GEAK-owned run coordinates is
    fully accounted for and needs no replay file. Set
    ``GEAK_STRICT_RECIPE_ENV=1`` to fail-close on an absent/empty record (useful
    in CI or when investigating a divergence).

    Returns the path written, or "" when there was nothing to write.
    """
    # Strictness follows the existence of a recipe alignment attempt, not how
    # the server script was discovered. An explicit handoff/env script still
    # launches against h["launch_recipe"] and must not bypass fail-close merely
    # because `source != "launch_recipe"`.
    strict = bool(str(h.get("launch_recipe") or "").strip()) and _env_flag(
        "GEAK_STRICT_RECIPE_ENV"
    )
    if not replay and owned:
        # The recipe did record an environment, but every entry is an explicit
        # run coordinate GEAK must replace (model/TP/port/GPU mask/etc.). There
        # is nothing to materialise, yet strict alignment is still satisfied:
        # the complete difference set is known and reported rather than absent.
        os.environ.pop("RECIPE_ENV_FILE", None)
        os.environ["RECIPE_ENV_SOURCE"] = str(h.get("launch_recipe") or "")
        os.environ["RECIPE_ENV_REPLAYED"] = ""
        os.environ["RECIPE_ENV_GEAK_OWNED"] = " ".join(owned)
        return ""
    if not replay:
        if strict:
            raise SystemExit(
                "!!! recipe alignment: the launch recipe\n"
                f"      {h.get('launch_recipe')}\n"
                "    records no envs: block, so the environment the reference server\n"
                "    ran with is unknown and this launch cannot be shown to match it.\n"
                "    Re-export the recipe with its envs, point BENCH_LAUNCHER=native at\n"
                "    an intentionally unaligned run, or unset GEAK_STRICT_RECIPE_ENV to\n"
                "    launch on defaults anyway."
            )
        if source == "launch_recipe":
            print(
                ">>> recipe alignment: the recipe carries no envs: block; launching "
                "with script defaults only. Set GEAK_STRICT_RECIPE_ENV=1 to refuse.",
                file=sys.stderr,
            )
        return ""

    # Resolve the eval dir the same way map_args does: h["eval_dir"] first, then
    # the GEAK_EVAL_DIR main() pins from ps_args (map_args never writes eval_dir
    # back into h, so on the --dry-run path h["eval_dir"] can be empty even
    # though the run has a real eval dir). When BOTH are empty there is no run
    # directory to own, so write a UNIQUE temp file instead of a fixed
    # gettempdir()/recipe_env.nul that two concurrent runs would clobber -- the
    # exact fixed-path collision that made a dry-run's .nul land where a later
    # inspection could not find it.
    out_dir = (
        str(h.get("eval_dir") or "").strip()
        or os.environ.get("GEAK_EVAL_DIR", "").strip()
    )
    path = ""
    tmp_path = ""
    try:
        if out_dir:
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            path = str(Path(out_dir) / "recipe_env.nul")
            fd, tmp_path = tempfile.mkstemp(
                prefix=".recipe_env.", suffix=".tmp", dir=out_dir
            )
        else:
            fd, path = tempfile.mkstemp(prefix="geak_recipe_env_", suffix=".nul")
            tmp_path = path
        with os.fdopen(fd, "wb") as fh:
            for key in sorted(replay):
                fh.write(f"{key}={replay[key]}".encode("utf-8") + b"\0")
            fh.flush()
            os.fsync(fh.fileno())
        if out_dir:
            os.replace(tmp_path, path)
            tmp_path = ""
    except OSError as exc:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        if strict:
            raise SystemExit(
                f"!!! recipe alignment: cannot materialise the replay env ({exc}). "
                "Refusing to launch a server whose environment would silently differ "
                "from the reference."
            ) from exc
        return ""

    os.environ["RECIPE_ENV_FILE"] = path
    os.environ["RECIPE_ENV_SOURCE"] = str(h.get("launch_recipe") or "")
    os.environ["RECIPE_ENV_REPLAYED"] = " ".join(sorted(replay))
    os.environ["RECIPE_ENV_GEAK_OWNED"] = " ".join(owned)
    # stderr, not stdout: callers evaluate this function's stdout as shell
    # (run_ab.sh does `eval "$(python3 ... apply_bench_launcher ...)"`), so a
    # progress line on stdout is parsed as a command and kills the run.
    print(
        f">>> recipe alignment: replaying {len(replay)} recorded env var(s) "
        f"[{' '.join(sorted(replay))}]"
        + (f"; GEAK owns [{' '.join(owned)}]" if owned else ""),
        file=sys.stderr,
    )
    return path


def apply_alignment_flags(h: dict) -> dict:
    """Export optional cold/hot measurement-alignment flags so bench_e2e.sh inherits them.

    Currently: ``BENCH_COLD_FINAL`` — when on, bench_e2e.sh also measures ONE cold
    full round per bench (surfaced as ``cold_output_throughput_tok_s`` in each
    bench_summary.json and folded into ``result.json.alignment_metrics``).

    Default OFF. The round is labelled "cold" but only the FIRST bench of a
    session ever runs on a genuinely cold box: by the time the final leg is
    benched, the JIT/HIP kernel caches, torch.compile artifacts and the page
    cache are all warm, so its "cold" round is a warm round wearing the label.
    That makes the two cold numbers incomparable — the baseline's cold round
    pays the full cache-fill cost and the final's pays almost none — and the
    asymmetry has shown up as a double-digit phantom speedup. It stays available
    as a diagnostic (explicit truthy ``handoff.bench_cold_final`` or
    ``$BENCH_COLD_FINAL=1``) but no longer costs an extra full round per bench
    by default, and never decides the promoted number.
    Returns the flags it exported.
    """
    exported: dict[str, str] = {}
    raw = h.get("bench_cold_final")
    if raw is None:
        raw = os.environ.get("BENCH_COLD_FINAL")
    # Default OFF: enabled only on an explicit truthy value.
    if raw is None or str(raw).strip() == "":
        on = False
    else:
        on = str(raw).strip().lower() in {"1", "true", "yes", "on"}
    os.environ["BENCH_COLD_FINAL"] = "1" if on else "0"
    exported["BENCH_COLD_FINAL"] = "1" if on else "0"
    return exported


# ---------------------------------------------------------------------------
# Bench-protocol measurement alignment (measurement knobs, not the client).
# ---------------------------------------------------------------------------
# handoff.bench_protocol key -> bench_e2e.sh / client-adapter env var.
_BENCH_PROTOCOL_ENV = {
    "random_range_ratio": "RANDOM_RANGE_RATIO",
    "num_prompts": "NUM_PROMPTS",
    "num_warmups": "NUM_WARMUPS",
    "seed": "SEED",
}


def apply_bench_protocol(h: dict) -> dict:
    """Export the caller's measurement protocol so workflow bench_e2e.sh inherits it.

    ``handoff.bench_protocol`` carries the EXACT bench knobs the external
    orchestrator (Hyperloom) measured with — chiefly ``random_range_ratio``
    (fixed vs variable sequence lengths), ``num_prompts``, ``num_warmups`` and
    ``seed``. We export each PROVIDED key into the environment (same mechanism
    as :func:`apply_bench_client`), so every ``bench_e2e.sh`` invocation the
    agents make overrides its built-in default with the orchestrator's value.

    IMPORTANT: only keys actually present in the handoff are exported. When
    ``bench_protocol`` is absent (e.g. GEAK run standalone, no external
    orchestrator), nothing is exported and ``bench_e2e.sh`` keeps its own
    defaults — so the standalone path is unchanged.

    Returns the dict of {env_var: value} it exported (for --dry-run / logging).
    """
    protocol = h.get("bench_protocol") or {}
    exported: dict[str, str] = {}
    if not isinstance(protocol, dict):
        return exported
    for key, env_var in _BENCH_PROTOCOL_ENV.items():
        if key not in protocol:
            continue
        val = protocol[key]
        if val is None or str(val).strip() == "":
            continue
        os.environ[env_var] = str(val)
        exported[env_var] = str(val)
    return exported


# ---------------------------------------------------------------------------
# Invocation: SDK preferred, CLI fallback.
# ---------------------------------------------------------------------------
def _iter_message_text(msg: Any) -> list[str]:
    """Best-effort extraction of every text fragment from one SDK message.

    The workflow return (the JSON object carrying ``eval_dir``) can surface in
    different places across SDK versions / message shapes: the assistant's
    final text, a ``text`` content block, or the ``Workflow`` tool's
    ``tool_result`` payload. Collecting from ALL of them (instead of only the
    last assistant ``.text``) makes the handoff capture robust to the agent
    ending its turn on a tool/result block rather than a plain text echo.

    Returns every string fragment found on the message (never raises).
    """
    out: list[str] = []

    def _take(v: Any) -> None:
        if isinstance(v, str) and v.strip():
            out.append(v)

    # 1) Flat ``.text`` / ``.result`` attributes.
    _take(getattr(msg, "text", None))
    _take(getattr(msg, "result", None))
    # 2) Structured ``.content`` blocks (assistant text + tool_result content).
    content = getattr(msg, "content", None)
    if isinstance(content, str):
        _take(content)
    elif isinstance(content, (list, tuple)):
        for block in content:
            _take(getattr(block, "text", None))
            if isinstance(block, dict):
                _take(block.get("text"))
                inner = block.get("content")
                if isinstance(inner, str):
                    _take(inner)
                elif isinstance(inner, (list, tuple)):
                    for ib in inner:
                        _take(getattr(ib, "text", None))
                        if isinstance(ib, dict):
                            _take(ib.get("text"))
    # 3) Dict-shaped messages (some SDK builds yield plain dicts).
    if isinstance(msg, dict):
        _take(msg.get("text"))
        _take(msg.get("result"))
    return out


def _workflow_done_on_disk(eval_dir: str | None) -> bool:
    """True once the workflow wrote a TERMINAL marker (its very last on-disk act).

    Two terminal markers, both written AT/AFTER the final Validate leg:
      * ``workflow_return.json`` — the canonical schema-validated return the
        workflow persists as its FINAL action (see e2e_workflow.js). This is the
        authoritative "everything finished" signal and the file run_e2e.py reads
        first. It is the LAST thing the workflow writes, so it is the ideal gate.
      * ``director_e2e_validation.json`` — the Validate director's marker, written
        just before. Kept as an alternative in case the canonical persist step
        (an agent Write) failed.

    ``final/final_launch.sh`` is intentionally NOT terminal: it is written by the
    EARLIER Finalize phase, BEFORE Report/Validate. Treating it as done made the
    SDK completion gate fire one or two phases early and SKIP the grace poll that
    keeps the client (and the still-running, detached Validate leg) alive —
    orphaning the director before it could write its json. Keying off the two
    post-Validate markers is what lets the grace poll wait for the real last leg.
    """
    if not eval_dir:
        return False
    p = Path(eval_dir)
    return (p / WORKFLOW_RETURN_FILE).is_file() or (
        p / "director_e2e_validation.json"
    ).is_file()


def _invoke_via_sdk(prompt: str, timeout_s: int, eval_dir: str | None = None) -> str:
    """Drive the JS workflow through the SDK, version-robustly.

    Why not a one-shot ``query()``? Newer Claude Code builds (CLI >=2.1.183)
    route a ``Workflow`` invocation to a NON-BLOCKING *background task*: the
    main agent turn ends almost immediately ("...running in the background"),
    so ``query()``'s async iterator completes and the runner used to return —
    tearing the still-running workflow down with it. Older builds (<=2.1.181)
    run the same workflow synchronously inside the turn. Pinning an SDK version
    papers over this; it does not survive the next update.

    This implementation does NOT depend on whether the workflow blocks the
    turn. It uses the persistent ``ClaudeSDKClient`` (keeping the CLI process —
    and therefore any background workflow — alive) and consumes the FULL
    message stream, driving completion off the SDK's documented background-task
    lifecycle (``TaskStartedMessage`` -> ``TaskNotificationMessage``) plus the
    workflow's own on-disk terminal marker. It returns the joined transcript so
    the existing ``_parse_last_json_line`` scrape still works; when the return
    JSON is not in the transcript (the background path surfaces it via the
    task's ``output_file``/``summary``, which we append), main() falls back to
    the scrape-independent on-disk recovery against the same pinned eval_dir.
    """
    import anyio
    from claude_agent_sdk import ClaudeAgentOptions

    try:
        from claude_agent_sdk import ClaudeSDKClient
    except ImportError:  # very old SDK without the streaming client
        ClaudeSDKClient = None  # type: ignore[assignment]

    def _opts() -> "ClaudeAgentOptions":
        extra: dict = {}
        if CLAUDE_EFFORT in VALID_EFFORTS:
            extra["effort"] = CLAUDE_EFFORT
        sdk_env: dict[str, str] = {}
        # Claude Code refuses bypassPermissions under root unless it is running
        # in an explicit sandbox. Scope this to the SDK child process only.
        if hasattr(os, "geteuid") and os.geteuid() == 0:
            sdk_env["IS_SANDBOX"] = "1"
        return ClaudeAgentOptions(
            model=CLAUDE_MODEL,
            allowed_tools=ALLOWED_TOOLS,
            permission_mode="bypassPermissions",
            settings=WORKFLOW_SETTINGS,
            extra_args=extra,
            cwd=str(E2E_DIR),
            env=sdk_env,
            **({"cli_path": CLAUDE_BIN} if CLAUDE_BIN else {}),
        )

    async def _run_client() -> str:
        # Accumulate the FULL transcript (every text fragment from every
        # message) so the workflow-return JSON is recoverable wherever it
        # surfaced. Track background tasks by class NAME (the Task* message
        # types exist in both old and new SDKs, so name-matching keeps one code
        # path working across versions without import coupling).
        chunks: list[str] = []
        pending: set[str] = set()   # started-but-unfinished background tasks
        bg_started = False          # did the workflow ever background a task?
        terminal_task = False       # saw a TaskNotification (completed/failed)
        saw_result = False          # the main turn's ResultMessage arrived
        # Enforce the orchestrator's budget INSIDE the SDK path so we self-stop
        # before Hyperloom's outer kill_timeout SIGKILLs us (a SIGKILL would
        # skip result.json flushing entirely). anyio raises TimeoutError on
        # expiry, which main() maps to error_class="timeout".
        with anyio.fail_after(timeout_s):
            async with ClaudeSDKClient(options=_opts()) as client:
                await client.query(prompt)
                async for msg in client.receive_messages():
                    chunks.extend(_iter_message_text(msg))
                    name = type(msg).__name__
                    if name == "TaskStartedMessage":
                        tid = getattr(msg, "task_id", None)
                        if tid:
                            pending.add(tid)
                            bg_started = True
                    elif name == "TaskNotificationMessage":
                        terminal_task = True
                        pending.discard(getattr(msg, "task_id", None))
                        # The background path surfaces the workflow return via
                        # the task's output_file / summary rather than the main
                        # transcript — fold them in so the scrape can find it.
                        of = getattr(msg, "output_file", None)
                        if of:
                            try:
                                chunks.append(Path(of).read_text(encoding="utf-8"))
                            except OSError:
                                pass
                        summ = getattr(msg, "summary", None)
                        if isinstance(summ, str) and summ.strip():
                            chunks.append(summ)
                    elif name == "ResultMessage":
                        saw_result = True

                    # ---- completion gate (independent of turn blocking) ----
                    # Never stop while a background task is still running.
                    if pending:
                        continue
                    # Authoritative: the optimizer wrote its terminal marker.
                    # This is the ONLY hard "the workflow finished a measured
                    # leg" signal and is independent of HOW the agent ran it.
                    if _workflow_done_on_disk(eval_dir):
                        break
                    # Pure synchronous path: the turn ended and no background
                    # task was EVER spawned — the workflow ran fully in-turn, so
                    # the turn's ResultMessage is itself terminal. (A missing
                    # marker here means an in-turn crash; disk-recovery judges.)
                    if saw_result and not bg_started:
                        break
                    # Background path "looks done": a task notified terminal AND
                    # the main turn produced a ResultMessage — BUT the workflow
                    # has not yet written its on-disk terminal marker. It may
                    # still be finishing a DETACHED leg (the integrate A/B
                    # reference/candidate bench is launched as a child process
                    # and outlives the task notification). Returning now would
                    # orphan that bench and discard a still-completing A/B. Stop
                    # consuming messages, but DO NOT close the client yet: fall
                    # through to the bounded grace poll below, which keeps the
                    # CLI (and the backgrounded workflow) alive while waiting for
                    # the authoritative marker to land.
                    if terminal_task and saw_result:
                        break

                # Grace window: we exited the message loop on the weak
                # background signal without an on-disk terminal marker. Keep the
                # persistent client open (so the detached integrate/Validate leg
                # keeps running) and poll the disk until the marker appears or
                # the bounded grace expires. The enclosing fail_after(timeout_s)
                # still caps total time, so this can never exceed the hard budget.
                if (
                    terminal_task
                    and saw_result
                    and bg_started
                    and not _workflow_done_on_disk(eval_dir)
                ):
                    deadline = time.monotonic() + DONE_GRACE_S
                    while time.monotonic() < deadline:
                        if _workflow_done_on_disk(eval_dir):
                            break
                        await anyio.sleep(DONE_POLL_S)
        return "\n".join(chunks)

    async def _run_query() -> str:
        # Legacy fallback for SDKs lacking ClaudeSDKClient. Behaves like the
        # original one-shot query (works for the synchronous in-turn path).
        from claude_agent_sdk import query
        chunks: list[str] = []
        with anyio.fail_after(timeout_s):
            async for msg in query(prompt=prompt, options=_opts()):
                chunks.extend(_iter_message_text(msg))
        return "\n".join(chunks)

    return anyio.run(_run_client if ClaudeSDKClient is not None else _run_query)


def _invoke_via_cli(prompt: str, timeout_s: int) -> str:
    claude = shutil.which("claude") or os.environ.get("CLAUDE_BIN", "claude")
    cmd = [
        claude, "-p", prompt,
        "--output-format", "json",
        "--settings", WORKFLOW_SETTINGS,
        "--model", CLAUDE_MODEL,
        "--allowed-tools", ",".join(ALLOWED_TOOLS),
        "--permission-mode", "auto",
    ]
    if CLAUDE_EFFORT in VALID_EFFORTS:
        cmd += ["--effort", CLAUDE_EFFORT]
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


def invoke_workflow(prompt: str, timeout_s: int, eval_dir: str | None = None) -> dict:
    """Run the JS workflow and return its parsed JSON return value."""
    try:
        import claude_agent_sdk  # noqa: F401
        raw = _invoke_via_sdk(prompt, timeout_s, eval_dir)
    except ImportError:
        raw = _invoke_via_cli(prompt, timeout_s)
    return _parse_last_json_line(raw)


class WorkflowParseError(RuntimeError):
    """The agent output carried no parseable workflow return (no ``eval_dir``)."""


def _iter_json_objects(raw: str):
    """Yield every parseable top-level JSON object in ``raw`` (in order).

    Robust to the workflow return arriving as: a single compact line, a value
    fenced in a ```json block, or a pretty-printed multi-line object possibly
    followed by trailing prose. Uses a brace-matching scan (string/escape
    aware) so multi-line objects are recovered, then also tries each physical
    line for the common single-line case. Never raises.
    """
    text = raw or ""
    # 1) Brace-matched scan: find balanced {...} spans and try to parse each.
    depth = 0
    start = -1
    in_str = False
    esc = False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    span = text[start : i + 1]
                    try:
                        obj = json.loads(span)
                    except json.JSONDecodeError:
                        obj = None
                    if isinstance(obj, dict):
                        yield obj
                    start = -1
    # 2) Per-line fallback (cheap; catches compact single-line returns the scan
    #    above already covers, but keeps behaviour stable on odd inputs).
    for line in (text.splitlines()):
        s = line.strip()
        if not (s.startswith("{") and s.endswith("}")):
            continue
        try:
            obj = json.loads(s)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            yield obj


def _parse_last_json_line(raw: str) -> dict:
    """Extract the workflow return (last JSON object carrying ``eval_dir``).

    Scans the whole transcript (not just the last line) so the handoff is
    recovered regardless of where/how the agent emitted it. Raises
    :class:`WorkflowParseError` only when no eval_dir-bearing object exists.
    """
    found: dict | None = None
    for obj in _iter_json_objects(raw):
        if obj.get("eval_dir"):
            found = obj  # keep scanning: the LAST one wins
    if found is not None:
        return found
    raise WorkflowParseError(
        "Could not parse a JSON workflow return (with eval_dir) from the agent "
        f"output. Last 2000 chars:\n{(raw or '')[-2000:]}"
    )


def _classify_error(exc: BaseException) -> str:
    """Map an internal failure onto a stable ``error_class`` for Hyperloom.

    Hyperloom's session-breakdown GEAK collector reads ``error_class`` to
    attribute *why* an e2e run missed. Keep these values stable; unknown
    failures fall back to ``runner_error``.
    """
    # anyio.fail_after raises builtins.TimeoutError on budget expiry.
    if isinstance(exc, TimeoutError):
        return "timeout"
    if isinstance(exc, WorkflowParseError):
        return "workflow_parse_error"
    if isinstance(exc, ImportError):
        return "sdk_import_failed"
    msg = str(exc)
    if "claude CLI failed" in msg:
        return "cli_failed"
    return "runner_error"


# ---------------------------------------------------------------------------
# Normalize workflow artifacts -> stable result.json
# ---------------------------------------------------------------------------
def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _safe_ratio(num: float | None, den: float | None) -> float | None:
    """num/den rounded to 4dp, or None when either side is missing/non-positive."""
    try:
        n, d = float(num or 0.0), float(den or 0.0)
    except (TypeError, ValueError):
        return None
    return round(n / d, 4) if (n > 0 and d > 0) else None


def read_orchestrator_hot_baseline(h: dict) -> float:
    """Read Hyperloom's HOT baseline throughput from its ``state.json`` (best-effort).

    Hyperloom's double-run baseline records BOTH a COLD round (``baseline_tput`` —
    the leaderboard denominator, forwarded to us as ``handoff.raw_baseline_tput``)
    and a HOT round (``baseline_hot_tput``). Only the cold one rides in the handoff,
    so for a hot-to-hot cross-check we read the hot one straight off ``state.json``.
    ``state.json`` lives at the SESSION dir (an ancestor of ``exp_root``); probe a
    couple of levels up. Returns 0.0 when unavailable (standalone / no orchestrator),
    so the alignment metrics simply degrade to None instead of raising.
    """
    exp_root = str(h.get("exp_root") or "").strip()
    if not exp_root:
        return 0.0
    p = Path(exp_root)
    for cand in (p / "state.json", p.parent / "state.json",
                 p.parent.parent / "state.json"):
        st = _read_json(cand)
        if not st:
            continue
        v = st.get("baseline_hot_tput")
        if not v:
            base = st.get("baseline") if isinstance(st.get("baseline"), dict) else {}
            v = base.get("baseline_hot_tput")
        try:
            if v and float(v) > 0:
                return float(v)
        except (TypeError, ValueError):
            continue
    return 0.0


def _wf_best_accepted_delta_pct(wf: dict) -> float:
    """Largest positive ``e2e_delta_pct`` claimed by an accepted head/kernel.

    The workflow return carries the heads/kernels it ACCEPTED (each with the
    measured same-session A/B ``e2e_delta_pct``). This is the ground-truth signal
    that a real, parity-checked win exists — independent of whatever final
    throughput/speedup the return also reports. Returns 0.0 when nothing claims a
    positive gain.
    """
    best = 0.0
    for item in (wf.get("accepted_heads") or []) + (wf.get("accepted_kernels") or []):
        if not isinstance(item, dict):
            continue
        try:
            d = float(item.get("e2e_delta_pct") or 0.0)
        except (TypeError, ValueError):
            d = 0.0
        if d > best:
            best = d
    return best


def _best_ledger_win(wf: dict) -> dict | None:
    """Largest positive-``e2e_delta_pct`` direction the run actually MEASURED.

    Read from the workflow return's in-run experience ledger
    (``state.history.ledger`` — the Architect's per-direction record). This is the
    honest source for the winning op's identity when the accepted lists were
    dropped. Returns ``None`` when the ledger is absent or holds no positive
    direction (never fabricates a winner).
    """
    ledger = (((wf.get("state") or {}).get("history") or {}).get("ledger") or [])
    best: dict | None = None
    best_delta = 0.0
    for entry in ledger:
        if not isinstance(entry, dict):
            continue
        if not (entry.get("direction") or entry.get("short_name")):
            continue
        try:
            delta = float(entry.get("e2e_delta_pct") or 0.0)
        except (TypeError, ValueError):
            delta = 0.0
        if delta > best_delta:
            best, best_delta = entry, delta
    return best


def _state_op_names(wf: dict, queue: str) -> set[str]:
    """``short_name`` set of the return's ``state.<queue>`` (headQueue|kernelQueue)."""
    names: set[str] = set()
    for op in ((wf.get("state") or {}).get(queue) or []):
        if isinstance(op, dict) and op.get("short_name"):
            names.add(str(op["short_name"]))
    return names


def _divergence_pct(measured: Any, reference: Any) -> float | None:
    """Return percentage divergence from a positive reference."""
    try:
        measured_value = float(measured)
        reference_value = float(reference)
    except (TypeError, ValueError):
        return None
    if (
        not math.isfinite(measured_value)
        or not math.isfinite(reference_value)
        or measured_value <= 0.0
        or reference_value <= 0.0
    ):
        return None
    return round(
        100.0 * (measured_value - reference_value) / reference_value,
        2,
    )


def _positive_finite_float(value: Any) -> float:
    """Normalize external numeric input for strict JSON serialization."""
    try:
        normalized = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(normalized) or normalized <= 0.0:
        return 0.0
    return normalized


def _build_baseline_alignment(
    same_config_divergence_pct: float | None,
    recipe_aligned: bool = True,
) -> dict[str, Any]:
    """Classify cross-harness alignment using only the same-config metric.

    ``recipe_aligned`` says whether this run launched its servers through the
    orchestrator's own launch script. When it did not, the two harnesses served
    DIFFERENT stacks (the orchestrator's script owns the platform kernel preset,
    ``--trust-remote-code`` and the gpu-mem-util default), so a divergence is
    evidence about the launch recipe, not about the box or the bench client.
    Saying that in the status keeps the number from being read as "GEAK
    measured slow".
    """
    if same_config_divergence_pct is None:
        status = "unavailable"
    elif abs(same_config_divergence_pct) > SAME_CONFIG_DIVERGENCE_WARN_PCT:
        status = "warning" if recipe_aligned else "warning_recipe_unaligned"
    else:
        status = "aligned"
    return {
        "status": status,
        "primary_metric": "current_best_same_config_divergence_pct",
        "divergence_pct": same_config_divergence_pct,
        "warning_threshold_pct": SAME_CONFIG_DIVERGENCE_WARN_PCT,
        "raw_session_divergence_is_measurement_signal": False,
        "recipe_aligned_with_orchestrator": recipe_aligned,
    }


# Kernel-selection lines a serving stack prints at startup. Substring matching
# is backend-agnostic on purpose: the goal is not to enumerate backends but to
# record WHICH kernels the engine actually chose, so a launch that silently fell
# back to a slower stack is visible in result.json instead of only in a
# thousand-line server log.
_STACK_SIGNAL_PATTERNS = (
    "Final IR op priority",
    "for Fp8LinearMethod",
    "Fp8 MoE backend",
    "ttention backend",  # "Attention backend" / "attention backend"
    "server_args=ServerArgs(",
)
_STACK_SIGNAL_MAX_PICKS = 8


def _serving_stack_signals(log_path: Path) -> dict[str, Any]:
    """Summarize the kernel stack a server actually came up with.

    Returns ``{}`` when the log is unreadable (leg never ran / was cleaned), so
    the caller can carry the field harmlessly on a standalone run.
    """
    picks: list[str] = []
    aiter_mentions = 0
    try:
        # Stream line-by-line: server.log can reach GBs, and the old
        # read_text()+splitlines()+text.lower() held three full O(n) copies at
        # once. Iterating the handle keeps at most one line resident (O(1)).
        with log_path.open(encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                # Whole-log count (semantics unchanged vs the old text.lower().count).
                aiter_mentions += line.lower().count("aiter")
                # Check the cap BEFORE appending so _STACK_SIGNAL_MAX_PICKS is a
                # real bound (the old code appended then checked, so switching the
                # break to a continue would have let picks grow without limit).
                if len(picks) < _STACK_SIGNAL_MAX_PICKS and any(
                    pattern in line for pattern in _STACK_SIGNAL_PATTERNS
                ):
                    # Drop the pid / timestamp / module prefix so the pick is legible.
                    picks.append(line.rsplit("] ", 1)[-1].strip()[:220])
    except OSError:
        return {}
    return {
        # A raw count, not a verdict: the accelerated-kernel stack is chatty at
        # startup, so a near-zero count is the signature of a launch that never
        # enabled it. Observed on one 122B session: 5499 mentions on the
        # orchestrator's server vs 4 on GEAK's, for the identical config.
        "aiter_mentions": aiter_mentions,
        "kernel_picks": picks,
    }


def _cold_penalty_pct(cold: Any, hot: Any) -> float | None:
    """How far a leg's cold round fell below that same leg's hot median, in %.

    Negative is the expected direction (the cold round pays JIT / graph-capture
    / page-cache costs). Comparing the two legs' penalties is what reveals
    whether their cold rounds were measured in comparable thermal states.
    """
    cold_value, hot_value = _positive_finite_float(cold), _positive_finite_float(hot)
    if cold_value <= 0.0 or hot_value <= 0.0:
        return None
    return round((cold_value - hot_value) / hot_value * 100.0, 3)


def _overlay_has_loadable_code(path: Path) -> bool:
    """True when ``path`` is an overlay a consumer could actually PYTHONPATH into.

    The Finalize phase creates ``final/overlay`` unconditionally and drops a
    marker file (``README.txt``, ``EMPTY_NO_ACCEPTED_OVERLAY.txt``) into it when
    nothing was accepted, so directory existence proves nothing. What makes an
    overlay real is importable code: a top-level module, the manifest the
    overlay loader reads, or an accepted ``cand_*`` subtree.
    """
    if not path.is_dir():
        return False
    if (path / "_overlay_manifest.json").is_file() or (path / "sitecustomize.py").is_file():
        return True
    if any(p.suffix == ".py" for p in path.iterdir() if p.is_file()):
        return True
    return any(
        d.is_dir() and (
            (d / "_overlay_manifest.json").is_file()
            or (d / "sitecustomize.py").is_file()
        )
        for d in path.glob("cand_*")
    )


def _patch_has_hunks(path: Path) -> bool:
    """True when ``path`` is a unified diff that would actually change something.

    ``final_patch.diff`` is written even when there is nothing to patch — the
    header/provenance preamble alone runs to over a kilobyte — so a size check
    passes on a patch that applies zero edits. A unified diff changes a file
    only through an ``@@`` hunk, so that marker is the real test.
    """
    if not path.is_file():
        return False
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return any(line.startswith("@@") for line in text.splitlines())


def _material_overlay_path(eval_dir: Path, wf: dict) -> str:
    """Path to a real overlay, or ``""`` when this run produced none.

    Advertising a path that holds nothing (or does not exist at all) makes a
    caller believe there is something to reuse; an empty string tells it the
    truth. The path recorded on the return wins when it still resolves,
    otherwise we look for the product under the eval_dir we were handed.
    """
    recorded = str(wf.get("final_overlay") or "").strip()
    candidates: list[Path] = [Path(recorded)] if recorded else []
    candidates += [eval_dir / "final" / "overlay", eval_dir / "final"]
    for candidate in candidates:
        if _overlay_has_loadable_code(candidate):
            return str(candidate)
    return ""


def _material_patch_path(eval_dir: Path, wf: dict) -> str:
    """Path to a patch that carries at least one hunk, or ``""``."""
    recorded = str(wf.get("final_patch") or "").strip()
    candidates: list[Path] = [Path(recorded)] if recorded else []
    candidates.append(eval_dir / "final" / "final_patch.diff")
    for candidate in candidates:
        if _patch_has_hunks(candidate):
            return str(candidate)
    return ""


def _same_session_baseline(
    base_leg_summary: dict, validation: dict
) -> tuple[float, str]:
    """The baseline leg measured in the SAME Validate session as the final leg.

    ``baseline/bench_summary.json`` is minted during Setup, often hours before
    the final leg runs, and the box moves under us in between (we have observed
    -4%..+15% drift within a single session). Dividing a Validate-time final by
    a Setup-time baseline therefore reports drift as optimization. The Validate
    phase re-measures the unpatched engine right next to the patched one, so
    that pair — and only that pair — is a valid A/B.

    Preference order, most to least direct:
      1. ``validation/base/bench_summary.json`` — the re-measured base leg.
      2. Director's ``base_block.throughput_tok_s_median`` — same leg, as the
         Director recorded it (used when the summary file was not kept).
      3. Director's ``drift_corrected_baseline_tok_s`` — the Director's own
         drift correction, whose key name is stable across schema versions
         (unlike ``baseline_throughput_tok_s``, which means the Setup baseline
         in some versions and the corrected one in others).

    Returns ``(0.0, "")`` when no same-session leg exists, which leaves the
    caller on its previous baseline.
    """
    value = _positive_finite_float(base_leg_summary.get("throughput_tok_s_median"))
    if value > 0.0:
        return value, "validation_base_bench_summary"
    base_block = validation.get("base_block") or {}
    value = _positive_finite_float(base_block.get("throughput_tok_s_median"))
    if value > 0.0:
        return value, "director_base_block"
    value = _positive_finite_float(validation.get("drift_corrected_baseline_tok_s"))
    if value > 0.0:
        return value, "director_drift_corrected"
    return 0.0, ""


def normalize_result(h: dict, wf: dict) -> dict:
    eval_dir = Path(wf["eval_dir"])
    validation = _read_json(eval_dir / "director_e2e_validation.json")
    baseline_summary = _read_json(eval_dir / "baseline" / "bench_summary.json")
    final_summary = _read_json(eval_dir / "validation" / "final" / "bench_summary.json")
    # The unpatched leg re-measured during Validate, next to the final leg.
    base_leg_summary = _read_json(eval_dir / "validation" / "base" / "bench_summary.json")

    # ── Reconcile a return with NO final measurement (do-no-harm guard for the
    # Hyperloom interface) ───────────────────────────────────────────────────
    # A run can accept a head/kernel (a positive e2e_delta_pct with a complete
    # integrate A/B on disk) and then lose the final number entirely — the
    # Validate bench crashes in engine-core init and the Director reports 0. A
    # real, parity-checked same-session win must not be thrown away because the
    # last bench died, so backfill throughput / speedup / baseline / latency /
    # overlay from the best accepted intermediate A/B on disk and tag the
    # provenance. Only for a LIVE return (not one we already recovered).
    #
    # A MEASURED final of <= 1.0x is NOT this case. That is Validate's verdict:
    # it re-ran the accepted change against a fresh base and did not confirm the
    # gain, which is precisely the check the intermediate A/B cannot perform for
    # itself. Overriding it here would promote the number the arbitration just
    # rejected, so the trigger is the ABSENCE of a final, never a low one. The
    # disagreement is still worth recording — see intermediate_win_not_confirmed.
    wf_speedup_raw = float(wf.get("throughput_speedup") or validation.get("throughput_speedup") or 1.0)
    wf_final_raw = float(
        wf.get("final_throughput_tok_s")
        or validation.get("director_verified_throughput_tok_s")
        or 0.0
    )
    live_accepted_win = (
        not wf.get("recovered_from_disk")
        and _wf_best_accepted_delta_pct(wf) > 0.0
    )
    intermediate_win_not_confirmed = (
        live_accepted_win and wf_final_raw > 0.0 and wf_speedup_raw <= 1.0
    )
    if live_accepted_win and wf_final_raw <= 0.0:
        recovered = _recover_best_intermediate_win(eval_dir)
        if recovered is not None and float(recovered.get("throughput_speedup") or 0.0) > 1.0:
            merged = dict(wf)
            for k in ("throughput_speedup", "baseline_throughput_tok_s",
                      "final_throughput_tok_s", "output_parity", "ttft_ms", "tpot_ms"):
                if recovered.get(k) is not None:
                    merged[k] = recovered[k]
            # The recovered win's own overlay (C7); "" means config-only, in
            # which case whatever the return already carried still stands.
            if recovered.get("final_overlay"):
                merged["final_overlay"] = recovered["final_overlay"]
            merged["recovered_intermediate"] = True   # provenance -> disk_intermediate_win
            merged["validate_final_missing"] = True
            wf = merged
            # The Director json (the crashed bench) is kept, not erased: it is
            # the evidence for WHY we fell back here, and validation_evidence
            # reports it. Every number above is already sourced from wf first.

    speedup = float(wf.get("throughput_speedup") or validation.get("throughput_speedup") or 1.0)
    status = "ok" if speedup > 1.0 else "no_gain"

    final_launch = (
        wf.get("final_launch_script")
        or validation.get("final_launch_script")
        or str(eval_dir / "final" / "final_launch.sh")
    )
    workload = h.get("workload") or {"isl": 1024, "osl": 1024, "conc": 64}

    # Provenance of the numbers below, so Hyperloom can gauge confidence:
    #   workflow_return        — the canonical schema-validated artifact / scraped
    #                            return (full Validate-arbitrated result).
    #   disk_director_validation — rebuilt from director_e2e_validation.json.
    #   disk_intermediate_win  — best accepted integrate A/B (no final Validate).
    #   disk_no_gain_synthesis — baseline measured, nothing accepted (do-no-harm).
    if wf.get("recovered_no_gain"):
        result_source = "disk_no_gain_synthesis"
    elif wf.get("recovered_intermediate"):
        # disk_stack_provisional — salvaged from candidates the integrator gated
        # "stack" (carry forward to compound), i.e. none of them cleared its bar
        # for a standalone win and no Director ever arbitrated the combination.
        result_source = (
            "disk_stack_provisional" if wf.get("recovered_stack_provisional")
            else "disk_intermediate_win"
        )
    elif wf.get("recovered_from_disk"):
        result_source = "disk_director_validation"
    else:
        result_source = "workflow_return"

    # ── Reconcile a VALIDATED win whose ATTRIBUTION was dropped (director
    # override) ───────────────────────────────────────────────────────────────
    # A live return can carry a real, Director-validated speedup
    # (validation_status == "validated_win") yet ship EMPTY accepted_heads AND
    # accepted_kernels — e.g. the head-track Amdahl plausibility guard marked the
    # winning direction "dead_end" (scored on a single head's mass) and the
    # Director's later validated_win never wrote it back, so result.json credits a
    # real win to nothing. Recover the winner's IDENTITY from the in-run ledger
    # (state.history.ledger — the direction the run actually MEASURED) and route it
    # to accepted_heads / accepted_kernels by which queue it came from. Do-no-harm:
    # fires ONLY on a live workflow_return with a positive validated speedup and
    # BOTH accepted lists empty; never overrides a populated list; never fabricates
    # (no ledger winner => no change). Tagged accepted_via="director_override" so the
    # provenance is auditable. See interface/run_e2e.md + GEAK#377.
    if (
        result_source == "workflow_return"
        and speedup > 1.0
        and (validation.get("validation_status") == "validated_win"
             or wf.get("validation_status") == "validated_win")
        and not (wf.get("accepted_kernels") or wf.get("accepted_heads"))
    ):
        _win = _best_ledger_win(wf)
        if _win is not None:
            _name = str(_win.get("direction") or _win.get("short_name") or "")
            _entry = {
                "short_name": _name,
                "e2e_delta_pct": _win.get("e2e_delta_pct"),
                "isolated": _win.get("isolated_speedup"),
                "backend": "geak",
                "accepted_via": "director_override",
                "note": _win.get("lesson"),
            }
            _kernels = _state_op_names(wf, "kernelQueue")
            _heads = _state_op_names(wf, "headQueue")
            wf = dict(wf)   # don't mutate the caller's return object
            if _name and _name in _kernels and _name not in _heads:
                wf["accepted_kernels"] = [_entry]
            else:
                wf["accepted_heads"] = [_entry]
            wf["attribution_backfilled"] = True

    # Cross-harness measurement-protocol check. GEAK's measured baseline is
    # seeded with the upstream orchestrator's accepted config, so compare it
    # separately with the raw session baseline and the same-config current best.
    # The A/B pair must be measured in the same session (see
    # _same_session_baseline). A recovered intermediate win already carries its
    # own paired legs (ref_med/cand_med from one integrate A/B) and a synthesized
    # no-gain deliberately reports baseline == final, so neither may be re-based.
    same_session_base, baseline_basis_source = 0.0, ""
    if not (wf.get("recovered_intermediate") or wf.get("recovered_no_gain")):
        same_session_base, baseline_basis_source = _same_session_baseline(
            base_leg_summary, validation
        )
    setup_baseline = _positive_finite_float(
        baseline_summary.get("throughput_tok_s_median")
    )
    geak_baseline = _positive_finite_float(
        same_session_base
        or wf.get("baseline_throughput_tok_s")
        or validation.get("baseline_throughput_tok_s")
        or 0.0
    )
    if not baseline_basis_source and geak_baseline > 0.0:
        baseline_basis_source = (
            "recovered_intermediate_ab"
            if wf.get("recovered_intermediate")
            else "no_gain_synthesis"
            if wf.get("recovered_no_gain")
            # A baseline rebuilt from a disk Director artifact (no more direct
            # same-session source) is NOT a live Setup measurement; label its
            # provenance for what it is so the A/B basis is auditable, mirroring
            # result_source == "disk_director_validation" above.
            else "disk_director_validation_baseline"
            if wf.get("recovered_from_disk")
            else "setup_baseline"
        )
    geak_final = _positive_finite_float(
        wf.get("final_throughput_tok_s")
        or validation.get("director_verified_throughput_tok_s")
        or 0.0
    )
    orch_baseline = _positive_finite_float(h.get("raw_baseline_tput"))
    # Orchestrator throughput measured on the SAME config GEAK seeds with
    # (the upstream orchestrator's current-best config). When present it isolates
    # the PURE
    # cross-harness measurement residue (identical config, both harnesses) from
    # the explore/framework config gain that is baked into the raw-baseline
    # comparison. It remains unavailable when absent from older handoffs.
    orch_same_cfg = _positive_finite_float(
        h.get("orchestrator_best_tput_same_config")
    )
    raw_session_divergence_pct = _divergence_pct(geak_baseline, orch_baseline)
    same_config_divergence_pct = _divergence_pct(geak_baseline, orch_same_cfg)

    # ── serving-stack provenance ─────────────────────────────────────────────
    # WHO launched the server, and WHAT kernels it selected. A cross-harness
    # comparison only means something when both sides served the same stack, and
    # the biggest way that breaks is silent: the orchestrator's launch script
    # exports the platform kernel preset and GEAK's own adapter does not, so the
    # identical config serves measurably slower. Recording the launcher next to
    # the kernels it produced makes that visible in the interface file.
    launcher = os.environ.get("BENCH_LAUNCHER", "native")
    recipe_aligned = launcher == "magpie"
    serving_stack = {
        "launcher": launcher,
        "launch_script": os.environ.get("MAGPIE_LAUNCH_SCRIPT", ""),
        "launch_script_source": os.environ.get("MAGPIE_LAUNCH_SCRIPT_SOURCE", ""),
        "recipe_aligned_with_orchestrator": recipe_aligned,
        "baseline": _serving_stack_signals(eval_dir / "baseline" / "server.log"),
        "validation_base": _serving_stack_signals(
            eval_dir / "validation" / "base" / "server.log"
        ),
    }
    baseline_alignment = _build_baseline_alignment(
        same_config_divergence_pct, recipe_aligned
    )
    baseline_basis = {
        # GEAK's own measured baseline (Hyperloom-accepted config = fair engagement baseline; gating uses this).
        "geak_measured_baseline_tok_s": geak_baseline or None,
        # Which leg the denominator above came from, so a reviewer can tell a
        # same-session A/B from a Setup-time comparison at a glance.
        "baseline_basis_source": baseline_basis_source or None,
        # The Setup-time baseline, kept for audit even when it is no longer the
        # denominator, plus how far the box moved between Setup and Validate.
        # A large drift means the Setup number was never a valid denominator.
        "setup_baseline_tok_s": setup_baseline or None,
        "baseline_drift_pct": (
            round((geak_baseline - setup_baseline) / setup_baseline * 100.0, 3)
            if (geak_baseline > 0.0 and setup_baseline > 0.0) else None
        ),
        # Hyperloom's own measured baseline forwarded in the handoff (the orchestrator reference).
        "orchestrator_baseline_tok_s": orch_baseline or None,
        # Audit-only comparison with the RAW session baseline. This includes
        # accepted upstream config gain and is not a measurement-drift signal.
        "raw_session_baseline_divergence_pct": raw_session_divergence_pct,
        # PURE cross-harness measurement residue: GEAK baseline vs the
        # orchestrator's throughput on the SAME (accepted) config. Both sides
        # run the identical config, so this isolates client/protocol/warm-cold
        # differences from the config gain. This is the primary alignment metric.
        "current_best_same_config_divergence_pct": same_config_divergence_pct,
        # Backward-compatible alias consumed by existing orchestrators.
        "measurement_divergence_pct": same_config_divergence_pct,
        "orchestrator_best_tput_same_config": orch_same_cfg or None,
        # Gain measured against the ORCHESTRATOR baseline (what Hyperloom sees end-to-end).
        "gain_vs_orchestrator_baseline": (
            round(geak_final / orch_baseline, 4)
            if (geak_final > 0 and orch_baseline > 0) else None
        ),
        # Measurement-protocol provenance so the comparison is self-describing.
        "bench_client": os.environ.get("BENCH_CLIENT", "native"),
        "bench_protocol": h.get("bench_protocol") or {},
        "baseline_config": {
            "accepted_flags": h.get("accepted_flags", "") or "",
            "accepted_env": h.get("accepted_env", "") or "",
        },
    }

    # ── cold/hot alignment metrics (double-check; never changes the primary
    # final_throughput_tok_s / throughput_speedup Hyperloom promotes) ─────────
    # Hyperloom's leaderboard anchor baseline_tput is a COLD single round; GEAK's
    # final is a HOT median, so the promoted cold-to-... comparison mixes thermal
    # states. We surface every well-defined speedup so a reviewer can tell a real
    # win from a warm/cold measurement artefact:
    #   * hot_speedup      = GEAK hot final  / Hyperloom HOT baseline  (hot-to-hot, cross-harness)
    #   * hot_geak_speedup = GEAK hot final  / GEAK  hot baseline      (within-GEAK, harness-internal)
    #   * cold_speedup     = GEAK cold final / Hyperloom COLD baseline (cold-to-cold, matches leaderboard state)
    #   * cold_geak_speedup= GEAK cold final / GEAK  cold baseline     (within-GEAK cold, if measured)
    # The cold numbers are populated only when BENCH_COLD_FINAL=1 added a cold
    # round to bench_e2e.sh (else None). All ratios are None when an input is
    # missing, so a standalone / orchestrator-less run carries the block harmlessly.
    orch_hot_baseline = read_orchestrator_hot_baseline(h)
    geak_hot_final = geak_final
    geak_hot_baseline = geak_baseline
    geak_cold_final = final_summary.get("cold_output_throughput_tok_s")
    # Pair the cold legs the same way the hot ones are paired. The Setup-time
    # cold round ran on a genuinely cold box; the Validate-time one did not, so
    # dividing the second by the first charges the baseline for a cache fill the
    # final never paid and returns the difference as speedup. Prefer the base leg
    # re-measured alongside the final; fall back to Setup only when Validate did
    # not run one, and let cold_penalty_pct_* below expose how far apart the two
    # legs' cold rounds really are.
    if base_leg_summary.get("cold_output_throughput_tok_s") is not None:
        geak_cold_baseline = base_leg_summary.get("cold_output_throughput_tok_s")
        cold_baseline_hot_leg = base_leg_summary.get("throughput_tok_s_median")
        cold_pairing = "same_session"
    else:
        geak_cold_baseline = baseline_summary.get("cold_output_throughput_tok_s")
        cold_baseline_hot_leg = baseline_summary.get("throughput_tok_s_median")
        cold_pairing = "setup_vs_validate" if geak_cold_baseline is not None else None
    alignment_metrics = {
        "geak_hot_final_tok_s": geak_hot_final or None,
        "geak_hot_baseline_tok_s": geak_hot_baseline or None,
        "geak_cold_final_tok_s": geak_cold_final,
        "geak_cold_baseline_tok_s": geak_cold_baseline,
        "orchestrator_cold_baseline_tok_s": orch_baseline or None,   # == handoff.raw_baseline_tput (leaderboard anchor)
        "orchestrator_hot_baseline_tok_s": orch_hot_baseline or None,
        "hot_speedup": _safe_ratio(geak_hot_final, orch_hot_baseline),
        "hot_geak_speedup": _safe_ratio(geak_hot_final, geak_hot_baseline),
        "cold_speedup": _safe_ratio(geak_cold_final, orch_baseline),
        "cold_geak_speedup": _safe_ratio(geak_cold_final, geak_cold_baseline),
        # Which two rounds cold_geak_speedup divides: "same_session" is a valid
        # A/B, "setup_vs_validate" is a comparison across thermal states and the
        # ratio should be read as such (or ignored).
        "cold_pairing": cold_pairing,
        # How much each leg's cold round lost against its OWN hot median. These
        # tell you whether the cold rounds are comparable at all: similar
        # penalties mean both paid a similar cache-fill cost, while a large
        # baseline penalty next to a small final one is the signature of a
        # "cold" round that ran on an already-warm box.
        "cold_penalty_pct_baseline": _cold_penalty_pct(
            geak_cold_baseline, cold_baseline_hot_leg
        ),
        "cold_penalty_pct_final": _cold_penalty_pct(
            geak_cold_final, final_summary.get("throughput_tok_s_median")
        ),
    }

    # ── final-throughput BASIS ─────────────────────────────────────────────────
    # Always the HOT median, i.e. the same basis the baseline above is measured
    # on. This used to switch to the cold round whenever the cold ratio looked
    # like a gain, which replaced the numerator and the reported speedup but
    # left the hot baseline in place as the denominator — so result.json shipped
    # a cold-over-cold ratio next to a cold-over-hot pair and the two disagreed
    # by up to ten points. The cold rounds could not carry the comparison
    # anyway: only the first bench of a session runs cold (see bench_e2e.sh), so
    # the baseline's cold round pays a cache-fill cost the final's never does.
    # Cold numbers remain in alignment_metrics as a diagnostic.
    final_tput_out = geak_final
    final_basis = "hot"
    alignment_metrics["final_basis"] = final_basis

    # ── speedup self-consistency invariant ───────────────────────────────────
    # Everything above can move the numerator or the denominator independently:
    # the final is pinned to the hot number, the same-session re-base can replace
    # the baseline, and the reported speedup arrives precomputed from a third place.
    # Whatever the path, the last word belongs to the pair we actually publish —
    # a consumer that recomputes final/baseline must land on the same number we
    # printed. When they disagree the reported ratio is describing some other
    # pair, so rebuild it from ours and re-derive the ok/no_gain gate with it.
    promoted_final = _positive_finite_float(final_tput_out)
    speedup_basis = "workflow_return"
    speedup_as_returned = speedup
    if geak_baseline > 0.0 and promoted_final > 0.0:
        pair_ratio = promoted_final / geak_baseline
        if abs(pair_ratio - speedup) > SPEEDUP_SELF_CONSISTENCY_TOL:
            speedup = round(pair_ratio, 6)
            status = "ok" if speedup > 1.0 else "no_gain"
            speedup_basis = "final_over_baseline"
    alignment_metrics["speedup_basis"] = speedup_basis
    # Only populated when we had to override, so its presence is the signal.
    alignment_metrics["speedup_as_returned"] = (
        speedup_as_returned if speedup_basis == "final_over_baseline" else None
    )

    # ── validation evidence ──────────────────────────────────────────────────
    # A speedup number on its own cannot be judged: 1.01x is a solid win on a
    # bench that repeats within 0.2% and pure noise on one that swings 3%. This
    # block carries what a reader needs to decide — the arbitration verdict, the
    # run-to-run spread of each leg, the Director's noise band, and whether the
    # delta clears them. Reported, never enforced: nothing here changes status.
    delta_pct = round((speedup - 1.0) * 100.0, 3)
    base_spread_pct = _positive_finite_float(
        base_leg_summary.get("output_throughput_tok_s_spread_pct")
        or (validation.get("base_block") or {}).get("spread_pct")
        or baseline_summary.get("output_throughput_tok_s_spread_pct")
    )
    final_spread_pct = _positive_finite_float(
        final_summary.get("output_throughput_tok_s_spread_pct")
        or (validation.get("final_block") or {}).get("spread_pct")
    )
    noise_band_pct = _positive_finite_float(validation.get("noise_band_pct"))
    # A delta smaller than the benches' own scatter, or than the band the
    # Director declared, is not measurable with the data we have.
    significance_threshold_pct = max(noise_band_pct, base_spread_pct, final_spread_pct)
    validation_evidence = {
        "validation_status": (
            validation.get("validation_status")
            or wf.get("validation_status")
            or None
        ),
        "speedup_basis": speedup_basis,
        "delta_pct": delta_pct,
        "noise_band_pct": noise_band_pct or None,
        "baseline_spread_pct": base_spread_pct or None,
        "final_spread_pct": final_spread_pct or None,
        "significance_threshold_pct": significance_threshold_pct or None,
        "delta_exceeds_noise": (
            abs(delta_pct) > significance_threshold_pct
            if significance_threshold_pct > 0.0 else None
        ),
        # Stricter than the threshold above: do the two legs' spread intervals
        # (median +/- spread/2) stay apart? Overlapping intervals mean a single
        # pair of runs could have produced either ordering. None unless BOTH
        # legs reported a spread — one unknown side cannot be assumed tight.
        "spreads_non_overlapping": (
            geak_baseline * (1.0 + base_spread_pct / 200.0)
            < promoted_final * (1.0 - final_spread_pct / 200.0)
            if (geak_baseline > 0.0 and promoted_final > 0.0
                and base_spread_pct > 0.0 and final_spread_pct > 0.0) else None
        ),
        "beats_orchestrator_same_config": (
            promoted_final > orch_same_cfg if orch_same_cfg > 0.0 else None
        ),
        # An intermediate A/B claimed a win that the arbitrated re-check did not
        # confirm. We keep Validate's verdict (see the reconcile above); this
        # flag is how the disagreement stays visible instead of disappearing.
        "intermediate_win_not_confirmed": intermediate_win_not_confirmed or None,
        # The Validate bench produced no final at all and the number above came
        # from the best accepted intermediate A/B on disk.
        "validate_final_missing": bool(wf.get("validate_final_missing")) or None,
        # How a disk-salvaged headline was chosen out of the candidate pool
        # (see _recover_best_intermediate_win). Absent for an arbitrated result.
        "recovery": wf.get("recovery_evidence") or None,
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "result_source": result_source,
        "eval_dir": str(eval_dir),
        "baseline_throughput_tok_s": geak_baseline,
        # Promoted final is ALWAYS the HOT median (see the final-basis selection
        # above); final_throughput_basis is therefore always "hot". Cold rounds,
        # when BENCH_COLD_FINAL enables them, live only in alignment_metrics as a
        # diagnostic and never become the headline.
        "final_throughput_tok_s": float(final_tput_out or 0.0),
        "final_throughput_basis": final_basis,
        "throughput_speedup": speedup,
        "output_parity": wf.get("output_parity") or validation.get("output_parity") or "unknown",
        # Latency measurement protocol (median ms), aligned field names with Hyperloom. Prefer the
        # value carried on the workflow return / recovered win (e.g. the accepted
        # A/B's candidate leg), then the same-session final/baseline summaries.
        "ttft_ms": wf.get("ttft_ms") or final_summary.get("ttft_ms_median") or baseline_summary.get("ttft_ms_median"),
        "tpot_ms": wf.get("tpot_ms") or final_summary.get("tpot_ms_median") or baseline_summary.get("tpot_ms_median"),
        # Sweep-reuse handles (see interface/run_e2e.md).
        "final_launch_script": final_launch,
        "bench_script": str(eval_dir / "bench_e2e.sh"),
        # Empty string == this run produced no reusable artifact of that kind.
        # Never a path to a directory/diff that holds nothing (see
        # _material_overlay_path / _material_patch_path).
        "final_patch": _material_patch_path(eval_dir, wf),
        "final_overlay": _material_overlay_path(eval_dir, wf),
        # Measurement basis: read back from the bench_summary.json that actually produced these numbers
        # (bench_e2e.sh records "aggregate_output_tok_s" or "aggregate_total_token_tok_s" per E2E_METRIC),
        # so the label never lies about the basis. Falls back to output when neither summary carries it.
        # See run_e2e.md alignment table.
        "metric_basis": (
            final_summary.get("metric_basis")
            or baseline_summary.get("metric_basis")
            or "aggregate_output_tok_s"
        ),
        # Which bench client measured these numbers. "inferencex" => identical
        # client to Hyperloom/Magpie (benchmark_serving.py); "native" => the
        # backend's own client (small cross-harness differences may remain).
        "bench_client": os.environ.get("BENCH_CLIENT", "native"),
        # Measurement protocol forwarded from the handoff, surfaced at TOP LEVEL
        # (not only inside baseline_basis) so a sweep/validated reuse can pin the
        # SAME num_prompts / random_range_ratio / num_warmups / seed the headline
        # result was measured with. Empty {} when running standalone (no handoff),
        # in which case the reuse path keeps bench_e2e.sh's per-conc defaults.
        "bench_protocol": h.get("bench_protocol") or {},
        # The kernels are only extracted/validated at this single workload point;
        # the caller must redo parity on out-of-regime sweep points.
        "validated_regimes": [workload],
        # What the kernel phase actually did (req: report must carry this).
        "accepted_kernels": wf.get("accepted_kernels") or [],
        "accepted_heads": wf.get("accepted_heads") or [],
        "accepted_config": wf.get("accepted_config") or {},
        # Self-describing baseline measurement-protocol + Hyperloom cross-check (see baseline_basis above).
        "baseline_basis": baseline_basis,
        # Reliability classification is independent of the optimization status.
        "baseline_alignment": baseline_alignment,
        # WHO launched the servers these numbers were measured on, and which
        # kernels those servers selected. This is what tells a reviewer whether
        # baseline_alignment's divergence is a measurement signal at all.
        "serving_stack": serving_stack,
        # Whether the reported delta is distinguishable from measurement noise,
        # and what the arbitration actually concluded (see above). Audit only.
        "validation_evidence": validation_evidence,
        # Cold/hot speedup cross-checks (double-check only; see alignment_metrics above).
        # Does NOT change the promoted final_throughput_tok_s / throughput_speedup.
        "alignment_metrics": alignment_metrics,
        "report_path": wf.get("report_path") or str(eval_dir / "final_report.md"),
    }


def _format_optional_number(
    value: Any,
    *,
    digits: int = 2,
    suffix: str = "",
) -> str:
    try:
        return f"{float(value):.{digits}f}{suffix}"
    except (TypeError, ValueError):
        return "unavailable"


def _render_baseline_alignment_section(result: dict[str, Any]) -> str:
    """Render a deterministic, same-config-first report section."""
    basis = result.get("baseline_basis") or {}
    alignment = result.get("baseline_alignment") or {}
    geak_baseline = _format_optional_number(
        basis.get("geak_measured_baseline_tok_s"), digits=3, suffix=" tok/s"
    )
    same_config_baseline = _format_optional_number(
        basis.get("orchestrator_best_tput_same_config"),
        digits=3,
        suffix=" tok/s",
    )
    same_config_divergence = _format_optional_number(
        basis.get("current_best_same_config_divergence_pct"),
        digits=2,
        suffix="%",
    )
    raw_baseline = _format_optional_number(
        basis.get("orchestrator_baseline_tok_s"), digits=3, suffix=" tok/s"
    )
    raw_divergence = _format_optional_number(
        basis.get("raw_session_baseline_divergence_pct"),
        digits=2,
        suffix="%",
    )
    threshold = _format_optional_number(
        alignment.get("warning_threshold_pct"), digits=1, suffix="%"
    )
    status = str(alignment.get("status") or "unavailable")
    stack = result.get("serving_stack") or {}
    launcher = str(stack.get("launcher") or "unknown")
    recipe_aligned = bool(alignment.get("recipe_aligned_with_orchestrator", True))
    recipe_caveat = (
        []
        if recipe_aligned
        else [
            "",
            (
                "The servers behind these numbers were launched by GEAK's own "
                "backend adapter, not by the upstream orchestrator's launch "
                "script. That script owns the platform kernel preset, "
                "`--trust-remote-code` and the gpu-memory-utilization default, so "
                "the two harnesses did not serve the same stack. Read the "
                "same-config divergence above as a launch-recipe difference, not "
                "as a measurement or hardware signal, and check "
                "`serving_stack.baseline.kernel_picks` for which kernels each "
                "side actually selected."
            ),
        ]
    )
    return "\n".join(
        [
            BASELINE_ALIGNMENT_BEGIN,
            "## Baseline alignment",
            "",
            "Primary same-config comparison:",
            "",
            f"- GEAK measured baseline: {geak_baseline}",
            (
                "- Upstream current-best baseline on the same config: "
                f"{same_config_baseline}"
            ),
            f"- Same-config divergence: {same_config_divergence}",
            f"- Alignment status: `{status}` (warning threshold: ±{threshold})",
            f"- Server launch recipe: `{launcher}`",
            *recipe_caveat,
            "",
            "Raw-session audit comparison:",
            "",
            f"- Upstream raw-session baseline: {raw_baseline}",
            f"- Raw-session baseline divergence: {raw_divergence}",
            "",
            (
                "The raw-session baseline predates accepted upstream configuration "
                "changes and is not expected to match the GEAK baseline. Its "
                "divergence includes previously accepted configuration gains and "
                "is not a pure measurement-drift signal."
            ),
            BASELINE_ALIGNMENT_END,
        ]
    )


def _upsert_marked_markdown_section(
    text: str,
    section: str,
    *,
    begin_marker: str,
    end_marker: str,
) -> str:
    """Replace a generated section or insert it after the first H1."""
    begin = text.find(begin_marker)
    end = text.find(end_marker)
    if begin >= 0 and end >= begin:
        end += len(end_marker)
        return text[:begin] + section + text[end:]

    lines = text.splitlines(keepends=True)
    for index, line in enumerate(lines):
        if line.startswith("# "):
            return (
                "".join(lines[: index + 1])
                + "\n"
                + section
                + "\n\n"
                + "".join(lines[index + 1 :])
            )
    return section + "\n\n" + text


def _update_baseline_alignment_reports(result: dict[str, Any]) -> list[str]:
    """Idempotently add baseline alignment to every existing primary report."""
    section = _render_baseline_alignment_section(result)
    eval_dir = str(result.get("eval_dir") or "").strip()
    if not eval_dir:
        return []
    eval_root = Path(eval_dir).resolve(strict=False)

    candidates: list[Path] = []
    report_path = str(result.get("report_path") or "").strip()
    if report_path:
        report_candidate = Path(report_path)
        if not report_candidate.is_absolute():
            report_candidate = eval_root / report_candidate
        candidates.append(report_candidate)
    candidates.append(eval_root / "final_report.md")

    updated: list[str] = []
    seen: set[str] = set()
    for path in candidates:
        resolved_path = path.resolve(strict=False)
        try:
            resolved_path.relative_to(eval_root)
        except ValueError:
            continue
        key = str(resolved_path)
        if key in seen:
            continue
        seen.add(key)
        if not resolved_path.is_file():
            continue
        original = resolved_path.read_text(encoding="utf-8")
        rendered = _upsert_marked_markdown_section(
            original,
            section,
            begin_marker=BASELINE_ALIGNMENT_BEGIN,
            end_marker=BASELINE_ALIGNMENT_END,
        )
        if rendered != original:
            tmp = resolved_path.with_name(resolved_path.name + ".tmp")
            tmp.write_text(rendered, encoding="utf-8")
            os.replace(tmp, resolved_path)
        updated.append(str(resolved_path))
    return updated


# ---------------------------------------------------------------------------
# Handoff resilience: persist + disk-recover the workflow return.
#
# The workflow return (carrying eval_dir + accepted_kernels/config) is the ONE
# fragile link — it is scraped from the agent transcript. When that scrape
# fails the whole run was historically discarded as ``workflow_parse_error``
# even though the optimizer's artifacts (director_e2e_validation.json, final/
# bundle, +gain) are all on disk. These helpers (a) persist the parsed return
# next to the artifacts so a re-run/recovery never re-scrapes, and (b) rebuild
# it from the on-disk artifacts when the scrape failed. Both are GENERAL: no
# model/run-specific assumptions — they key only off the stable artifact
# layout the workflow always writes.
# ---------------------------------------------------------------------------
WORKFLOW_RETURN_FILE = "workflow_return.json"
KERNEL_JOURNEY_FILE = "kernel_journey.json"


def _git_short_sha(root: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:
        return ""


def _discover_eval_dir(exp_root: Path) -> Path | None:
    """Find the workflow's eval_dir under ``exp_root`` without the scraped return.

    The workflow always creates ``<exp_root>/e2e_*`` and writes
    ``director_e2e_validation.json`` (Validate phase) / a ``final/`` bundle into
    it. Pick the most-recently-modified ``e2e_*`` dir that carries one of those
    completion markers; fall back to the newest ``e2e_*`` dir.

    A pinned ``GEAK_EVAL_DIR`` (set by main() from the single eval_dir
    map_args minted for this run) short-circuits the glob/guess: recovery then
    targets EXACTLY the dir this run used, never a sibling from another run.
    """
    pinned = os.environ.get("GEAK_EVAL_DIR", "").strip()
    if pinned and Path(pinned).is_dir():
        return Path(pinned)
    if not exp_root.is_dir():
        return None
    cands = sorted(
        (p for p in exp_root.glob("e2e_*") if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not cands:
        return None
    for p in cands:
        if (p / "director_e2e_validation.json").is_file() or (p / "final").is_dir():
            return p
    return cands[0]


def _enumerate_overlay_kernels(eval_dir: Path) -> list[str]:
    """Recover accepted authored-kernel names from the stable overlay layout.

    Each accepted authored kernel leaves an ``overlay/cand_<name>`` directory;
    this is a deterministic on-disk enumeration usable when the scraped return
    (the only structured ``accepted_kernels`` source) was lost.
    """
    names: list[str] = []
    for base in (eval_dir / "overlay", eval_dir / "final" / "overlay"):
        if not base.is_dir():
            continue
        for d in sorted(base.glob("cand_*")):
            if d.is_dir():
                name = d.name[len("cand_"):]
                if name and name not in names:
                    names.append(name)
    return names


def _recover_workflow_return(exp_root: Path) -> dict | None:
    """Rebuild the workflow return from on-disk artifacts (scrape-independent).

    Returns ``None`` when no completed eval_dir is discoverable (e.g. the run
    died before Validate, so there is genuinely nothing to keep). Otherwise
    returns a workflow-return-shaped dict good enough for
    :func:`normalize_result` (which itself reads most numbers from disk).
    """
    eval_dir = _discover_eval_dir(exp_root)
    if eval_dir is None:
        return None
    # Prefer the WORKFLOW's own canonical artifact (e2e_workflow.js persists its
    # schema-validated return to workflow_return.json as its final act; main()
    # also persists a successfully-scraped live return there). Trust it ONLY when
    # it is that authoritative artifact — i.e. it has NO recovery markers. A file
    # we previously wrote from our OWN best-effort disk recovery (recovered_*
    # flags) must be re-derived fresh here, otherwise a stale reconstruction would
    # permanently shadow later recovery improvements (e.g. newly-extracted latency).
    persisted = _read_json(eval_dir / WORKFLOW_RETURN_FILE)
    if persisted.get("eval_dir") and not any(
        persisted.get(k)
        for k in ("recovered_from_disk", "recovered_intermediate", "recovered_no_gain")
    ):
        return persisted
    validation = _read_json(eval_dir / "director_e2e_validation.json")
    if not validation:
        # No final Validate marker => the director never synthesized its json
        # (run killed mid-Validate, or torn down before it wrote). Recover in
        # priority order so a COMPLETED run is NEVER discarded as a parse error:
        #   1. the best gate==accepted intermediate win (a real measured gain),
        #   2. else, if a baseline was measured but nothing was accepted, a
        #      legitimate NO_GAIN run (the optimizer correctly did no harm).
        win = _recover_best_intermediate_win(eval_dir)
        if win is not None:
            return win
        return _recover_completed_no_gain(eval_dir)
    serving = validation.get("serving_config") or {}
    accepted_config = {
        "flags": serving.get("final_flags") or serving.get("baseline_flags") or "",
        "env": serving.get("final_env") or serving.get("baseline_env") or "",
    }

    # The director records throughput/speedup in NESTED blocks (the top-level
    # keys normalize_result would otherwise read don't exist here). Pull them
    # from the known blocks with general fallbacks; never fabricate.
    def _first(*vals: Any) -> Any:
        for v in vals:
            if isinstance(v, (int, float)) and v:
                return float(v)
        return None

    # The director schema has evolved across workflow versions. Read it
    # SCHEMA-ROBUSTLY so the recovered numbers (and the kernel_journey e2e
    # attribution below) survive a schema change:
    #   * current (VALIDATE_SCHEMA, flat): baseline_throughput_tok_s,
    #     director_verified_throughput_tok_s, throughput_speedup
    #   * 20260615-era (flat, different names): provided_baseline_throughput,
    #     final.median / drift_corrected_baseline.median, delta_pct_drift_corrected
    #   * earlier (nested blocks): vs_provided_baseline.*, base_block.*, etc.
    # Take the FIRST present (never fabricate). _nest() reads a nested median.
    def _nest(key: str) -> Any:
        v = validation.get(key)
        return v.get("median") if isinstance(v, dict) else None

    vs_base = validation.get("vs_provided_baseline") or {}
    arb = validation.get("arbitration") or {}
    drift = validation.get("drift_corrected_same_session") or {}
    final_block = validation.get("final_block") or {}
    base_block = validation.get("base_block") or {}
    baseline_tput = _first(
        validation.get("baseline_throughput_tok_s"),       # current flat
        validation.get("provided_baseline_throughput"),    # 20260615 flat
        _nest("drift_corrected_baseline"),
        vs_base.get("baseline_throughput_tok_s"),           # nested (legacy)
        base_block.get("warm_median_tok_s"),
    )
    final_tput = _first(
        validation.get("director_verified_throughput_tok_s"),  # current flat
        _nest("final"),                                        # 20260615 flat
        validation.get("claimed_throughput"),
        arb.get("director_verified_throughput_tok_s"),         # nested (legacy)
        vs_base.get("final_warm_median_tok_s"),
        final_block.get("warm_median_tok_s"),
    )
    speedup = _first(
        validation.get("throughput_speedup"),               # current + 20260615 flat
        vs_base.get("speedup"),
        drift.get("speedup_warm"),
    )
    if speedup is None and baseline_tput and final_tput:
        speedup = final_tput / baseline_tput
    overall_delta_pct = _first(
        validation.get("delta_pct_drift_corrected"),        # 20260615 flat
        vs_base.get("delta_pct"),
        drift.get("delta_pct_warm"),
    )
    if overall_delta_pct is None and speedup is not None:
        overall_delta_pct = (speedup - 1.0) * 100.0

    # accepted_kernels structured data only lived in the scraped return; recover
    # names from the overlay layout so the kernel_journey still names what landed.
    names = _enumerate_overlay_kernels(eval_dir)
    accepted_kernels = [
        {"short_name": n, "kind": "authored", "backend": "geak"} for n in names
    ]
    # Sound general attribution: when EXACTLY one kernel was accepted it is, by
    # definition, responsible for the whole measured e2e delta — credit it.
    # With >1 we cannot split, so leave per-kernel gain null (the headline gain
    # still folds via the geak section + cumulative_gain).
    if len(accepted_kernels) == 1 and overall_delta_pct is not None:
        accepted_kernels[0]["e2e_delta_pct"] = overall_delta_pct

    return {
        "eval_dir": str(eval_dir),
        "throughput_speedup": speedup,
        "baseline_throughput_tok_s": baseline_tput,
        "final_throughput_tok_s": final_tput,
        "output_parity": validation.get("output_parity"),
        "validation_status": validation.get("validation_status"),
        "final_overlay": validation.get("final_overlay"),
        "final_launch_script": validation.get("final_launch_script"),
        "accepted_config": accepted_config,
        "accepted_kernels": accepted_kernels,
        "accepted_heads": [],
        "recovered_from_disk": True,
    }


def _ir_get(ir: dict, *keys: str) -> Any:
    """Read a field from an ``integrate_result.json`` that may be FLAT or NESTED.

    The e2e_integrator writes the measured numbers in either shape across
    workflow versions:
      * FLAT (older / test fixtures): ``e2e_delta_pct``, ``e2e_throughput_tok_s``,
        ``ref_med``, ``cand_med``, ``apply_env``, ``apply_flags`` at top level.
      * NESTED (current integrator output): the numbers live under an ``e2e``
        block (``delta_pct``, ``cand_median_tok_s``, ``ref_median_tok_s``) and the
        config under an ``accepted_config`` block (``apply_env`` / ``apply_flags``).
    Reading both is what lets a real accepted win be recovered regardless of which
    shape the integrator emitted (a nested-only result used to read as 0 delta /
    0 tput and get silently skipped). Returns the first present non-None value.
    """
    sources: list[dict] = [ir]
    for sub in ("e2e", "accepted_config"):
        v = ir.get(sub)
        if isinstance(v, dict):
            sources.append(v)
    for k in keys:
        for src in sources:
            val = src.get(k)
            if val is not None:
                return val
    return None


def _ir_float(ir: dict, *keys: str) -> float:
    v = _ir_get(ir, *keys)
    try:
        return float(v) if v is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def _amdahl_ceiling_pct(pct_gpu_time: float, isolated: float) -> float:
    """Largest e2e gain an op can produce, from its GPU-time share and isolated speedup.

    Port of ``amdahlCeilingPct`` in e2e_workflow.js. ``inf`` when either input is
    missing, so an unknown profile never flags anything (fail-open).
    """
    share = max(0.0, min(1.0, (pct_gpu_time or 0.0) / 100.0))
    speedup = isolated if (isolated and isolated > 1.0) else 1.0
    if share <= 0.0 or speedup <= 1.0:
        return math.inf
    return (1.0 / (1.0 - share * (1.0 - 1.0 / speedup)) - 1.0) * 100.0


def _parity_is_soft(ir: dict) -> bool:
    """Whether the candidate's acceptance rests on a SAMPLED accuracy probe.

    Byte-exact parity is a hard correctness guarantee, so a speedup above the
    Amdahl ceiling is believed. A sampled accuracy gate can be squeaked past by
    degenerate output, so an impossible speedup there is treated as corruption.

    Diverges from ``parityIsSoft`` in e2e_workflow.js in one place: the JS falls
    back to the run's ``accuracy_gate`` arg when the label is absent or
    unrecognized, and that arg is not in any on-disk artifact. An unlabelled
    candidate is therefore treated as soft — the guard still needs the delta to
    exceed twice the ceiling before it fires, and a missing profile makes the
    ceiling infinite, so this cannot drop a win on schema grounds alone.
    """
    kind = str(_ir_get(ir, "parity_kind") or "").strip().lower()
    if kind in ("byte_exact", "none"):
        return False
    return True


def _is_implausible_speedup(ir: dict, delta_pct: float, ceiling_pct: float) -> bool:
    """A gain so far past the op's ceiling that it reads as corruption, not a win."""
    if not _parity_is_soft(ir) or not math.isfinite(ceiling_pct):
        return False
    return delta_pct > ceiling_pct * (1.0 + IMPLAUSIBLE_SPEEDUP_MARGIN) + 1e-9


def _integrate_candidates(eval_dir: Path) -> list[dict]:
    """Every integrate A/B on disk, normalized and judged.

    Each entry keeps the candidate's OWN paired legs (``ref_med`` / ``cand_med``)
    separate from ``e2e_throughput_tok_s``, which is not this candidate's
    measurement at all: the workflow uses it as the running stack throughput
    (``curTput``) and carries the previous value forward on a reject. Mixing the
    two — a cumulative numerator over a local denominator — describes no A/B
    that was ever run.

    Eligibility mirrors what the live workflow requires before it banks a
    candidate (``integAccepted`` + the parity check), so a candidate the live
    path would refuse cannot slip in through recovery.
    """
    records: list[dict] = []
    seen: set[str] = set()
    for base in (eval_dir / "overlay", eval_dir / "final" / "overlay"):
        if not base.is_dir():
            continue
        for cand in sorted(base.glob("cand_*")):
            if cand.name in seen:
                continue
            ir = _read_json(cand / "integrate_result.json")
            if not ir:
                continue
            seen.add(cand.name)
            gate = str(ir.get("gate") or "")
            delta_pct = _ir_float(ir, "e2e_delta_pct", "delta_pct")
            ceiling_pct = _amdahl_ceiling_pct(
                _ir_float(ir, "pct_gpu_time"), _ir_float(ir, "isolated_speedup")
            )
            implausible = _is_implausible_speedup(ir, delta_pct, ceiling_pct)
            parity = str(ir.get("output_parity") or "")
            separated = ir.get("separated")
            if separated is None:
                separated = (ir.get("pooled_all_repeats") or {}).get("separated")
            if separated is None:
                separated = _ir_get(ir, "non_overlapping")
            records.append({
                "dir": cand,
                "ir": ir,
                "short_name": str(ir.get("short_name") or cand.name[len("cand_"):]),
                "gate": gate,
                "delta_pct": delta_pct,
                "ref_med": _ir_float(ir, "ref_med", "ref_median_tok_s"),
                "cand_med": _ir_float(ir, "cand_med", "cand_median_tok_s"),
                "stack_tput": _ir_float(ir, "e2e_throughput_tok_s"),
                "amdahl_ceiling_pct": (
                    round(ceiling_pct, 3) if math.isfinite(ceiling_pct) else None
                ),
                "implausible": implausible,
                "output_parity": parity or None,
                "separated": separated,
                # winner_kind in {"env","config","flags"} => config-only.
                "is_kernel": _ir_get(ir, "winner_kind") not in ("env", "config", "flags"),
                "eligible": (
                    gate in ("accepted", "stack")
                    and delta_pct > 0.0
                    and parity != "fail"
                    and ir.get("ab_complete") is not False
                    and not implausible
                ),
            })
    return records


def _recover_best_intermediate_win(eval_dir: Path) -> dict | None:
    """Salvage the best accepted intermediate win when the run died BEFORE Validate.

    The whole-pipeline workflow records each accepted config/kernel integrate as
    ``overlay/<cand>/integrate_result.json`` with a measured e2e delta + gate.
    When no ``director_e2e_validation.json`` exists (the run was killed
    mid-pipeline), salvage the best one so a real, parity-checked win is NEVER
    silently discarded.

    Selection follows the integrator's own vocabulary and its own numbers:

    * ``accepted`` outranks ``stack``. The integrator writes ``stack`` to mean
      "non-negative, engaged, parity-safe — carry it forward to compound, but it
      is NOT a standalone win and the Director decides the headline". Promoting
      one as the headline says the opposite of what the gate says, so a
      stack-only salvage is returned flagged as provisional.
    * Rank by each candidate's own ``e2e_delta_pct``, not by absolute
      throughput. Every candidate was measured against its own reference leg at
      its own point in the session, so absolute values are not comparable
      across candidates — ranking by them preferentially picks whichever
      candidate happened to draw the slowest reference, which is a property of
      the box, not of the kernel.
    * Report the winner's own two legs as the pair. See
      :func:`_integrate_candidates` for why ``e2e_throughput_tok_s`` is not one
      of them.

    Schema-robust: the integrator's integrate_result.json may carry the numbers
    flat or nested under ``e2e`` / ``accepted_config`` (see :func:`_ir_get`); both
    are read. Returns a workflow-return-shaped dict (status derived later by
    :func:`normalize_result`) or ``None`` when nothing acceptable is on disk.
    """
    candidates = _integrate_candidates(eval_dir)
    eligible = [c for c in candidates if c["eligible"]]
    if not eligible:
        return None
    # Candidates sharing a short_name are competing IMPLEMENTATIONS of one kernel
    # — the workflow benches several backends per op and banks at most one — not
    # stacked changes. Collapse them so the run is credited with distinct
    # kernels rather than with one kernel once per backend attempted.
    by_name: dict[str, dict] = {}
    for cand in eligible:
        incumbent = by_name.get(cand["short_name"])
        if incumbent is None or cand["delta_pct"] > incumbent["delta_pct"]:
            by_name[cand["short_name"]] = cand
    banked = sorted(by_name.values(), key=lambda c: -c["delta_pct"])
    accepted = [c for c in banked if c["gate"] == "accepted"]
    stack_only = not accepted
    best = max(
        accepted or banked,
        key=lambda c: (c["delta_pct"], c["cand_med"] or c["stack_tput"]),
    )

    ir = best["ir"]
    delta_pct = best["delta_pct"]
    if best["cand_med"] > 0.0 and best["ref_med"] > 0.0:
        final_tput, ref_med = best["cand_med"], best["ref_med"]
    else:
        # Only the cumulative stack number survived. Pair it with the
        # denominator its own delta implies rather than with a reference leg
        # from a different comparison.
        final_tput = best["cand_med"] or best["stack_tput"]
        ref_med = final_tput / (1.0 + delta_pct / 100.0)
    speedup = (final_tput / ref_med) if ref_med > 0 else (1.0 + delta_pct / 100.0)
    # A win recovered from an intermediate A/B never reached Finalize, so there is
    # no ``final/overlay`` bundle — but the code that produced the win is on disk
    # in the candidate's own overlay, and the integrator names it in
    # ``accepted_overlay`` (which for a stacked gate points at the base of the
    # stack, not at this candidate). Carrying it forward is the difference between
    # a reusable win and a number the caller cannot reproduce. Re-root by name
    # under this eval_dir when the recorded path no longer resolves (the run was
    # archived or moved), and fall back to the winning candidate's own directory.
    overlay_out = ""
    recorded_overlay = str(_ir_get(ir, "accepted_overlay") or "").strip()
    for cand_path in (
        Path(recorded_overlay) if recorded_overlay else None,
        (eval_dir / "overlay" / Path(recorded_overlay).name) if recorded_overlay else None,
        best["dir"],
    ):
        if cand_path is not None and _overlay_has_loadable_code(cand_path):
            overlay_out = str(cand_path)
            break
    # The deployed overlay is the STACK, so every eligible candidate is credited,
    # not just the one whose pair became the headline. Reporting one of several
    # stacked kernels loses both the attribution and the fact that more than one
    # change is live.
    accepted_kernels = [
        {"short_name": c["short_name"], "kind": "authored", "backend": "geak",
         "e2e_delta_pct": c["delta_pct"], "gate": c["gate"],
         "headline": c is best}
        for c in banked if c["is_kernel"] and c["short_name"]
    ]
    # Same reasoning for env/flags: a config win banked before a later kernel win
    # is still live on the server, and dropping it makes every downstream reuse
    # relaunch an UN-optimized server. The integrator emits these under either
    # ``apply_*`` (flat/nested-e2e schema) or ``accepted_*`` (summary schema).
    flags: list[str] = []
    env: list[str] = []
    for c in banked:
        for value, sink in (
            (str(_ir_get(c["ir"], "apply_flags", "accepted_flags") or ""), flags),
            (str(_ir_get(c["ir"], "apply_env", "accepted_env") or ""), env),
        ):
            if value and value not in sink:
                sink.append(value)
    return {
        "eval_dir": str(eval_dir),
        "throughput_speedup": speedup,
        "baseline_throughput_tok_s": ref_med,
        "final_throughput_tok_s": final_tput,
        "output_parity": ir.get("output_parity"),
        "validation_status": "recovered_intermediate",
        # Latency from the candidate (accepted) A/B leg when the integrator recorded
        # it (flat or nested) — so result.json carries real ttft/tpot even without a
        # final Validate bench. None when absent (never fabricated).
        "ttft_ms": _ir_get(ir, "ttft_ms_cand", "ttft_ms_median", "cand_ttft_ms"),
        "tpot_ms": _ir_get(ir, "tpot_ms_cand", "tpot_ms_median", "cand_tpot_ms"),
        # The accepted candidate's overlay, or "" for a config-only win (applied
        # through env/flags, so there is no overlay to hand back).
        "final_overlay": overlay_out,
        "final_launch_script": "",
        "accepted_config": {"flags": " ".join(flags), "env": " ".join(env)},
        "accepted_kernels": accepted_kernels,
        "accepted_heads": [],
        "recovered_from_disk": True,
        "recovered_intermediate": True,
        # No candidate cleared the integrator's own bar for a standalone win.
        "recovered_stack_provisional": stack_only or None,
        # How the headline was chosen, so picking a maximum out of N noisy
        # candidates is visible rather than implicit.
        "recovery_evidence": {
            "candidates_considered": len(candidates),
            "candidates_eligible": len(eligible),
            "distinct_kernels_banked": len(banked),
            "selected": best["short_name"],
            "selected_gate": best["gate"],
            "selection_basis": "max_e2e_delta_pct",
            "amdahl_ceiling_pct": best["amdahl_ceiling_pct"],
            "delta_over_amdahl_ceiling": (
                round(delta_pct / best["amdahl_ceiling_pct"], 2)
                if best["amdahl_ceiling_pct"] else None
            ),
            "distributions_separated": best["separated"],
            "stack_only": stack_only or None,
            "excluded_as_implausible": [
                c["short_name"] for c in candidates if c["implausible"]
            ] or None,
        },
    }


def _recover_completed_no_gain(eval_dir: Path) -> dict | None:
    """Synthesize a NO_GAIN return when a baseline was measured but nothing won.

    A run that measured a baseline and then REJECTED / failed to e2e-accept
    every candidate (e.g. the live op is already SOTA, or every integrate A/B
    was cut off) is a LEGITIMATE ``no_gain`` outcome — the optimizer correctly
    did no harm — NOT a runner error. The earlier recovery tiers only handle a
    present ``director_e2e_validation.json`` or a ``gate==accepted`` intermediate,
    so a clean no-win run used to fall through to ``None`` and get misreported as
    ``workflow_parse_error`` even though every artifact (measured baseline, final
    bundle, report) is on disk. This recovers the authoritative baseline so
    result.json reports ``no_gain`` instead.

    With NO accepted change the served path is unchanged, so final == baseline by
    construction (do-no-harm); speedup 1.0 -> :func:`normalize_result` => no_gain.
    Returns ``None`` only when no baseline throughput was ever measured (the run
    genuinely produced nothing to keep).
    """
    official = _read_json(eval_dir / "baseline" / "baseline_official.json")
    summary = _read_json(eval_dir / "baseline" / "bench_summary.json")
    baseline_tput = (
        official.get("baseline_throughput_tok_s")
        or official.get("plateau_median_tok_s")
        or summary.get("output_throughput_tok_s_median")
    )
    if not baseline_tput:
        return None
    try:
        baseline_tput = float(baseline_tput)
    except (TypeError, ValueError):
        return None
    return {
        "eval_dir": str(eval_dir),
        "throughput_speedup": 1.0,
        "baseline_throughput_tok_s": baseline_tput,
        # No accepted overlay/config => served path unchanged => final == baseline.
        "final_throughput_tok_s": baseline_tput,
        "output_parity": "n/a",
        "validation_status": "recovered_no_gain",
        "final_overlay": "",
        "final_launch_script": "",
        "accepted_config": {
            "flags": str(official.get("server_flags") or ""),
            "env": str(official.get("server_env") or ""),
        },
        "accepted_kernels": [],
        "accepted_heads": [],
        "recovered_from_disk": True,
        "recovered_no_gain": True,
    }


def _persist_workflow_return(eval_dir: Path, wf: dict) -> None:
    """Persist the authoritative workflow return beside the artifacts (best-effort)."""
    try:
        (eval_dir / WORKFLOW_RETURN_FILE).write_text(
            json.dumps(wf, indent=2), encoding="utf-8")
    except OSError:
        pass


# ---------------------------------------------------------------------------
# kernel_journey contract (KERNEL_JOURNEY_SCHEMA.md producer side).
#
# GEAK-e2e is a whole-pipeline e2e optimizer, so it never went through the
# per-kernel SDK recorder path — its authored kernels were invisible in the
# orchestrator's kernel_journey (only tracelens discovery showed). We emit a
# self-contained kernel_journey.json whose per-kernel sub-objects are shaped
# EXACTLY as the recorder's record_kernel_{dispatch,backend_result,e2e} inputs,
# so the orchestrator can replay them verbatim (all mapping lives here, once).
# ---------------------------------------------------------------------------
_BACKEND_ENUM = {"geak", "claude", "codex", "forge"}


def _norm_backend(b: Any) -> str:
    b = str(b or "").strip().lower()
    return b if b in _BACKEND_ENUM else "geak"


def _parity_passed(parity: Any) -> bool | None:
    """Normalize the workflow's output_parity into a correctness bool (or None)."""
    if isinstance(parity, dict):
        parity = parity.get("status")
    s = str(parity or "").strip().lower()
    if s in ("pass", "passed", "ok", "identical", "true"):
        return True
    if s in ("fail", "failed", "mismatch", "false"):
        return False
    return None


def _norm_kname(s: Any) -> str:
    """Normalize a kernel short_name for cross-source matching. The profiler keeps
    leading underscores (``_fwd_grouped_kernel_stage1``) that the overlay dir name
    strips (``cand_fwd_grouped_kernel_stage1``); compare case/underscore-insensitively."""
    return str(s or "").lstrip("_").lower()


def _canon_kid(s: Any) -> str:
    """Canonical ``kernel_id`` shared by the discovery and the kernels[] substreams.

    The profiler keeps a leading underscore (``_fwd_grouped_kernel_stage1``) that
    the overlay dir name strips (``cand_fwd_grouped_kernel_stage1`` ->
    ``fwd_grouped_kernel_stage1``). If discovery emits the underscored id while the
    kernels[] entry emits the stripped one, the orchestrator's assembler folds them
    into TWO journey entries for ONE kernel (one ``discovered``-only, one
    ``adopted``-without-discovery-fields). Emitting the SAME canonical id on both
    sides is what keeps a kernel a single, fully-populated journey entry. We strip
    leading underscores (the only documented divergence) and preserve case so the
    human-readable ``name`` still carries the raw profiler spelling."""
    return str(s or "").lstrip("_")


def _fuzzy_kid_key(s: Any) -> str:
    """Stronger cross-source match key than :func:`_norm_kname`: in addition to
    stripping the leading underscore and lowercasing, it drops the generic
    ``kernel`` filler token. The profiler symbol (``_fwd_grouped_kernel_stage1``)
    and the overlay dir short (``cand_fwd_grouped_stage1`` -> ``fwd_grouped_stage1``)
    can differ by that INFIX token, not only the documented leading underscore;
    without this they canonicalize to two different ids (``fwd_grouped_kernel_stage1``
    vs ``fwd_grouped_stage1``) and the assembler splits ONE kernel into two journey
    entries. Used ONLY to MATCH the two substreams — the emitted ``kernel_id`` stays
    the profiler's :func:`_canon_kid` spelling so discovery and kernels[] fold into one."""
    toks = [t for t in str(s or "").lstrip("_").lower().split("_") if t and t != "kernel"]
    return "_".join(toks)


def _journey_profile_topn(eval_dir: Path) -> dict:
    """Newest ``profile/round_*/profile_topN.json`` (rocprofv3 discovery), or {}."""
    pbase = eval_dir / "profile"
    if pbase.is_dir():
        for r in sorted(pbase.glob("round_*"), key=lambda p: p.name, reverse=True):
            d = _read_json(r / "profile_topN.json")
            if d.get("top_kernels"):
                return d
    return _read_json(pbase / "profile_topN.json")


def _journey_selected_names(eval_dir: Path) -> set[str]:
    """Match keys (BOTH norm + fuzzy) for short_names that had an optimization
    overlay built (= selected). Carrying the fuzzy key too lets the discovery
    substream flag ``selected_for_optimization`` even when the overlay dir short
    differs from the profiler symbol by the ``kernel`` infill (see
    :func:`_fuzzy_kid_key`)."""
    sel: set[str] = set()
    for base in (eval_dir / "overlay", eval_dir / "final" / "overlay"):
        if base.is_dir():
            for d in base.glob("cand_*"):
                if d.is_dir():
                    short = d.name[len("cand_"):]
                    sel.add(_norm_kname(short))
                    sel.add(_fuzzy_kid_key(short))
    return sel


def _journey_discovery_runs(eval_dir: Path, selected: set[str]) -> list[dict]:
    """Reconstruct the stage-1 discovery substream (schema §3/§5) from the on-disk
    rocprofv3 ``profile_topN.json`` — the real hot-kernel table the optimizer saw.
    Never fabricates: fields the profiler does not carry (roofline AI, source_file)
    stay ``None``. ``source='bypass'`` because GEAK profiles via rocprofv3, not
    tracelens."""
    prof = _journey_profile_topn(eval_dir)
    tops = prof.get("top_kernels") or []
    if not tops:
        return []
    hot: list[dict] = []
    seen_ids: dict[str, int] = {}
    for i, k in enumerate(tops):
        short = str(k.get("short_name") or k.get("name") or "")
        sel = _norm_kname(short) in selected or _fuzzy_kid_key(short) in selected
        # kernel_id MUST be the SAME canonical token the kernels[] entries use
        # (the overlay dir name strips the profiler's leading underscore);
        # otherwise the orchestrator's assembler folds discovery and the
        # optimized kernel into TWO entries for one kernel. Canonicalize here;
        # the raw profiler spelling (underscores intact) stays in ``name``.
        canon = _canon_kid(short)
        # The profiler can emit the SAME short_name for genuinely distinct kernels
        # (e.g. two CK attention mask variants, two Tensile GEMM configs). Keep
        # kernel_id UNIQUE (schema §1) by suffixing the real profiler rank on a
        # repeat; the full unmangled name is preserved in ``name``.
        seen_ids[canon] = seen_ids.get(canon, 0) + 1
        kid = canon if seen_ids[canon] == 1 else f"{canon}#{k.get('rank') or i}"
        hot.append({
            "kernel_id": kid,
            "name": str(k.get("name") or short),
            "gpu_pct": k.get("pct_gpu_time"),
            "time_ms": k.get("total_ms"),
            "bound_type": "",                     # rocprofv3 carries no roofline bound; backfilled later
            "arithmetic_intensity": None,
            "flops_per_byte": None,
            "efficiency_percent": None,
            "reusable_native_kernel": bool(k.get("editable")),
            "source_file": None,
            # GEAK is the only optimization backend; recommend it only for the
            # editable kernels it actually selected (overlay built).
            "recommended_backends": ["geak"] if (sel and k.get("editable")) else [],
            "selected_for_optimization": sel,
            # schema §5 ❌ field the producer is asked to backfill (kernel class).
            "kernel_category": k.get("classification"),
        })
    return [{
        "source": "bypass",
        "status": "success",
        "duration_sec": None,
        "scan": {"candidates_path": f"geak:{eval_dir}",
                 "profiler": prof.get("source") or "rocprofv3",
                 "num_distinct_kernels": prof.get("num_distinct_kernels")},
        "hot_kernel_count": len(hot),
        "hot_kernels": hot,
        "error": None,
    }]


def _journey_overlay_entry(eval_dir: Path, short: str, ir: dict, wf: dict,
                           geak_sha: str, overall_parity: bool | None,
                           gpu_pct_prof: Any, display_name: str | None = None,
                           kernel_id_override: str | None = None) -> dict:
    """One ``kernels[]`` entry for an optimization overlay, driven by its
    integrate_result.json. Honest per gate state:
      * accepted/stack -> succeeded + KEEP + integrated e2e (config win routes its
        flags into ``e2e.extra_server_args``, authored win into ``patch_path``),
      * rejected       -> succeeded attempt but REVERT + e2e REJECTED (do-no-harm),
      * incomplete A/B (no integrate_result) -> dispatch only, no e2e (outcome the
        assembler computes is ``dispatched``); never fabricates a KEEP/FAIL it
        cannot prove.

    ``display_name`` is the profiler's real kernel symbol (with its leading
    underscore intact) resolved from the discovery table; when given it is used
    as ``name`` so the kernels[] entry, the discovery hot_kernel, and the
    geak accepted-kernel backfill all carry the SAME human-readable name.
    The overlay dir name (``short``, underscore-stripped for filesystem safety)
    is only a fallback when the kernel was not in the profiler table.
    """
    kernel_id = kernel_id_override or _canon_kid(short)
    name = display_name or short
    backend = "geak"
    gate = str(ir.get("gate") or "").lower() if isinstance(ir, dict) else ""
    gpu_pct = _ir_get(ir, "pct_gpu_time") if ir else None
    if gpu_pct is None:
        gpu_pct = gpu_pct_prof
    micro = _ir_get(ir, "isolated_speedup", "micro_speedup", "speedup") if ir else None
    delta = _ir_get(ir, "e2e_delta_pct", "delta_pct") if ir else None
    # The delta is a ratio; these are the two numbers it is a ratio of, read
    # off the SAME integrate_result. A consumer holding the pair can restate
    # the win in points of one fixed baseline. Holding only the percentage it
    # can do nothing but add figures measured against different denominators.
    base_tput = _ir_get(ir, "ref_med", "ref_median_tok_s") if ir else None
    new_tput = _ir_get(ir, "cand_med", "cand_median_tok_s") if ir else None
    winner_kind = str(_ir_get(ir, "winner_kind") or "").lower() if ir else ""
    is_config = winner_kind in ("env", "config", "flags")
    flags = str(_ir_get(ir, "apply_flags") or "") if ir else ""
    tuned_file = _ir_get(ir, "tuned_config_file") if ir else None
    patch = None if (is_config or not ir) else (_ir_get(ir, "final_patch", "patch_path"))
    target_file = (tuned_file if is_config else _ir_get(ir, "target_file", "target_callable")) if ir else None
    parity = (_parity_passed(ir.get("output_parity"))
              if (ir and ir.get("output_parity") is not None) else overall_parity)

    entry: dict = {
        "kernel_id": kernel_id,
        "name": name,
        "gpu_pct": gpu_pct,
        "micro_speedup": micro,
        "dispatch": {
            "dispatched": True,
            "backends": [backend],
            "skip_reason": "",
            "orchestration_commit": "",
            "task_group": None,
        },
    }
    if not ir:
        # Verified-isolated candidate whose e2e A/B never completed (cut off): it
        # WAS dispatched, but no measured backend/e2e result exists. Record only
        # what is true; leave attempts empty and emit no e2e (not KEEP, not FAIL).
        entry["backend_result"] = {
            "kernel_id": kernel_id, "run_id": str(eval_dir),
            "attempts": [], "verification": {},
            "metadata": {"root_dir": str(GEAK_ROOT), "version": geak_sha,
                         "note": "e2e A/B incomplete (cut off before result)"},
        }
        entry["dispatch"]["task_group"] = "ab_incomplete"
        return entry

    accepted = gate in ("accepted", "stack")
    attempt_id = f"{kernel_id}-{backend}-0"
    entry["backend_result"] = {
        "kernel_id": kernel_id, "run_id": str(eval_dir),
        "attempts": [{
            "backend": backend, "attempt_id": attempt_id,
            "status": "succeeded",
            "decision": "KEEP" if accepted else "REVERT",
            "micro_speedup": micro,
            # A config tune is not compiled -> null (not fabricated True); an
            # authored kernel that reached the A/B did compile.
            "compile_passed": None if is_config else True,
            "correctness_passed": parity,
            "optimized_files": [tuned_file] if (is_config and tuned_file)
                               else ([patch] if patch else []),
            "error": None, "error_type": None, "ts": None, "duration_sec": None,
        }],
        "verification": {"micro_speedup": micro, "best_attempt_id": attempt_id,
                         "best_backend": backend},
        "metadata": {"root_dir": str(GEAK_ROOT), "version": geak_sha},
    }
    entry["e2e"] = {
        "kernel_id": kernel_id,
        "integrated": accepted,
        "e2e_gain_pct": delta,
        "base_tput": base_tput,
        "new_tput": new_tput,
        "validated": True,                        # an A/B gate ran either way
        "decision": "KEEP" if accepted else "REJECTED",
        "patch_path": patch,
        "target_file": target_file,
        "extra_server_args": flags if accepted else "",
        "ts": None,
    }
    return entry


def _journey_return_entry(eval_dir: str, k: dict, idx: int, wf: dict,
                          geak_sha: str, parity: bool | None,
                          kernel_id_override: str | None = None) -> dict:
    """One ``kernels[]`` entry from an accepted kernel named in the workflow return
    (used when there is no overlay on disk to read — e.g. the live path)."""
    name = str(k.get("short_name") or k.get("name") or k.get("op_kind") or f"kernel{idx}")
    # Canonical id (matches the discovery + overlay substreams); ``name`` keeps
    # the raw spelling so the assembler folds this kernel into a single entry.
    # An override adopts the profiler symbol's id when this kernel was fuzzy-matched
    # to a discovery hot_kernel (infix/underscore divergence) — see build_kernel_journey.
    kid = kernel_id_override or _canon_kid(name)
    backend = _norm_backend(k.get("backend") or k.get("source"))
    isolated = k.get("isolated") or k.get("micro_speedup") or k.get("verified_isolated_speedup")
    patch = k.get("final_patch") or None
    attempt_id = f"{kid}-{backend}-{idx}"
    return {
        "kernel_id": kid, "name": name, "gpu_pct": k.get("pct_gpu_time"),
        "micro_speedup": isolated,
        "dispatch": {"dispatched": True, "backends": [backend], "skip_reason": "",
                     "orchestration_commit": "", "task_group": None},
        "backend_result": {
            "kernel_id": kid, "run_id": str(k.get("kernel_eval_dir") or eval_dir),
            "attempts": [{
                "backend": backend, "attempt_id": attempt_id, "status": "succeeded",
                "decision": "KEEP", "micro_speedup": isolated, "compile_passed": True,
                "correctness_passed": parity, "optimized_files": [patch] if patch else [],
                "error": None, "error_type": None, "ts": None, "duration_sec": None,
            }],
            "verification": {"micro_speedup": isolated, "best_attempt_id": attempt_id,
                             "best_backend": backend},
            "metadata": {"root_dir": str(GEAK_ROOT), "version": geak_sha},
        },
        "e2e": {
            "kernel_id": kid, "integrated": True, "e2e_gain_pct": k.get("e2e_delta_pct"),
            # Same pair as the overlay path, carried through the workflow return.
            "base_tput": k.get("base_tput"), "new_tput": k.get("new_tput"),
            "validated": True, "decision": "KEEP", "patch_path": patch,
            "target_file": k.get("target_file") or k.get("target_callable"),
            "extra_server_args": str((wf.get("accepted_config") or {}).get("flags") or ""),
            "ts": None,
        },
    }


def _overlay_claim(ir: Any) -> dict | None:
    """What an INTEGRATED overlay says it optimized, or None when the overlay is
    not integrated (rejected / A/B never completed) and so claims nothing.

    ``integrate_result.json`` is the only place that records both spellings of a
    kernel: ``cand_tag`` is the overlay directory's name and ``short_name`` is the
    symbol the workflow return uses. A claim carries the symbol plus the
    integrated e2e delta, and is consumed at most once (``used``) so one overlay
    can never account for two distinct acceptances.
    """
    if not isinstance(ir, dict):
        return None
    if str(ir.get("gate") or "").lower() not in ("accepted", "stack"):
        return None
    gain = ir.get("e2e_delta_pct")
    return {
        "sym": _norm_kname(str(ir.get("short_name") or "")),
        "gain": float(gain) if isinstance(gain, (int, float))
                and not isinstance(gain, bool) else None,
        "used": False,
    }


def _claim_for(name: str, gain: Any, claims: list[dict]) -> dict | None:
    """The overlay claim that already covers this return-named acceptance, or None
    when the return names a kernel no overlay on disk accounted for.

    Matched on the symbol first — the same normalization the profiler match uses,
    so a spelling difference does not split one kernel in two. Falling back to the
    integrated e2e delta covers the runs where the overlay recorded no usable
    symbol: the return does not recompute that number, it copies the overlay's own
    A/B result, so an EXACT hit is the same measurement rather than a coincidence.
    The delta is only allowed to fold when exactly one unconsumed overlay claims
    it; an ambiguous delta never merges two kernels.
    """
    nk = _norm_kname(name)
    if nk:
        for c in claims:
            if c["sym"] and c["sym"] == nk:
                return c
    if isinstance(gain, (int, float)) and not isinstance(gain, bool):
        hits = [c for c in claims if not c["used"] and c["gain"] == float(gain)]
        if len(hits) == 1:
            return hits[0]
    return None


def build_kernel_journey(wf: dict, normalized: dict) -> dict:
    """Build the kernel_journey handoff (recorder-input shapes the orchestrator
    replays through the SBD SDK — KERNEL_JOURNEY_SCHEMA.md §2).

    Reconstructs the FULL journey the run actually produced, from disk truth:
      * ``discovery_runs`` from rocprofv3 ``profile_topN.json`` (the real hot-kernel
        table; ``selected_for_optimization`` set for kernels that got an overlay),
      * one ``kernels[]`` entry PER optimization overlay (accepted / rejected /
        incomplete A/B), so a CONFIG-only win (no authored patch) is still recorded
        as the optimized hot kernel — its flags land in ``e2e.extra_server_args``.
    Falls back to workflow-return-named kernels when no overlay/profile is on disk
    (e.g. the live path or a unit fixture). Empty ``kernels``/``discovery_runs`` is
    valid and honest only when nothing was discovered/attempted.
    """
    eval_dir_str = str(normalized.get("eval_dir") or wf.get("eval_dir") or "")
    eval_dir = Path(eval_dir_str) if eval_dir_str else None
    geak_sha = _git_short_sha(GEAK_ROOT)
    overall_parity = _parity_passed(wf.get("output_parity") or normalized.get("output_parity"))

    selected = _journey_selected_names(eval_dir) if eval_dir else set()
    discovery_runs = _journey_discovery_runs(eval_dir, selected) if eval_dir else []
    prof = _journey_profile_topn(eval_dir) if eval_dir else {}
    # Profiler index for cross-source matching. Each entry carries the canonical
    # kernel_id DISCOVERY assigns (mirrors _journey_discovery_runs: bare canon on
    # first sight, ``canon#rank`` on a repeat) plus the display name + gpu%. An
    # overlay/return kernel resolves to its profiler symbol via _match_profiler:
    # EXACT norm key first, then a UNIQUE fuzzy key (filler-token-insensitive).
    # On a match the kernels[] entry ADOPTS the profiler's kernel_id/name/gpu% so
    # discovery and kernels[] always fold into ONE journey entry — fixing both the
    # leading-underscore and the ``kernel`` infix divergences.
    prof_index: list[dict] = []
    _seen_canon: dict[str, int] = {}
    for i, k in enumerate(prof.get("top_kernels") or []):
        sh = str(k.get("short_name") or k.get("name") or "")
        if not sh:
            continue
        canon = _canon_kid(sh)
        _seen_canon[canon] = _seen_canon.get(canon, 0) + 1
        kid_p = canon if _seen_canon[canon] == 1 else f"{canon}#{k.get('rank') or i}"
        prof_index.append({
            "norm": _norm_kname(sh), "fuzzy": _fuzzy_kid_key(sh),
            "kid": kid_p, "name": str(k.get("name") or sh),
            "pct": k.get("pct_gpu_time"),
        })

    def _match_profiler(short: str) -> dict | None:
        """Resolve an overlay/return short_name to its profiler symbol: exact norm
        key first, then a UNIQUE fuzzy-key match (ambiguous fuzzy -> no match,
        never guess)."""
        nk, fk = _norm_kname(short), _fuzzy_kid_key(short)
        exact = [p for p in prof_index if p["norm"] == nk]
        if exact:
            return exact[0]
        fuzzy = [p for p in prof_index if p["fuzzy"] == fk]
        return fuzzy[0] if len(fuzzy) == 1 else None

    kernels: list[dict] = []
    seen: set[str] = set()  # dedup on the FINAL emitted kernel_id
    # What each INTEGRATED overlay says it optimized (see _overlay_claim). The id
    # dedup above cannot carry pass 2, because the two substreams name a kernel
    # differently: an overlay dir is named for its CANDIDATE TAG (``cand_c0_triton``)
    # and the workflow return for the KERNEL SYMBOL
    # (``dsa_sparse_attn_prefill_main_kernel``). integrate_result.json is the file
    # that ties the two together, so the claim is read from there, never from the
    # directory name.
    claims: list[dict] = []

    # 1) Disk truth: one entry per optimization overlay, driven by integrate_result.
    if eval_dir:
        for base in (eval_dir / "overlay", eval_dir / "final" / "overlay"):
            if not base.is_dir():
                continue
            for cand in sorted(base.glob("cand_*")):
                if not cand.is_dir():
                    continue
                short = cand.name[len("cand_"):]
                if not short:
                    continue
                ir = _read_json(cand / "integrate_result.json")
                claim = _overlay_claim(ir)
                if claim:
                    claims.append(claim)
                m = _match_profiler(short)
                kid = m["kid"] if m else _canon_kid(short)
                if kid in seen:
                    continue
                seen.add(kid)
                kernels.append(_journey_overlay_entry(
                    eval_dir, short, ir, wf, geak_sha, overall_parity,
                    m["pct"] if m else None, m["name"] if m else None,
                    kernel_id_override=kid))

    # 2) Augment with accepted kernels named only in the workflow return (live path
    #    / no overlay on disk), deduped against the overlay entries above: by id,
    #    and by the identity those overlays claimed, which is what catches the same
    #    acceptance spelled as a candidate tag on disk and as a symbol in the return.
    accepted = list(wf.get("accepted_kernels") or []) + list(wf.get("accepted_heads") or [])
    synth_hot: list[dict] = []
    for idx, k in enumerate(accepted):
        if not isinstance(k, dict):
            continue
        name = str(k.get("short_name") or k.get("name") or k.get("op_kind") or f"kernel{idx}")
        claimed = _claim_for(name, k.get("e2e_delta_pct"), claims)
        if claimed is not None:
            claimed["used"] = True
            continue
        m = _match_profiler(name)
        kid = m["kid"] if m else _canon_kid(name)
        if kid in seen:
            continue
        seen.add(kid)
        kernels.append(_journey_return_entry(
            eval_dir_str, k, idx, wf, geak_sha, overall_parity,
            kernel_id_override=kid))
        synth_hot.append({
            "kernel_id": kid, "name": (m["name"] if m else name),
            "gpu_pct": k.get("pct_gpu_time"),
            "bound_type": str(k.get("bound_type") or k.get("op_kind") or ""),
            "source_file": k.get("target_file") or k.get("target_callable"),
            "recommended_backends": [_norm_backend(k.get("backend") or k.get("source"))],
            "selected_for_optimization": True,
        })
    # When there is no on-disk profiler discovery (live path), synthesize a minimal
    # discovery run from the accepted kernels so they are not orphaned.
    if not discovery_runs and synth_hot:
        discovery_runs = [{
            "source": "bypass", "status": "success", "duration_sec": None,
            "scan": {"candidates_path": f"geak:{eval_dir_str}"},
            "hot_kernel_count": len(synth_hot), "hot_kernels": synth_hot, "error": None,
        }]

    return {
        "schema_version": KERNEL_JOURNEY_SCHEMA_VERSION,
        "producer": "kernel-agent",
        "eval_dir": eval_dir_str,
        "versions": _geak_versions(),
        "discovery_runs": discovery_runs,
        "kernels": kernels,
    }


def _geak_versions() -> dict:
    """The top-level ``versions`` section (schema §1) — GEAK's authoritative tool
    version, shared by the full and the empty-kernels journey shapes."""
    geak_sha = _git_short_sha(GEAK_ROOT)
    return {
        "geak": {
            "tool": "geak",
            "root_dir": str(GEAK_ROOT),
            "commit": geak_sha,
            "version": geak_sha,
        }
    }


def _empty_journey(eval_dir: Path, normalized: dict) -> dict:
    """A VALID, kernels-empty journey (schema-compliant: missing data is ``[]``,
    never fabricated). Carries the run ``status``/``error`` so a consumer ALWAYS
    finds a parseable file and can see WHY nothing landed — used on an error/
    timeout/no-recovery run, or as the fallback when the full build raises."""
    return {
        "schema_version": KERNEL_JOURNEY_SCHEMA_VERSION,
        "producer": "kernel-agent",
        "eval_dir": str(eval_dir),
        "versions": _geak_versions(),
        # Diagnostic context (extra to schema's discovery_runs/kernels/versions):
        # honest provenance for an empty journey, ignored by strict consumers.
        "status": normalized.get("status"),
        "error_class": normalized.get("error_class"),
        "error": normalized.get("error"),
        "discovery_runs": [],
        "kernels": [],
    }


def _write_kernel_journey(eval_dir: Path, wf: dict | None, normalized: dict) -> str:
    """Write an HONEST kernel_journey.json into eval_dir; return its path.

    GUARANTEED-EMIT (parallel to result.json): writes a parseable file in EVERY
    case so a consumer always finds one:
      * a FULL journey (one entry per accepted kernel/head) when a workflow
        result was recovered (``wf`` is not None),
      * else an EMPTY-kernels journey carrying the run status/error_class.
    If building the full journey raises, fall back to the empty-kernels shape
    rather than dropping the file (never fabricates kernels). Raises ONLY when the
    filesystem write itself fails — the caller records that into result.json as
    ``kernel_journey_error`` instead of letting it pass silently.
    """
    try:
        journey = build_kernel_journey(wf, normalized) if wf is not None \
            else _empty_journey(eval_dir, normalized)
    except Exception:  # full build failed: degrade to a valid empty journey.
        journey = _empty_journey(eval_dir, normalized)
    eval_dir.mkdir(parents=True, exist_ok=True)
    path = eval_dir / KERNEL_JOURNEY_FILE
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(journey, indent=2), encoding="utf-8")
    os.replace(tmp, path)  # atomic: a kill mid-write never yields a partial file
    return str(path)


def _publish_protected_pgids() -> str:
    """Tell the benchmark teardown which process groups it must NEVER signal.

    ``e2e_workflow/scripts/server_teardown.sh`` refuses a group kill against any pgid
    in ``GEAK_PROTECTED_PGIDS``. Nothing was ever publishing that list, so the hook
    only ever held its two built-in defaults. We are the process the caller (Hyperloom)
    launches, so we are the one place that knows the two groups a bench cleanup must
    never reach: OUR group (the whole GEAK subtree runs in it) and our PARENT's — the
    orchestrator that was TERMed by a bench teardown in issue #397. pid 1 is included
    explicitly because a container's init IS that orchestrator in the deployment where
    this happened.

    Purely additive: a group kill is only ever allowed for a server that leads its own
    session (pgid == pid), which can never be one of these groups, so no legitimate
    teardown is downgraded by this.

    Returns:
        str: the published, space-separated pgid list (also set in ``os.environ``).
    """
    protected: set[str] = {"1"}
    for pid in (0, os.getppid()):
        try:
            protected.add(str(os.getpgid(pid)))
        except OSError:  # racing parent exit / unsupported platform
            pass
    protected.update(
        tok for tok in os.environ.get("GEAK_PROTECTED_PGIDS", "").split() if tok.isdigit()
    )
    value = " ".join(sorted(protected, key=int))
    os.environ["GEAK_PROTECTED_PGIDS"] = value
    return value


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
DEFAULT_E2E_TIMEOUT_S = 43200  # 12h — used only when NOBODY states a budget


def _int_or_none(raw: str | None, what: str) -> int | None:
    """Positive int, or None (with a note) for junk — never abort on a bad budget."""
    if not raw:
        return None
    try:
        return int(raw) if int(raw) > 0 else None
    except ValueError:
        sys.stderr.write(f"ignoring non-integer {what}: {raw!r}\n")
        return None


def _resolve_timeout_s(argv: list[str]) -> tuple[list[str], set[str], int]:
    """Split argv into positionals + flags, and resolve the wall-clock budget.

    `--timeout-s N` is the wall-clock at which the orchestrator kills us. Its
    VALUE is not a "--" token, so it used to land in an ignored third positional
    and we paced against GEAK_E2E_TIMEOUT_S's 12h default instead — a caller
    killing at 8h tore the run down mid-optimization (Hyperloom #1202). Flag and
    env var both name a REAL kill, so the min() wins; the default applies only
    when neither is stated.
    """
    positional: list[str] = []
    flags: set[str] = set()
    stated: dict[str, int] = {}
    expect_value = False
    for tok in argv:
        if expect_value:
            expect_value = False
            v = _int_or_none(tok, "--timeout-s value")
        elif tok == "--timeout-s":
            expect_value = True
            continue
        elif tok.startswith("--timeout-s="):
            v = _int_or_none(tok.split("=", 1)[1], "--timeout-s value")
        elif tok.startswith("--"):
            flags.add(tok)
            continue
        else:
            positional.append(tok)
            continue
        if v is not None:
            stated["--timeout-s"] = v
    v = _int_or_none(os.environ.get("GEAK_E2E_TIMEOUT_S"), "GEAK_E2E_TIMEOUT_S")
    if v is not None:
        stated["GEAK_E2E_TIMEOUT_S"] = v
    budget = min(stated.values()) if stated else DEFAULT_E2E_TIMEOUT_S
    sys.stderr.write(
        f"[budget] wall-clock {budget}s "
        f"({', '.join(f'{k}={v}' for k, v in stated.items()) or 'nobody stated one; default'})\n"
    )
    return positional, flags, budget


def main(argv: list[str]) -> int:
    args, flags, timeout_s = _resolve_timeout_s(argv)
    if len(args) < 2:
        sys.stderr.write(
            "usage: run_e2e.py <handoff.json> <result.json> "
            "[--timeout-s N] [--dry-run]\n"
        )
        return 2
    handoff_path, result_path = Path(args[0]), Path(args[1])

    h = _read_json(handoff_path)
    if not h:
        sys.stderr.write(f"empty/invalid handoff: {handoff_path}\n")
        return 2

    ps_args = map_args(h, timeout_s)
    # Pin the single eval_dir into the environment so BOTH the live completion
    # check (_workflow_done_on_disk) and the scrape-independent disk recovery
    # (_discover_eval_dir) target EXACTLY this run's dir, deterministically.
    os.environ["GEAK_EVAL_DIR"] = ps_args["eval_dir"]
    _publish_protected_pgids()
    bench_client = apply_bench_client(h)
    bench_launcher = apply_bench_launcher(h)
    bench_protocol = apply_bench_protocol(h)
    alignment_flags = apply_alignment_flags(h)
    prompt = build_prompt(ps_args)

    if "--dry-run" in flags:
        print(json.dumps({"mapped_args": ps_args, "bench_client": bench_client,
                          "bench_launcher": bench_launcher,
                          "magpie_launch_script": os.environ.get("MAGPIE_LAUNCH_SCRIPT", ""),
                          "magpie_launch_script_source": os.environ.get("MAGPIE_LAUNCH_SCRIPT_SOURCE", ""),
                          "recipe_env_file": os.environ.get("RECIPE_ENV_FILE", ""),
                          "recipe_env_source": os.environ.get("RECIPE_ENV_SOURCE", ""),
                          "recipe_env_replayed": os.environ.get("RECIPE_ENV_REPLAYED", ""),
                          "recipe_env_geak_owned": os.environ.get("RECIPE_ENV_GEAK_OWNED", ""),
                          "bench_protocol": bench_protocol,
                          "alignment_flags": alignment_flags,
                          "inferencex_path": os.environ.get("INFERENCEX_PATH", ""),
                          "prompt": prompt, "e2e_script": str(E2E_SCRIPT)}, indent=2))
        return 0

    exp_root = Path(h.get("exp_root") or "")
    eval_dir_hint = ps_args["eval_dir"]

    # ── Guaranteed interface-file emission ──────────────────────────────────
    # CONTRACT: as long as GEAK produced ANY measured E2E effect on disk,
    # result.json (+ kernel_journey.json) MUST be written. No termination,
    # timeout, signal, or exception may leave the interface files missing.
    #   * idempotent (writes once), best-effort (never raises),
    #   * ATOMIC write (tmp + os.replace) so a kill mid-write never yields a
    #     partial/corrupt result.json,
    #   * recovers from disk (incl. the best accepted intermediate win) when no
    #     explicit workflow return is available.
    _emit_state: dict[str, Any] = {"done": False, "out": {}}

    def _emit(wf: dict | None = None, *, error: object = None,
              error_class: str | None = None) -> dict:
        if _emit_state["done"]:
            return _emit_state["out"]
        # A second SIGTERM must not interrupt the flush we are about to do.
        try:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
        except Exception:
            pass
        if wf is None:
            try:
                wf = _recover_workflow_return(exp_root)
            except Exception:
                wf = None
        try:
            if wf is not None:
                out = normalize_result(h, wf)
                if wf.get("recovered_from_disk"):
                    out["recovered_from_disk"] = True
            else:
                out = {
                    "schema_version": SCHEMA_VERSION,
                    "status": "timeout" if error_class == "timeout" else "error",
                    "error_class": error_class or "runner_error",
                    "error": str(error or ""),
                }
        except Exception as norm_exc:  # normalize blew up: still emit an error.
            out = {
                "schema_version": SCHEMA_VERSION,
                "status": "error",
                "error_class": "normalize_failed",
                "error": f"{type(norm_exc).__name__}: {norm_exc}",
            }
        # kernel_journey.json is a GUARANTEED interface file too (same contract as
        # result.json). Resolve eval_dir even on the error path (eval_dir_hint is
        # this run's pinned dir) so the journey always has a home, persist the
        # canonical workflow return when we have one, then ALWAYS write an honest
        # journey. A build/write failure is surfaced into result.json rather than
        # silently dropping the file.
        eval_dir_str = str(out.get("eval_dir") or eval_dir_hint or "")
        if eval_dir_str:
            eval_dir = Path(eval_dir_str)
            if wf is not None:
                try:
                    _persist_workflow_return(eval_dir, wf)
                except Exception:
                    pass
            try:
                out["kernel_journey_path"] = _write_kernel_journey(eval_dir, wf, out)
            except Exception as kj_exc:
                out["kernel_journey_error"] = f"{type(kj_exc).__name__}: {kj_exc}"
        if out.get("baseline_basis"):
            try:
                updated_reports = _update_baseline_alignment_reports(out)
                if updated_reports:
                    out["baseline_alignment_report_paths"] = updated_reports
            except Exception as report_exc:
                out["baseline_alignment_report_error"] = (
                    f"{type(report_exc).__name__}: {report_exc}"
                )
        try:
            result_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = result_path.with_name(result_path.name + ".tmp")
            tmp.write_text(json.dumps(out, indent=2), encoding="utf-8")
            os.replace(tmp, result_path)  # atomic
            _emit_state.update(done=True, out=out)
        except Exception:
            try:  # last-ditch non-atomic write: never leave NOTHING behind.
                result_path.parent.mkdir(parents=True, exist_ok=True)
                result_path.write_text(json.dumps(out), encoding="utf-8")
                _emit_state.update(done=True, out=out)
            except Exception:
                pass
        return out

    # Safety net: any exit path that somehow skipped _emit still leaves a file.
    atexit.register(
        lambda: None if _emit_state["done"]
        else _emit(error="process exiting without an emit",
                   error_class="interrupted")
    )

    # SIGTERM (the outer runner's graceful-stop) -> break out of the workflow
    # wait as a TimeoutError so the finally below emits from on-disk artifacts.
    def _on_term(signum, _frame):
        raise TimeoutError(f"signal {signum}: self-stop to flush interface files")
    signal.signal(signal.SIGTERM, _on_term)

    # ── Resume-from-cache short-circuit ──────────────────────────────────────
    # If a prior invocation already drove THIS (pinned) eval_dir to a terminal
    # marker, re-emit result.json from the on-disk artifacts instead of re-running
    # the entire workflow. General, not case-by-case: it keys off the workflow's
    # own terminal markers via _workflow_done_on_disk, so it fires for ANY re-entry
    # against a completed eval_dir (e.g. an orchestrator resume that re-delegates
    # the KERNEL phase). A fresh run mints an empty eval_dir, so the marker is
    # absent and this never trips — byte-identical to a first-time run.
    if _workflow_done_on_disk(eval_dir_hint):
        sys.stderr.write(
            f"GEAK e2e: eval_dir already terminal on disk "
            f"({eval_dir_hint}); recovering without re-running the workflow.\n"
        )
        try:
            cached_wf = _recover_workflow_return(exp_root)
        except Exception:
            cached_wf = None
        cached_out = _emit(wf=cached_wf)
        print(json.dumps({"status": cached_out.get("status"),
                          "result_json": str(result_path),
                          "speedup": cached_out.get("throughput_speedup")}))
        return 0 if cached_out.get("status") != "error" else 1

    out: dict = {}
    wf: dict | None = None
    err: object = None
    err_class: str | None = None
    try:
        wf = invoke_workflow(prompt, timeout_s, ps_args["eval_dir"])
    except Exception as e:  # scrape/crash/timeout/SIGTERM: recover from disk.
        err = e
        err_class = _classify_error(e)
        try:
            wf = _recover_workflow_return(exp_root)
        except Exception:
            wf = None
        if wf is not None:
            sys.stderr.write(
                f"GEAK e2e: workflow handoff failed [{err_class}]; "
                f"recovered from disk artifacts ({wf.get('eval_dir')}).\n"
            )
        else:
            sys.stderr.write(f"GEAK e2e failed [{err_class}]: {e}\n")
    finally:
        out = _emit(wf=wf, error=err, error_class=err_class)

    print(json.dumps({"status": out.get("status"),
                      "result_json": str(result_path),
                      "speedup": out.get("throughput_speedup")}))
    return 0 if out.get("status") != "error" else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
