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
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1

# interface/ is a sibling of e2e_workflow/ under the repo root.
INTERFACE_DIR = Path(__file__).resolve().parent
GEAK_ROOT = INTERFACE_DIR.parent
E2E_DIR = GEAK_ROOT / "e2e_workflow"
E2E_SCRIPT = E2E_DIR / "e2e_workflow.js"
BENCH_SCRIPT = E2E_DIR / "scripts" / "bench_e2e.sh"

# Workflow primitives are only available at this effort tier (see README).
CLAUDE_EFFORT = os.environ.get("GEAK_CLAUDE_EFFORT", "ultracode")
CLAUDE_MODEL = os.environ.get("GEAK_CLAUDE_MODEL", "claude-opus-4-8")
ALLOWED_TOOLS = ["Workflow", "Bash", "Read", "Write"]

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
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        eval_dir = str(Path(h["exp_root"]) / f"e2e_{model_name}_{ts}")
    ps_args["eval_dir"] = eval_dir
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
        "Invoke the Workflow tool exactly once with:\n"
        f'  scriptPath: "{E2E_SCRIPT}"\n'
        f"  args: {json.dumps(ps_args)}\n"
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
      * explicit ``handoff.bench_launcher`` / ``$BENCH_LAUNCHER`` wins;
      * else enable ``magpie`` ONLY when a script is discoverable
        (``handoff.launch_server_script``, or generic ``$MAGPIE_LAUNCH_SCRIPT``,
        or per-backend ``$MAGPIE_<BACKEND>_SCRIPT`` e.g. ``$MAGPIE_VLLM_SCRIPT``)
        AND the backend is one Magpie supports; otherwise ``native``.

    When nothing is discoverable the native backend launch is kept, so the
    standalone / unaligned path is byte-identical to before.

    Returns the resolved launcher name (for --dry-run / logging).
    """
    requested = str(
        h.get("bench_launcher") or os.environ.get("BENCH_LAUNCHER", "") or ""
    ).strip().lower()
    backend = str(h.get("framework", "sglang") or "sglang").strip().lower()
    # Discover the Magpie launch script: explicit handoff, generic env, then the
    # per-backend env (MAGPIE_SGLANG_SCRIPT / MAGPIE_VLLM_SCRIPT / ...).
    script = str(
        h.get("launch_server_script")
        or os.environ.get("MAGPIE_LAUNCH_SCRIPT", "")
        or os.environ.get(f"MAGPIE_{backend.upper()}_SCRIPT", "")
        or ""
    ).strip()
    if script:
        # Normalise onto the generic var the backend-agnostic launcher reads.
        os.environ["MAGPIE_LAUNCH_SCRIPT"] = script

    if requested and requested != "auto":
        launcher = requested
    elif script and backend in _MAGPIE_BACKENDS:
        launcher = "magpie"
    else:
        launcher = "native"
    os.environ["BENCH_LAUNCHER"] = launcher
    return launcher


def apply_alignment_flags(h: dict) -> dict:
    """Export optional cold/hot measurement-alignment flags so bench_e2e.sh inherits them.

    Currently: ``BENCH_COLD_FINAL`` — when on, bench_e2e.sh also measures ONE cold
    full round per bench (surfaced as ``cold_output_throughput_tok_s`` in each
    bench_summary.json, folded into ``result.json.alignment_metrics``, and used by
    the cold-preferred final-basis selection in :func:`normalize_result`). Default
    ON — the cold round is what enables the cold-to-cold promotion, so we opt IN by
    default; a caller disables it with an explicit falsey ``handoff.bench_cold_final``
    or ``$BENCH_COLD_FINAL=0`` (e.g. to save the one extra full round per bench).
    Returns the flags it exported.
    """
    exported: dict[str, str] = {}
    raw = h.get("bench_cold_final")
    if raw is None:
        raw = os.environ.get("BENCH_COLD_FINAL")
    # Default ON: enabled unless an explicit falsey value is given.
    if raw is None or str(raw).strip() == "":
        on = True
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


def normalize_result(h: dict, wf: dict) -> dict:
    eval_dir = Path(wf["eval_dir"])
    validation = _read_json(eval_dir / "director_e2e_validation.json")
    baseline_summary = _read_json(eval_dir / "baseline" / "bench_summary.json")
    final_summary = _read_json(eval_dir / "validation" / "final" / "bench_summary.json")

    # ── Reconcile a CONTRADICTORY return (do-no-harm guard for the Hyperloom
    # interface) ────────────────────────────────────────────────────────────
    # result.json must NEVER report no_gain over a real, parity-checked
    # same-session win. A return can be internally inconsistent: it ACCEPTED a
    # head/kernel (accepted_heads/kernels carry a positive e2e_delta_pct + a
    # complete integrate A/B is on disk) yet reports a degenerate final/speedup
    # — e.g. the final Validate bench crashed in engine-core init, so the
    # Director number came back 0. When that happens, backfill the throughput /
    # speedup / baseline / latency from the best accepted intermediate A/B on
    # disk (the same source _recover_best_intermediate_win trusts), and tag the
    # provenance so Hyperloom sees the number came from the disk A/B. We only do
    # this for a LIVE return (not one we already recovered from disk).
    wf_speedup_raw = float(wf.get("throughput_speedup") or validation.get("throughput_speedup") or 1.0)
    wf_final_raw = float(
        wf.get("final_throughput_tok_s")
        or validation.get("director_verified_throughput_tok_s")
        or 0.0
    )
    if (
        not wf.get("recovered_from_disk")
        and _wf_best_accepted_delta_pct(wf) > 0.0
        and (wf_speedup_raw <= 1.0 or wf_final_raw <= 0.0)
    ):
        recovered = _recover_best_intermediate_win(eval_dir)
        if recovered is not None and float(recovered.get("throughput_speedup") or 0.0) > 1.0:
            merged = dict(wf)
            for k in ("throughput_speedup", "baseline_throughput_tok_s",
                      "final_throughput_tok_s", "output_parity", "ttft_ms", "tpot_ms"):
                if recovered.get(k) is not None:
                    merged[k] = recovered[k]
            merged["recovered_intermediate"] = True   # provenance -> disk_intermediate_win
            wf = merged
            validation = {}   # the on-disk Director json (if any) was the crashed bench

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
        result_source = "disk_intermediate_win"
    elif wf.get("recovered_from_disk"):
        result_source = "disk_director_validation"
    else:
        result_source = "workflow_return"

    # baseline measurement-protocol cross-check (GEAK-E2E vs Hyperloom-E2E divergence).
    # GEAK's measured baseline is already seeded with Hyperloom's accepted config (map_args forwards
    # accepted_flags/accepted_env), so it should match Hyperloom's own baseline. Surface BOTH numbers
    # plus their divergence so the caller can tell a real win from a measurement mismatch (different
    # client/protocol/warm-vs-cold) instead of trusting a gain measured against a different baseline.
    geak_baseline = float(
        wf.get("baseline_throughput_tok_s")
        or validation.get("baseline_throughput_tok_s")
        or 0.0
    )
    geak_final = float(
        wf.get("final_throughput_tok_s")
        or validation.get("director_verified_throughput_tok_s")
        or 0.0
    )
    try:
        orch_baseline = float(h.get("raw_baseline_tput") or 0.0)
    except (TypeError, ValueError):
        orch_baseline = 0.0
    # Orchestrator throughput measured on the SAME config GEAK seeds with
    # (== Hyperloom current_best config). When present it isolates the PURE
    # cross-harness measurement residue (identical config, both harnesses) from
    # the explore/framework config gain that is baked into the raw-baseline
    # comparison. Falls back to raw baseline when absent (older handoffs).
    try:
        orch_same_cfg = float(h.get("orchestrator_best_tput_same_config") or 0.0)
    except (TypeError, ValueError):
        orch_same_cfg = 0.0
    baseline_basis = {
        # GEAK's own measured baseline (Hyperloom-accepted config = fair engagement baseline; gating uses this).
        "geak_measured_baseline_tok_s": geak_baseline or None,
        # Hyperloom's own measured baseline forwarded in the handoff (the orchestrator reference).
        "orchestrator_baseline_tok_s": orch_baseline or None,
        # How far GEAK's baseline drifted from Hyperloom's RAW baseline. NOTE:
        # this conflates the explore/framework CONFIG gain (GEAK seeds with the
        # accepted config, raw baseline does not) with measurement residue, so it
        # is NOT a clean measurement-mismatch signal. Kept for continuity.
        "baseline_divergence_pct": (
            round(100.0 * (geak_baseline - orch_baseline) / orch_baseline, 2)
            if (geak_baseline > 0 and orch_baseline > 0) else None
        ),
        # PURE cross-harness measurement residue: GEAK baseline vs the
        # orchestrator's throughput on the SAME (accepted) config. Both sides
        # run the identical config, so this isolates client/protocol/warm-cold
        # differences from the config gain. This is the value promote-side gating
        # should use. None when the same-config baseline was not forwarded.
        "measurement_divergence_pct": (
            round(100.0 * (geak_baseline - orch_same_cfg) / orch_same_cfg, 2)
            if (geak_baseline > 0 and orch_same_cfg > 0) else None
        ),
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
    geak_cold_baseline = baseline_summary.get("cold_output_throughput_tok_s")
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
    }

    # ── final-throughput BASIS selection (cold-preferred when it's a real cold win) ──
    # When a COLD full round was measured (BENCH_COLD_FINAL=1), prefer the COLD final
    # as the PROMOTED number: it is the SAME thermal state as Hyperloom's cold
    # baseline_tput denominator, so the promoted gain becomes a fair cold-to-cold
    # ratio. BUT an authored-kernel overlay pays a one-off JIT / cuda-graph capture
    # cost on the cold round that does not amortize in a single pass, so a genuine
    # steady-state win can surface as a cold LOSS. Guard against promoting that:
    # only switch to cold when the cold measurement is itself a NON-NEGATIVE gain
    # (cold_speedup >= 1.0 vs the orchestrator cold baseline it will be compared
    # against; fall back to the within-GEAK cold ratio when running standalone).
    # Otherwise keep the HOT median (today's behaviour). Default (no cold round
    # measured) => HOT, byte-identical to before.
    final_tput_out = geak_final          # hot median (== the pre-change promoted value)
    final_basis = "hot"
    cold_gate = (
        alignment_metrics["cold_speedup"]
        if alignment_metrics["cold_speedup"] is not None
        else alignment_metrics["cold_geak_speedup"]
    )
    if geak_cold_final and cold_gate is not None and cold_gate >= 1.0:
        final_tput_out = float(geak_cold_final)
        final_basis = "cold"
        # Keep GEAK's own speedup field + status consistent with the chosen basis.
        # (Hyperloom recomputes the promoted gain from final_throughput_tok_s /
        # baseline_tput itself; this only keeps result.json self-consistent and the
        # ok/no_gain status gate correct for the number we actually promote.)
        cold_geak_sp = alignment_metrics["cold_geak_speedup"]
        if cold_geak_sp is not None:
            speedup = float(cold_geak_sp)
            status = "ok" if speedup > 1.0 else "no_gain"
    alignment_metrics["final_basis"] = final_basis

    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "result_source": result_source,
        "eval_dir": str(eval_dir),
        "baseline_throughput_tok_s": float(
            wf.get("baseline_throughput_tok_s")
            or validation.get("baseline_throughput_tok_s")
            or 0.0
        ),
        # Promoted final: the COLD final when it is a real cold win, else the HOT
        # median (see the final-basis selection above). final_throughput_basis says
        # which one this is, so a consumer can tell cold-to-cold from hot-to-cold.
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
        "final_patch": str(eval_dir / "final" / "final_patch.diff"),
        "final_overlay": wf.get("final_overlay") or str(eval_dir / "final" / "overlay"),
        # Measurement basis: reports aggregate output tok/s (not per-GPU),
        # matching Hyperloom's Magpie output_throughput. See run_e2e.md alignment table.
        "metric_basis": "aggregate_output_tok_s",
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
        # Cold/hot speedup cross-checks (double-check only; see alignment_metrics above).
        # Does NOT change the promoted final_throughput_tok_s / throughput_speedup.
        "alignment_metrics": alignment_metrics,
        "report_path": wf.get("report_path") or str(eval_dir / "final_report.md"),
    }


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


def _recover_best_intermediate_win(eval_dir: Path) -> dict | None:
    """Salvage the best accepted intermediate win when the run died BEFORE Validate.

    The whole-pipeline workflow records each accepted config/kernel integrate as
    ``overlay/<cand>/integrate_result.json`` with a measured e2e delta + gate.
    When no ``director_e2e_validation.json`` exists (the run was killed
    mid-pipeline), pick the BEST accepted, positive-delta intermediate so a real,
    parity-checked win is NEVER silently discarded.

    Schema-robust: the integrator's integrate_result.json may carry the numbers
    flat or nested under ``e2e`` / ``accepted_config`` (see :func:`_ir_get`); both
    are read. Returns a workflow-return-shaped dict (status derived later by
    :func:`normalize_result`) or ``None`` when nothing acceptable is on disk.
    """
    best: dict | None = None
    best_tput = 0.0
    for base in (eval_dir / "overlay", eval_dir / "final" / "overlay"):
        if not base.is_dir():
            continue
        for cand in sorted(base.glob("cand_*")):
            ir = _read_json(cand / "integrate_result.json")
            if not ir or ir.get("gate") not in ("accepted", "stack"):
                continue
            delta = _ir_float(ir, "e2e_delta_pct", "delta_pct")
            tput = _ir_float(ir, "e2e_throughput_tok_s", "cand_median_tok_s", "cand_med")
            if delta > 0.0 and tput > best_tput:
                best_tput, best = tput, ir
    if best is None:
        return None

    ref_med = _ir_float(best, "ref_med", "ref_median_tok_s")
    final_tput = best_tput
    delta_pct = _ir_float(best, "e2e_delta_pct", "delta_pct")
    speedup = (final_tput / ref_med) if ref_med > 0 else (1.0 + delta_pct / 100.0)
    name = str(best.get("short_name") or "")
    # winner_kind in {"env","config","flags"} => config-only (no authored kernel).
    is_kernel = _ir_get(best, "winner_kind") not in ("env", "config", "flags")
    return {
        "eval_dir": str(eval_dir),
        "throughput_speedup": speedup,
        "baseline_throughput_tok_s": ref_med,
        "final_throughput_tok_s": final_tput,
        "output_parity": best.get("output_parity"),
        "validation_status": "recovered_intermediate",
        # Latency from the candidate (accepted) A/B leg when the integrator recorded
        # it (flat or nested) — so result.json carries real ttft/tpot even without a
        # final Validate bench. None when absent (never fabricated).
        "ttft_ms": _ir_get(best, "ttft_ms_cand", "ttft_ms_median", "cand_ttft_ms"),
        "tpot_ms": _ir_get(best, "tpot_ms_cand", "tpot_ms_median", "cand_tpot_ms"),
        "final_overlay": "",                 # config-only: applied via env/flags
        "final_launch_script": "",
        "accepted_config": {
            # The integrator emits the winning config under either key family:
            # ``apply_flags``/``apply_env`` (flat/nested-e2e schema) OR
            # ``accepted_flags``/``accepted_env`` (the integrate_result.json
            # summary schema). Read BOTH so a disk-recovered win never loses its
            # server flags/env — otherwise the recovered result.json carries an
            # empty accepted_config and every downstream reuse (sweep /
            # conc_sweep) relaunches an UN-optimized server. General across every
            # env/flags/config winner, not model-specific.
            "flags": str(_ir_get(best, "apply_flags", "accepted_flags") or ""),
            "env": str(_ir_get(best, "apply_env", "accepted_env") or ""),
        },
        "accepted_kernels": (
            [{"short_name": name, "kind": "authored", "backend": "geak",
              "e2e_delta_pct": delta_pct}]
            if is_kernel and name else []
        ),
        "accepted_heads": [],
        "recovered_from_disk": True,
        "recovered_intermediate": True,
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
            "validated": True, "decision": "KEEP", "patch_path": patch,
            "target_file": k.get("target_file") or k.get("target_callable"),
            "extra_server_args": str((wf.get("accepted_config") or {}).get("flags") or ""),
            "ts": None,
        },
    }


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
                m = _match_profiler(short)
                kid = m["kid"] if m else _canon_kid(short)
                if kid in seen:
                    continue
                seen.add(kid)
                ir = _read_json(cand / "integrate_result.json")
                kernels.append(_journey_overlay_entry(
                    eval_dir, short, ir, wf, geak_sha, overall_parity,
                    m["pct"] if m else None, m["name"] if m else None,
                    kernel_id_override=kid))

    # 2) Augment with accepted kernels named only in the workflow return (live path
    #    / no overlay on disk), deduped against the overlay entries above (by id).
    accepted = list(wf.get("accepted_kernels") or []) + list(wf.get("accepted_heads") or [])
    synth_hot: list[dict] = []
    for idx, k in enumerate(accepted):
        if not isinstance(k, dict):
            continue
        name = str(k.get("short_name") or k.get("name") or k.get("op_kind") or f"kernel{idx}")
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
        "schema_version": SCHEMA_VERSION,
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
        "schema_version": SCHEMA_VERSION,
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
    timeout_s = int(os.environ.get("GEAK_E2E_TIMEOUT_S", "43200"))  # 12h

    h = _read_json(handoff_path)
    if not h:
        sys.stderr.write(f"empty/invalid handoff: {handoff_path}\n")
        return 2

    ps_args = map_args(h, timeout_s)
    # Pin the single eval_dir into the environment so BOTH the live completion
    # check (_workflow_done_on_disk) and the scrape-independent disk recovery
    # (_discover_eval_dir) target EXACTLY this run's dir, deterministically.
    os.environ["GEAK_EVAL_DIR"] = ps_args["eval_dir"]
    bench_client = apply_bench_client(h)
    bench_launcher = apply_bench_launcher(h)
    bench_protocol = apply_bench_protocol(h)
    alignment_flags = apply_alignment_flags(h)
    prompt = build_prompt(ps_args)

    if "--dry-run" in flags:
        print(json.dumps({"mapped_args": ps_args, "bench_client": bench_client,
                          "bench_launcher": bench_launcher,
                          "magpie_launch_script": os.environ.get("MAGPIE_LAUNCH_SCRIPT", ""),
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
