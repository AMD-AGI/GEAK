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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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

# Public claude builds (>=2.1.x) REJECT "--effort ultracode". The Workflow /
# parallel / phase primitives that e2e_workflow.js needs are instead gated behind
# the `enableWorkflows` + `ultracode` settings keys (the highest-priority "flag
# settings" layer, == CLI `--settings`). Inject them so the Workflow tool truly
# executes the JS pipeline instead of the agent merely "backgrounding" it.
VALID_EFFORTS = {"low", "medium", "high", "xhigh", "max"}
WORKFLOW_SETTINGS = os.environ.get(
    "PERFSKILLS_CLAUDE_SETTINGS",
    json.dumps({"enableWorkflows": True, "ultracode": True}),
)
# Override which claude binary the SDK drives. The claude_agent_sdk otherwise
# prefers its OWN bundled CLI (claude_agent_sdk/_bundled/claude) over $PATH, so
# swapping the system claude alone has no effect on the SDK path. Set
# PERFSKILLS_CLAUDE_BIN to pin a specific build (e.g. an older native version).
CLAUDE_BIN = os.environ.get("PERFSKILLS_CLAUDE_BIN", "").strip()


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
    # Pin ONE EVAL_DIR for the whole run (workflow reads A.eval_dir ->
    # EVAL_DIR_OVERRIDE). Without it, every PHASE=setup invocation mints a fresh
    # timestamped dir, so a re-entered setup leaves an abandoned preflight-only
    # scaffold beside the authoritative run. Honor an explicit handoff/env
    # override first (resume); otherwise mint a single fresh dir here so BOTH
    # the preflight smoke and the real baseline/profile/kernel land under it.
    eval_dir = str(h.get("eval_dir") or os.environ.get("PERFSKILLS_EVAL_DIR", "")).strip()
    if not eval_dir:
        model_name = Path(h["model_path"]).name
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        eval_dir = str(Path(h["exp_root"]) / f"e2e_{model_name}_{ts}")
    ps_args["eval_dir"] = eval_dir
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
# Bench-protocol 口径 alignment (measurement knobs, not the client).
# ---------------------------------------------------------------------------
# handoff.bench_protocol key -> bench_e2e.sh / client-adapter env var.
_BENCH_PROTOCOL_ENV = {
    "random_range_ratio": "RANDOM_RANGE_RATIO",
    "num_prompts": "NUM_PROMPTS",
    "num_warmups": "NUM_WARMUPS",
    "seed": "SEED",
}


def apply_bench_protocol(h: dict) -> dict:
    """Export the caller's measurement 口径 so workflow bench_e2e.sh inherits it.

    ``handoff.bench_protocol`` carries the EXACT bench knobs the external
    orchestrator (Hyperloom) measured with — chiefly ``random_range_ratio``
    (fixed vs variable sequence lengths), ``num_prompts``, ``num_warmups`` and
    ``seed``. We export each PROVIDED key into the environment (same mechanism
    as :func:`apply_bench_client`), so every ``bench_e2e.sh`` invocation the
    agents make overrides its built-in default with the orchestrator's value.

    IMPORTANT: only keys actually present in the handoff are exported. When
    ``bench_protocol`` is absent (e.g. PerfSkills run standalone, no external
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
    """True once the workflow has written a TERMINAL completion marker.

    The Validate phase writes ``director_e2e_validation.json`` last, and the
    Finalize phase emits ``final/final_launch.sh``. Either is an authoritative
    "the optimizer finished a measured leg" signal that is independent of HOW
    the agent ran the workflow (in-turn vs background task). The pinned
    ``eval_dir`` (see :func:`map_args`) is what makes this a deterministic,
    single-path check rather than a guess.
    """
    if not eval_dir:
        return False
    p = Path(eval_dir)
    return (p / "director_e2e_validation.json").is_file() or (
        p / "final" / "final_launch.sh"
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
        return ClaudeAgentOptions(
            model=CLAUDE_MODEL,
            allowed_tools=ALLOWED_TOOLS,
            permission_mode="bypassPermissions",
            settings=WORKFLOW_SETTINGS,
            extra_args=extra,
            cwd=str(E2E_DIR),
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
                    if _workflow_done_on_disk(eval_dir):
                        break
                    # Background path concluded (success OR failure): stop and
                    # let downstream disk-recovery/normalize judge the result.
                    if terminal_task and saw_result:
                        break
                    # Pure synchronous path: the turn ended and no background
                    # task was ever spawned — the workflow ran in-turn.
                    if saw_result and not bg_started:
                        break
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

    Hyperloom's session-breakdown PerfSkills collector reads ``error_class`` to
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

    A pinned ``PERFSKILLS_EVAL_DIR`` (set by main() from the single eval_dir
    map_args minted for this run) short-circuits the glob/guess: recovery then
    targets EXACTLY the dir this run used, never a sibling from another run.
    """
    pinned = os.environ.get("PERFSKILLS_EVAL_DIR", "").strip()
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
    # Prefer a previously-persisted authoritative return (full structured data).
    persisted = _read_json(eval_dir / WORKFLOW_RETURN_FILE)
    if persisted.get("eval_dir"):
        return persisted
    validation = _read_json(eval_dir / "director_e2e_validation.json")
    if not validation:
        # No completion marker => the optimizer did not finish a measured leg.
        return None
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
    # still folds via the perfskills section + cumulative_gain).
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


def build_kernel_journey(wf: dict, normalized: dict) -> dict:
    """Build the schema-shaped kernel_journey contract from a workflow return.

    One entry per accepted kernel/head; each carries ``dispatch`` /
    ``backend_result`` / ``e2e`` sub-objects ready to feed the recorder. Empty
    ``kernels`` when nothing was accepted (still valid — the orchestrator then
    records nothing). General: iterates whatever the optimizer accepted.
    """
    eval_dir = str(normalized.get("eval_dir") or wf.get("eval_dir") or "")
    final_patch = normalized.get("final_patch") or ""
    parity = _parity_passed(wf.get("output_parity") or normalized.get("output_parity"))
    geak_sha = _git_short_sha(PERFSKILLS_ROOT)
    versions = {
        "geak": {
            "tool": "geak",
            "root_dir": str(PERFSKILLS_ROOT),
            "commit": geak_sha,
            "version": geak_sha,
        }
    }

    accepted = list(wf.get("accepted_kernels") or []) + list(wf.get("accepted_heads") or [])
    kernels: list[dict] = []
    # Discovery-shape projection (KERNEL_JOURNEY_SCHEMA.md §3/§5). GEAK-e2e
    # profiles via rocprofv3 (not tracelens), so the discovery ROUTE is
    # ``bypass``. The assembler backfills each kernel's discovery-sourced fields
    # (name / gpu_pct / bound_type / source_file) from these hot_kernels — they
    # are dropped otherwise, since dispatch/backend_result/e2e don't carry them.
    hot_kernels: list[dict] = []
    for idx, k in enumerate(accepted):
        if not isinstance(k, dict):
            continue
        name = str(k.get("short_name") or k.get("name") or k.get("op_kind") or f"kernel{idx}")
        kernel_id = name
        backend = _norm_backend(k.get("backend") or k.get("source"))
        isolated = k.get("isolated") or k.get("micro_speedup") or k.get("verified_isolated_speedup")
        e2e_delta = k.get("e2e_delta_pct")
        gpu_pct = k.get("pct_gpu_time")
        patch = k.get("final_patch") or final_patch or None
        attempt_id = f"{kernel_id}-{backend}-{idx}"
        hot_kernels.append({
            "kernel_id": kernel_id,
            "name": name,
            "gpu_pct": gpu_pct,
            "bound_type": str(k.get("bound_type") or k.get("op_kind") or ""),
            "source_file": k.get("target_file") or k.get("target_callable"),
            "recommended_backends": [backend],
            "selected_for_optimization": True,
        })
        attempt = {
            "backend": backend,
            "attempt_id": attempt_id,
            "status": "succeeded",
            "decision": "KEEP",
            "micro_speedup": isolated,
            "compile_passed": True,
            "correctness_passed": parity,
            "optimized_path": patch,
            "error": None,
            "error_type": None,
        }
        kernels.append({
            "kernel_id": kernel_id,
            "name": name,
            "gpu_pct": gpu_pct,
            "dispatch": {
                "dispatched": True,
                "backends": [backend],
                "skip_reason": "",
                "task_group": None,
            },
            # Shape == record_kernel_backend_result's ``result`` input.
            "backend_result": {
                "kernel_id": kernel_id,
                "run_id": str(k.get("kernel_eval_dir") or eval_dir),
                "attempts": [attempt],
                "verification": {
                    "micro_speedup": isolated,
                    "best_attempt_id": attempt_id,
                    "best_backend": backend,
                },
                "metadata": {"root_dir": str(PERFSKILLS_ROOT), "version": geak_sha},
            },
            # Shape == record_kernel_e2e's keyword inputs.
            "e2e": {
                "integrated": True,
                "e2e_gain_pct": e2e_delta,
                "validated": True,
                "decision": "KEEP",
                "patch_path": patch,
                "target_file": k.get("target_file") or k.get("target_callable"),
                "extra_server_args": str((wf.get("accepted_config") or {}).get("flags") or ""),
            },
        })

    return {
        "schema_version": SCHEMA_VERSION,
        "producer": "kernel-agent",
        "eval_dir": eval_dir,
        "versions": versions,
        # Stage-1 discovery substream (schema §3) so Hyperloom's recorder can
        # backfill the discovery-sourced kernel fields. Empty when nothing was
        # accepted (still valid).
        "discovery_runs": [
            {
                "source": "bypass",
                "status": "success",
                "scan": {"candidates_path": f"perfskills:{eval_dir}"},
                "hot_kernels": hot_kernels,
            }
        ] if hot_kernels else [],
        "kernels": kernels,
    }


def _write_kernel_journey(eval_dir: Path, wf: dict, normalized: dict) -> str:
    """Write kernel_journey.json into eval_dir; return its path ("" on failure)."""
    try:
        journey = build_kernel_journey(wf, normalized)
        path = eval_dir / KERNEL_JOURNEY_FILE
        path.write_text(json.dumps(journey, indent=2), encoding="utf-8")
        return str(path)
    except Exception:  # noqa: BLE001
        return ""


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
    timeout_s = int(os.environ.get("PERFSKILLS_E2E_TIMEOUT_S", "43200"))  # 12h

    h = _read_json(handoff_path)
    if not h:
        sys.stderr.write(f"empty/invalid handoff: {handoff_path}\n")
        return 2

    ps_args = map_args(h)
    # Pin the single eval_dir into the environment so BOTH the live completion
    # check (_workflow_done_on_disk) and the scrape-independent disk recovery
    # (_discover_eval_dir) target EXACTLY this run's dir, deterministically.
    os.environ["PERFSKILLS_EVAL_DIR"] = ps_args["eval_dir"]
    bench_client = apply_bench_client(h)
    bench_protocol = apply_bench_protocol(h)
    prompt = build_prompt(ps_args)

    if "--dry-run" in flags:
        print(json.dumps({"mapped_args": ps_args, "bench_client": bench_client,
                          "bench_protocol": bench_protocol,
                          "inferencex_path": os.environ.get("INFERENCEX_PATH", ""),
                          "prompt": prompt, "e2e_script": str(E2E_SCRIPT)}, indent=2))
        return 0

    recovered = False
    try:
        wf = invoke_workflow(prompt, timeout_s, ps_args["eval_dir"])
    except Exception as e:  # scrape/crash/timeout: try a disk recovery first.
        error_class = _classify_error(e)
        # The optimizer's artifacts (director_e2e_validation.json + final/
        # bundle + gain) are on disk even when the transcript scrape failed;
        # rebuild the return from them so a real win is never discarded over a
        # lost handoff line. Recovery yields None only when no completed
        # eval_dir exists (the run genuinely produced nothing to keep).
        wf = _recover_workflow_return(Path(h.get("exp_root") or ""))
        if wf is None:
            status = "timeout" if error_class == "timeout" else "error"
            result_path.parent.mkdir(parents=True, exist_ok=True)
            result_path.write_text(json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "status": status,
                    "error_class": error_class,
                    "error": str(e),
                },
                indent=2), encoding="utf-8")
            sys.stderr.write(f"PerfSkills e2e failed [{error_class}]: {e}\n")
            return 1
        recovered = True
        sys.stderr.write(
            f"PerfSkills e2e: transcript handoff failed [{error_class}]; "
            f"recovered the workflow return from disk artifacts "
            f"({wf.get('eval_dir')}).\n"
        )

    out = normalize_result(h, wf)

    # Persist the authoritative return + the kernel_journey contract beside the
    # artifacts so (a) a future recovery never re-scrapes and (b) the
    # orchestrator can replay the per-kernel journey into its breakdown.
    eval_dir = Path(out["eval_dir"])
    _persist_workflow_return(eval_dir, wf)
    kj_path = _write_kernel_journey(eval_dir, wf, out)
    if kj_path:
        out["kernel_journey_path"] = kj_path
    if recovered:
        out["recovered_from_disk"] = True

    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({"status": out["status"], "result_json": str(result_path),
                      "speedup": out["throughput_speedup"]}))
    return 0 if out["status"] != "error" else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
