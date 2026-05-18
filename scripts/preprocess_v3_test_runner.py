#!/usr/bin/env python3
"""GEAK v3 preprocessing test runner — HIP 3-kernel sweep.

Drives the v3 preprocess pipeline (orchestrator + 3 always-on subagents
+ Path-A short-circuit + per-language KB) against a YAML test plan
covering 3 AKA HIP kernels x 4 prompt variants + 1 verifier-escalation
fixture.

The runner has TWO process modes:

* **Parent mode** (default, when ``--plan ...`` is supplied) — iterates
  the test plan, spawns a child Python process per scenario via
  ``subprocess.run(..., timeout=600)``. The child process isolation
  gives us:

  - hard wall-clock cap per invocation (no signal handling required);
  - clean state between scenarios (no LLM-client/state bleed);
  - independent stdout/stderr capture per scenario for postmortem.

* **Child mode** (``--run-scenario``) — receives kernel + scenario
  configuration via stdin (JSON), runs the v3 orchestrator
  **in-process** (importing :mod:`minisweagent.run.preprocess_v3`
  directly, bypassing the ``geak`` CLI), serialises the resulting
  :class:`PreprocessResult` plus the agent's full message history to
  disk, and exits ``0`` regardless of test outcome (assertions are run
  by the parent against the on-disk artefacts).

Why bypass the ``geak`` CLI:

* The CLI also drives the optimisation loop (``run_pipeline``) which is
  out of scope for preprocess testing and would consume the entire
  4-hour budget on the first 1-2 kernels.
* In-process orchestrator access gives us the full
  :class:`PreprocessResult` dataclass — ``path_taken``, ``tool_calls``,
  ``subagent_runs``, ``errors`` — without log-scraping.
* The kernel target / repo path / task / GPU plumbing matches
  ``mem_homo/batch_test_hip_kernel.sh`` argument-shape-for-argument-
  shape; we use the same task strings and same kernel-url + repo +
  num-parallel + gpu-ids semantics, just with a cleaner observation
  surface.

Per-scenario assertions (driven by ``scenario.kind`` in the YAML):

* ``path_a`` — ``result.path_taken == "A"``; ``harness_path is None``;
  COMMANDMENT.md contains the user's command verbatim.
* ``path_b_determinism`` — runs the same prompt N times (default 3);
  hashes the harness file from each run; pass iff all N hashes match.
* ``path_b_coverage`` — single run; parse harness shape list; assert
  harness shapes (by integer-prefix signature) cover the oracle.
* ``task_override`` — single run; assert harness contains the single
  user-specified shape and (warning, not failure) no other shapes.
* ``path_a_partial`` — ``result.path_taken == "A"``; COMMANDMENT.md
  contains ``PATH_A_PARTIAL_COVERAGE`` markers for the modes the
  user's command did not cover.
* ``cross_language`` — piggybacks on a Path-B run's transcript;
  asserts HIP-canonical phrases present, Triton-canonical absent.
* ``verifier_escalation`` — kernel fixture with renamed entry function;
  ``result.success == False``, ``len(result.errors) > 0``,
  ``subagent_runs`` shows >= 3 ``harness-verifier`` dispatches.

Outputs:

* ``<output_root>/<kernel>/<scenario>/run<i>/`` — orchestrator artefact
  bundle (COMMANDMENT.md, harness-*.py, baseline_metrics.json, etc.)
  + ``preprocess_result.json`` (the runner-written serialisation of
  :class:`PreprocessResult`) + ``agent_messages.json`` (full LLM
  conversation transcript, used for cross-language phrase checks) +
  ``run_meta.json`` (wall-clock, exit code, stdout/stderr tails).
* ``<output_root>/summary.json`` — one record per kernel × scenario
  with ``status`` (pass / warn / fail / error) + per-assertion
  details.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from preprocess_v3_oracle import load_oracle_for_kernel, shape_signature  # noqa: E402

# Per-scenario wall-clock cap. Tightened from 1800s to 600s alongside the
# ``--preprocess-only`` CLI flag work: this runner already short-circuits
# at PreprocessResult (the child process runs the v3 orchestrator
# in-process and exits BEFORE any round loop), so a healthy run completes
# in 1-5 minutes wall-clock and a 600s budget surfaces hangs / runaway
# LLM loops without burning 30 minutes per scenario. Note: this runner
# does NOT spawn a ``geak`` subprocess (see module docstring); the
# ``--preprocess-only`` flag is therefore not threaded through here, the
# in-process invocation already has the same effect.
DEFAULT_PER_RUN_TIMEOUT_S = 600
DEFAULT_DETERMINISM_RUNS = 3

HIP_CANONICAL_PHRASES = ("hipLaunchKernelGGL", "__global__", "hip/hip_runtime", "hipMalloc")
TRITON_CANONICAL_PHRASES = ("@triton.jit", "tl.program_id", "tl.load", "tl.store")


# ---------------------------------------------------------------------------
# Child-mode entry point — runs ONE scenario in-process and dumps to disk.
# ---------------------------------------------------------------------------


def child_run_scenario(scenario_payload: dict[str, Any]) -> int:
    """Execute one scenario in this process; serialise the result.

    Called only when the script is invoked with ``--run-scenario``. The
    parent passes a JSON-encoded payload on stdin describing kernel +
    scenario + output_dir. We import GEAK lazily (after ``--run-scenario``
    is detected) so that the parent's argparse + plan-loading path stays
    clean even when GEAK fails to import for some reason.
    """
    ensure_amd_llm_key()
    output_dir = Path(scenario_payload["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    kernel_path_str = scenario_payload["kernel_path"]
    repo_root_str = scenario_payload["repo_root"]
    task = scenario_payload["task"]
    gpu_id = int(scenario_payload.get("gpu_id", 0))

    from minisweagent.kernel_languages.base import KernelLanguage  # noqa: F401
    from minisweagent.models import get_model
    from minisweagent.run.preprocess_v3.lang import detect_language, detect_language_for_repo
    from minisweagent.run.preprocess_v3.orchestrator import (
        PreprocessOrchestratorAgent,
        PreprocessOrchestratorConfig,
    )
    from minisweagent.run.preprocess_v3.tools import register_default_tools

    kernel_path = Path(kernel_path_str).resolve()
    detected_language = detect_language(kernel_path)
    if detected_language.name == "unknown":
        detected_language = detect_language_for_repo(Path(repo_root_str).resolve())

    # Reproduce the model config that ``run/mini.py`` builds from
    # ``config/mini_kernel_strategy_list.yaml`` + ``config/geak.yaml``.
    # Loading the YAML files directly keeps the child agnostic to
    # mini.py's heavy CLI initialisation (parse_pipeline_params,
    # parse_task_info, watchdog setup) — none of which is needed for
    # preprocess-only testing.
    model_config = _load_model_config()
    if scenario_payload.get("model"):
        model_config["model_name"] = scenario_payload["model"]
    model = get_model(model_config.get("model_name"), model_config)

    config = PreprocessOrchestratorConfig(
        gpu_id=gpu_id,
        repo=Path(repo_root_str).resolve(),
    )
    agent = PreprocessOrchestratorAgent(model=model, config=config)
    register_default_tools(agent, kernel_language=detected_language)

    t0 = time.monotonic()
    error_payload: dict[str, Any] | None = None
    try:
        result = agent.run(
            task=task,
            kernel_path=kernel_path,
            repo_root=str(Path(repo_root_str).resolve()),
            kernel_language=detected_language,
            source_language=detected_language.name,
            target_language=detected_language.name,
            output_dir=output_dir,
            gpu_id=gpu_id,
        )
    except Exception as exc:
        error_payload = {
            "exception_type": type(exc).__name__,
            "exception_repr": repr(exc),
            "traceback": traceback.format_exc(),
        }
        result = None
    elapsed_s = round(time.monotonic() - t0, 3)

    result_dump = _serialise_result(result, error_payload, elapsed_s, kernel_path, detected_language)
    (output_dir / "preprocess_result.json").write_text(
        json.dumps(result_dump, indent=2, default=str), encoding="utf-8"
    )
    (output_dir / "agent_messages.json").write_text(
        json.dumps(agent.messages, indent=2, default=str), encoding="utf-8"
    )
    sys.stdout.write(f"CHILD_DONE elapsed_s={elapsed_s}\n")
    return 0


def ensure_amd_llm_key() -> None:
    """Make sure ``AMD_LLM_API_KEY`` is set to a key the gateway accepts.

    Some host setups expose the AMD subscription key inside
    ``ANTHROPIC_CUSTOM_HEADERS`` (formatted ``key:value, key:value``)
    rather than as a bare ``AMD_LLM_API_KEY`` env var. Cursor's
    Anthropic-routing setup is one of those. When that's the case and
    the existing ``AMD_LLM_API_KEY`` is empty / fails authentication,
    we fall back to extracting the ``Ocp-Apim-Subscription-Key`` field
    from ``ANTHROPIC_CUSTOM_HEADERS`` and exporting it as
    ``AMD_LLM_API_KEY`` so :class:`AmdLlmModel` picks it up via its
    standard env-var lookup. Idempotent — safe to call multiple times.
    """
    headers = os.environ.get("ANTHROPIC_CUSTOM_HEADERS", "")
    if not headers:
        return
    match = re.search(r"Ocp-Apim-Subscription-Key\s*:\s*([0-9a-fA-F]{16,})", headers)
    if not match:
        return
    extracted = match.group(1)
    current = os.environ.get("AMD_LLM_API_KEY", "")
    if current == extracted:
        return
    os.environ["AMD_LLM_API_KEY"] = extracted


def _load_model_config() -> dict[str, Any]:
    """Load the model config from GEAK's standard config files.

    Mirrors ``run/mini.py``'s deep-merge of
    ``config/mini_kernel_strategy_list.yaml`` (base) over
    ``config/geak.yaml`` (final override). Returns the merged ``model``
    sub-dict, with at minimum ``model_class`` and ``model_name`` set so
    :func:`get_model` constructs the AMD-router-backed model the
    orchestrator expects (rather than the bare LiteLLM fallback that
    would fail on ``claude-opus-4.6`` without a provider prefix).
    """
    base_path = REPO_ROOT / "src" / "minisweagent" / "config" / "mini_kernel_strategy_list.yaml"
    geak_path = REPO_ROOT / "src" / "minisweagent" / "config" / "geak.yaml"
    base_cfg = yaml.safe_load(base_path.read_text(encoding="utf-8")) or {} if base_path.is_file() else {}
    geak_cfg = yaml.safe_load(geak_path.read_text(encoding="utf-8")) or {} if geak_path.is_file() else {}
    merged: dict[str, Any] = dict(base_cfg)
    for key, value in geak_cfg.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    model_cfg = dict(merged.get("model") or {})
    model_cfg.setdefault("model_class", "amd_llm")
    model_cfg.setdefault("model_name", "claude-opus-4.6")
    return model_cfg


def _serialise_result(
    result: Any,
    error_payload: dict[str, Any] | None,
    elapsed_s: float,
    kernel_path: Path,
    detected_language: Any,
) -> dict[str, Any]:
    """Project a :class:`PreprocessResult` (or a crash) to a JSON-safe dict."""
    if result is None:
        return {
            "success": False,
            "elapsed_s": elapsed_s,
            "kernel_path": str(kernel_path),
            "kernel_language": getattr(detected_language, "name", None),
            "errors": [error_payload["exception_repr"]] if error_payload else ["unknown_failure"],
            "child_error": error_payload,
            "tool_calls": [],
            "subagent_runs": [],
            "harness_path": None,
            "commandment_path": None,
            "path_taken": "B",
        }
    out: dict[str, Any] = {
        "success": bool(result.success),
        "elapsed_s": elapsed_s,
        "orchestrator_elapsed_s": float(result.elapsed_s),
        "kernel_path": str(result.kernel_path) if result.kernel_path else None,
        "kernel_language": getattr(result.kernel_language, "name", None),
        "harness_path": str(result.harness_path) if result.harness_path else None,
        "commandment_path": str(result.commandment_path) if result.commandment_path else None,
        "path_taken": result.path_taken,
        "errors": list(result.errors or []),
        "tool_calls": [{"name": tc.get("name"), "args": tc.get("args")} for tc in (result.tool_calls or [])],
        "subagent_runs": [_dump_dataclass(r) for r in (result.subagent_runs or [])],
    }
    if result.baseline is not None:
        out["baseline"] = _dump_dataclass(result.baseline)
    if result.profile is not None:
        out["profile"] = _dump_dataclass(result.profile)
    return out


def _dump_dataclass(value: Any) -> Any:
    """Best-effort serialisation: dataclass -> dict, dict-as-is, else str."""
    if value is None:
        return None
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return value
    if isinstance(value, (list, tuple)):
        return [_dump_dataclass(v) for v in value]
    return str(value)


# ---------------------------------------------------------------------------
# Parent-mode plumbing — spawns child processes, applies assertions.
# ---------------------------------------------------------------------------


def _spawn_child(
    *,
    kernel_path: str,
    repo_root: str,
    task: str,
    output_dir: Path,
    gpu_id: int,
    timeout_s: int,
    model: str | None,
    log_file: Path,
) -> dict[str, Any]:
    """Run one scenario as a child process; return ``{exit_code, wall_s, stdout_tail, stderr_tail}``."""
    payload = {
        "kernel_path": kernel_path,
        "repo_root": repo_root,
        "task": task,
        "output_dir": str(output_dir),
        "gpu_id": gpu_id,
        "model": model,
    }
    cmd = [sys.executable, str(Path(__file__).resolve()), "--run-scenario"]
    log_file.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )
        wall_s = round(time.monotonic() - t0, 3)
        log_file.write_text(
            f"=== STDOUT ===\n{proc.stdout}\n\n=== STDERR ===\n{proc.stderr}\n",
            encoding="utf-8",
        )
        return {
            "exit_code": proc.returncode,
            "wall_s": wall_s,
            "timed_out": False,
            "stdout_tail": (proc.stdout or "")[-4000:],
            "stderr_tail": (proc.stderr or "")[-4000:],
        }
    except subprocess.TimeoutExpired as exc:
        wall_s = round(time.monotonic() - t0, 3)
        stdout = (exc.stdout or b"").decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = (exc.stderr or b"").decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        log_file.write_text(
            f"=== TIMEOUT after {timeout_s}s ===\n=== STDOUT ===\n{stdout}\n\n=== STDERR ===\n{stderr}\n",
            encoding="utf-8",
        )
        return {
            "exit_code": -1,
            "wall_s": wall_s,
            "timed_out": True,
            "stdout_tail": stdout[-4000:],
            "stderr_tail": stderr[-4000:],
        }


# ---------------------------------------------------------------------------
# Harness-shape parsing
# ---------------------------------------------------------------------------


_SHAPE_LIST_NAMES = ("ALL_SHAPES", "TEST_SHAPES", "SHAPES", "BENCHMARK_SHAPES", "DEFAULT_SHAPES")


def _coerce_node(node: ast.AST) -> Any:
    """Like :func:`_to_python_value` in oracle, but tolerant of names + calls.

    Names resolve to ``None`` (we just want the literal structure). Calls
    are flattened to their argument list (so ``torch.Size((B, N, 3))``
    becomes ``(B, N, 3)`` ish). Anything we can't parse becomes ``None``.
    """
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _coerce_node(node.operand)
        return -inner if isinstance(inner, (int, float)) else None
    if isinstance(node, ast.Tuple):
        return tuple(_coerce_node(e) for e in node.elts)
    if isinstance(node, ast.List):
        return [_coerce_node(e) for e in node.elts]
    if isinstance(node, ast.Dict):
        return {
            _coerce_node(k) if k is not None else None: _coerce_node(v) for k, v in zip(node.keys, node.values, strict=False)
        }
    if isinstance(node, ast.Call):
        return tuple(_coerce_node(a) for a in node.args)
    return None


def parse_harness_shapes(harness_path: str | Path) -> list[tuple]:
    """Extract a flat list of shape tuples from a harness Python file.

    Strategy:

    1. AST walk the module body; for each top-level / nested-function-
       level assignment whose target name matches one of
       :data:`_SHAPE_LIST_NAMES`, decode the right-hand side via
       :func:`_coerce_node`.
    2. Flatten all decoded lists/tuples to a list of "leaf" tuples.
    3. As a regex fallback (when AST decoding yields nothing), grep for
       parenthesised number sequences inside an obvious shapes block.

    The list returned is what the runner diffs against the oracle for
    the coverage / task-override assertions.
    """
    p = Path(harness_path)
    if not p.is_file():
        return []
    text = p.read_text(encoding="utf-8")
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return _regex_shape_fallback(text)

    candidates: list[Any] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in _SHAPE_LIST_NAMES:
                    candidates.append(_coerce_node(node.value))
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id in _SHAPE_LIST_NAMES and node.value is not None:
                candidates.append(_coerce_node(node.value))

    out: list[tuple] = []
    for cand in candidates:
        out.extend(_flatten_to_shape_tuples(cand))
    if out:
        return out
    return _regex_shape_fallback(text)


def _flatten_to_shape_tuples(value: Any) -> list[tuple]:
    """Walk a (possibly nested) list/tuple and yield leaf tuples-of-numbers."""
    out: list[tuple] = []
    if value is None:
        return out
    if isinstance(value, (list, tuple)):
        if value and all(isinstance(v, (int, float)) for v in value):
            out.append(tuple(value))
            return out
        for elem in value:
            out.extend(_flatten_to_shape_tuples(elem))
    return out


_SHAPE_REGEX = re.compile(r"\((\s*\d+\s*(?:,\s*\d+\s*){1,6})\)")


def _regex_shape_fallback(text: str) -> list[tuple]:
    """Last-resort shape extraction for harnesses that hide shapes in calls."""
    out: list[tuple] = []
    for match in _SHAPE_REGEX.finditer(text):
        body = match.group(1)
        try:
            tup = tuple(int(x.strip()) for x in body.split(",") if x.strip())
        except ValueError:
            continue
        if 2 <= len(tup) <= 8 and all(t > 0 for t in tup):
            out.append(tup)
    return out


def _hash_file(path: str | Path) -> str | None:
    p = Path(path)
    if not p.is_file():
        return None
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Per-scenario assertions
# ---------------------------------------------------------------------------


def _load_run_artifacts(run_dir: Path) -> dict[str, Any]:
    """Read the per-run JSON dumps + COMMANDMENT.md + harness file."""
    out: dict[str, Any] = {"run_dir": str(run_dir)}
    pr_path = run_dir / "preprocess_result.json"
    out["result"] = json.loads(pr_path.read_text(encoding="utf-8")) if pr_path.is_file() else None

    msg_path = run_dir / "agent_messages.json"
    if msg_path.is_file():
        try:
            out["messages"] = json.loads(msg_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            out["messages"] = []
    else:
        out["messages"] = []

    cmd_path = run_dir / "COMMANDMENT.md"
    out["commandment_text"] = cmd_path.read_text(encoding="utf-8") if cmd_path.is_file() else ""

    harness_paths = sorted(run_dir.glob("harness*.py"))
    out["harness_files"] = [str(p) for p in harness_paths]
    return out


def assert_path_a(scenario_cfg: dict[str, Any], artifacts: dict[str, Any], run_meta: dict[str, Any]) -> dict[str, Any]:
    """Path-A assertion: short-circuit + COMMANDMENT rendered, no harness file expected.

    Strictness of each check:

    * ``path_taken == A`` — hard. The orchestrator must have called
      ``commandment_from_user_command`` (which is what flips
      ``path_taken`` to ``A``).
    * ``commandment file exists`` — hard. The whole point of Path A is
      the COMMANDMENT.md artefact; if it's missing the run is broken.
    * ``COMMANDMENT contains user command substring`` — hard. The
      rendered commandment must carry the user's literal command.
    * ``harness_path is None or == kernel_path`` — warn-only. The v3
      contract ``_finalize_success`` allows ``harness_path`` to be
      ``None`` on Path A (the harness IS the user's command). In
      practice the LLM occasionally surfaces ``kernel_path`` through
      ``finish_preprocess(harness_path=...)``; we record but don't fail
      on that, since it doesn't affect downstream consumers.
    """
    result = artifacts.get("result") or {}
    checks: list[dict[str, Any]] = []

    path_taken = result.get("path_taken")
    checks.append({
        "name": "path_taken == A",
        "ok": path_taken == "A",
        "detail": f"observed path_taken={path_taken!r}",
    })

    commandment_path = result.get("commandment_path")
    text = artifacts.get("commandment_text") or ""
    checks.append({
        "name": "COMMANDMENT.md rendered to disk",
        "ok": bool(commandment_path) and bool(text),
        "detail": f"commandment_path={commandment_path!r}, text_len={len(text)}",
    })

    harness_path = result.get("harness_path")
    kernel_path = result.get("kernel_path")
    harness_ok = (
        harness_path in (None, "")
        or (kernel_path is not None and harness_path == kernel_path)
    )
    checks.append({
        "name": "harness_path None or == kernel_path",
        "ok": harness_ok,
        "detail": f"harness_path={harness_path!r}, kernel_path={kernel_path!r}",
        "warn_only": True,
    })

    expected_substring = scenario_cfg.get("expect_command_substring", "")
    if expected_substring:
        ok = expected_substring in text
        checks.append({
            "name": "COMMANDMENT contains user command substring",
            "ok": ok,
            "detail": f"substring={expected_substring!r} present={ok}",
        })
    return _bundle_checks(checks, run_meta)


def assert_path_a_partial(
    scenario_cfg: dict[str, Any], artifacts: dict[str, Any], run_meta: dict[str, Any]
) -> dict[str, Any]:
    """Path-A partial assertion: short-circuit + PATH_A_PARTIAL_COVERAGE markers."""
    result = artifacts.get("result") or {}
    checks: list[dict[str, Any]] = []

    path_taken = result.get("path_taken")
    checks.append({
        "name": "path_taken == A",
        "ok": path_taken == "A",
        "detail": f"observed path_taken={path_taken!r}",
    })

    text = artifacts.get("commandment_text") or ""
    expected_modes = scenario_cfg.get("expect_partial_modes", ["correctness", "full_benchmark", "profile"])
    missing_markers: list[str] = []
    for mode in expected_modes:
        marker_present = (
            f"PATH_A_PARTIAL_COVERAGE: {mode} not covered" in text
            or f"PATH_A_PARTIAL_COVERAGE: {mode} inferred" in text
        )
        if not marker_present:
            missing_markers.append(mode)
    checks.append({
        "name": "PATH_A_PARTIAL_COVERAGE markers present for uncovered modes",
        "ok": not missing_markers,
        "detail": (
            f"expected modes with marker: {expected_modes}; missing markers: {missing_markers}"
        ),
    })
    return _bundle_checks(checks, run_meta)


def assert_path_b_coverage(
    scenario_cfg: dict[str, Any],
    artifacts: dict[str, Any],
    run_meta: dict[str, Any],
    oracle: dict[str, Any],
) -> dict[str, Any]:
    """Path-B coverage assertion: harness shapes >= oracle shapes (by signature)."""
    result = artifacts.get("result") or {}
    checks: list[dict[str, Any]] = []

    path_taken = result.get("path_taken")
    checks.append({
        "name": "path_taken == B",
        "ok": path_taken == "B",
        "detail": f"observed path_taken={path_taken!r}",
    })

    harness_files = artifacts.get("harness_files") or []
    if not harness_files:
        checks.append({"name": "harness file produced", "ok": False, "detail": "no harness*.py in run dir"})
        return _bundle_checks(checks, run_meta)
    harness_path = harness_files[0]
    parsed_shapes = parse_harness_shapes(harness_path)
    oracle_sigs_lists = oracle.get("signatures") or []
    oracle_sigs = {tuple(s) for s in oracle_sigs_lists}

    parsed_sigs = {shape_signature(tuple(s)) for s in parsed_shapes}
    parsed_sigs.discard(())
    missing = sorted(oracle_sigs - parsed_sigs, key=lambda t: (len(t), t))
    extras = sorted(parsed_sigs - oracle_sigs, key=lambda t: (len(t), t))

    if missing:
        ok = False
        status = "missing oracle signatures"
    elif extras:
        ok = True
        status = "harness covers oracle (with extras)"
    else:
        ok = True
        status = "harness signatures == oracle signatures"
    checks.append({
        "name": "harness signatures cover oracle",
        "ok": ok,
        "detail": f"{status}; oracle={sorted(oracle_sigs)}, parsed={sorted(parsed_sigs)}, missing={missing}, extras={extras}",
        "warn_only": ok and bool(extras),
    })
    return _bundle_checks(checks, run_meta)


def assert_task_override(
    scenario_cfg: dict[str, Any], artifacts: dict[str, Any], run_meta: dict[str, Any]
) -> dict[str, Any]:
    """Task-override assertion: harness contains ONLY the user-specified shape."""
    result = artifacts.get("result") or {}
    checks: list[dict[str, Any]] = []

    expected_shape = tuple(scenario_cfg["expect_shape"])
    path_taken = result.get("path_taken")
    checks.append({
        "name": "path_taken == B",
        "ok": path_taken == "B",
        "detail": f"observed path_taken={path_taken!r}",
    })

    harness_files = artifacts.get("harness_files") or []
    if not harness_files:
        checks.append({"name": "harness file produced", "ok": False, "detail": "no harness*.py in run dir"})
        return _bundle_checks(checks, run_meta)
    harness_path = harness_files[0]
    parsed_shapes = parse_harness_shapes(harness_path)
    parsed_sigs = {shape_signature(tuple(s)) for s in parsed_shapes}
    parsed_sigs.discard(())
    expected_sig = shape_signature(expected_shape)

    contains_expected = expected_sig in parsed_sigs
    extras = sorted(parsed_sigs - {expected_sig}, key=lambda t: (len(t), t))

    checks.append({
        "name": "harness contains user-specified shape",
        "ok": contains_expected,
        "detail": f"expected_sig={expected_sig}, parsed={sorted(parsed_sigs)}",
    })
    checks.append({
        "name": "no shapes leaked from source discovery",
        "ok": not extras,
        "detail": f"extra signatures present: {extras}",
        "warn_only": True,
    })
    return _bundle_checks(checks, run_meta)


def assert_cross_language(
    scenario_cfg: dict[str, Any], artifacts: dict[str, Any], run_meta: dict[str, Any]
) -> dict[str, Any]:
    """Cross-language assertion: HIP markers present, Triton markers absent."""
    expected_present = scenario_cfg.get("expected_phrases_present", list(HIP_CANONICAL_PHRASES))
    expected_absent = scenario_cfg.get("expected_phrases_absent", list(TRITON_CANONICAL_PHRASES))
    transcript_text = json.dumps(artifacts.get("messages") or [], default=str)

    found_present = [p for p in expected_present if p in transcript_text]
    found_absent_violations = [p for p in expected_absent if p in transcript_text]

    checks: list[dict[str, Any]] = [
        {
            "name": "expected HIP-canonical markers present in transcript",
            "ok": bool(found_present),
            "detail": f"hits={found_present}, looked_for={expected_present}",
        },
        {
            "name": "Triton-canonical markers absent from transcript",
            "ok": not found_absent_violations,
            "detail": f"unexpected hits={found_absent_violations}, looked_for={expected_absent}",
        },
    ]
    return _bundle_checks(checks, run_meta)


def assert_verifier_escalation(
    scenario_cfg: dict[str, Any], artifacts: dict[str, Any], run_meta: dict[str, Any]
) -> dict[str, Any]:
    """Verifier-escalation assertion: failure expected, with retries logged."""
    result = artifacts.get("result") or {}
    checks: list[dict[str, Any]] = []

    success = bool(result.get("success"))
    checks.append({
        "name": "result.success == False",
        "ok": not success,
        "detail": f"observed success={success}",
    })

    errors = result.get("errors") or []
    checks.append({
        "name": "result.errors populated",
        "ok": bool(errors),
        "detail": f"errors={errors}",
    })

    runs = result.get("subagent_runs") or []
    verifier_runs = [r for r in runs if isinstance(r, dict) and r.get("name") == "harness-verifier"]
    rejected = [r for r in verifier_runs if not r.get("success")]
    threshold = int(scenario_cfg.get("min_verifier_rejections", 1))
    checks.append({
        "name": f">= {threshold} verifier rejections recorded",
        "ok": len(rejected) >= threshold,
        "detail": f"verifier_runs={len(verifier_runs)}, rejected={len(rejected)}",
    })
    return _bundle_checks(checks, run_meta)


def _bundle_checks(checks: list[dict[str, Any]], run_meta: dict[str, Any]) -> dict[str, Any]:
    """Reduce a list of checks to a status string + carry the run metadata.

    Status legend:
    * ``error``  — the run timed out / child crashed before assertions ran.
    * ``fail``   — at least one non-warning check returned ``ok=False``.
    * ``warn``   — only warning-flagged checks failed; treat as soft signal.
    * ``pass``   — every check passed.
    """
    has_fail = any(not c["ok"] and not c.get("warn_only") for c in checks)
    has_warn = any(not c["ok"] and c.get("warn_only") for c in checks)
    if run_meta.get("timed_out"):
        status = "error"
    elif run_meta.get("exit_code", 0) != 0 and not any(c["ok"] for c in checks):
        status = "error"
    elif has_fail:
        status = "fail"
    elif has_warn:
        status = "warn"
    else:
        status = "pass"
    return {"status": status, "checks": checks, "run_meta": run_meta}


# ---------------------------------------------------------------------------
# Test-plan iteration
# ---------------------------------------------------------------------------


def _build_task_for_scenario(scenario: dict[str, Any], kernel: dict[str, Any]) -> str:
    """Construct the orchestrator's free-form task string per scenario kind.

    Mirrors ``mem_homo/batch_test_hip_kernel.sh``'s task shape for
    ``path_a`` and adds 3 descriptive variants for the other scenarios.
    The literal strings live here (not in the YAML) so they stay in sync
    with the orchestrator's Step 0 indicator catalogue and are diff-able
    in the runner.
    """
    repo_path = kernel["repo_path"]
    kernel_url = kernel["kernel_path"]
    kernel_name = kernel["name"]
    kind = scenario["kind"]

    if kind == "path_a":
        return (
            f"Optimize the repository {repo_path}, test command is "
            "python3 scripts/task_runner.py compile && "
            "python3 scripts/task_runner.py correctness && "
            "python3 scripts/task_runner.py performance"
        )
    if kind == "path_a_partial":
        return (
            f"Optimize the repository {repo_path}, test command is "
            "python3 scripts/task_runner.py performance"
        )
    if kind == "path_b_determinism" or kind == "path_b_coverage" or kind == "cross_language":
        return (
            f"Optimize the {kernel_name} HIP kernel located at {kernel_url} "
            f"in the repository {repo_path}. Benchmark across representative "
            "shapes for this kernel's intended workload."
        )
    if kind == "task_override":
        shape = scenario["expect_shape"]
        return (
            f"Optimize the {kernel_name} HIP kernel located at {kernel_url} in "
            f"{repo_path}. Use ONLY these shapes for all benchmarking: "
            f"{tuple(shape)}. Do not benchmark any other shapes; ignore "
            "shape lists discovered in source files."
        )
    if kind == "verifier_escalation":
        fixture_kernel_url = scenario.get("kernel_path_override") or kernel_url
        fixture_repo = scenario.get("repo_path_override") or repo_path
        return (
            f"Optimize the {kernel_name} HIP kernel located at {fixture_kernel_url} "
            f"in the repository {fixture_repo}. Benchmark across representative "
            "shapes for this kernel's intended workload."
        )
    raise ValueError(f"Unknown scenario.kind: {kind!r}")


def _resolve_kernel_url(scenario: dict[str, Any], kernel: dict[str, Any]) -> tuple[str, str]:
    """Pick the right kernel_path/repo_root pair (escalation fixtures override)."""
    kernel_url = scenario.get("kernel_path_override") or kernel["kernel_path"]
    repo_root = scenario.get("repo_path_override") or kernel["repo_path"]
    return kernel_url, repo_root


def _scenario_run_count(scenario: dict[str, Any]) -> int:
    if scenario["kind"] == "path_b_determinism":
        return int(scenario.get("runs", DEFAULT_DETERMINISM_RUNS))
    return 1


def _apply_assertions(
    scenario: dict[str, Any],
    artifacts: dict[str, Any],
    run_meta: dict[str, Any],
    oracle: dict[str, Any],
) -> dict[str, Any]:
    kind = scenario["kind"]
    if kind == "path_a":
        return assert_path_a(scenario, artifacts, run_meta)
    if kind == "path_a_partial":
        return assert_path_a_partial(scenario, artifacts, run_meta)
    if kind == "path_b_coverage":
        return assert_path_b_coverage(scenario, artifacts, run_meta, oracle)
    if kind == "task_override":
        return assert_task_override(scenario, artifacts, run_meta)
    if kind == "cross_language":
        return assert_cross_language(scenario, artifacts, run_meta)
    if kind == "verifier_escalation":
        return assert_verifier_escalation(scenario, artifacts, run_meta)
    if kind == "path_b_determinism":
        return assert_path_b_run_health(scenario, artifacts, run_meta)
    return {"status": "error", "checks": [], "run_meta": run_meta, "note": f"unknown kind {kind!r}"}


def assert_path_b_run_health(
    scenario_cfg: dict[str, Any], artifacts: dict[str, Any], run_meta: dict[str, Any]
) -> dict[str, Any]:
    """Per-run health check for ``path_b_determinism`` — ensure each run got
    far enough that hash + coverage piggyback assertions are meaningful.

    The hash-match across runs is computed in :func:`_summarise_scenario`;
    coverage and cross-language are piggybacked in :func:`run_plan` against
    run1's artefacts. This per-run check just verifies the orchestrator
    actually walked Path B and produced a harness file (if not, the
    downstream per-scenario assertions have nothing to compare).
    """
    result = artifacts.get("result") or {}
    checks: list[dict[str, Any]] = [
        {
            "name": "path_taken == B",
            "ok": result.get("path_taken") == "B",
            "detail": f"observed path_taken={result.get('path_taken')!r}",
        },
        {
            "name": "harness file produced",
            "ok": bool(artifacts.get("harness_files")),
            "detail": f"harness_files={artifacts.get('harness_files')!r}",
        },
    ]
    return _bundle_checks(checks, run_meta)


def run_plan(
    plan_path: Path,
    output_root: Path,
    *,
    timeout_s: int = DEFAULT_PER_RUN_TIMEOUT_S,
    only_kernels: set[str] | None = None,
    only_scenarios: set[str] | None = None,
    model: str | None = None,
    gpu_id: int = 0,
) -> dict[str, Any]:
    """Iterate the YAML plan, spawn children, run assertions, write summary."""
    plan = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
    output_root.mkdir(parents=True, exist_ok=True)
    summary_records: list[dict[str, Any]] = []
    determinism_records: list[dict[str, Any]] = []

    sweep_t0 = time.monotonic()

    for kernel in plan.get("kernels", []):
        if not kernel.get("enabled", True):
            continue
        if only_kernels is not None and kernel["name"] not in only_kernels:
            continue
        oracle = load_oracle_for_kernel(kernel)
        kernel_dir = output_root / kernel["name"]
        kernel_dir.mkdir(parents=True, exist_ok=True)
        for scenario in kernel.get("scenarios", []):
            if not scenario.get("enabled", True):
                continue
            if only_scenarios is not None and scenario["kind"] not in only_scenarios:
                continue
            scenario_dir = kernel_dir / scenario["kind"]
            scenario_dir.mkdir(parents=True, exist_ok=True)
            n_runs = _scenario_run_count(scenario)
            run_results: list[dict[str, Any]] = []
            for i in range(1, n_runs + 1):
                run_dir = scenario_dir / f"run{i}"
                run_dir.mkdir(parents=True, exist_ok=True)
                kernel_url, repo_root = _resolve_kernel_url(scenario, kernel)
                task = _build_task_for_scenario(scenario, kernel)
                log_file = run_dir / "child.log"
                run_meta = _spawn_child(
                    kernel_path=kernel_url,
                    repo_root=repo_root,
                    task=task,
                    output_dir=run_dir,
                    gpu_id=gpu_id,
                    timeout_s=timeout_s,
                    model=model,
                    log_file=log_file,
                )
                (run_dir / "run_meta.json").write_text(
                    json.dumps(run_meta, indent=2), encoding="utf-8"
                )
                artifacts = _load_run_artifacts(run_dir)
                run_record = _apply_assertions(scenario, artifacts, run_meta, oracle)
                run_record["run_dir"] = str(run_dir)
                run_record["task"] = task
                run_results.append(run_record)
                print(
                    f"  [{kernel['name']}/{scenario['kind']}] run{i}: status={run_record['status']} "
                    f"wall_s={run_meta['wall_s']:.1f} exit={run_meta['exit_code']}",
                    flush=True,
                )

            scenario_summary = _summarise_scenario(kernel, scenario, run_results, scenario_dir)
            scenario_summary["task"] = run_results[0]["task"] if run_results else None
            summary_records.append(scenario_summary)
            if scenario["kind"] == "path_b_determinism":
                determinism_records.append({
                    "kernel": kernel["name"],
                    "hashes": scenario_summary.get("determinism_hashes", []),
                    "all_match": scenario_summary.get("all_hashes_match", False),
                })
                # Piggyback assertions on run1 of the determinism scenario:
                # both coverage (vs oracle) and cross-language (HIP markers)
                # are properties of any clean Path-B run, so we extract them
                # for free instead of spawning two extra child invocations.
                if run_results:
                    run1_dir = Path(run_results[0]["run_dir"])
                    run1_meta = run_results[0]["run_meta"]
                    run1_artifacts = _load_run_artifacts(run1_dir)
                    coverage_record = assert_path_b_coverage(scenario, run1_artifacts, run1_meta, oracle)
                    coverage_record["kernel"] = kernel["name"]
                    coverage_record["scenario"] = "path_b_coverage"
                    coverage_record["scenario_dir"] = str(run1_dir)
                    coverage_record["task"] = run_results[0]["task"]
                    coverage_record["piggyback_on"] = "path_b_determinism/run1"
                    summary_records.append(coverage_record)

                    crosslang_record = assert_cross_language(
                        {"expected_phrases_present": list(HIP_CANONICAL_PHRASES),
                         "expected_phrases_absent": list(TRITON_CANONICAL_PHRASES)},
                        run1_artifacts, run1_meta,
                    )
                    crosslang_record["kernel"] = kernel["name"]
                    crosslang_record["scenario"] = "cross_language"
                    crosslang_record["scenario_dir"] = str(run1_dir)
                    crosslang_record["task"] = run_results[0]["task"]
                    crosslang_record["piggyback_on"] = "path_b_determinism/run1"
                    summary_records.append(crosslang_record)

    sweep_elapsed = round(time.monotonic() - sweep_t0, 3)
    summary = {
        "plan_path": str(plan_path.resolve()),
        "elapsed_s": sweep_elapsed,
        "records": summary_records,
        "determinism": determinism_records,
        "host": {"python": sys.version.split()[0], "platform": sys.platform},
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSWEEP DONE elapsed_s={sweep_elapsed} records={len(summary_records)}", flush=True)
    return summary


def _summarise_scenario(
    kernel: dict[str, Any],
    scenario: dict[str, Any],
    run_results: list[dict[str, Any]],
    scenario_dir: Path,
) -> dict[str, Any]:
    """Roll up multiple-run scenarios (determinism) into a single status."""
    statuses = [r["status"] for r in run_results]
    if not statuses:
        overall = "error"
    elif scenario["kind"] == "path_b_determinism":
        hashes = []
        for r in run_results:
            run_dir = Path(r["run_dir"])
            harness_files = sorted(run_dir.glob("harness*.py"))
            if harness_files:
                hashes.append(_hash_file(harness_files[0]))
            else:
                hashes.append(None)
        all_present = all(h is not None for h in hashes)
        all_match = all_present and len(set(hashes)) == 1
        if not all_present:
            overall = "error"
        elif all_match:
            overall = "pass"
        else:
            overall = "fail"
        return {
            "kernel": kernel["name"],
            "scenario": scenario["kind"],
            "status": overall,
            "scenario_dir": str(scenario_dir),
            "per_run": run_results,
            "determinism_hashes": hashes,
            "all_hashes_match": all_match,
        }
    elif "fail" in statuses or "error" in statuses:
        overall = "fail" if "fail" in statuses else "error"
    elif "warn" in statuses:
        overall = "warn"
    else:
        overall = "pass"
    return {
        "kernel": kernel["name"],
        "scenario": scenario["kind"],
        "status": overall,
        "scenario_dir": str(scenario_dir),
        "per_run": run_results,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="GEAK v3 preprocessing test runner — HIP sweep.")
    parser.add_argument("--plan", type=Path, help="Path to test_plan.yaml.")
    parser.add_argument("--output", type=Path, help="Output root directory.")
    parser.add_argument("--timeout-s", type=int, default=DEFAULT_PER_RUN_TIMEOUT_S)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--only-kernels", type=str, default="", help="Comma-separated kernel names.")
    parser.add_argument("--only-scenarios", type=str, default="", help="Comma-separated scenario kinds.")
    parser.add_argument("--model", type=str, default=None, help="Model override; default = router/env.")
    parser.add_argument("--run-scenario", action="store_true", help="Child-mode: run one scenario from stdin JSON.")
    args = parser.parse_args()

    if args.run_scenario:
        payload = json.loads(sys.stdin.read())
        return child_run_scenario(payload)

    if not args.plan or not args.output:
        parser.error("--plan and --output are required in parent mode")

    # Set the gateway key in the parent so the spawned children inherit
    # it. ``ensure_amd_llm_key`` is idempotent and a no-op when the key
    # is already valid, so calling it unconditionally is safe.
    ensure_amd_llm_key()

    only_kernels = {s.strip() for s in args.only_kernels.split(",") if s.strip()} or None
    only_scenarios = {s.strip() for s in args.only_scenarios.split(",") if s.strip()} or None

    summary = run_plan(
        args.plan.resolve(),
        args.output.resolve(),
        timeout_s=args.timeout_s,
        only_kernels=only_kernels,
        only_scenarios=only_scenarios,
        model=args.model,
        gpu_id=args.gpu_id,
    )
    failures = [r for r in summary["records"] if r["status"] in ("fail", "error")]
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
