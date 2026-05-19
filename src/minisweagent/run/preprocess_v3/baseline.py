"""Step 4 — baseline + profile collection for the v3 preprocess pipeline.

Two deterministic primitives, lifted from
``run/preprocess/phases/baseline.py`` + ``run/preprocess/kernel_profile.py``
+ ``run/preprocess/profiler_runner.py`` and given a tighter v3 surface:

* :func:`collect_baseline_metrics` — invokes the harness in
  ``--benchmark`` mode ``repeats`` times via ``subprocess.run`` and
  extracts ``GEAK_RESULT_LATENCY_MS=<float>`` (or any of the legacy
  parsers) from each invocation. Returns a frozen
  :class:`BaselineMetrics` carrying ``median_ms``, ``samples_ms``,
  ``stdev_ms``, the harness path, and the raw stdout/stderr from
  every run for downstream debugging.

* :func:`collect_profile` — wraps ``profiler-mcp``'s
  ``profile_kernel`` on the harness in ``--profile`` mode, returning
  a :class:`ProfileResult` with the parsed profile JSON, the wrapper
  command, and optionally the path the JSON was written to.

Both functions share the env contract used by
``run/preprocess/run_harness.py``: ``PYTHONPATH`` is prefixed with
the work_dir, ``HIP_VISIBLE_DEVICES`` is set, ``PYTHONUNBUFFERED=1``
is forced, and the harness is invoked via ``bash -lc`` so login-shell
profile fragments (FlyDSL setup, etc.) are sourced. This keeps
agent-side and preprocessor-side measurements directly comparable.

No LLM calls. The legacy variance / retry logic was a single shot
in the legacy code; the only addition here is the ``repeats`` knob,
which simply re-invokes the harness ``repeats`` times in series and
computes the median across samples — the benchmark output parsers
themselves are imported from the legacy module unchanged.
"""

from __future__ import annotations

import logging
import os
import shlex
import statistics
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# TODO(commit-set-5): inline; old preprocess/ goes away
from minisweagent.run.preprocess.benchmark_parsing import extract_latency_ms

# TODO(commit-set-5): inline; old preprocess/ goes away
from minisweagent.run.preprocess.repo_paths import ensure_preprocess_mcp_importable

logger = logging.getLogger(__name__)

#: Default profiler-mcp parameters — pinned to match the legacy
#: ``BaselinePhase._call_profiler`` defaults so v3 / legacy comparisons
#: of the same harness produce comparable numbers.
_DEFAULT_BACKEND = "metrix"
_DEFAULT_NUM_REPLAYS = 3
_DEFAULT_QUICK = False

#: Per-mode subprocess timeouts (seconds). Match
#: ``run/preprocess/run_harness.MODE_TIMEOUTS``.
_BENCHMARK_TIMEOUT_S = 600
_PROFILE_TIMEOUT_S = 120


@dataclass(frozen=True)
class BaselineMetrics:
    """Wall-clock benchmark statistics for a harness run.

    Attributes:
        harness_path:
            Absolute path to the harness script that produced these
            samples.
        median_ms:
            Median latency across ``samples_ms``. ``None`` when no
            sample produced a parseable latency (e.g. every run
            failed or the harness output didn't match any legacy
            extractor).
        samples_ms:
            Per-run latency_ms values, in invocation order.
            Successful runs that produced a parseable marker
            contribute one entry each; failures do not contribute
            to this list (they're still recorded in ``raw_outputs``).
        stdev_ms:
            Sample standard deviation of ``samples_ms``. ``None``
            when fewer than two samples were collected (Python's
            :func:`statistics.stdev` requires n >= 2).
        repeats:
            Number of times the harness was invoked. Matches the
            ``repeats`` argument to :func:`collect_baseline_metrics`.
        command:
            The exact shell command used per invocation (for audit /
            reproducibility). All ``repeats`` invocations use the
            same command — we replay it ``repeats`` times rather
            than parametrising over different shapes.
        raw_outputs:
            Per-invocation ``{returncode, stdout, stderr, duration_s,
            latency_ms}`` dicts. ``latency_ms`` is ``None`` when the
            extractor didn't match. Carried verbatim so downstream
            debugging never has to re-run the benchmark.
    """

    harness_path: Path
    median_ms: float | None
    samples_ms: list[float]
    stdev_ms: float | None
    repeats: int
    command: str
    raw_outputs: list[dict[str, Any]] = field(default_factory=list)

    @property
    def success(self) -> bool:
        """``True`` when at least one invocation produced a parseable latency."""
        return self.median_ms is not None


@dataclass(frozen=True)
class ProfileResult:
    """Structured profiler-mcp output.

    Attributes:
        harness_path:
            Absolute path to the harness script that was profiled.
        command:
            The wrapper command passed to profiler-mcp's
            ``profile_kernel``. Recorded so the downstream consumer
            can re-run the same profile invocation manually.
        profile:
            The structured profile JSON returned by profiler-mcp.
            ``None`` when the profiler was unavailable / failed.
        profile_path:
            On-disk path the JSON was written to (when
            ``out_path`` was supplied to :func:`collect_profile`).
            ``None`` when only an in-memory result was requested.
        backend:
            Profiler backend used (``"metrix"`` or
            ``"rocprof-compute"``).
        success:
            ``True`` when ``profile`` is non-``None`` and the
            profiler reported ``success`` (or didn't include the
            field, which the legacy code also treats as success).
    """

    harness_path: Path
    command: str
    profile: dict[str, Any] | None
    profile_path: Path | None = None
    backend: str = _DEFAULT_BACKEND

    @property
    def success(self) -> bool:
        if self.profile is None:
            return False
        # Legacy convention: ``success`` defaults to True when the
        # profiler returned a structured result without explicitly
        # marking failure.
        return bool(self.profile.get("success", True))


def _build_env(
    work_dir: Path | None,
    *,
    gpu_id: int,
    extra: dict[str, str] | None = None,
) -> dict[str, str]:
    """Build a subprocess env matching ``run/preprocess/run_harness._build_env``.

    Adds:
      * ``PYTHONPATH`` — work_dir prepended (when supplied) to the
        existing value.
      * ``HIP_VISIBLE_DEVICES`` — pinned to ``gpu_id``.
      * ``PYTHONUNBUFFERED=1`` — so streaming logs aren't lost.

    Optionally merges any extra overrides last. Returns a fresh dict
    (does not mutate ``os.environ``).
    """
    env = os.environ.copy()
    if work_dir is not None:
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{work_dir}:{existing}" if existing else str(work_dir)
    env["HIP_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONUNBUFFERED"] = "1"
    if extra:
        env.update(extra)
    return env


def _benchmark_command(harness_path: Path) -> list[str]:
    """Build the harness ``--benchmark`` argv (interpreter + path + flag).

    Wraps with ``bash -lc`` so login-shell profile fragments are
    sourced — matches ``run/preprocess/run_harness._run_single``.
    """
    inner = " ".join(shlex.quote(c) for c in [sys.executable, str(harness_path), "--benchmark"])
    return ["bash", "-lc", inner]


def _profile_command(harness_path: Path) -> str:
    """Build the harness ``--profile`` shell snippet for profiler-mcp.

    profiler-mcp expects a single-string command; we mirror what the
    legacy ``run_baseline_profile`` produced (``"python {harness}
    --profile"``) so the wrapper line in ``ProfileResult.command``
    matches what users see when inspecting historical pipelines.
    """
    return f"python3 {harness_path} --profile"


def _run_benchmark_once(
    harness_path: Path,
    *,
    work_dir: Path | None,
    gpu_id: int,
    timeout_s: int,
) -> dict[str, Any]:
    """Run the harness in ``--benchmark`` mode once and capture output.

    Returns:
        ``{returncode, stdout, stderr, duration_s, latency_ms}`` —
        ``latency_ms`` is the value extracted via
        :func:`extract_latency_ms`, or ``None`` if no marker matched.
        On subprocess failure ``returncode`` is non-zero and
        ``latency_ms`` is ``None``.
    """
    import time as _time

    cmd = _benchmark_command(harness_path)
    env = _build_env(work_dir, gpu_id=gpu_id)
    cwd = str(work_dir) if work_dir is not None else None

    t0 = _time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=env,
            cwd=cwd,
        )
        duration_s = round(_time.monotonic() - t0, 3)
        latency_ms = extract_latency_ms(proc.stdout or "") if proc.returncode == 0 else None
        return {
            "returncode": proc.returncode,
            "stdout": proc.stdout or "",
            "stderr": proc.stderr or "",
            "duration_s": duration_s,
            "latency_ms": latency_ms,
        }
    except subprocess.TimeoutExpired as exc:
        duration_s = round(_time.monotonic() - t0, 3)
        return {
            "returncode": -1,
            "stdout": (exc.stdout or "")
            if isinstance(exc.stdout, str)
            else (exc.stdout or b"").decode(errors="replace"),
            "stderr": f"TIMEOUT after {timeout_s}s",
            "duration_s": duration_s,
            "latency_ms": None,
        }
    except Exception as exc:
        duration_s = round(_time.monotonic() - t0, 3)
        return {
            "returncode": -1,
            "stdout": "",
            "stderr": str(exc),
            "duration_s": duration_s,
            "latency_ms": None,
        }


def collect_baseline_metrics(
    harness_path: Path,
    *,
    repeats: int = 5,
    work_dir: Path | None = None,
    gpu_id: int = 0,
) -> BaselineMetrics:
    """Run the harness ``repeats`` times in ``--benchmark`` mode.

    Each invocation is parsed with the same legacy benchmark-output
    extractors used by ``BaselinePhase`` so the v3 measurement is
    apples-to-apples with what the old pipeline produced.

    Args:
        harness_path:
            Absolute path to a validated harness. Must exist; missing
            files raise :class:`FileNotFoundError`.
        repeats:
            Number of independent invocations. Defaults to 5; values
            below 1 are clamped to 1 (a single sample yields a
            ``stdev_ms`` of ``None`` — see
            :class:`BaselineMetrics`).
        work_dir:
            Working directory + ``PYTHONPATH`` prefix. ``None``
            inherits the caller's CWD and PYTHONPATH unchanged.
        gpu_id:
            ``HIP_VISIBLE_DEVICES`` value for each invocation.
            Defaults to GPU 0 to match the legacy default.

    Returns:
        A :class:`BaselineMetrics` summarising the run.

    Raises:
        FileNotFoundError: If ``harness_path`` is not a regular file.
    """
    harness_path = Path(harness_path)
    if not harness_path.is_file():
        raise FileNotFoundError(f"collect_baseline_metrics: harness not found: {harness_path}")

    repeats = max(1, int(repeats))
    cmd = _benchmark_command(harness_path)
    cmd_str = " ".join(shlex.quote(c) for c in cmd)

    raw_outputs: list[dict[str, Any]] = []
    samples_ms: list[float] = []
    for i in range(repeats):
        result = _run_benchmark_once(
            harness_path,
            work_dir=work_dir,
            gpu_id=gpu_id,
            timeout_s=_BENCHMARK_TIMEOUT_S,
        )
        raw_outputs.append(result)
        if result["latency_ms"] is not None:
            samples_ms.append(float(result["latency_ms"]))
        else:
            logger.warning(
                "collect_baseline_metrics: sample %d/%d produced no latency (rc=%s)",
                i + 1,
                repeats,
                result["returncode"],
            )

    median_ms = statistics.median(samples_ms) if samples_ms else None
    stdev_ms = statistics.stdev(samples_ms) if len(samples_ms) >= 2 else None

    return BaselineMetrics(
        harness_path=harness_path.resolve(),
        median_ms=median_ms,
        samples_ms=samples_ms,
        stdev_ms=stdev_ms,
        repeats=repeats,
        command=cmd_str,
        raw_outputs=raw_outputs,
    )


def _invoke_profiler_mcp(
    command: str,
    *,
    backend: str,
    num_replays: int,
    quick: bool,
    gpu_devices: str,
    workdir: str | None,
) -> dict[str, Any] | None:
    """Call ``profiler_mcp.server.profile_kernel``.

    Factored into a helper so tests can monkeypatch one symbol
    without having to round-trip through the real profiler-mcp
    package (which depends on hip / metrix at import time on real
    GPU hosts).

    Returns the structured profile result, or ``None`` if the
    profiler is unavailable / raises.
    """
    try:
        ensure_preprocess_mcp_importable(
            "mcp_tools/profiler-mcp/src",
            "mcp_tools/metrix-mcp/src",
        )
        import importlib

        profiler_server = importlib.import_module("profiler_mcp.server")
        profile_kernel = profiler_server.profile_kernel
        # FastMCP wraps the function; the underlying callable lives
        # at ``.fn`` on the wrapped object. Match the legacy lookup
        # so we work whether profiler-mcp is wrapped or bare.
        profile_fn = getattr(profile_kernel, "fn", profile_kernel)
        kwargs: dict[str, Any] = {
            "command": command,
            "backend": backend,
            "num_replays": num_replays,
            "quick": quick,
            "gpu_devices": gpu_devices,
        }
        if workdir is not None:
            kwargs["workdir"] = workdir
        return profile_fn(**kwargs)
    except Exception as exc:
        logger.warning("profiler-mcp invocation failed: %s", exc, exc_info=True)
        return None


def collect_profile(
    harness_path: Path,
    *,
    work_dir: Path | None = None,
    gpu_id: int = 0,
    backend: str = _DEFAULT_BACKEND,
    num_replays: int = _DEFAULT_NUM_REPLAYS,
    quick: bool = _DEFAULT_QUICK,
    out_path: Path | None = None,
) -> ProfileResult:
    """Profile the harness via ``profiler-mcp.profile_kernel``.

    Mirrors the legacy ``BaselinePhase._call_profiler`` /
    ``run_baseline_profile`` invocation: same backend default
    (``metrix``), same replay count default (3), same env contract
    via :func:`_build_env`. The only difference is that the result
    is returned as a structured dataclass rather than written into
    a phase context.

    Args:
        harness_path:
            Absolute path to a validated harness. Must exist.
        work_dir:
            Working directory passed through as profiler-mcp's
            ``workdir`` argument and also used to set the wrapper's
            CWD.
        gpu_id:
            ``HIP_VISIBLE_DEVICES`` for the profiler. Forwarded to
            profiler-mcp via the ``gpu_devices`` parameter.
        backend:
            ``"metrix"`` (default) or ``"rocprof-compute"``.
        num_replays:
            Number of replays the profiler runs internally.
        quick:
            Quick-mode flag (3 metrics, 1 pass) for the metrix
            backend. Defaults to ``False`` to match the legacy
            ``BaselinePhase`` setting.
        out_path:
            When supplied, the profile JSON is also written here as
            indented JSON. Idempotent across repeated calls.

    Returns:
        A :class:`ProfileResult` carrying the profile dict and the
        wrapper command. ``ProfileResult.profile`` is ``None`` when
        the profiler is unavailable or failed; check
        ``ProfileResult.success`` for a boolean view.

    Raises:
        FileNotFoundError: If ``harness_path`` is not a regular file.
    """
    import json

    harness_path = Path(harness_path)
    if not harness_path.is_file():
        raise FileNotFoundError(f"collect_profile: harness not found: {harness_path}")

    command = _profile_command(harness_path.resolve())
    profile = _invoke_profiler_mcp(
        command,
        backend=backend,
        num_replays=num_replays,
        quick=quick,
        gpu_devices=str(gpu_id),
        workdir=str(work_dir) if work_dir is not None else None,
    )

    profile_path: Path | None = None
    if profile is not None and out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(profile, indent=2, default=str), encoding="utf-8")
        profile_path = out_path.resolve()

    return ProfileResult(
        harness_path=harness_path.resolve(),
        command=command,
        profile=profile,
        profile_path=profile_path,
        backend=backend,
    )


__all__ = [
    "BaselineMetrics",
    "ProfileResult",
    "collect_baseline_metrics",
    "collect_profile",
]
