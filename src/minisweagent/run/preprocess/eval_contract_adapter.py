"""Deterministic adapter: repo shell eval → universal harness contract.

Maps arbitrary ``eval_command`` / ``correctness_command`` + ``performance_command``
into a single Python harness that exposes:

  ``--correctness`` | ``--profile`` | ``--benchmark`` | ``--full-benchmark``

and prints ``GEAK_RESULT_LATENCY_MS`` / ``GEAK_RESULT_SPEEDUP`` as required by
``kernel_languages/contract.py`` and ``harness_utils.validate_harness``.

The generated harness captures subprocess stdout and parses it for kernel-level
timing (GPU event measurements, per-shape latencies, aggregate markers) before
falling back to wall-clock timing.  This ensures ``GEAK_RESULT_LATENCY_MS``
reflects actual kernel performance rather than subprocess overhead.

This is **not** the legacy monolithic ``preprocessor.py`` — it is a small code
generator so HIP / AgentKernelArena ``scripts/task_runner.py`` workflows join
the same contract surface as Triton harnesses **without** an LLM HarnessBuilder
when the eval is already explicit shell.

Kernel-language-specific preprocess (Discovery → ``KernelLanguage``) still
runs first; this layer only **wraps** the resolved shell commands.

**Relationship to HarnessBuilder (HarnessPhase layer 6):** open-ended prompts
and heterogeneous repos (pytest, custom scripts, Makefile flows, etc.) are
handled by the **HarnessBuilder** subagent — iterative LLM + language Jinja
template until the universal contract passes. This module is the **fast,
deterministic** path when Explore/COMMANDMENT already materialized explicit
``correctness_command`` / ``performance_command`` (or a single ``eval_command``
with ``&&``), so no LLM is needed to map shell → contract.
"""

from __future__ import annotations

import re
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from minisweagent.run.preprocess.phases.base import PhaseContext


def resolve_shell_eval_commands(ctx: PhaseContext) -> tuple[str | None, str | None]:
    """Return ``(correctness_shell, performance_shell)`` from ctx, if any."""
    cc = getattr(ctx, "correctness_command", None)
    pc = getattr(ctx, "performance_command", None)
    if cc is not None and pc is not None:
        cs = cc if isinstance(cc, str) else " && ".join(cc)
        ps = pc if isinstance(pc, str) else " && ".join(pc)
        cs, ps = cs.strip(), ps.strip()
        if cs and ps:
            return cs, ps

    ev = getattr(ctx, "eval_command", None)
    if isinstance(ev, str) and "&&" in ev:
        left, right = ev.rsplit("&&", 1)
        ls, rs = left.strip(), right.strip()
        if ls and rs:
            return ls, rs
    return None, None


# ---------------------------------------------------------------------------
# Kernel-level timing extraction (used both in-process and embedded in
# the generated harness).
# ---------------------------------------------------------------------------


def _extract_kernel_latency_ms(text: str) -> tuple[float | None, str]:
    """Parse subprocess output for kernel-level timing.

    Three-layer strategy:

    Layer 1 — Pass-through:
        If the subprocess already emitted ``GEAK_RESULT_LATENCY_MS``, return
        ``(None, "kernel")`` to signal the caller should not add another marker.

    Layer 2 — Summary extraction:
        Try well-known aggregate patterns (TOTAL_KERNEL_TIME_MS,
        median_latency_ms, Geomean, Google Benchmark, universal keyword scan).

    Layer 3 — Per-line timing aggregation:
        Broad regex for lines containing ``<number> (ms|us|µs)``.  Computes
        SUM of all extracted values (handles mixed ms/µs units).

    Returns ``(latency_ms, source_tag)`` where *source_tag* is one of
    ``"kernel"`` (pass-through), ``"subprocess_parsed"``, or ``"wall_clock"``
    (when returning None to signal wall-clock fallback).
    """
    # Strip ANSI escape sequences — SGR codes like \x1b[92m contain digits
    # followed by 'm' which the per-line regex misreads as "<N> ms".
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)

    # Layer 1: subprocess already emitted the canonical marker
    if re.search(r"GEAK_RESULT_LATENCY_MS=[\d.]+", text):
        return None, "kernel"

    # Layer 2: summary / aggregate patterns (priority order matches
    # benchmark_parsing.extract_latency_ms)
    _summary_patterns: list[tuple[str, int]] = [
        (r"(?:TOTAL_KERNEL_TIME_MS|BENCHMARK_LATENCY_MS)\s*:\s*([\d.]+(?:e[+-]?\d+)?)", 0),
        (r"BENCHMARK_METRIC:\s*median_latency_ms=([\d.]+(?:e[+-]?\d+)?)", 0),
        (r"median_latency_ms:\s*([\d.]+(?:e[+-]?\d+)?)", 0),
        (r"Geomean\s*\(ms\)\s*:\s*([\d.]+(?:e[+-]?\d+)?)", 0),
        (
            r"(?:[Mm]edian\s+(?:latency|time)[\w\s]*|total\s+median\s+time)"
            r"\s*:\s*([\d.]+(?:e[+-]?\d+)?)\s*ms",
            0,
        ),
    ]
    for pat, _flags in _summary_patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return float(m.group(1)), "subprocess_parsed"

    # Google Benchmark format: <name> <iters> <latency> ms
    m = re.search(r"^\S+\s+\d+\s+([\d.]+(?:e[+-]?\d+)?)\s+ms", text, re.MULTILINE)
    if m:
        return float(m.group(1)), "subprocess_parsed"

    # Universal keyword scanner (last 30 lines, near latency-related keywords)
    _keywords = {"median", "overall", "geomean", "latency", "total"}
    _candidates: list[float] = []
    lines = text.strip().splitlines()
    for line in lines[-30:]:
        lower = line.lower()
        if not any(kw in lower for kw in _keywords):
            continue
        for m in re.finditer(r"([\d.]+(?:e[+-]?\d+)?)\s*ms", line):
            val = float(m.group(1))
            if 0.0001 < val < 100_000:
                _candidates.append(val)
    if _candidates:
        return _candidates[-1], "subprocess_parsed"

    # Layer 3: per-line timing aggregation
    _per_line_re = re.compile(
        r"([\d.]+(?:e[+-]?\d+)?)\s*(ms|us|µs)(?:/launch|/iter)?",
        re.IGNORECASE,
    )
    per_line_ms: list[float] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        lower = stripped.lower()
        if lower.startswith(("geak_", "#", "---", "===", "total number")):
            continue
        # Skip compilation / hipify status lines
        if "->" in stripped and ("[ok]" in lower or "[skipped" in lower):
            continue
        if stripped.startswith("[") and stripped.endswith("]"):
            continue
        m = _per_line_re.search(stripped)
        if m:
            val = float(m.group(1))
            unit = m.group(2).lower()
            if unit in ("us", "µs"):
                val /= 1000.0
            if 0.0 < val < 100_000:
                per_line_ms.append(val)

    if per_line_ms:
        return sum(per_line_ms), "subprocess_parsed"

    return None, "wall_clock"


# ---------------------------------------------------------------------------
# String constant of the parser for embedding into the generated harness.
# Kept in sync with _extract_kernel_latency_ms above.
# ---------------------------------------------------------------------------

_PARSER_SOURCE = textwrap.dedent('''\
    def _extract_kernel_latency_ms(text):
        """Parse subprocess output for kernel-level timing (self-contained)."""
        import re as _re

        # Strip ANSI escape sequences (e.g. \\x1b[92m → digits + 'm' fools the ms regex)
        text = _re.sub(r'\\x1b\\[[0-9;]*m', '', text)

        # Layer 1: pass-through
        if _re.search(r"GEAK_RESULT_LATENCY_MS=[\\d.]+", text):
            return None, "kernel"

        # Layer 2: summary patterns
        _summary = [
            r"(?:TOTAL_KERNEL_TIME_MS|BENCHMARK_LATENCY_MS)\\s*:\\s*([\\d.]+(?:e[+-]?\\d+)?)",
            r"BENCHMARK_METRIC:\\s*median_latency_ms=([\\d.]+(?:e[+-]?\\d+)?)",
            r"median_latency_ms:\\s*([\\d.]+(?:e[+-]?\\d+)?)",
            r"Geomean\\s*\\(ms\\)\\s*:\\s*([\\d.]+(?:e[+-]?\\d+)?)",
            r"(?:[Mm]edian\\s+(?:latency|time)[\\w\\s]*|total\\s+median\\s+time)"
            r"\\s*:\\s*([\\d.]+(?:e[+-]?\\d+)?)\\s*ms",
        ]
        for pat in _summary:
            m = _re.search(pat, text, _re.IGNORECASE)
            if m:
                return float(m.group(1)), "subprocess_parsed"

        m = _re.search(r"^\\S+\\s+\\d+\\s+([\\d.]+(?:e[+-]?\\d+)?)\\s+ms", text, _re.MULTILINE)
        if m:
            return float(m.group(1)), "subprocess_parsed"

        _keywords = {"median", "overall", "geomean", "latency", "total"}
        _candidates = []
        lines = text.strip().splitlines()
        for line in lines[-30:]:
            lower = line.lower()
            if not any(kw in lower for kw in _keywords):
                continue
            for m in _re.finditer(r"([\\d.]+(?:e[+-]?\\d+)?)\\s*ms", line):
                val = float(m.group(1))
                if 0.0001 < val < 100000:
                    _candidates.append(val)
        if _candidates:
            return _candidates[-1], "subprocess_parsed"

        # Layer 3: per-line timing aggregation
        _plre = _re.compile(r"([\\d.]+(?:e[+-]?\\d+)?)\\s*(ms|us|µs)(?:/launch|/iter)?", _re.IGNORECASE)
        per_line_ms = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            lower = stripped.lower()
            if lower.startswith(("geak_", "#", "---", "===", "total number")):
                continue
            if "->" in stripped and ("[ok]" in lower or "[skipped" in lower):
                continue
            if stripped.startswith("[") and stripped.endswith("]"):
                continue
            m = _plre.search(stripped)
            if m:
                val = float(m.group(1))
                unit = m.group(2).lower()
                if unit in ("us", "µs"):
                    val /= 1000.0
                if 0.0 < val < 100000:
                    per_line_ms.append(val)

        if per_line_ms:
            return sum(per_line_ms), "subprocess_parsed"

        return None, "wall_clock"
''')


def materialize_shell_contract_harness(
    *,
    output_dir: Path,
    repo_root: str,
    correctness_shell: str,
    performance_shell: str,
) -> Path:
    """Write ``_geak_shell_contract_harness.py`` and return its path."""
    out = Path(output_dir) / "_geak_shell_contract_harness.py"
    # Embed commands as Python string literals (safe with repr).
    c_lit = repr(correctness_shell)
    p_lit = repr(performance_shell)
    r_lit = repr(str(Path(repo_root).resolve()))

    head = textwrap.dedent(
        f'''\
        #!/usr/bin/env python3
        """GEAK universal shell adapter — generated by eval_contract_adapter.

        Delegates to the repo's own compile/correctness/performance commands.
        Parses subprocess output for kernel-level timing before falling back
        to wall-clock measurement.
        """
        from __future__ import annotations

        import argparse
        import os
        import re
        import subprocess
        import sys
        import time

        REPO_ROOT = {r_lit}
        CORRECTNESS_CMD = {c_lit}
        PERFORMANCE_CMD = {p_lit}


        def _run_shell(cmd: str, cwd: str) -> tuple:
            env = os.environ.copy()
            result = subprocess.run(
                cmd, shell=True, cwd=cwd, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            )
            return result.returncode, result.stdout or ""

        '''
    )

    # The main() function uses runtime f-strings for elapsed_ms/kernel_ms
    # which must NOT be interpolated by this generator's f-string.  We use
    # a plain string (no f-prefix) for the tail.
    tail = textwrap.dedent(
        """\

        def main() -> None:
            parser = argparse.ArgumentParser()
            group = parser.add_mutually_exclusive_group(required=True)
            group.add_argument("--correctness", action="store_true")
            group.add_argument("--profile", action="store_true")
            group.add_argument("--benchmark", action="store_true")
            group.add_argument("--full-benchmark", action="store_true")
            args = parser.parse_args()
            # Honour GEAK_WORK_DIR when present so the wrapper runs inside the
            # caller's worktree (preflight, agent worktree) rather than always
            # the repo root baked at synthesis time. Falls back to REPO_ROOT
            # when no work-dir override is set.
            cwd = os.environ.get("GEAK_WORK_DIR") or REPO_ROOT

            if args.correctness:
                rc, captured = _run_shell(CORRECTNESS_CMD, cwd)
                sys.stdout.write(captured)
                sys.stdout.flush()
                print("OK" if rc == 0 else "FAIL")
                sys.exit(rc)

            if args.profile or args.benchmark or args.full_benchmark:
                t0 = time.perf_counter()
                rc, captured = _run_shell(PERFORMANCE_CMD, cwd)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0

                # Forward original output for downstream consumers
                # (config matching, shape extraction, etc.)
                sys.stdout.write(captured)
                sys.stdout.flush()

                kernel_ms, timing_source = _extract_kernel_latency_ms(captured)

                if timing_source == "kernel":
                    # Subprocess already emitted GEAK_RESULT_LATENCY_MS
                    pass
                elif kernel_ms is not None:
                    print(f"GEAK_RESULT_LATENCY_MS={kernel_ms:.6f}")
                    print(f"GEAK_RESULT_TIMING_SOURCE={timing_source}")
                else:
                    print(f"GEAK_RESULT_LATENCY_MS={elapsed_ms:.6f}")
                    print("GEAK_RESULT_TIMING_SOURCE=wall_clock")

                if args.full_benchmark:
                    print("GEAK_RESULT_SPEEDUP=1.0")
                sys.exit(rc)

            sys.exit(2)


        if __name__ == "__main__":
            main()
        """
    )
    body = head + "\n" + _PARSER_SOURCE + "\n" + tail
    out.write_text(body, encoding="utf-8")
    try:
        out.chmod(out.stat().st_mode | 0o111)
    except OSError:
        pass
    return out
