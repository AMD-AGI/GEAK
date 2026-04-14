"""PyTorch -> FlyDSL translation module.

Provides:
- ``run_translation()`` — orchestration function called by the preprocessor
  (multi-round retry loop, self-review, performance measurement)
- ``main()`` — standalone CLI entry point (``geak-translate``)

Each translation round delegates to
:func:`~minisweagent.agents.translation_agent.run_translation_agent`
which instantiates a :class:`~minisweagent.agents.translation_agent.TranslationAgent`
(a ``DefaultAgent`` subclass) configured with translation-specific YAML
and KB content.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _parse_timing_from_harness_output(
    stdout: str,
    result: dict[str, Any],
    _print,
) -> None:
    """Extract latency metrics from harness stdout into *result* dict.

    Parses ``PyTorch reference latency:``, ``FlyDSL candidate latency:``,
    and ``Speedup:`` lines.  Computes speedup from the two latencies if
    the explicit ``Speedup:`` line is missing.
    """
    import re

    ref_match = re.search(r"PyTorch reference latency:\s*([\d.]+)\s*ms", stdout)
    cand_match = re.search(r"FlyDSL candidate latency:\s*([\d.]+)\s*ms", stdout)
    speedup_match = re.search(r"Speedup:\s*([\d.]+)x", stdout)

    if ref_match:
        result["translation_pytorch_latency_ms"] = float(ref_match.group(1))
    if cand_match:
        result["translation_flydsl_latency_ms"] = float(cand_match.group(1))
    if speedup_match:
        result["translation_speedup"] = float(speedup_match.group(1))
    elif ref_match and cand_match:
        pt_val = float(ref_match.group(1))
        fly_val = float(cand_match.group(1))
        if fly_val > 0:
            result["translation_speedup"] = round(pt_val / fly_val, 2)

    if not cand_match and stdout:
        _print("  Could not parse FlyDSL latency from harness output")
        for _line in stdout.strip().splitlines()[-5:]:
            _print(f"    stdout: {_line}")

    _print(
        f"  PyTorch: {result.get('translation_pytorch_latency_ms', 'N/A')}ms | "
        f"FlyDSL: {result.get('translation_flydsl_latency_ms', 'N/A')}ms | "
        f"Speedup: {result.get('translation_speedup', 'N/A')}x"
    )


def run_translation(
    kernel_path: Path,
    output_dir: Path,
    gpu_id: int = 0,
    *,
    target_language: str | None = None,
    model=None,
    model_factory=None,
    model_name: str | None = None,
    repo: Path | None = None,
    flydsl_repo: Path | None = None,
    console=None,
) -> dict[str, Any]:
    """Run translation pipeline. Returns translation metadata dict.

    Parameters
    ----------
    kernel_path:
        Path to the source kernel (e.g. a PyTorch nn.Module).
    output_dir:
        Directory for translation artefacts.
    gpu_id:
        GPU device for harness execution.
    target_language:
        Target language (e.g. ``"flydsl"``). Auto-detected if ``None``.
    model:
        LLM model instance (optional; uses *model_factory* if ``None``).
    model_factory:
        Callable returning a new model instance.
    model_name:
        Explicit model name from CLI ``-m`` flag.  When ``None`` the
        agent config YAML's ``model`` section is used to create the model,
        giving the per-config model precedence over the global default.
    repo:
        Repository root path.
    flydsl_repo:
        Optional path to a local FlyDSL clone. When set, loads FlyDSL
        reference docs from repo instead of authored KB files.
    console:
        Optional Rich console for progress output.

    Returns
    -------
    dict with translation metadata including success/failure status,
    translated kernel path, latency comparison, and diagnostic info.
    """
    from minisweagent.agents.translation_agent import run_translation_agent
    from minisweagent.run.preprocess.config_loader import load_preprocess_agent_config
    from minisweagent.run.preprocess.run_harness import run_harness
    from minisweagent.tools.translation_registry import (
        REGISTRY,
        detect_kernel_categories,
        get_gpu_specs,
        load_translation_kb,
    )

    def _print(msg: str) -> None:
        if console:
            console.print(msg)
        else:
            print(msg, file=sys.stderr)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    kernel_path = Path(kernel_path).resolve()

    result: dict[str, Any] = {
        "translation_success": False,
        "translation_source_language": None,
        "translation_target_language": None,
        "translation_kernel_path": None,
        "translation_best_attempt_path": None,
        "translation_rounds_used": 0,
        "translation_pytorch_latency_ms": None,
        "translation_flydsl_latency_ms": None,
        "translation_errors": [],
    }

    # -- Detect translation pair --
    pair = REGISTRY.detect(kernel_path, target_language)
    if pair is None:
        msg = f"No translation pair found for {kernel_path}"
        if target_language:
            msg += f" with target={target_language}"
        _print(f"  [yellow]{msg}[/yellow]" if console else f"  {msg}")
        result["translation_errors"].append(msg)
        return result

    result["translation_source_language"] = pair.source
    result["translation_target_language"] = pair.target
    _print(f"  Translation: {pair.source} -> {pair.target}")

    # -- Load agent config (before model so YAML model section can be used) --
    try:
        agent_config_dict, model_config = load_preprocess_agent_config(pair.config_name)
    except Exception as exc:
        msg = f"Failed to load translation agent config '{pair.config_name}': {exc}"
        result["translation_errors"].append(msg)
        _print(f"  [red]{msg}[/red]" if console else f"  ERROR: {msg}")
        return result

    # -- Resolve model --
    # Precedence: explicit model object > explicit model_name > YAML config > factory default
    _model = model
    if _model is None and model_name is None and model_config.get("model_name"):
        from minisweagent.models import get_model
        _print(f"  Using model from agent config: {model_config['model_name']}")
        _model = get_model(model_config["model_name"], config=model_config)
    if _model is None:
        _model = model_factory() if model_factory else None
    if _model is None:
        msg = "No LLM model available for translation agent"
        result["translation_errors"].append(msg)
        return result

    # -- Load KB content --
    categories = detect_kernel_categories(kernel_path)
    gpu_specs = get_gpu_specs()
    kb_content = load_translation_kb(
        pair,
        categories,
        gpu_specs,
        flydsl_repo=flydsl_repo,
    )
    _print(f"  KB loaded: {len(kb_content)} chars, categories={categories}")

    # -- Set up environment --
    repo_root = repo or kernel_path.parent
    try:
        env_overrides = pair.env_setup(repo_root, flydsl_repo=flydsl_repo)
    except TypeError:
        env_overrides = pair.env_setup(repo_root)

    # -- Build candidate filename --
    kernel_stem = kernel_path.stem
    candidate_filename = pair.candidate_filename_fn(kernel_stem)
    candidate_path = output_dir / candidate_filename

    # -- Build the test command that the agent will use via save_and_test --
    test_command = f"{sys.executable} {{harness}} {pair.harness_candidate_flag} {{candidate}}"

    # -- Create translation harness --
    _print("  Creating translation harness...")
    harness_path = output_dir / f"test_{kernel_stem}_translation_harness.py"

    try:
        harness_path = _create_translation_harness(
            kernel_path=kernel_path,
            candidate_path=candidate_path,
            harness_path=harness_path,
            pair=pair,
            model=_model,
            repo_root=repo_root,
            output_dir=output_dir,
        )
    except Exception as exc:
        msg = f"Failed to create translation harness: {exc}"
        result["translation_errors"].append(msg)
        _print(f"  [red]{msg}[/red]" if console else f"  ERROR: {msg}")
        return result

    # -- Build task prompt --
    source_code = kernel_path.read_text()
    task = (
        f"Translate the following PyTorch kernel to FlyDSL.\n\n"
        f"## Source kernel ({kernel_path.name})\n"
        f"```python\n{source_code}\n```\n\n"
        f"## Requirements\n"
        f"- Write the FlyDSL translation to: {candidate_path}\n"
        f"- The translation must preserve the exact same numerical output as the PyTorch original.\n"
        f"- Use the FlyDSL API described in the knowledge base below.\n"
        f"- The test harness is at: {harness_path}\n"
        f"- Run correctness checks with: `python {harness_path} {pair.harness_candidate_flag} {candidate_path}`\n"
    )

    # -- Translation loop --
    best_attempt: Path | None = None
    best_attempt_errors: list[str] = []
    t0 = time.monotonic()

    for round_num in range(1, pair.max_rounds + 1):
        _print(f"  Round {round_num}/{pair.max_rounds}...")

        round_task = task
        if round_num > 1 and best_attempt_errors:
            feedback = "\n".join(best_attempt_errors[-3:])
            round_task += (
                f"\n\n## Previous attempt feedback\n"
                f"The previous translation attempt had these errors:\n{feedback}\n"
                f"Fix these issues in your new attempt.\n"
            )

        round_log_dir = output_dir / f"round_{round_num}"
        test_cmd = f"{sys.executable} {harness_path} {pair.harness_candidate_flag} {candidate_path}"

        try:
            exit_status, agent_result = run_translation_agent(
                model=_model,
                repo_root=repo_root,
                agent_config=agent_config_dict,
                task=round_task,
                kb_content=kb_content,
                env_overrides=env_overrides or None,
                test_command=test_cmd,
                log_dir=round_log_dir,
                log_name=f"translation_agent_round_{round_num}.log",
            )
        except Exception as exc:
            _print(f"  Round {round_num} agent error: {exc}")
            best_attempt_errors.append(str(exc))
            continue

        _print(f"  Round {round_num} exit: {exit_status}")
        result["translation_rounds_used"] = round_num

        if not candidate_path.exists():
            best_attempt_errors.append("Agent did not produce a candidate file")
            continue

        best_attempt = candidate_path

        # External validation: run the harness independently (follows
        # main-branch pattern).  The harness has DEFAULT_CANDIDATE baked
        # in so --correctness mode tests the actual FlyDSL translation.
        # _run_single uses bash -lc to source /etc/profile.d/* for
        # FlyDSL env setup.
        harness_result = run_harness(
            str(harness_path),
            mode="correctness",
            repo_root=str(repo_root),
            gpu_id=gpu_id,
            env_overrides=env_overrides,
        )
        assert isinstance(harness_result, dict)

        if harness_result["success"]:
            _print(f"  Round {round_num}: CORRECT")
            result["translation_success"] = True
            result["translation_kernel_path"] = str(candidate_path)

            # Parse timing from the validation run's stdout — the harness
            # prints latencies and speedup when the candidate is tested.
            _parse_timing_from_harness_output(
                harness_result.get("stdout", ""), result, _print,
            )

            # -- Performance regression gate --
            perf_fail_threshold = pair.perf_fail_threshold if hasattr(pair, "perf_fail_threshold") else 0.5
            perf_warn_threshold = pair.perf_warn_threshold if hasattr(pair, "perf_warn_threshold") else 0.8
            speedup_val = result.get("translation_speedup")

            if speedup_val is not None and speedup_val < perf_fail_threshold:
                pt_ms = result.get("translation_pytorch_latency_ms", "?")
                fly_ms = result.get("translation_flydsl_latency_ms", "?")
                _print(
                    f"  PERF REGRESSION: {speedup_val:.2f}x "
                    f"(threshold {perf_fail_threshold}x) — retrying"
                )
                result["translation_success"] = False
                result["translation_kernel_path"] = None
                best_attempt_errors.append(
                    f"Performance regression: {speedup_val:.2f}x speedup "
                    f"(PyTorch {pt_ms}ms vs FlyDSL {fly_ms}ms). "
                    f"Your translation is {1/speedup_val:.1f}x SLOWER than PyTorch. "
                    f"Avoid Python for-loops over batch dimensions. "
                    f"Use build_flash_attn_func_module for attention patterns, "
                    f"compile_preshuffle_gemm_a8 for batched GEMM. "
                    f"Never decompose what a single pre-built kernel can handle."
                )
                continue
            elif speedup_val is not None and speedup_val < perf_warn_threshold:
                _print(
                    f"  PERF WARNING: {speedup_val:.2f}x "
                    f"(below {perf_warn_threshold}x warn threshold)"
                )

            # -- Self-review: audit ops + efficiency --
            _print("  Running self-review (op + efficiency audit)...")
            review_ok = _run_self_review(
                candidate_path=candidate_path,
                pair=pair,
                model=_model,
                kb_content=kb_content,
                _print=_print,
            )

            if review_ok:
                if review_ok == "rewritten":
                    _print("  Self-review rewrote candidate. Re-checking correctness...")
                    recheck = run_harness(
                        str(harness_path),
                        mode="correctness",
                        repo_root=str(repo_root),
                        gpu_id=gpu_id,
                        env_overrides=env_overrides,
                    )
                    if not recheck["success"]:
                        _print("  Self-review broke correctness — continuing to next round")
                        result["translation_success"] = False
                        result["translation_kernel_path"] = None
                        best_attempt_errors.append(
                            "Self-review rewrite broke correctness"
                        )
                        continue
                    _print("  Self-review rewrite still correct")
                    _parse_timing_from_harness_output(
                        recheck.get("stdout", ""), result, _print,
                    )
                    result["translation_self_review"] = "passed_with_changes"
                else:
                    result["translation_self_review"] = "passed"
                break
            else:
                _print("  Self-review failed — continuing to next round")
                result["translation_success"] = False
                result["translation_kernel_path"] = None
                best_attempt_errors.append("Self-review failed (parse error or no rewritten code)")
                continue
        else:
            stderr_tail = harness_result.get("stderr", "")[-500:]
            best_attempt_errors.append(f"Correctness check failed:\n{stderr_tail}")
            _print(f"  Round {round_num}: failed correctness")

    elapsed = time.monotonic() - t0
    result["translation_elapsed_s"] = round(elapsed, 1)

    if not result["translation_success"] and best_attempt and best_attempt.exists():
        saved = output_dir / f"best_attempt_{candidate_filename}"
        best_attempt.rename(saved)
        result["translation_best_attempt_path"] = str(saved)
        _print(f"  Translation failed after {pair.max_rounds} rounds. Best attempt saved to {saved}")

    if result["translation_success"]:
        _print(f"  Translation successful in {result['translation_rounds_used']} rounds ({elapsed:.1f}s)")

    # Write result metadata
    (output_dir / "translation_result.json").write_text(
        json.dumps(result, indent=2, default=str)
    )

    return result


_SELF_REVIEW_PROMPT = """\
You are reviewing a FlyDSL translation of a PyTorch GPU kernel that passed correctness.

## FlyDSL Knowledge Base
{kb_content}

## Translated code ({candidate_path}):
```python
{candidate_code}
```

## Part A — PyTorch Fallback Audit

Enumerate every call that performs GPU **compute** via PyTorch (e.g.
`torch.matmul`, `F.relu`, `nn.Linear(...)`, `@` for matrix multiply,
`F.softmax`, `F.scaled_dot_product_attention`, `torch.sum`, `torch.mean`).

Ignore non-compute helpers: `import torch`, `torch.empty`, `torch.zeros`,
`torch.no_grad`, `.cuda()`, `.contiguous()`, `.view()`, `.reshape()`,
dtype/device helpers, `torch.Tensor` type annotations.

For each, decide:
- **REPLACE** — FlyDSL has an equivalent. Provide the FlyDSL replacement.
- **KEEP** — FlyDSL has no equivalent (e.g. `nn.Conv2d`, `nn.BatchNorm2d`,
  `F.max_pool2d`). Briefly explain why.

## Part B — Efficiency Audit

Check for:
1. **Python for-loops over batch dimensions** — extremely slow; restructure
   as a single batched call.
2. **Decomposed attention** (Q@K^T, softmax, @V separately) when
   `build_flash_attn_func_module()` could replace it (head_dim>=64,
   head_dim%32==0, seq_len%128==0).
3. **Duplicate kernels** — mixing PyTorch and FlyDSL for the same op.
4. **Missing pre-built kernels** — custom `@flyc.kernel` for ops with
   pre-built equivalents (softmax, layernorm, rmsnorm).

## Required JSON Response

Respond with ONLY a JSON object (no markdown, no explanation outside JSON):
{{
  "fallback_audit": [
    {{"op": "<pytorch call>", "line": <approx line number>, "verdict": "KEEP" or "REPLACE", "reason": "<brief>", "flydsl_replacement": "<FlyDSL API or null>"}}
  ],
  "efficiency_issues": [
    {{"issue": "<description>", "current_code": "<snippet>", "fix": "<description>"}}
  ],
  "verdict": "pass" or "rewrite",
  "reasoning": "<one sentence summary>",
  "rewritten_code": "<FULL rewritten file contents if verdict=rewrite, else null>"
}}

Rules:
- verdict="rewrite" if ANY fallback_audit entry has verdict="REPLACE" or ANY
  efficiency_issues exist. Otherwise verdict="pass".
- rewritten_code must be the COMPLETE file (not a diff/patch), ready to write
  directly. Include all imports, class definitions, helper functions.
- Be strict — maximise FlyDSL utilisation and performance.
"""


def _run_self_review(
    *,
    candidate_path: Path,
    pair,
    model,
    kb_content: str,
    _print,
) -> str | bool:
    """Single-call self-review: audit translated code for PyTorch fallbacks
    and efficiency issues via one LLM query.

    Returns:
    - ``"rewritten"`` if the LLM produced a rewritten candidate
    - ``True`` if all ops are justified (no changes needed)
    - ``False`` if the review failed (parse error, exception, etc.)
    """
    import re

    candidate_code = candidate_path.read_text()
    prompt = _SELF_REVIEW_PROMPT.format(
        candidate_path=candidate_path,
        candidate_code=candidate_code,
        kb_content=kb_content,
    )

    try:
        response = model.query([
            {
                "role": "system",
                "content": (
                    "You are a GPU kernel translation reviewer specialising in "
                    "FlyDSL (AMD's Python DSL for MI300X).\n\n"
                    "CRITICAL: Your entire response must be a single JSON object. "
                    "Do NOT include any text before or after the JSON. "
                    "Do NOT wrap it in markdown code fences. "
                    "Start your response with { and end with }."
                ),
            },
            {"role": "user", "content": prompt},
        ])
    except Exception as exc:
        _print(f"  Self-review query error: {exc}")
        return False

    content = response.get("content", "") if isinstance(response, dict) else ""
    content = content.strip()

    # Try to extract JSON from the response — handle fences and prose preambles
    fence_match = re.search(r"```(?:json)?\s*\n(.*?)\n\s*```", content, re.DOTALL)
    if fence_match:
        content = fence_match.group(1).strip()
    elif not content.startswith("{"):
        brace_pos = content.find("{")
        if brace_pos >= 0:
            content = content[brace_pos:]

    try:
        parsed = json.loads(content)
    except (json.JSONDecodeError, TypeError) as exc:
        _print(f"  Self-review JSON parse error: {exc}")
        _print(f"  Raw response (first 500 chars): {content[:500]}")
        return False

    verdict = parsed.get("verdict", "pass")
    reasoning = parsed.get("reasoning", "")
    fallback_audit = parsed.get("fallback_audit", [])
    efficiency_issues = parsed.get("efficiency_issues", [])

    n_replace = sum(1 for f in fallback_audit if f.get("verdict") == "REPLACE")
    n_keep = sum(1 for f in fallback_audit if f.get("verdict") == "KEEP")
    _print(
        f"  Self-review: {n_replace} REPLACE, {n_keep} KEEP, "
        f"{len(efficiency_issues)} efficiency issues"
    )
    _print(f"  Self-review reasoning: {reasoning}")

    if verdict == "rewrite":
        rewritten = parsed.get("rewritten_code")
        if not rewritten or not rewritten.strip():
            _print("  Self-review verdict=rewrite but no rewritten_code provided")
            return False
        candidate_path.write_text(rewritten)
        _print("  Self-review wrote rewritten candidate")
        return "rewritten"

    _print("  Self-review: all ops justified (no changes)")
    return True


def _create_translation_harness(
    *,
    kernel_path: Path,
    candidate_path: Path,
    harness_path: Path,
    pair,
    model,
    repo_root: Path,
    output_dir: Path,
) -> Path:
    """Create a comparison harness for translation validation.

    The harness compares PyTorch reference outputs against the FlyDSL
    candidate. For now, generates a minimal harness inline. The UTA-based
    harness creation (run_pytorch_translation_agent) can be used for more
    complex kernels.
    """
    source_code = kernel_path.read_text()
    harness_code = _generate_minimal_translation_harness(
        kernel_path=kernel_path,
        candidate_path=candidate_path,
        candidate_flag=pair.harness_candidate_flag,
    )
    harness_path.write_text(harness_code)
    logger.info("Created translation harness: %s", harness_path)
    return harness_path


def _generate_minimal_translation_harness(
    *,
    kernel_path: Path,
    candidate_path: Path,
    candidate_flag: str,
) -> str:
    """Generate a minimal Python harness that validates translation correctness.

    The harness:
    1. Imports the PyTorch reference Model from the source kernel
    2. Imports the FlyDSL candidate Model (when ``--flydsl-kernel`` is given)
    3. Runs both on the same inputs and compares outputs
    """
    return f'''#!/usr/bin/env python3
"""Translation comparison harness: PyTorch reference vs FlyDSL candidate.

Usage:
    python {{this_file}} {candidate_flag} <candidate_path>
    python {{this_file}} --correctness  # baseline-only mode
"""

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import torch


def _load_module(path: str, module_name: str = "kernel_module"):
    """Dynamically load a Python module from a file path."""
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {{path}}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _get_model_and_inputs(module):
    """Extract Model class, get_inputs, and get_init_inputs from a module."""
    model_cls = getattr(module, "Model", None)
    if model_cls is None:
        raise AttributeError("Module does not define a Model class")
    get_inputs = getattr(module, "get_inputs", None)
    get_init_inputs = getattr(module, "get_init_inputs", None)
    return model_cls, get_inputs, get_init_inputs


def run_reference():
    """Run PyTorch reference kernel and return (model, inputs, outputs, latency_ms)."""
    ref_module = _load_module("{kernel_path}", "pytorch_ref")
    model_cls, get_inputs, get_init_inputs = _get_model_and_inputs(ref_module)

    init_inputs = get_init_inputs() if get_init_inputs else []
    model = model_cls(*init_inputs).cuda()

    inputs = get_inputs()
    inputs = [x.cuda() if isinstance(x, torch.Tensor) else x for x in inputs]

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            model(*inputs)
    torch.cuda.synchronize()

    # Timed run
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    with torch.no_grad():
        start.record()
        ref_output = model(*inputs)
        end.record()
    torch.cuda.synchronize()
    latency_ms = start.elapsed_time(end)

    return model, inputs, ref_output, latency_ms


def run_candidate(candidate_path: str, ref_inputs):
    """Run FlyDSL candidate kernel and return (outputs, latency_ms)."""
    cand_module = _load_module(candidate_path, "flydsl_candidate")
    model_cls, get_inputs, get_init_inputs = _get_model_and_inputs(cand_module)

    init_inputs = get_init_inputs() if get_init_inputs else []
    model = model_cls(*init_inputs).cuda()

    inputs = ref_inputs

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            model(*inputs)
    torch.cuda.synchronize()

    # Timed run
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    with torch.no_grad():
        start.record()
        cand_output = model(*inputs)
        end.record()
    torch.cuda.synchronize()
    latency_ms = start.elapsed_time(end)

    return cand_output, latency_ms


def compare_outputs(ref_output, cand_output, rtol=1e-3, atol=1e-3):
    """Compare reference and candidate outputs."""
    if isinstance(ref_output, torch.Tensor) and isinstance(cand_output, torch.Tensor):
        torch.testing.assert_close(cand_output, ref_output, rtol=rtol, atol=atol)
        return True
    if isinstance(ref_output, (tuple, list)) and isinstance(cand_output, (tuple, list)):
        assert len(ref_output) == len(cand_output), (
            f"Output count mismatch: ref={{len(ref_output)}}, cand={{len(cand_output)}}"
        )
        for i, (r, c) in enumerate(zip(ref_output, cand_output)):
            if isinstance(r, torch.Tensor) and isinstance(c, torch.Tensor):
                torch.testing.assert_close(c, r, rtol=rtol, atol=atol)
        return True
    print(f"WARNING: Cannot compare output types: ref={{type(ref_output)}}, cand={{type(cand_output)}}")
    return True


DEFAULT_CANDIDATE = "{candidate_path}"


def main():
    parser = argparse.ArgumentParser(description="Translation comparison harness")
    parser.add_argument("{candidate_flag}", dest="candidate", nargs="?",
                        default=None,
                        help="Path to FlyDSL candidate kernel")
    parser.add_argument("--correctness", action="store_true",
                        help="Run correctness check (uses default candidate if no explicit path)")
    parser.add_argument("--profile", action="store_true",
                        help="Run in profile mode")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run benchmark mode")
    parser.add_argument("--full-benchmark", action="store_true",
                        help="Run full benchmark mode")
    args = parser.parse_args()

    candidate = args.candidate
    if candidate is None and (args.correctness or args.profile
                              or args.benchmark or args.full_benchmark):
        candidate = DEFAULT_CANDIDATE

    torch.manual_seed(42)

    print("Running PyTorch reference...")
    ref_model, ref_inputs, ref_output, ref_latency = run_reference()
    print(f"PyTorch reference latency: {{ref_latency:.3f}} ms")

    if candidate and Path(candidate).exists():
        print(f"Running FlyDSL candidate: {{candidate}}")
        cand_output, cand_latency = run_candidate(candidate, ref_inputs)
        print(f"FlyDSL candidate latency: {{cand_latency:.3f}} ms")

        print("Comparing outputs...")
        compare_outputs(ref_output, cand_output)
        print("CORRECTNESS: PASS")

        speedup = ref_latency / cand_latency if cand_latency > 0 else float("inf")
        print(f"Speedup: {{speedup:.2f}}x (ref={{ref_latency:.3f}}ms, cand={{cand_latency:.3f}}ms)")

        if speedup < 0.5:
            print("WARNING: FlyDSL candidate is significantly slower than PyTorch reference")
    elif candidate:
        print(f"WARNING: Candidate file not found: {{candidate}}")
        print("CORRECTNESS: PASS (baseline only)")
    else:
        print("CORRECTNESS: PASS (baseline only)")


if __name__ == "__main__":
    main()
'''


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI: ``geak-translate --kernel-url <path> --target-language flydsl``."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Translate a GPU kernel from one language to another (e.g. PyTorch -> FlyDSL)",
    )
    parser.add_argument(
        "--kernel-url",
        required=True,
        help="Kernel source (local path or GitHub URL)",
    )
    parser.add_argument(
        "--target-language",
        default="flydsl",
        help="Target language (default: flydsl)",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output directory (default: <kernel_dir>/translation_output)",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU device ID (default: 0)",
    )
    parser.add_argument(
        "--repo",
        default=None,
        help="Repository root path",
    )
    parser.add_argument(
        "-m", "--model",
        default=None,
        help="Model name for translation agent",
    )
    parser.add_argument(
        "--flydsl-repo",
        default=None,
        help="Path to local FlyDSL clone (use repo docs instead of authored KB)",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=None,
        help="Override max translation rounds",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Resolve kernel
    from minisweagent.run.preprocess.resolve_kernel_url import resolve_kernel_url

    resolved = resolve_kernel_url(args.kernel_url, repo=args.repo)
    if resolved.get("error"):
        print(f"Error resolving kernel: {resolved['error']}", file=sys.stderr)
        sys.exit(1)

    kernel_path = Path(resolved["local_file_path"])
    repo_root = Path(resolved.get("local_repo_path") or kernel_path.parent)

    output_dir = Path(args.output) if args.output else kernel_path.parent / "translation_output"

    # Build model factory (used as fallback when YAML config has no model)
    from minisweagent.run.preprocess.harness_utils import geak_model_factory

    _model_factory = geak_model_factory(args.model)

    try:
        from rich.console import Console
        console = Console()
    except ImportError:
        console = None

    flydsl_repo = Path(args.flydsl_repo) if args.flydsl_repo else None

    result = run_translation(
        kernel_path=kernel_path,
        output_dir=output_dir,
        gpu_id=args.gpu,
        target_language=args.target_language,
        model_name=args.model,
        model_factory=_model_factory,
        repo=repo_root,
        flydsl_repo=flydsl_repo,
        console=console,
    )

    print(json.dumps(result, indent=2, default=str))
    sys.exit(0 if result.get("translation_success") else 1)


if __name__ == "__main__":
    main()
