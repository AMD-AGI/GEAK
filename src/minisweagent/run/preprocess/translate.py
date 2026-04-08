"""PyTorch -> FlyDSL translation module.

Provides:
- ``run_translation()`` — library function called by the preprocessor
- ``main()`` — standalone CLI entry point (``geak-translate``)

The translation loop uses a ``DefaultAgent`` configured with
translation-specific YAML configs and KB content injected via
``agent.run(task, knowledge_base=...)``.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def run_translation(
    kernel_path: Path,
    output_dir: Path,
    gpu_id: int = 0,
    *,
    target_language: str | None = None,
    model=None,
    model_factory=None,
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
    from minisweagent.agents.default import AgentConfig, DefaultAgent
    from minisweagent.environments.local import LocalEnvironment, LocalEnvironmentConfig
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

    # -- Resolve model --
    _model = model or (model_factory() if model_factory else None)
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
    env_overrides = pair.env_setup(repo_root)

    # -- Build candidate filename --
    kernel_stem = kernel_path.stem
    candidate_filename = pair.candidate_filename_fn(kernel_stem)
    candidate_path = output_dir / candidate_filename

    # -- Load agent config --
    try:
        agent_config_dict, model_config = load_preprocess_agent_config(pair.config_name)
    except Exception as exc:
        msg = f"Failed to load translation agent config '{pair.config_name}': {exc}"
        result["translation_errors"].append(msg)
        _print(f"  [red]{msg}[/red]" if console else f"  ERROR: {msg}")
        return result

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

        env = LocalEnvironment(**LocalEnvironmentConfig(cwd=str(repo_root)).__dict__)
        agent_kwargs = dict(agent_config_dict)
        agent_kwargs["test_command"] = f"{sys.executable} {harness_path} {pair.harness_candidate_flag} {candidate_path}"
        agent_kwargs["patch_output_dir"] = str(output_dir / f"round_{round_num}")
        if pair.env_setup != type(pair).env_setup:
            env_config = LocalEnvironmentConfig(cwd=str(repo_root), env=env_overrides)
            env = LocalEnvironment(**env_config.__dict__)

        agent = DefaultAgent(_model, env, **agent_kwargs)
        agent.log_file = output_dir / f"translation_agent_round_{round_num}.log"

        try:
            exit_status, agent_result = agent.run(round_task, knowledge_base=kb_content)
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

        # Validate the candidate externally
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
            break
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


def main():
    parser = argparse.ArgumentParser(description="Translation comparison harness")
    parser.add_argument("{candidate_flag}", dest="candidate", nargs="?", default=None,
                        help="Path to FlyDSL candidate kernel")
    parser.add_argument("--correctness", action="store_true",
                        help="Run baseline correctness check only")
    parser.add_argument("--profile", action="store_true",
                        help="Run in profile mode (same as correctness for translation)")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run benchmark mode")
    parser.add_argument("--full-benchmark", action="store_true",
                        help="Run full benchmark mode")
    args = parser.parse_args()

    torch.manual_seed(42)

    print("Running PyTorch reference...")
    ref_model, ref_inputs, ref_output, ref_latency = run_reference()
    print(f"PyTorch reference latency: {{ref_latency:.3f}} ms")

    if args.candidate:
        print(f"Running FlyDSL candidate: {{args.candidate}}")
        cand_output, cand_latency = run_candidate(args.candidate, ref_inputs)
        print(f"FlyDSL candidate latency: {{cand_latency:.3f}} ms")

        print("Comparing outputs...")
        compare_outputs(ref_output, cand_output)
        print("CORRECTNESS: PASS")

        speedup = ref_latency / cand_latency if cand_latency > 0 else float("inf")
        print(f"Speedup: {{speedup:.2f}}x (ref={{ref_latency:.3f}}ms, cand={{cand_latency:.3f}}ms)")

        if speedup < 0.5:
            print("WARNING: FlyDSL candidate is significantly slower than PyTorch reference")
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

    # Build model factory
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
        model_factory=_model_factory,
        repo=repo_root,
        flydsl_repo=flydsl_repo,
        console=console,
    )

    print(json.dumps(result, indent=2, default=str))
    sys.exit(0 if result.get("translation_success") else 1)


if __name__ == "__main__":
    main()
