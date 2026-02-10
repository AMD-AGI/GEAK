# Copyright(C)[2025] Advanced Micro Devices, Inc. All rights reserved.

"""
COMMANDMENT-based deterministic evaluator for OpenEvolve.

COMMANDMENT.md is the UNIVERSAL CONTRACT between the orchestrator (agent)
and OpenEvolve.  It specifies exact shell commands for:
  - SETUP: environment preparation, directory creation, GPU warmup
  - CORRECTNESS: any verification commands (exit 0 = pass, non-zero = fail)
  - PROFILE: Metrix / rocprofv3 commands for hardware-level profiling

This evaluator does NOT know or care what language the kernel is written in
(Triton, CK, HIP, ASM, etc.), how many files it spans, or how correctness
is checked.  It just executes the commands in COMMANDMENT.md and parses the
profiling output for metrics.

The COMMANDMENT.md is created by the caller (mini-SWE-agent, run.sh, etc.)
BEFORE OpenEvolve starts, and is FROZEN for the entire evolution.  The
agent is responsible for:
  1. Creating correctness checking scripts tailored to the specific kernel
  2. Creating profiling/benchmarking commands (using Metrix / kernel-profile)
  3. Validating that these commands work on the baseline kernel
  4. Writing the validated commands to COMMANDMENT.md

The CommandmentGenerator is a convenience helper for programmatically
writing COMMANDMENT.md files -- the agent can also write them directly.
"""

import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Known benign warnings that should NOT cause a command to be treated as
# a failure.  These are regex patterns matched against stderr.
# ---------------------------------------------------------------------------
_BENIGN_STDERR_PATTERNS = [
    # ROCm/HIP does not support expandable_segments; PyTorch emits a
    # UserWarning but the kernel still runs correctly.
    r"(?i)expandable.?segments",
    # Triton cache / compilation info messages
    r"(?i)triton.*compilation",
    # General Python UserWarnings (non-fatal)
    r"UserWarning:",
    # hipBLAS/rocBLAS informational messages
    r"(?i)rocblas.*info",
    # torch.cuda setup messages
    r"(?i)setting.*cuda.*device",
]

import re as _re
_BENIGN_RE = _re.compile("|".join(_BENIGN_STDERR_PATTERNS))


def _filter_benign_stderr(stderr: str) -> str:
    """
    Remove lines matching known benign warning patterns from stderr.

    Returns the filtered stderr string.  If all lines are benign, returns
    an empty string -- the caller can then avoid treating the command as
    failed just because stderr was non-empty.
    """
    filtered = []
    for line in stderr.splitlines():
        if not _BENIGN_RE.search(line):
            filtered.append(line)
    return "\n".join(filtered)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CommandmentSection:
    """A parsed section from COMMANDMENT.md."""
    name: str  # SETUP, CORRECTNESS, or PROFILE
    commands: List[str] = field(default_factory=list)
    description: str = ""


@dataclass
class EvaluationResult:
    """Result of evaluating a candidate program."""
    success: bool
    correctness_passed: bool = False
    metrics: Dict[str, Any] = field(default_factory=dict)
    speedup: float = 1.0
    stdout: str = ""
    stderr: str = ""
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# CommandmentEvaluator
# ---------------------------------------------------------------------------

class CommandmentEvaluator:
    """
    Evaluates candidate programs using commands from a COMMANDMENT.md file.

    Pipeline:
      1. Write candidate program files to a working directory
      2. Execute SETUP commands
      3. Execute CORRECTNESS commands (must all pass)
      4. Execute PROFILE commands (parse metrics from output)
      5. Compare against baseline metrics to compute speedup
    """

    def __init__(
        self,
        commandment_path: str,
        baseline_metrics_path: Optional[str] = None,
        kernel_dir: Optional[str] = None,
        timeout: int = 300,
    ):
        """
        Args:
            commandment_path: Path to COMMANDMENT.md
            baseline_metrics_path: Path to baseline metrics.json (for speedup)
            kernel_dir: Original kernel directory (for reference files)
            timeout: Timeout per command in seconds
        """
        # Resolve from env vars if not provided
        self.commandment_path = commandment_path or os.environ.get(
            "GEAK_COMMANDMENT_PATH", ""
        )
        self.baseline_metrics_path = baseline_metrics_path or os.environ.get(
            "GEAK_BASELINE_METRICS", ""
        )
        self.kernel_dir = kernel_dir or os.environ.get("GEAK_KERNEL_DIR", "")
        self.timeout = timeout

        # Parse the commandment file
        self.sections: Dict[str, CommandmentSection] = {}
        if self.commandment_path and os.path.isfile(self.commandment_path):
            self.sections = self._parse_commandment(self.commandment_path)
            logger.info(
                f"Loaded COMMANDMENT with sections: {list(self.sections.keys())}"
            )
        else:
            logger.warning(
                f"COMMANDMENT file not found: {self.commandment_path}"
            )

        # Load baseline metrics
        self.baseline_metrics: Dict[str, Any] = {}
        if self.baseline_metrics_path and os.path.isfile(self.baseline_metrics_path):
            with open(self.baseline_metrics_path, "r") as f:
                self.baseline_metrics = json.load(f)
            logger.info(f"Loaded baseline metrics from {self.baseline_metrics_path}")

    def evaluate(
        self,
        program_files: Dict[str, str],
        program_id: str,
        work_dir: Optional[str] = None,
        gpu_id: Optional[int] = None,
    ) -> EvaluationResult:
        """
        Evaluate a candidate program using COMMANDMENT commands.

        Args:
            program_files: Dict of relative_path -> content
            program_id: Unique identifier for this evaluation
            work_dir: Optional working directory (temp dir created if None)
            gpu_id: GPU device ID for this evaluation.  When provided,
                    GEAK_GPU_DEVICE and HIP_VISIBLE_DEVICES are set to this
                    value in every subprocess, ensuring exclusive GPU access.
                    COMMANDMENT.md commands should use ${GEAK_GPU_DEVICE}
                    instead of hardcoded GPU IDs.

        Returns:
            EvaluationResult with success/failure, metrics, and speedup.
        """
        # Create working directory
        cleanup = False
        if work_dir is None:
            work_dir = tempfile.mkdtemp(prefix=f"oe_eval_{program_id[:8]}_")
            cleanup = True

        try:
            # Step 1: Write candidate files
            self._write_program_files(work_dir, program_files)

            # Step 2: SETUP
            if "SETUP" in self.sections:
                ok, stdout, stderr = self._run_section(
                    "SETUP", self.sections["SETUP"], work_dir, gpu_id=gpu_id
                )
                if not ok:
                    return EvaluationResult(
                        success=False,
                        error=f"SETUP failed: {stderr[-500:]}",
                        stdout=stdout,
                        stderr=stderr,
                    )

            # Step 3: CORRECTNESS
            correctness_passed = True
            correctness_stdout = ""
            correctness_stderr = ""
            if "CORRECTNESS" in self.sections:
                ok, stdout, stderr = self._run_section(
                    "CORRECTNESS", self.sections["CORRECTNESS"], work_dir,
                    gpu_id=gpu_id,
                )
                correctness_passed = ok
                correctness_stdout = stdout
                correctness_stderr = stderr
                if not ok:
                    return EvaluationResult(
                        success=False,
                        correctness_passed=False,
                        error=f"CORRECTNESS failed: {stderr[-500:]}",
                        stdout=stdout,
                        stderr=stderr,
                    )

            # Step 4: PROFILE
            metrics: Dict[str, Any] = {}
            profile_stdout = ""
            profile_stderr = ""
            if "PROFILE" in self.sections:
                ok, stdout, stderr = self._run_section(
                    "PROFILE", self.sections["PROFILE"], work_dir,
                    gpu_id=gpu_id,
                )
                profile_stdout = stdout
                profile_stderr = stderr
                if ok:
                    metrics = self._parse_profiling_output(stdout + "\n" + stderr)
                else:
                    logger.warning(f"PROFILE section failed for {program_id}")
                    # Still continue - profiling failure is not fatal
                    metrics = {}

            # Step 5: Calculate speedup
            speedup = self._calculate_speedup(metrics)
            metrics["speedup"] = speedup

            return EvaluationResult(
                success=True,
                correctness_passed=correctness_passed,
                metrics=metrics,
                speedup=speedup,
                stdout=correctness_stdout + "\n" + profile_stdout,
                stderr=correctness_stderr + "\n" + profile_stderr,
            )

        except Exception as e:
            logger.exception(f"Evaluation failed for {program_id}: {e}")
            return EvaluationResult(
                success=False,
                error=str(e),
            )
        finally:
            if cleanup and os.path.isdir(work_dir):
                try:
                    shutil.rmtree(work_dir)
                except Exception:
                    pass

    def _write_program_files(
        self, work_dir: str, program_files: Dict[str, str]
    ) -> None:
        """Write candidate program files to the working directory."""
        for rel_path, content in program_files.items():
            file_path = os.path.join(work_dir, rel_path)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "w") as f:
                f.write(content)

    def _run_section(
        self,
        section_name: str,
        section: CommandmentSection,
        work_dir: str,
        gpu_id: Optional[int] = None,
    ) -> Tuple[bool, str, str]:
        """
        Execute all commands in a COMMANDMENT section.

        Args:
            section_name: Name of the section (SETUP, CORRECTNESS, PROFILE)
            section: Parsed COMMANDMENT section
            work_dir: Working directory for the evaluation
            gpu_id: GPU device ID for exclusive access.  Sets both
                    GEAK_GPU_DEVICE and HIP_VISIBLE_DEVICES so that
                    COMMANDMENT commands using ${GEAK_GPU_DEVICE} get the
                    right GPU, and any CUDA/HIP call is also isolated.

        Returns:
            Tuple of (success, combined_stdout, combined_stderr)
        """
        all_stdout = []
        all_stderr = []

        env = os.environ.copy()
        env["GEAK_KERNEL_DIR"] = self.kernel_dir or work_dir
        env["GEAK_WORK_DIR"] = work_dir

        # On ROCm/HIP, remove PYTORCH_CUDA_ALLOC_CONF if it contains
        # expandable_segments -- this option is unsupported on ROCm and
        # triggers a UserWarning that can cause false failures.
        alloc_conf = env.get("PYTORCH_CUDA_ALLOC_CONF", "")
        if "expandable_segments" in alloc_conf and env.get("HIP_VISIBLE_DEVICES") is not None:
            env.pop("PYTORCH_CUDA_ALLOC_CONF", None)

        # GPU isolation: set per-subprocess environment variables
        # This is thread-safe because each subprocess.run() gets its own env dict.
        if gpu_id is not None:
            env["GEAK_GPU_DEVICE"] = str(gpu_id)
            env["HIP_VISIBLE_DEVICES"] = str(gpu_id)
            # Also strip expandable_segments for ROCm
            if "PYTORCH_CUDA_ALLOC_CONF" in env and "expandable_segments" in env.get("PYTORCH_CUDA_ALLOC_CONF", ""):
                env.pop("PYTORCH_CUDA_ALLOC_CONF", None)

        for cmd in section.commands:
            if not cmd.strip():
                continue

            logger.debug(f"[{section_name}] Running: {cmd}")
            try:
                result = subprocess.run(
                    cmd,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    cwd=work_dir,
                    env=env,
                )
                all_stdout.append(result.stdout)
                all_stderr.append(result.stderr)

                if result.returncode != 0:
                    # Filter out known benign warnings before reporting
                    # failure -- some warnings (e.g. expandable_segments
                    # on ROCm) produce a non-zero exit code but the
                    # kernel actually ran correctly.
                    filtered_stderr = _filter_benign_stderr(result.stderr)
                    logger.warning(
                        f"[{section_name}] Command failed (rc={result.returncode}): {cmd}\n"
                        f"stderr: {filtered_stderr[-300:]}"
                    )
                    return False, "\n".join(all_stdout), "\n".join(all_stderr)

            except subprocess.TimeoutExpired:
                logger.warning(
                    f"[{section_name}] Command timed out after {self.timeout}s: {cmd}"
                )
                all_stderr.append(f"TIMEOUT after {self.timeout}s")
                return False, "\n".join(all_stdout), "\n".join(all_stderr)
            except Exception as e:
                logger.warning(f"[{section_name}] Command error: {e}")
                all_stderr.append(str(e))
                return False, "\n".join(all_stdout), "\n".join(all_stderr)

        return True, "\n".join(all_stdout), "\n".join(all_stderr)

    def _parse_commandment(self, path: str) -> Dict[str, CommandmentSection]:
        """Parse a COMMANDMENT.md file into sections."""
        with open(path, "r") as f:
            content = f.read()

        sections: Dict[str, CommandmentSection] = {}
        current_section: Optional[CommandmentSection] = None

        for line in content.split("\n"):
            stripped = line.strip()

            # Detect section headers: ## SETUP, ## CORRECTNESS, ## PROFILE
            section_match = re.match(r"^##\s+(SETUP|CORRECTNESS|PROFILE)", stripped)
            if section_match:
                section_name = section_match.group(1)
                current_section = CommandmentSection(name=section_name)
                sections[section_name] = current_section
                continue

            # Skip empty lines and comments
            if not stripped:
                continue

            # --- separator means end of command sections
            if stripped.startswith("---"):
                current_section = None
                continue

            # Any ## header that isn't SETUP/CORRECTNESS/PROFILE ends the
            # current section (e.g., ## BASELINE METRICS)
            if stripped.startswith("##") and not section_match:
                current_section = None
                continue

            # Top-level # headers
            if stripped.startswith("#") and not stripped.startswith("##"):
                continue

            # Detect code blocks
            if stripped.startswith("```"):
                continue

            # Detect status markers (e.g., "Status: VALIDATED")
            if stripped.lower().startswith("status:"):
                if current_section:
                    current_section.description = stripped
                continue

            # Detect config hash
            if stripped.lower().startswith("config hash:"):
                continue

            # Add command lines to current section
            if current_section and stripped:
                # Strip leading "$ " or "- " from commands
                cmd = stripped
                if cmd.startswith("$ "):
                    cmd = cmd[2:]
                elif cmd.startswith("- "):
                    cmd = cmd[2:]
                current_section.commands.append(cmd)

        return sections

    def _parse_profiling_output(self, output: str) -> Dict[str, Any]:
        """
        Parse profiling output to extract metrics.

        Supports:
        - JSON-formatted metrics
        - Key-value pairs (e.g., "latency_us: 123.45")
        - Metrix output format
        """
        metrics: Dict[str, Any] = {}

        # Try to find JSON in the output
        json_match = re.search(r"\{[^{}]*\}", output, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
                if isinstance(parsed, dict):
                    metrics.update(parsed)
            except json.JSONDecodeError:
                pass

        # Also parse key-value pairs
        kv_pattern = r"(\w+(?:_\w+)*)\s*[:=]\s*([\d.]+(?:e[+-]?\d+)?)\s*(\w*)"
        for match in re.finditer(kv_pattern, output):
            key = match.group(1).lower()
            try:
                value = float(match.group(2))
                metrics.setdefault(key, value)
            except ValueError:
                pass

        return metrics

    # Maximum allowed regression factor.  Candidates slower than
    # baseline by more than this factor are flagged as "catastrophic
    # regression" and their speedup is clamped to a very low value
    # instead of receiving the raw number.  This prevents extreme
    # outliers (e.g. 25,000x regressions from fused_rms_fp8) from
    # polluting the evolutionary population.
    _MAX_REGRESSION_FACTOR = 5.0
    _REGRESSION_FLOOR_SPEEDUP = 0.01  # assigned to catastrophic regressions

    def _calculate_speedup(self, metrics: Dict[str, Any]) -> float:
        """Calculate speedup vs baseline metrics.

        Checks multiple latency key names (duration_us, latency_us,
        latency_avg_us) because Metrix may report under any of these.
        Falls back to throughput-based if latency is unavailable.

        **Regression guard**: if the candidate is more than
        ``_MAX_REGRESSION_FACTOR`` times slower than baseline, the
        speedup is clamped to ``_REGRESSION_FLOOR_SPEEDUP`` and a
        warning is logged.  This prevents catastrophic regressions from
        receiving a near-zero (but positive) speedup that might still
        survive selection pressure in the evolutionary population.
        """
        if not self.baseline_metrics:
            return 1.0

        # Try latency-based speedup -- check several common key names
        latency_keys = ("duration_us", "latency_us", "latency_avg_us")

        baseline_latency = 0.0
        for key in latency_keys:
            val = self.baseline_metrics.get(key, 0.0)
            if val and float(val) > 0:
                baseline_latency = float(val)
                break

        target_latency = 0.0
        for key in latency_keys:
            val = metrics.get(key, 0.0)
            if val and float(val) > 0:
                target_latency = float(val)
                break

        if baseline_latency > 0 and target_latency > 0:
            raw_speedup = baseline_latency / target_latency

            # --- Regression guard ---
            if raw_speedup < (1.0 / self._MAX_REGRESSION_FACTOR):
                regression_factor = target_latency / baseline_latency
                logger.warning(
                    f"CATASTROPHIC REGRESSION: candidate is {regression_factor:.0f}x "
                    f"slower than baseline (latency {target_latency:.1f} us vs "
                    f"{baseline_latency:.1f} us baseline). Clamping speedup to "
                    f"{self._REGRESSION_FLOOR_SPEEDUP}."
                )
                metrics["_regression"] = True
                metrics["_regression_factor"] = regression_factor
                return self._REGRESSION_FLOOR_SPEEDUP

            return raw_speedup

        # Try throughput-based speedup
        baseline_tflops = self.baseline_metrics.get("tflops", 0.0)
        target_tflops = metrics.get("tflops", 0.0)

        if baseline_tflops > 0 and target_tflops > 0:
            raw_speedup = target_tflops / baseline_tflops

            # --- Regression guard (throughput) ---
            if raw_speedup < (1.0 / self._MAX_REGRESSION_FACTOR):
                logger.warning(
                    f"CATASTROPHIC REGRESSION (throughput): candidate tflops "
                    f"{target_tflops:.2f} vs baseline {baseline_tflops:.2f}. "
                    f"Clamping speedup to {self._REGRESSION_FLOOR_SPEEDUP}."
                )
                metrics["_regression"] = True
                return self._REGRESSION_FLOOR_SPEEDUP

            return raw_speedup

        return 1.0


# ---------------------------------------------------------------------------
# CommandmentGenerator
# ---------------------------------------------------------------------------

class CommandmentGenerator:
    """
    Generates a COMMANDMENT.md file from validated, working commands.

    Pipeline flow:
      1. The caller builds the command lists (setup, correctness, profile)
      2. The caller validates commands by running them on the baseline kernel
      3. ONLY AFTER validation passes does the caller call this generator
      4. The resulting COMMANDMENT.md is FROZEN and never modified again

    Every candidate during OpenEvolve evolution is scored using exactly the
    commands in the frozen COMMANDMENT.md -- 100% deterministic evaluation.
    """

    def generate(
        self,
        kernel_dir: str,
        setup_commands: Optional[List[str]] = None,
        correctness_commands: Optional[List[str]] = None,
        profile_commands: Optional[List[str]] = None,
        profiling_results: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Generate COMMANDMENT.md content.

        Args:
            kernel_dir: Path to the kernel directory
            setup_commands: List of setup shell commands
            correctness_commands: List of correctness check commands
            profile_commands: List of profiling commands
            profiling_results: Baseline profiling results dict

        Returns:
            COMMANDMENT.md content as a string
        """
        lines = [
            "# COMMANDMENT.md",
            "",
            f"Kernel Directory: `{kernel_dir}`",
            f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Status: VALIDATED",
            "",
        ]

        # SETUP section
        lines.append("## SETUP")
        lines.append("")
        if setup_commands:
            for cmd in setup_commands:
                lines.append(f"$ {cmd}")
        else:
            lines.append("$ export PYTHONPATH=${GEAK_WORK_DIR}:${PYTHONPATH}")
            lines.append("$ mkdir -p ${GEAK_WORK_DIR}/results")
        lines.append("")

        # CORRECTNESS section
        lines.append("## CORRECTNESS")
        lines.append("")
        if correctness_commands:
            for cmd in correctness_commands:
                lines.append(f"$ {cmd}")
        else:
            lines.append(
                "$ python -m pytest ${GEAK_WORK_DIR}/tests/ -v --tb=short"
            )
        lines.append("")

        # PROFILE section
        lines.append("## PROFILE")
        lines.append("")
        if profile_commands:
            for cmd in profile_commands:
                lines.append(f"$ {cmd}")
        else:
            lines.append(
                "$ python -c \"from metrix import Metrix; "
                "m = Metrix(arch='gfx942'); "
                "print(m.profile('${GEAK_WORK_DIR}/kernel.py'))\""
            )
        lines.append("")

        # Config hash for integrity
        content_for_hash = "\n".join(lines)
        config_hash = hashlib.md5(content_for_hash.encode()).hexdigest()[:16]
        lines.append("---")
        lines.append(f"Config Hash: {config_hash}")
        lines.append("")

        # Baseline metrics summary
        if profiling_results:
            lines.append("## BASELINE METRICS (reference)")
            lines.append("```json")
            lines.append(json.dumps(profiling_results, indent=2))
            lines.append("```")
            lines.append("")

        return "\n".join(lines)

    def save(
        self,
        output_path: str,
        kernel_dir: str,
        setup_commands: Optional[List[str]] = None,
        correctness_commands: Optional[List[str]] = None,
        profile_commands: Optional[List[str]] = None,
        profiling_results: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate and save COMMANDMENT.md to disk. Returns the file path."""
        content = self.generate(
            kernel_dir=kernel_dir,
            setup_commands=setup_commands,
            correctness_commands=correctness_commands,
            profile_commands=profile_commands,
            profiling_results=profiling_results,
        )
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            f.write(content)
        logger.info(f"Saved COMMANDMENT.md to {output_path}")
        return output_path
