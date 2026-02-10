# Modifications Copyright(C)[2025] Advanced Micro Devices, Inc. All rights reserved.
# https://github.com/algorithmicsuperintelligence/openevolve  - Apache License 2.0

"""
Prompt sampling for OpenEvolve
"""

import logging
import random
from typing import Any, Dict, List, Optional, Tuple, Union

from openevolve.config import PromptConfig
from openevolve.prompt.templates import TemplateManager
from openevolve.utils.format_utils import format_metrics_safe
from openevolve.utils.metrics_utils import safe_numeric_average

logger = logging.getLogger(__name__)


class PromptSampler:
    """Generates prompts for code evolution"""

    def __init__(self, config: PromptConfig):
        self.config = config
        self.template_manager = TemplateManager(config.template_dir)

        # Initialize the random number generator
        random.seed()

        # Store custom template mappings
        self.system_template_override = None
        self.user_template_override = None

        logger.info("Initialized prompt sampler")

    def set_templates(
        self, system_template: Optional[str] = None, user_template: Optional[str] = None
    ) -> None:
        """
        Set custom templates to use for this sampler

        Args:
            system_template: Template name for system message
            user_template: Template name for user message
        """
        self.system_template_override = system_template
        self.user_template_override = user_template
        logger.info(f"Set custom templates: system={system_template}, user={user_template}")

    def build_prompt(
        self,
        current_program: str = "",
        parent_program: str = "",
        program_metrics: Dict[str, float] = {},
        previous_programs: List[Dict[str, Any]] = [],
        top_programs: List[Dict[str, Any]] = [],
        inspirations: List[Dict[str, Any]] = [],  # Add inspirations parameter
        language: str = "python",
        evolution_round: int = 0,
        diff_based_evolution: bool = True,
        template_key: Optional[str] = None,
        program_artifacts: Optional[Dict[str, Union[str, bytes]]] = None,
        # Multi-file support
        is_multifile: bool = False,
        program_files: Optional[Dict[str, str]] = None,
        main_file: Optional[str] = None,
        baseline_profiling: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, str]:
        """
        Build a prompt for the LLM

        Args:
            current_program: Current program code
            parent_program: Parent program from which current was derived
            program_metrics: Dictionary of metric names to values
            previous_programs: List of previous program attempts
            top_programs: List of top-performing programs (best by fitness)
            inspirations: List of inspiration programs (diverse/creative examples)
            language: Programming language
            evolution_round: Current evolution round
            diff_based_evolution: Whether to use diff-based evolution (True) or full rewrites (False)
            template_key: Optional override for template key
            program_artifacts: Optional artifacts from program evaluation
            is_multifile: Whether this is a multi-file program
            program_files: Dict of relative_path -> content (for multi-file)
            main_file: Entry point file (for multi-file)
            baseline_profiling: Hardware profiling data from Metrix
            **kwargs: Additional keys to replace in the user prompt

        Returns:
            Dictionary with 'system' and 'user' keys
        """
        # Select template based on evolution mode (with overrides)
        if template_key:
            # Use explicitly provided template key
            user_template_key = template_key
        elif self.user_template_override:
            # Use the override set with set_templates
            user_template_key = self.user_template_override
        elif is_multifile and diff_based_evolution:
            # Auto-select multi-file template
            user_template_key = "diff_user_multifile"
        else:
            # Default behavior: diff-based vs full rewrite
            user_template_key = "diff_user" if diff_based_evolution else "full_rewrite_user"

        # Get the template
        user_template = self.template_manager.get_template(user_template_key)

        # Use system template override if set
        if self.system_template_override:
            system_message = self.template_manager.get_template(self.system_template_override)
            logger.info(f"🎯 Using system_message from template override: {self.system_template_override} ({len(system_message)} chars)")
        else:
            system_message = self.config.system_message
            # If system_message is a template name rather than content, get the template
            if system_message in self.template_manager.templates:
                system_message = self.template_manager.get_template(system_message)
                logger.info(f"🎯 Using system_message from template: 'system_message' ({len(system_message)} chars)")
            else:
                logger.info(f"🎯 Using system_message from config (inline): ({len(system_message)} chars)")

        # Format metrics
        metrics_str = self._format_metrics(program_metrics)

        # Identify areas for improvement
        improvement_areas = self._identify_improvement_areas(
            current_program, parent_program, program_metrics, previous_programs
        )

        # Format evolution history
        evolution_history = self._format_evolution_history(
            previous_programs, top_programs, inspirations, language
        )

        # Format artifacts section if enabled and available
        artifacts_section = ""
        if self.config.include_artifacts and program_artifacts:
            artifacts_section = self._render_artifacts(program_artifacts)

        # Apply stochastic template variations if enabled
        if self.config.use_template_stochasticity:
            user_template = self._apply_template_variations(user_template)

        # Build multi-file specific placeholders
        extra_format_kwargs = dict(kwargs)
        if is_multifile and program_files:
            extra_format_kwargs["file_listing"] = self._format_file_listing(
                program_files, main_file
            )
            extra_format_kwargs["file_contents"] = self._format_file_contents(
                program_files, main_file, language
            )
        elif is_multifile:
            # Fallback if files not provided
            extra_format_kwargs.setdefault("file_listing", "(single file)")
            extra_format_kwargs.setdefault("file_contents", f"```{language}\n{current_program}\n```")

        # Baseline profiling
        if baseline_profiling:
            extra_format_kwargs["baseline_profiling"] = self._format_baseline_profiling(
                baseline_profiling
            )
        else:
            extra_format_kwargs.setdefault(
                "baseline_profiling",
                "No baseline profiling data available. Focus on general optimization strategies."
            )

        # Format the final user message
        user_message = user_template.format(
            metrics=metrics_str,
            improvement_areas=improvement_areas,
            evolution_history=evolution_history,
            current_program=current_program,
            language=language,
            artifacts=artifacts_section,
            **extra_format_kwargs,
        )

        return {
            "system": system_message,
            "user": user_message,
        }

    def _format_metrics(self, metrics: Dict[str, float]) -> str:
        """Format metrics for the prompt using safe formatting"""
        # Use safe formatting to handle mixed numeric and string values
        formatted_parts = []
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                try:
                    formatted_parts.append(f"- {name}: {value:.4f}")
                except (ValueError, TypeError):
                    formatted_parts.append(f"- {name}: {value}")
            else:
                formatted_parts.append(f"- {name}: {value}")
        return "\n".join(formatted_parts)

    def _identify_improvement_areas(
        self,
        current_program: str,
        parent_program: str,
        metrics: Dict[str, float],
        previous_programs: List[Dict[str, Any]],
    ) -> str:
        """Identify potential areas for improvement"""
        # This method could be expanded to include more sophisticated analysis
        # For now, we'll use a simple approach

        improvement_areas = []

        # # Check program length
        # if len(current_program) > 500:
        #     improvement_areas.append(
        #         "Consider simplifying the code to improve readability and maintainability"
        #     )

        # Check for performance patterns in previous attempts
        if len(previous_programs) >= 2:
            recent_attempts = previous_programs[-2:]
            metrics_improved = []
            metrics_regressed = []

            for metric, value in metrics.items():
                # Only compare numeric metrics
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    continue

                improved = True
                regressed = True

                for attempt in recent_attempts:
                    attempt_value = attempt["metrics"].get(metric, 0)
                    # Only compare if both values are numeric
                    if isinstance(value, (int, float)) and isinstance(attempt_value, (int, float)):
                        if attempt_value <= value:
                            regressed = False
                        if attempt_value >= value:
                            improved = False
                    else:
                        # If either value is non-numeric, skip comparison
                        improved = False
                        regressed = False

                if improved and metric not in metrics_improved:
                    metrics_improved.append(metric)
                if regressed and metric not in metrics_regressed:
                    metrics_regressed.append(metric)

            if metrics_improved:
                improvement_areas.append(
                    f"Metrics showing improvement: {', '.join(metrics_improved)}. "
                    "Consider continuing with similar changes."
                )

            if metrics_regressed:
                improvement_areas.append(
                    f"Metrics showing regression: {', '.join(metrics_regressed)}. "
                    "Consider reverting or revising recent changes in these areas."
                )

        # If we don't have specific improvements to suggest
        if not improvement_areas:
            improvement_areas.append(
                "Focus on optimizing the code for better performance on the target metrics"
            )

        return "\n".join([f"- {area}" for area in improvement_areas])

    def _format_evolution_history(
        self,
        previous_programs: List[Dict[str, Any]],
        top_programs: List[Dict[str, Any]],
        inspirations: List[Dict[str, Any]],
        language: str,
    ) -> str:
        """Format the evolution history for the prompt"""
        # Get templates
        history_template = self.template_manager.get_template("evolution_history")
        previous_attempt_template = self.template_manager.get_template("previous_attempt")
        top_program_template = self.template_manager.get_template("top_program")

        # Format previous attempts (most recent first)
        previous_attempts_str = ""
        selected_previous = previous_programs[-min(3, len(previous_programs)) :]

        for i, program in enumerate(reversed(selected_previous)):
            attempt_number = len(previous_programs) - i
            changes = program.get("changes", None)
            if changes is None:
                continue
            # Format performance metrics using safe formatting
            performance_parts = []
            for name, value in program.get("metrics", {}).items():
                if isinstance(value, (int, float)):
                    try:
                        performance_parts.append(f"{name}: {value:.4f}")
                    except (ValueError, TypeError):
                        performance_parts.append(f"{name}: {value}")
                else:
                    performance_parts.append(f"{name}: {value}")
            performance_str = ", ".join(performance_parts)

            # Determine outcome based on comparison with parent (only numeric metrics)
            parent_metrics = program.get("parent_metrics", {})
            outcome = "Mixed results"

            # Safely compare only numeric metrics
            program_metrics = program.get("metrics", {})

            # Check if all numeric metrics improved
            numeric_comparisons_improved = []
            numeric_comparisons_regressed = []

            for m in program_metrics:
                prog_value = program_metrics.get(m, 0)
                parent_value = parent_metrics.get(m, 0)

                # Only compare if both values are numeric
                if isinstance(prog_value, (int, float)) and isinstance(parent_value, (int, float)):
                    if prog_value > parent_value:
                        numeric_comparisons_improved.append(True)
                    else:
                        numeric_comparisons_improved.append(False)

                    if prog_value < parent_value:
                        numeric_comparisons_regressed.append(True)
                    else:
                        numeric_comparisons_regressed.append(False)

            # Determine outcome based on numeric comparisons
            if numeric_comparisons_improved and all(numeric_comparisons_improved):
                outcome = "Improvement in all metrics"
            elif numeric_comparisons_regressed and all(numeric_comparisons_regressed):
                outcome = "Regression in all metrics"

            previous_attempts_str += (
                previous_attempt_template.format(
                    attempt_number=attempt_number,
                    changes=changes,
                    performance=performance_str,
                    # outcome=outcome,
                    reasoning=program.get("analysis", "No analysis provided"),
                )
                + "\n\n"
            )

        # Format top programs
        top_programs_str = ""
        selected_top = top_programs[: min(self.config.num_top_programs, len(top_programs))]

        for i, program in enumerate(selected_top):
            # Extract a snippet (first 10 lines) for display
            program_code = program.get("code", "")
            max_prog_lines = min(1000, len(program_code.split("\n")))
            program_snippet = "\n".join(program_code.split("\n")[:max_prog_lines])
            if len(program_code.split("\n")) > max_prog_lines:
                program_snippet += "\n# ... (truncated for brevity)"

            # Calculate a composite score using safe numeric average
            score = safe_numeric_average(program.get("metrics", {}))

            # Extract key features (this could be more sophisticated)
            key_features = program.get("key_features", [])
            if not key_features:
                key_features = []
                for name, value in program.get("metrics", {}).items():
                    if isinstance(value, (int, float)):
                        try:
                            key_features.append(f"Performs well on {name} ({value:.4f})")
                        except (ValueError, TypeError):
                            key_features.append(f"Performs well on {name} ({value})")
                    else:
                        key_features.append(f"Performs well on {name} ({value})")

            key_features_str = ", ".join(key_features)

            top_programs_str += (
                top_program_template.format(
                    program_number=i + 1,
                    score=f"{score:.4f}",
                    language=language,
                    program_snippet=program_snippet,
                    # key_features=key_features_str,
                    reasoning=program.get("analysis", "No analysis provided"),
                )
                + "\n\n"
            )

        # Format diverse programs using num_diverse_programs config
        diverse_programs_str = ""
        if (
            self.config.num_diverse_programs > 0
            and len(top_programs) > self.config.num_top_programs
        ):
            # Skip the top programs we already included
            remaining_programs = top_programs[self.config.num_top_programs :]

            # Sample diverse programs from the remaining
            num_diverse = min(self.config.num_diverse_programs, len(remaining_programs))
            if num_diverse > 0:
                # Use random sampling to get diverse programs
                diverse_programs = random.sample(remaining_programs, num_diverse)

                diverse_programs_str += "\n\n## Diverse Programs\n\n"

                for i, program in enumerate(diverse_programs):
                    # Extract a snippet (first 5 lines for diversity)
                    program_code = program.get("code", "")
                    program_snippet = "\n".join(program_code.split("\n")[:5])
                    if len(program_code.split("\n")) > 5:
                        program_snippet += "\n# ... (truncated)"

                    # Calculate a composite score using safe numeric average
                    score = safe_numeric_average(program.get("metrics", {}))

                    # Extract key features
                    key_features = program.get("key_features", [])
                    if not key_features:
                        key_features = [
                            f"Alternative approach to {name}"
                            for name in list(program.get("metrics", {}).keys())[
                                :2
                            ]  # Just first 2 metrics
                        ]

                    key_features_str = ", ".join(key_features)

                    diverse_programs_str += (
                        top_program_template.format(
                            program_number=f"D{i + 1}",
                            score=f"{score:.4f}",
                            language=language,
                            program_snippet=program_snippet,
                            key_features=key_features_str,
                            reasoning=program.get("analysis", "No analysis provided"),
                        )
                        + "\n\n"
                    )

        # Combine top and diverse programs
        combined_programs_str = top_programs_str #+ diverse_programs_str

        # Format inspirations section
        inspirations_section_str = self._format_inspirations_section(inspirations, language)

        # Combine into full history
        return history_template.format(
            previous_attempts=previous_attempts_str.strip(),
            top_programs=combined_programs_str.strip(),
            # inspirations_section=inspirations_section_str,
        )

    def _format_inspirations_section(
        self, inspirations: List[Dict[str, Any]], language: str
    ) -> str:
        """
        Format the inspirations section for the prompt
        
        Args:
            inspirations: List of inspiration programs
            language: Programming language
            
        Returns:
            Formatted inspirations section string
        """
        if not inspirations:
            return ""
            
        # Get templates
        inspirations_section_template = self.template_manager.get_template("inspirations_section")
        inspiration_program_template = self.template_manager.get_template("inspiration_program")
        
        inspiration_programs_str = ""
        
        for i, program in enumerate(inspirations):
            # Extract a snippet (first 8 lines) for display
            program_code = program.get("code", "")
            program_snippet = "\n".join(program_code.split("\n")[:8])
            if len(program_code.split("\n")) > 8:
                program_snippet += "\n# ... (truncated for brevity)"
            
            # Calculate a composite score using safe numeric average
            score = safe_numeric_average(program.get("metrics", {}))
            
            # Determine program type based on metadata and score
            program_type = self._determine_program_type(program)
            
            # Extract unique features (emphasizing diversity rather than just performance)
            unique_features = self._extract_unique_features(program)
            
            inspiration_programs_str += (
                inspiration_program_template.format(
                    program_number=i + 1,
                    score=f"{score:.4f}",
                    program_type=program_type,
                    language=language,
                    program_snippet=program_snippet,
                    unique_features=unique_features,
                    reasoning=program.get("analysis", "No analysis provided")
                )
                + "\n\n"
            )
        
        return inspirations_section_template.format(
            inspiration_programs=inspiration_programs_str.strip()
        )
        
    def _determine_program_type(self, program: Dict[str, Any]) -> str:
        """
        Determine the type/category of an inspiration program
        
        Args:
            program: Program dictionary
            
        Returns:
            String describing the program type
        """
        metadata = program.get("metadata", {})
        score = safe_numeric_average(program.get("metrics", {}))
        
        # Check metadata for explicit type markers
        if metadata.get("diverse", False):
            return "Diverse"
        if metadata.get("migrant", False):
            return "Migrant"
        if metadata.get("random", False):
            return "Random"
            
        # Classify based on score ranges
        if score >= 0.8:
            return "High-Performer"
        elif score >= 0.6:
            return "Alternative"
        elif score >= 0.4:
            return "Experimental"
        else:
            return "Exploratory"
            
    def _extract_unique_features(self, program: Dict[str, Any]) -> str:
        """
        Extract unique features of an inspiration program
        
        Args:
            program: Program dictionary
            
        Returns:
            String describing unique aspects of the program
        """
        features = []
        
        # Extract from metadata if available
        metadata = program.get("metadata", {})
        if "changes" in metadata:
            changes = metadata["changes"]
            if isinstance(changes, str) and len(changes) < 100:
                features.append(f"Modification: {changes}")
        
        # Analyze metrics for standout characteristics
        metrics = program.get("metrics", {})
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):
                if value >= 0.9:
                    features.append(f"Excellent {metric_name} ({value:.3f})")
                elif value <= 0.3:
                    features.append(f"Alternative {metric_name} approach")
        
        # Code-based features (simple heuristics)
        code = program.get("code", "")
        if code:
            code_lower = code.lower()
            if "class" in code_lower and "def __init__" in code_lower:
                features.append("Object-oriented approach")
            if "numpy" in code_lower or "np." in code_lower:
                features.append("NumPy-based implementation")
            if "for" in code_lower and "while" in code_lower:
                features.append("Mixed iteration strategies")
            if len(code.split("\n")) < 10:
                features.append("Concise implementation")
            elif len(code.split("\n")) > 50:
                features.append("Comprehensive implementation")
        
        # Default if no specific features found
        if not features:
            program_type = self._determine_program_type(program)
            features.append(f"{program_type} approach to the problem")
            
        return ", ".join(features[:3])  # Limit to top 3 features

    # --- Multi-file formatting helpers ---

    def _format_file_listing(
        self, files: Dict[str, str], main_file: Optional[str] = None
    ) -> str:
        """Format a listing of all files in the program."""
        lines = []
        sorted_paths = sorted(files.keys())
        for path in sorted_paths:
            size = len(files[path])
            marker = " (main)" if path == main_file else ""
            lines.append(f"- `{path}` ({size} chars){marker}")
        return "\n".join(lines)

    def _format_file_contents(
        self,
        files: Dict[str, str],
        main_file: Optional[str] = None,
        language: str = "python",
    ) -> str:
        """Format file contents for inclusion in the prompt."""
        parts = []
        # Main file first
        sorted_paths = sorted(files.keys())
        if main_file and main_file in files:
            sorted_paths.remove(main_file)
            sorted_paths.insert(0, main_file)

        for path in sorted_paths:
            marker = " (main entry point)" if path == main_file else ""
            parts.append(
                f"## File: `{path}`{marker}\n```{language}\n{files[path]}\n```"
            )
        return "\n\n".join(parts)

    @staticmethod
    def _normalize_profiling_keys(profiling: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize Metrix metric keys by stripping the ``memory.`` prefix so
        that downstream formatters can look up keys consistently.

        Metrix outputs keys like ``memory.coalescing_efficiency``,
        ``memory.hbm_bandwidth_utilization``, etc.  This method returns a
        *new* dict where those keys have the prefix stripped, while
        non-prefixed keys (``duration_us``, ``bottleneck``, ``kernel_name``,
        GPU hardware fields, etc.) are kept as-is.

        Original keys are also kept (lower priority) so that callers that
        already handle the prefixed form continue to work.
        """
        normalized: Dict[str, Any] = {}
        for key, val in profiling.items():
            # Always keep the original key
            normalized[key] = val
            # Strip "memory." prefix
            if key.startswith("memory."):
                stripped = key[len("memory."):]
                # Only add if not already present (prefer explicit unprefixed)
                normalized.setdefault(stripped, val)
        return normalized

    # Known AMD GPU specs lookup: architecture -> (name, peak_hbm_bw_gb_s, peak_fp32_tflops)
    _GPU_SPECS = {
        "gfx942": ("AMD Instinct MI300X", "5300.0", "163.4"),
        "gfx950": ("AMD Instinct MI325X", "6000.0", "163.4"),
        "gfx940": ("AMD Instinct MI300A", "5300.0", "122.6"),
        "gfx90a": ("AMD Instinct MI250X", "3276.8", "95.7"),
        "gfx908": ("AMD Instinct MI100", "1228.8", "23.1"),
    }

    @staticmethod
    def _detect_gpu_hardware() -> Dict[str, str]:
        """
        Detect GPU hardware info at runtime via rocm-smi / rocminfo.
        Returns a dict with keys: gpu, architecture, compute_units,
        peak_hbm_bw_gb_s, peak_fp32_tflops (any may be missing).
        """
        import subprocess, re
        info: Dict[str, str] = {}
        arch = ""

        # --- rocm-smi: get GFX version and product name ---
        try:
            result = subprocess.run(
                ["rocm-smi", "--showproductname"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    # GFX Version line (e.g., "GPU[0]\t\t: GFX Version:\t\tgfx950")
                    gfx_m = re.search(r"GFX\s+Version\s*:\s*(gfx\w+)", line, re.I)
                    if gfx_m:
                        arch = gfx_m.group(1)
                        info["architecture"] = arch
                    # Card series (may be "N/A")
                    series_m = re.search(
                        r"Card\s+Series\s*:\s*(.+)", line, re.I
                    )
                    if series_m:
                        val = series_m.group(1).strip()
                        if val and val.upper() != "N/A":
                            info["gpu"] = val
        except Exception:
            pass

        # --- rocminfo: get architecture + compute units + marketing name ---
        try:
            result = subprocess.run(
                ["rocminfo"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                in_gpu_agent = False
                for line in result.stdout.splitlines():
                    # Detect GPU agent blocks via "Device Type: GPU" or UUID
                    if "Device Type:" in line and "GPU" in line.upper():
                        in_gpu_agent = True
                        continue
                    elif "Device Type:" in line and "CPU" in line.upper():
                        in_gpu_agent = False
                        continue
                    # Also detect via Vendor Name (CPU vs AMD/GPU)
                    if "Vendor Name:" in line:
                        if "CPU" in line:
                            in_gpu_agent = False
                        elif "AMD" in line:
                            in_gpu_agent = True

                    if not in_gpu_agent:
                        continue

                    # Architecture from "Name: gfx950"
                    if re.match(r"\s+Name:", line) and "gfx" in line:
                        m = re.search(r"(gfx\d+)", line)
                        if m:
                            arch = m.group(1)
                            info["architecture"] = arch
                    # Marketing Name (may be empty)
                    if "Marketing Name:" in line:
                        m = re.search(r"Marketing Name:\s*(.+)", line)
                        if m:
                            val = m.group(1).strip()
                            if val:
                                info["gpu"] = val
                    # Compute Units
                    if "Compute Unit:" in line:
                        m = re.search(r"(\d+)", line)
                        if m:
                            info["compute_units"] = m.group(1)
                    # Stop after first GPU agent is fully parsed
                    if info.get("compute_units") and info.get("architecture"):
                        break
        except Exception:
            pass

        # --- Fill in known specs from architecture ---
        if arch in PromptSampler._GPU_SPECS:
            name, bw, tflops = PromptSampler._GPU_SPECS[arch]
            info.setdefault("gpu", name)
            info.setdefault("peak_hbm_bw_gb_s", bw)
            info.setdefault("peak_fp32_tflops", tflops)

        return info

    def _format_baseline_profiling(self, profiling: Dict[str, Any]) -> str:
        """
        Format baseline hardware profiling data from Metrix into a human-readable
        string for inclusion in the LLM prompt.

        Handles Metrix's ``memory.``-prefixed keys (e.g.
        ``memory.coalescing_efficiency``) by normalizing them, and includes GPU
        hardware specifications for context.

        Args:
            profiling: Dictionary of profiling metrics from Metrix/rocprofv3.

        Returns:
            Formatted profiling summary string with bottleneck analysis and
            GPU hardware context.
        """
        # Normalize Metrix "memory."-prefixed keys
        p = self._normalize_profiling_keys(profiling)
        sections = []

        # --- GPU Hardware Context ---
        hw = self._detect_gpu_hardware()
        # Also check if profiling dict itself has GPU info (e.g. from kernel-profile)
        gpu_name = p.get("gpu") or hw.get("gpu", "Unknown")
        arch = p.get("architecture") or hw.get("architecture", "Unknown")
        cus = p.get("compute_units") or hw.get("compute_units", "Unknown")
        peak_bw = p.get("peak_hbm_bw_gb_s") or hw.get("peak_hbm_bw_gb_s", "Unknown")
        peak_tflops = p.get("peak_fp32_tflops") or hw.get("peak_fp32_tflops", "Unknown")

        hw_section = (
            "## Target GPU Hardware\n"
            f"  - GPU: {gpu_name}\n"
            f"  - Architecture: {arch}\n"
            f"  - Compute Units: {cus}\n"
            f"  - Peak HBM Bandwidth: {peak_bw} GB/s\n"
            f"  - Peak FP32 Throughput: {peak_tflops} TFLOPS"
        )
        sections.append(hw_section)

        # --- Kernel info ---
        kernel_name = p.get("kernel_name", "")
        if kernel_name:
            sections.append(f"## Profiled Kernel: `{kernel_name}`")

        # --- Latency ---
        # Metrix uses "duration_us"; also check legacy latency_* keys
        latency_keys = ["duration_us", "latency_us", "latency_min_us", "latency_avg_us", "latency_max_us"]
        latency_parts = []
        for key in latency_keys:
            if key in p:
                val = p[key]
                if isinstance(val, (int, float)):
                    label = key.replace("_us", "").replace("_", " ").capitalize()
                    if label.lower() == "duration":
                        label = "Kernel Latency"
                    latency_parts.append(f"  - {label}: {val:.2f} us")
        if latency_parts:
            sections.append("## Latency\n" + "\n".join(latency_parts))

        # --- Memory Performance ---
        # Keys after normalization (memory. prefix stripped)
        mem_keys = {
            "hbm_bandwidth_utilization": ("HBM Bandwidth Utilization", "%"),
            "hbm_read_bandwidth": ("HBM Read Bandwidth", "GB/s"),
            "hbm_write_bandwidth": ("HBM Write Bandwidth", "GB/s"),
            "bytes_transferred_hbm": ("HBM Bytes Transferred", "bytes"),
            "global_load_efficiency": ("Global Load Efficiency", "%"),
            "global_store_efficiency": ("Global Store Efficiency", "%"),
            "coalescing_efficiency": ("Coalescing Efficiency", "%"),
            "lds_bank_conflicts": ("LDS Bank Conflicts", "count"),
            "atomic_latency": ("Atomic Latency", ""),
            # Legacy key names (in case someone uses them)
            "bandwidth_gb_s": ("Bandwidth", "GB/s"),
            "hbm_utilization": ("HBM Utilization", "%"),
        }
        mem_parts = []
        seen_labels = set()
        for key, (label, unit) in mem_keys.items():
            if key in p and label not in seen_labels:
                seen_labels.add(label)
                val = p[key]
                if isinstance(val, (int, float)):
                    if unit == "%":
                        mem_parts.append(f"  - {label}: {val:.1f}%")
                    elif unit == "GB/s":
                        mem_parts.append(f"  - {label}: {val:.2f} GB/s")
                    elif unit == "bytes":
                        mem_parts.append(f"  - {label}: {val:,.0f} bytes")
                    else:
                        mem_parts.append(f"  - {label}: {val:.2f}")
                else:
                    mem_parts.append(f"  - {label}: {val}")
        if mem_parts:
            sections.append("## Memory Performance\n" + "\n".join(mem_parts))

        # --- Compute Performance ---
        compute_keys = {
            "tflops": ("TFLOPS", "tflops"),
            "compute_busy": ("GPU Compute Busy", "%"),
            "valu_busy": ("VALU Busy", "%"),
            "mfma_busy": ("MFMA Busy", "%"),
        }
        compute_parts = []
        for key, (label, unit) in compute_keys.items():
            if key in p:
                val = p[key]
                if isinstance(val, (int, float)):
                    if unit == "%":
                        compute_parts.append(f"  - {label}: {val:.1f}%")
                    else:
                        compute_parts.append(f"  - {label}: {val:.2f}")
        if compute_parts:
            sections.append("## Compute Performance\n" + "\n".join(compute_parts))

        # --- Cache Performance ---
        cache_keys = {
            "l1_hit_rate": "L1 Cache Hit Rate",
            "l2_hit_rate": "L2 Cache Hit Rate",
            "l2_bandwidth": "L2 Bandwidth (GB/s)",
            "l2_read_hit_rate": "L2 Read Hit Rate",
            "l2_write_hit_rate": "L2 Write Hit Rate",
        }
        cache_parts = []
        for key, label in cache_keys.items():
            if key in p:
                val = p[key]
                if isinstance(val, (int, float)):
                    if "bandwidth" in key.lower():
                        cache_parts.append(f"  - {label}: {val:.2f} GB/s")
                    else:
                        cache_parts.append(f"  - {label}: {val:.1f}%")
        if cache_parts:
            sections.append("## Cache Performance\n" + "\n".join(cache_parts))

        # --- Bottleneck Analysis ---
        bottleneck = self._analyze_bottleneck(p)
        if bottleneck:
            sections.append(f"## Bottleneck Analysis & Recommended Algorithmic Strategies\n{bottleneck}")

        if not sections:
            return "No detailed profiling data available."

        return "\n\n".join(sections)

    def _analyze_bottleneck(self, profiling: Dict[str, Any]) -> str:
        """
        Analyze profiling data to identify the primary bottleneck and recommend
        **concrete algorithmic strategies** (NOT autotuning).

        Each recommendation targets a specific profiling signal and maps it to
        an algorithmic transformation the LLM should attempt.  Autotuning
        (BLOCK_SIZE, num_warps, etc.) is explicitly discouraged.

        Args:
            profiling: Normalized profiling dict (``memory.`` prefixes already
                       stripped by ``_normalize_profiling_keys``).
        """
        analysis = []

        # Read metrics (already normalized -- no memory. prefix)
        compute_busy = profiling.get("compute_busy")
        hbm_util = profiling.get("hbm_bandwidth_utilization")
        coalescing = profiling.get("coalescing_efficiency")
        l1_hit = profiling.get("l1_hit_rate")
        l2_hit = profiling.get("l2_hit_rate")
        l2_bw = profiling.get("l2_bandwidth")
        hbm_read_bw = profiling.get("hbm_read_bandwidth")
        hbm_write_bw = profiling.get("hbm_write_bandwidth")
        global_load_eff = profiling.get("global_load_efficiency")
        global_store_eff = profiling.get("global_store_efficiency")
        lds_conflicts = profiling.get("lds_bank_conflicts")
        duration_us = profiling.get("duration_us")
        # Metrix may also provide a textual bottleneck classification
        metrix_bottleneck = profiling.get("bottleneck", "")

        # ----- Primary bottleneck classification -----
        if metrix_bottleneck:
            analysis.append(f"**Metrix classification**: {metrix_bottleneck}-bound")

        _hbm = hbm_util if hbm_util is not None else 0
        _comp = compute_busy if compute_busy is not None else 0

        if _comp < 30 and _hbm > 60:
            analysis.append(
                "**Memory-bound**: GPU compute is idle while HBM bandwidth is saturated.\n"
                "  Algorithmic strategies:\n"
                "  - *Kernel fusion*: Merge consecutive kernels to avoid writing intermediate "
                "results to HBM. For example, fuse an elementwise + reduction kernel pair.\n"
                "  - *Operator reordering*: Reorder operations so that data produced by one "
                "computation is consumed immediately in registers/LDS rather than round-tripping "
                "through HBM.\n"
                "  - *Data compression*: Compute on fp16/bf16 instead of fp32 to halve memory "
                "traffic while maintaining accuracy (use Kahan summation for accumulations).\n"
                "  - *Tiling with LDS staging*: Load a tile of data into LDS (shared memory), "
                "then perform multiple operations on it before writing back."
            )
        elif _comp > 60 and _hbm < 30:
            analysis.append(
                "**Compute-bound**: ALU is saturated while memory bandwidth is underutilized.\n"
                "  Algorithmic strategies:\n"
                "  - *Strength reduction*: Replace expensive operations (div, sqrt, exp) with "
                "cheaper approximations or lookup tables. Use `tl.math.fast_expf` or "
                "polynomial approximations where precision allows.\n"
                "  - *Redundant computation elimination*: Precompute values outside the inner "
                "loop. Hoist invariants, factor common sub-expressions.\n"
                "  - *Algorithmic complexity reduction*: If the kernel is O(N^2), look for "
                "O(N log N) or O(N) alternatives (e.g., FFT-based convolution, flash attention "
                "tiling pattern).\n"
                "  - *FMA utilization*: Ensure multiply-add pairs use fused multiply-add "
                "(`tl.fma` or `a * b + c`) to get 2 flops per instruction."
            )
        elif _comp < 30 and _hbm < 30:
            latency_bound_msg = (
                "**Latency-bound**: Both compute and memory are severely underutilized.\n"
                "  Algorithmic strategies:\n"
                "  - *Work amplification*: The kernel does too little work per launch. Fuse "
                "multiple operations into a single kernel or increase per-thread work.\n"
                "  - *Pointer precomputation*: Move address calculations (offsets, strides) "
                "outside the inner loop to reduce instruction count.\n"
                "  - *Loop unrolling & software pipelining*: Manually unroll the innermost "
                "loop to overlap loads with computation (prefetch next iteration's data "
                "while processing current).\n"
                "  - *Occupancy-aware redesign*: Use fewer registers per thread to allow more "
                "concurrent wavefronts, or restructure to avoid barrier synchronizations."
            )
            # For severely latency-bound kernels (<5% HBM utilization), allow
            # tuning BLOCK_SIZE / num_warps as a SECONDARY strategy alongside
            # algorithmic changes.  These kernels are so underutilized that
            # increasing occupancy via parameter changes can genuinely help.
            if _hbm < 5:
                latency_bound_msg += (
                    "\n\n  **Note**: Because HBM utilization is extremely low (<5%), "
                    "you MAY also adjust `BLOCK_SIZE` and `num_warps` as a SECONDARY "
                    "strategy to improve occupancy -- but this must be accompanied by "
                    "at least one algorithmic change from the list above."
                )
            analysis.append(latency_bound_msg)

        # ----- Specific metric-driven recommendations -----
        if coalescing is not None and coalescing < 50:
            analysis.append(
                f"**Poor memory coalescing** ({coalescing:.0f}%):\n"
                "  - *Layout transformation*: Transpose from AoS (Array of Structs) to SoA "
                "(Struct of Arrays) so that consecutive threads access consecutive addresses.\n"
                "  - *Index remapping*: Reorder the loop iteration space so that the innermost "
                "dimension maps to thread indices (coalesced access pattern).\n"
                "  - *Padding*: Add padding to eliminate bank conflicts if the stride is a "
                "power of 2."
            )

        if global_load_eff is not None and global_load_eff < 25:
            analysis.append(
                f"**Low global load efficiency** ({global_load_eff:.0f}%):\n"
                "  - Many loaded bytes are wasted. Restructure loads to use full cache lines.\n"
                "  - Consider vectorized loads (`tl.load` with contiguous offsets) to maximize "
                "bytes useful per transaction."
            )

        if global_store_eff is not None and global_store_eff < 25:
            analysis.append(
                f"**Low global store efficiency** ({global_store_eff:.0f}%):\n"
                "  - Writes are scattered / partial cache-line writes. Accumulate results in "
                "registers and write full tiles at once."
            )

        if l2_hit is not None and l2_hit < 30:
            analysis.append(
                f"**Low L2 cache hit rate** ({l2_hit:.0f}%):\n"
                "  - *Temporal tiling*: Restructure the algorithm so the same data block is "
                "accessed multiple times within a short window (stays in L2).\n"
                "  - *Producer-consumer fusion*: If this kernel reads output of another, fuse "
                "them so data goes through L2 once instead of HBM round-trip."
            )

        if l1_hit is not None and l1_hit < 40:
            analysis.append(
                f"**Low L1 cache hit rate** ({l1_hit:.0f}%):\n"
                "  - Data is not being reused within a workgroup. Use LDS (shared memory) to "
                "stage frequently-accessed data explicitly."
            )

        if lds_conflicts is not None and lds_conflicts > 5:
            analysis.append(
                f"**LDS bank conflicts** ({lds_conflicts:.1f}):\n"
                "  - Add padding to shared memory arrays (e.g., `tl.make_block_ptr` with +1 "
                "column) to avoid bank conflicts.\n"
                "  - Swizzle access patterns within the tile."
            )

        if hbm_read_bw is not None and hbm_write_bw is not None:
            if hbm_write_bw > 2 * hbm_read_bw and hbm_write_bw > 50:
                analysis.append(
                    f"**Write-heavy kernel** (read={hbm_read_bw:.0f} GB/s, write={hbm_write_bw:.0f} GB/s):\n"
                    "  - Consider write-combining: accumulate partial results before writing.\n"
                    "  - If writing intermediate buffers, fuse with the next consumer kernel."
                )

        if duration_us is not None and duration_us < 10:
            analysis.append(
                f"**Very short kernel** ({duration_us:.1f} us):\n"
                "  - Kernel launch overhead may dominate. Fuse with adjacent kernels "
                "or batch more work into a single launch."
            )

        # ----- Kernel-type-specific guidance -----

        # Element-wise kernels: low bytes_transferred + low duration → limited headroom
        bytes_transferred = profiling.get("bytes_transferred_hbm")
        if (
            bytes_transferred is not None
            and duration_us is not None
            and duration_us < 50
            and _hbm < 40
            and _comp < 40
        ):
            analysis.append(
                "**Element-wise / lightweight kernel** (low data volume, short duration):\n"
                "  - Limited headroom for single-kernel optimization.\n"
                "  - *Kernel fusion*: The highest-impact strategy is fusing this kernel with "
                "adjacent operations (preceding producer or following consumer) to eliminate "
                "an entire memory round-trip.\n"
                "  - *Vectorized loads/stores*: Ensure you are using the widest possible "
                "vector width (e.g., 4-element vectors) to maximize per-thread throughput.\n"
                "  - If fusion is not possible from this file alone, note this limitation "
                "and focus on reducing instruction count within the kernel."
            )

        # Sequential / sorting-like algorithms (topk, argsort, etc.)
        # Heuristic: very low compute busy, very low HBM, moderate duration
        if (
            duration_us is not None
            and _comp < 15
            and _hbm < 15
            and duration_us > 20
        ):
            analysis.append(
                "**Potentially sequential / data-dependent algorithm**:\n"
                "  - Both compute and memory utilization are extremely low, suggesting "
                "the algorithm may be inherently sequential (e.g., sorting, top-k selection, "
                "scan with dependencies).\n"
                "  - *Consider a fundamentally different algorithm*: Rather than tweaking the "
                "existing implementation, propose an entirely different approach. For example:\n"
                "    * Replace iterative top-k with radix select or approximate top-k\n"
                "    * Replace sequential scan with work-efficient parallel prefix sum\n"
                "    * Replace comparison-based sort with radix sort\n"
                "  - *Parallelism extraction*: Identify independent sub-problems that can be "
                "solved concurrently across workgroups."
            )

        # LDS-bound kernels: high LDS bank conflicts
        if lds_conflicts is not None and lds_conflicts > 5:
            analysis.append(
                "**LDS-bound kernel** (high bank conflicts detected):\n"
                "  Concrete Triton patterns to reduce LDS bank conflicts:\n"
                "  - *Padding*: When allocating shared memory tiles, add +1 to the "
                "innermost dimension to avoid power-of-2 stride conflicts:\n"
                "    ```python\n"
                "    # Instead of:\n"
                "    tile = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.float32)\n"
                "    # Use padded layout (conceptually):\n"
                "    # Ensure BLOCK_K is not a multiple of 32 (bank count)\n"
                "    ```\n"
                "  - *Swizzled access*: XOR the row index with column index bits to "
                "spread accesses across banks:\n"
                "    ```python\n"
                "    # Swizzle pattern for LDS access\n"
                "    row_idx = tl.arange(0, BLOCK_M)\n"
                "    col_idx = tl.arange(0, BLOCK_K)\n"
                "    swizzled_col = col_idx ^ (row_idx[:, None] & 0x1F)\n"
                "    ```\n"
                "  - *Bank-conflict-free transpose*: If doing a tile transpose through "
                "LDS, use the padding + swizzle technique above."
            )

        # ----- EXPLICIT: what NOT to focus on -----
        analysis.append(
            "**DO NOT** focus on autotuning (BLOCK_SIZE, num_warps, num_stages, grid "
            "dimensions). These are parameter sweeps, not algorithmic improvements. "
            "The goal is to find fundamentally better algorithms or data-flow "
            "transformations that reduce total work or memory traffic."
        )

        return "\n\n".join(analysis) if analysis else ""

    def _apply_template_variations(self, template: str) -> str:
        """Apply stochastic variations to the template"""
        result = template

        # Apply variations defined in the config
        for key, variations in self.config.template_variations.items():
            if variations and f"{{{key}}}" in result:
                chosen_variation = random.choice(variations)
                result = result.replace(f"{{{key}}}", chosen_variation)

        return result

    def _render_artifacts(self, artifacts: Dict[str, Union[str, bytes]]) -> str:
        """
        Render artifacts for prompt inclusion

        Args:
            artifacts: Dictionary of artifact name to content

        Returns:
            Formatted string for prompt inclusion (empty string if no artifacts)
        """
        if not artifacts:
            return ""

        sections = []

        # Process all artifacts using .items()
        for key, value in artifacts.items():
            content = self._safe_decode_artifact(value)
            # Truncate if too long
            if len(content) > self.config.max_artifact_bytes:
                content = content[: self.config.max_artifact_bytes] + "\n... (truncated)"

            sections.append(f"### {key}\n```\n{content}\n```")

        if sections:
            return "## Last Execution Output\n\n" + "\n\n".join(sections)
        else:
            return ""

    def _safe_decode_artifact(self, value: Union[str, bytes]) -> str:
        """
        Safely decode an artifact value to string

        Args:
            value: Artifact value (string or bytes)

        Returns:
            String representation of the value
        """
        if isinstance(value, str):
            # Apply security filter if enabled
            if self.config.artifact_security_filter:
                return self._apply_security_filter(value)
            return value
        elif isinstance(value, bytes):
            try:
                decoded = value.decode("utf-8", errors="replace")
                if self.config.artifact_security_filter:
                    return self._apply_security_filter(decoded)
                return decoded
            except Exception:
                return f"<binary data: {len(value)} bytes>"
        else:
            return str(value)

    def _apply_security_filter(self, text: str) -> str:
        """
        Apply security filtering to artifact text

        Args:
            text: Input text

        Returns:
            Filtered text with potential secrets/sensitive info removed
        """
        import re

        # Remove ANSI escape sequences
        ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
        filtered = ansi_escape.sub("", text)

        # Basic patterns for common secrets (can be expanded)
        secret_patterns = [
            (r"[A-Za-z0-9]{32,}", "<REDACTED_TOKEN>"),  # Long alphanumeric tokens
            (r"sk-[A-Za-z0-9]{48}", "<REDACTED_API_KEY>"),  # OpenAI-style API keys
            (r"password[=:]\s*[^\s]+", "password=<REDACTED>"),  # Password assignments
            (r"token[=:]\s*[^\s]+", "token=<REDACTED>"),  # Token assignments
        ]

        for pattern, replacement in secret_patterns:
            filtered = re.sub(pattern, replacement, filtered, flags=re.IGNORECASE)

        return filtered
