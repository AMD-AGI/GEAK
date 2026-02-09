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

    def _format_baseline_profiling(self, profiling: Dict[str, Any]) -> str:
        """
        Format baseline hardware profiling data from Metrix into a human-readable
        string for inclusion in the LLM prompt.

        Args:
            profiling: Dictionary of profiling metrics from Metrix/rocprofv3.

        Returns:
            Formatted profiling summary string with bottleneck analysis.
        """
        sections = []

        # --- Latency ---
        latency_keys = ["latency_us", "latency_min_us", "latency_avg_us", "latency_max_us"]
        latency_parts = []
        for key in latency_keys:
            if key in profiling:
                label = key.replace("_us", "").replace("latency_", "").capitalize() or "Latency"
                latency_parts.append(f"  - {label}: {profiling[key]:.2f} us")
        if latency_parts:
            sections.append("## Latency\n" + "\n".join(latency_parts))

        # --- Memory Performance ---
        mem_keys = {
            "bandwidth_gb_s": "Bandwidth",
            "hbm_utilization": "HBM Utilization",
            "hbm_read_gb_s": "HBM Read",
            "hbm_write_gb_s": "HBM Write",
            "global_load_efficiency": "Global Load Efficiency",
            "global_store_efficiency": "Global Store Efficiency",
            "coalescing_efficiency": "Coalescing Efficiency",
            "lds_bank_conflicts": "LDS Bank Conflicts",
        }
        mem_parts = []
        for key, label in mem_keys.items():
            if key in profiling:
                val = profiling[key]
                if isinstance(val, float):
                    # Format percentages vs raw numbers
                    if "efficiency" in key or "utilization" in key:
                        mem_parts.append(f"  - {label}: {val:.1f}%")
                    else:
                        mem_parts.append(f"  - {label}: {val:.2f} GB/s")
                else:
                    mem_parts.append(f"  - {label}: {val}")
        if mem_parts:
            sections.append("## Memory Performance\n" + "\n".join(mem_parts))

        # --- Compute Performance ---
        compute_keys = {
            "tflops": "TFLOPS",
            "compute_busy": "GPU Compute Busy",
            "valu_busy": "VALU Busy",
            "mfma_busy": "MFMA Busy",
        }
        compute_parts = []
        for key, label in compute_keys.items():
            if key in profiling:
                val = profiling[key]
                if "busy" in key:
                    compute_parts.append(f"  - {label}: {val:.1f}%")
                else:
                    compute_parts.append(f"  - {label}: {val:.2f}")
        if compute_parts:
            sections.append("## Compute Performance\n" + "\n".join(compute_parts))

        # --- Cache Performance ---
        cache_keys = {
            "l1_hit_rate": "L1 Cache Hit Rate",
            "l2_hit_rate": "L2 Cache Hit Rate",
            "l2_read_hit_rate": "L2 Read Hit Rate",
            "l2_write_hit_rate": "L2 Write Hit Rate",
        }
        cache_parts = []
        for key, label in cache_keys.items():
            if key in profiling:
                cache_parts.append(f"  - {label}: {profiling[key]:.1f}%")
        if cache_parts:
            sections.append("## Cache Performance\n" + "\n".join(cache_parts))

        # --- Bottleneck Analysis ---
        bottleneck = self._analyze_bottleneck(profiling)
        if bottleneck:
            sections.append(f"## Bottleneck Analysis\n{bottleneck}")

        if not sections:
            return "No detailed profiling data available."

        return "\n\n".join(sections)

    def _analyze_bottleneck(self, profiling: Dict[str, Any]) -> str:
        """
        Automatically analyze profiling data to identify the primary bottleneck.
        """
        analysis = []

        compute_busy = profiling.get("compute_busy", None)
        hbm_util = profiling.get("hbm_utilization", None)
        bandwidth = profiling.get("bandwidth_gb_s", None)
        coalescing = profiling.get("coalescing_efficiency", None)
        l2_hit = profiling.get("l2_hit_rate", None)

        if compute_busy is not None and hbm_util is not None:
            if compute_busy < 30 and hbm_util > 60:
                analysis.append(
                    "**Memory-bound**: GPU compute is underutilized while memory bandwidth "
                    "is high. Focus on reducing memory traffic, improving data reuse, "
                    "and using shared memory (LDS)."
                )
            elif compute_busy > 60 and hbm_util < 30:
                analysis.append(
                    "**Compute-bound**: Memory bandwidth is underutilized while compute "
                    "is saturated. Focus on algorithmic improvements, reducing redundant "
                    "computation, and increasing parallelism."
                )
            elif compute_busy < 30 and hbm_util < 30:
                analysis.append(
                    "**Latency-bound**: Both compute and memory are underutilized. "
                    "The kernel may be limited by synchronization, launch overhead, "
                    "or low occupancy. Focus on increasing occupancy and reducing barriers."
                )

        if coalescing is not None and coalescing < 50:
            analysis.append(
                f"**Poor memory coalescing** ({coalescing:.0f}%): Memory accesses are "
                "not well-coalesced. Reorganize data layout or access patterns."
            )

        if l2_hit is not None and l2_hit < 30:
            analysis.append(
                f"**Low L2 cache hit rate** ({l2_hit:.0f}%): Data is not being reused "
                "effectively. Consider tiling strategies or adjusting block sizes."
            )

        return "\n".join(analysis) if analysis else ""

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
