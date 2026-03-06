# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
# SPDX-License-Identifier: Apache-2.0

"""
MCP tool for GPU kernel profiling using metrix.

Provides hardware metrics, bottleneck classification, and factual observations.
"""

import logging
import os
from typing import Any

from metrix import Metrix

logger = logging.getLogger(__name__)


class MetrixTool:
    """MCP tool for GPU kernel profiling using metrix Python API."""

    TOOL_NAME = "metrix"
    TOOL_DESCRIPTION = "Profile GPU kernels with detailed hardware metrics and factual observations."

    # Expected metrics from metrix "quick" profile
    # Expected metrics for full (memory) profile
    EXPECTED_METRICS_FULL = [
        "duration_us",
        "memory.hbm_bandwidth_utilization",
        "memory.hbm_read_bandwidth",
        "memory.hbm_write_bandwidth",
        "memory.bytes_transferred_hbm",
        "memory.l1_hit_rate",
        "memory.l2_hit_rate",
        "memory.l2_bandwidth",
        "memory.coalescing_efficiency",
        "memory.global_load_efficiency",
        "memory.global_store_efficiency",
        "memory.lds_bank_conflicts",
    ]

    # Expected metrics for quick profile
    EXPECTED_METRICS_QUICK = [
        "duration_us",
        "memory.hbm_bandwidth_utilization",
        "memory.l2_hit_rate",
    ]

    TOOL_SCHEMA = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "Command to execute for profiling (e.g., 'python3 kernel.py --profile')",
            },
            "num_replays": {
                "type": "integer",
                "description": "Number of profiling replays for statistics",
                "default": 3,
            },
            "kernel_filter": {
                "type": "string",
                "description": "Kernel name pattern to filter (e.g., '*topk*')",
                "default": None,
            },
            "auto_select": {
                "type": "boolean",
                "description": "Automatically select main kernel. If False and no kernel_filter, returns all kernels (default: False)",
                "default": False,
            },
            "quick": {
                "type": "boolean",
                "description": "Use quick profile (3 metrics, 1 pass) for speed. If False, uses memory profile (14 metrics, 2 passes) for comprehensive insights (default: False)",
                "default": False,
            },
        },
        "required": ["command"],
    }

    def __init__(self, gpu_devices: str | list[str] = None):
        """
        Initialize MetrixTool.

        Args:
            gpu_devices: Single GPU ID (e.g., "0") or list of GPU IDs (e.g., ["0", "1", "2"]).
                        Defaults to HIP_VISIBLE_DEVICES env var or "0".
        """
        if gpu_devices is None:
            gpu_devices = os.environ.get("HIP_VISIBLE_DEVICES") or "0"

        # Normalize to list for uniform handling
        if isinstance(gpu_devices, str):
            self.gpu_devices = [gpu_devices]
        else:
            self.gpu_devices = gpu_devices

        logger.info(f"Initializing MetrixTool with GPU device(s): {self.gpu_devices}")

        self.profiler = Metrix()  # Auto-detect GPU architecture

        # Get GPU info from metrix (much more reliable than amd-smi)
        self.gpu_info_map = {}
        for device in self.gpu_devices:
            self.gpu_info_map[device] = self._get_gpu_info_from_metrix(device)
            if self.gpu_info_map[device].get("detected"):
                gpu_info = self.gpu_info_map[device]
                logger.info(
                    f"GPU {device}: {gpu_info.get('model')} ({gpu_info.get('architecture')}), "
                    f"{gpu_info.get('compute_units')} CUs, "
                    f"{gpu_info.get('peak_hbm_bandwidth_gbs'):.0f} GB/s HBM"
                )

    def _get_gpu_info_from_metrix(self, device: str) -> dict[str, Any]:
        """
        Get GPU information from metrix backend.

        Metrix auto-detects GPU specs, which is more reliable than calling amd-smi/nvidia-smi.
        """
        try:
            # Metrix backend contains device_specs with comprehensive GPU info
            if hasattr(self.profiler, "backend") and hasattr(self.profiler.backend, "device_specs"):
                specs = self.profiler.backend.device_specs
                return {
                    "detected": True,
                    "device_id": device,
                    "vendor": "AMD",  # Metrix currently supports AMD GPUs
                    "architecture": specs.arch,
                    "model": specs.name,
                    "compute_units": specs.num_cu,
                    "peak_hbm_bandwidth_gbs": specs.hbm_bandwidth_gbs,
                    "peak_l2_bandwidth_gbs": specs.l2_bandwidth_gbs,
                    "l2_size_mb": specs.l2_size_mb,
                    "lds_size_per_cu_kb": specs.lds_size_per_cu_kb,
                    "wavefront_size": specs.wavefront_size,
                    "fp32_tflops": specs.fp32_tflops,
                    "fp64_tflops": specs.fp64_tflops,
                }
        except Exception as e:
            logger.warning(f"Failed to detect GPU info for device {device}: {e}")

        return {"detected": False, "device_id": device}

    def profile(
        self,
        command: str,
        num_replays: int = 3,
        kernel_filter: str = None,
        auto_select: bool = False,
        quick: bool = False,
    ) -> dict[str, Any]:
        """
        Profile a kernel using metrix.

        Args:
            command: Command to execute for profiling.
            num_replays: Number of profiling replays.
            kernel_filter: Kernel name pattern to filter.
            auto_select: If True, automatically select main kernel. If False, return all kernels.
            quick: If True, use quick profile (3 metrics, 1 pass). If False, use memory profile (12 metrics, 2 passes).

        Returns:
            Dict with consistent structure for single or multiple GPUs:
            {
                "results": [
                    {
                        "device_id": "0",
                        "gpu_info": {...},  # GPU model, architecture, memory
                        "kernels": [{
                            "name": ...,
                            "duration_us": ...,
                            "bottleneck": ...,
                            "observations": [...],  # Factual observations
                            "metrics": {...}
                        }, ...]
                    },
                    # ... more GPUs if multiple devices
                ]
            }
        """
        profile_mode = "quick" if quick else "memory (full)"
        logger.info(
            f"Starting profiling: {len(self.gpu_devices)} GPU(s), "
            f"profile={profile_mode}, replays={num_replays}, "
            f"auto_select={auto_select}, kernel_filter={kernel_filter or 'None'}"
        )

        # Always return a list for consistency
        results_list = []
        for device in self.gpu_devices:
            result = self._profile_single_gpu(device, command, num_replays, kernel_filter, auto_select, quick)
            # Add device_id to the result
            result["device_id"] = device
            results_list.append(result)

        logger.info(
            f"Profiling complete: {sum(len(r['kernels']) for r in results_list)} total kernels across {len(self.gpu_devices)} GPU(s)"
        )
        return {"results": results_list}

    def _profile_single_gpu(
        self,
        device: str,
        command: str,
        num_replays: int,
        kernel_filter: str,
        auto_select: bool,
        quick: bool,
    ) -> dict[str, Any]:
        """Profile on a single GPU."""
        logger.info(f"Profiling GPU {device}...")

        # Set HIP_VISIBLE_DEVICES for this specific GPU
        original_hip_devices = os.environ.get("HIP_VISIBLE_DEVICES")
        os.environ["HIP_VISIBLE_DEVICES"] = device

        try:
            profile_level = "quick" if quick else "memory"
            results = self.profiler.profile(
                command=command,
                profile=profile_level,
                num_replays=num_replays,
                aggregate_by_kernel=True,
                kernel_filter=kernel_filter,
            )
        finally:
            # Restore original HIP_VISIBLE_DEVICES
            if original_hip_devices is not None:
                os.environ["HIP_VISIBLE_DEVICES"] = original_hip_devices
            elif "HIP_VISIBLE_DEVICES" in os.environ:
                del os.environ["HIP_VISIBLE_DEVICES"]

        logger.debug(f"GPU {device}: Captured {len(results.kernels)} kernel(s)")

        # Determine which kernels to process
        if auto_select and not kernel_filter:
            main_kernel = self._find_main_kernel(results.kernels)
            if not main_kernel:
                raise RuntimeError("No kernel found in profiling results")
            kernels_to_process = [main_kernel]
            logger.info(
                f"GPU {device}: Auto-selected kernel '{main_kernel.name}' "
                f"(duration: {main_kernel.duration_us.avg:.2f} μs)"
            )
        else:
            kernels_to_process = results.kernels
            if kernel_filter:
                logger.info(
                    f"GPU {device}: Processing {len(kernels_to_process)} kernel(s) matching filter '{kernel_filter}'"
                )
            else:
                logger.info(f"GPU {device}: Processing all {len(kernels_to_process)} kernel(s)")

        if not kernels_to_process:
            raise RuntimeError("No kernels found in profiling results")

        # Get GPU info for contextual observations
        gpu_info = self.gpu_info_map.get(device, {"detected": False})

        # Build kernel data (always classify for useful insights)
        kernels_data = []
        for kernel in kernels_to_process:
            metrics = self._extract_all_metrics(kernel)
            self._validate_metrics(metrics, kernel.name, quick)
            bottleneck = self._classify_bottleneck(metrics, quick)
            logger.debug(f"GPU {device}: Kernel '{kernel.name[:60]}...' classified as {bottleneck}")

            kernels_data.append(
                {
                    "name": kernel.name,
                    "duration_us": metrics["duration_us"],
                    "bottleneck": bottleneck,
                    "observations": self._generate_observations(bottleneck, metrics, quick, gpu_info),
                    "metrics": metrics,
                }
            )

        return {
            "gpu_info": self.gpu_info_map.get(device, {"detected": False}),
            "kernels": kernels_data,
        }

    def _find_main_kernel(self, kernels):
        """
        Find main user kernel by filtering framework internals and selecting longest duration.

        Returns:
            Kernel object with longest duration, or None if no user kernel found.
        """
        skip_patterns = [
            "vectorized_elementwise",
            "distribution_",
            "reduce_kernel",
            "fillBuffer",
            "copyBuffer",
            "Cijk_",
            "at::native",
        ]

        main_kernel = None
        max_duration = 0

        for kernel in kernels:
            if any(pattern in kernel.name for pattern in skip_patterns):
                continue

            if kernel.duration_us.avg > max_duration:
                max_duration = kernel.duration_us.avg
                main_kernel = kernel

        return main_kernel

    def _extract_all_metrics(self, kernel) -> dict[str, float]:
        """Extract all metrics from kernel object, including duration."""
        metrics: dict[str, float] = {}

        dur = kernel.duration_us
        metrics["duration_us"] = dur.avg
        for stat_name in ("min", "max", "median"):
            val = getattr(dur, stat_name, None)
            if val is not None:
                metrics[f"duration_us_{stat_name}"] = float(val)

        kernel_metrics = kernel.metrics if hasattr(kernel, "metrics") else {}
        for name, value in kernel_metrics.items():
            metrics[name] = float(value.avg)

        return metrics

    def _validate_metrics(self, metrics: dict[str, float], kernel_name: str, quick: bool = False) -> None:
        """Validate that expected metrics are present."""
        expected = self.EXPECTED_METRICS_QUICK if quick else self.EXPECTED_METRICS_FULL
        missing = [m for m in expected if m not in metrics]
        if missing:
            logger.error(f"Metric validation failed for '{kernel_name[:60]}...': missing {len(missing)} metric(s)")
            raise RuntimeError(
                f"Missing expected metrics for kernel '{kernel_name}': {missing}\n"
                f"Available metrics: {list(metrics.keys())}"
            )
        logger.debug(f"Validated {len(expected)} metrics for '{kernel_name[:60]}...'")

    def _classify_bottleneck(self, metrics: dict[str, float], quick: bool = False) -> str:
        """Classify bottleneck based on metrics."""
        duration_us = metrics.get("duration_us", 0)

        if duration_us < 10:  # < 10 microseconds
            return "latency"

        # If no metrics captured, can't classify
        if not metrics:
            return "balanced"

        # Extract common metrics
        hbm_bw = metrics.get("memory.hbm_bandwidth_utilization", 0)
        l2_hit = metrics.get("memory.l2_hit_rate", 0)

        # Full profile: leverage additional metrics for better classification
        if not quick:
            coalescing = metrics.get("memory.coalescing_efficiency", 100)
            lds_conflicts = metrics.get("memory.lds_bank_conflicts", 0)
            l1_hit = metrics.get("memory.l1_hit_rate", 0)
            load_eff = metrics.get("memory.global_load_efficiency", 100)
            store_eff = metrics.get("memory.global_store_efficiency", 100)

            # LDS bottleneck: high bank conflicts
            if lds_conflicts > 0.1:  # > 0.1 conflicts per instruction
                return "lds"

            # Memory bottleneck with poor coalescing
            if hbm_bw > 30 and coalescing < 50:
                return "memory"

            # Memory bottleneck with poor load/store efficiency
            if hbm_bw > 30 and (load_eff < 50 or store_eff < 50):
                return "memory"

            # Compute-bound: low HBM, high cache hits, good efficiency
            if hbm_bw < 5 and l1_hit > 80 and l2_hit > 80:
                return "compute"

        # Basic heuristics (quick mode or fallback):
        # High HBM bandwidth (>30%) → Memory-bound
        # Low HBM bandwidth (<5%) + high L2 hit (>80%) → Compute-bound (reusing data)
        # Low HBM bandwidth (<5%) → Latency-bound

        if hbm_bw > 30:
            return "memory"
        if hbm_bw > 0 and hbm_bw < 5 and l2_hit > 80:
            return "compute"
        if hbm_bw > 0 and hbm_bw < 5:
            return "latency"

        return "balanced"

    def _generate_observations(
        self,
        bottleneck: str,
        metrics: dict[str, float],
        quick: bool = False,
        gpu_info: dict[str, Any] = None,
    ) -> list[str]:
        """Generate factual observations based on metrics with GPU context."""
        logger.debug(
            f"Generating observations for {bottleneck} bottleneck "
            f"(quick={quick}, gpu_detected={gpu_info.get('detected') if gpu_info else False})"
        )
        observations = []
        gpu_info = gpu_info or {}

        hbm = metrics.get("memory.hbm_bandwidth_utilization", 0)
        l2 = metrics.get("memory.l2_hit_rate", 0)

        # Extract additional metrics if available (full profile)
        coalescing = metrics.get("memory.coalescing_efficiency", None)
        lds_conflicts = metrics.get("memory.lds_bank_conflicts", None)
        l1_hit = metrics.get("memory.l1_hit_rate", None)
        load_eff = metrics.get("memory.global_load_efficiency", None)
        store_eff = metrics.get("memory.global_store_efficiency", None)
        hbm_read_bw = metrics.get("memory.hbm_read_bandwidth", None)
        hbm_write_bw = metrics.get("memory.hbm_write_bandwidth", None)
        metrics.get("memory.l2_bandwidth", None)

        # State the bottleneck classification
        if bottleneck == "memory":
            observations.append(f"Classified as memory-bound (HBM util: {hbm:.1f}%)")

            # Add context from GPU specs
            if not quick and gpu_info.get("detected") and hbm_read_bw is not None and hbm_write_bw is not None:
                achieved_bw = hbm_read_bw + hbm_write_bw
                peak_bw = gpu_info.get("peak_hbm_bandwidth_gbs", 0)
                if peak_bw > 0:
                    observations.append(f"Achieved HBM bandwidth: {achieved_bw:.1f} GB/s (peak: {peak_bw:.0f} GB/s)")

            if not quick and coalescing is not None:
                desc = "poor" if coalescing < 50 else "good" if coalescing > 80 else "moderate"
                observations.append(f"Coalescing efficiency: {coalescing:.1f}% ({desc})")
            if not quick and load_eff is not None:
                desc = "inefficient" if load_eff < 50 else "efficient" if load_eff > 80 else "moderate"
                observations.append(f"Global load efficiency: {load_eff:.1f}% ({desc})")
            if not quick and store_eff is not None:
                desc = "inefficient" if store_eff < 50 else "efficient" if store_eff > 80 else "moderate"
                observations.append(f"Global store efficiency: {store_eff:.1f}% ({desc})")

        elif bottleneck == "compute":
            observations.append("Classified as compute-bound (high data reuse)")
            if not quick and l1_hit is not None:
                desc = "excellent" if l1_hit > 90 else "good" if l1_hit > 70 else "low"
                observations.append(f"L1 hit rate: {l1_hit:.1f}% ({desc})")
            desc = "excellent" if l2 > 90 else "good" if l2 > 70 else "low"
            observations.append(f"L2 hit rate: {l2:.1f}% ({desc})")

        elif bottleneck == "lds":
            observations.append("Classified as LDS-bound")
            if lds_conflicts is not None:
                severity = "severe" if lds_conflicts > 1.0 else "high" if lds_conflicts > 0.1 else "moderate"
                observations.append(f"Bank conflicts: {lds_conflicts:.3f} per instruction ({severity})")
            if gpu_info.get("detected"):
                lds_per_cu = gpu_info.get("lds_size_per_cu_kb", 0)
                if lds_per_cu > 0:
                    observations.append(f"LDS available: {lds_per_cu:.0f} KB per CU")

        elif bottleneck == "latency":
            observations.append(f"Classified as latency-bound (HBM util: {hbm:.1f}%, severely under-utilized)")
            if gpu_info.get("detected") and not quick:
                peak_bw = gpu_info.get("peak_hbm_bandwidth_gbs", 0)
                if peak_bw > 0 and hbm_read_bw is not None and hbm_write_bw is not None:
                    achieved_bw = hbm_read_bw + hbm_write_bw
                    observations.append(f"Using only {achieved_bw:.1f} of {peak_bw:.0f} GB/s available HBM bandwidth")
        else:
            observations.append("Classified as balanced")
            if not quick and l1_hit is not None:
                observations.append(f"L1 hit rate: {l1_hit:.1f}%")
            if l2 > 0:
                observations.append(f"L2 hit rate: {l2:.1f}%")
            if hbm > 0:
                observations.append(f"HBM utilization: {hbm:.1f}%")

        logger.debug(f"Generated {len(observations)} observation(s) for {bottleneck} bottleneck")
        return observations

    def get_tool_definition(self) -> dict[str, Any]:
        """Return MCP tool definition."""
        return {
            "name": self.TOOL_NAME,
            "description": self.TOOL_DESCRIPTION,
            "inputSchema": self.TOOL_SCHEMA,
        }

    def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Execute the tool with given arguments."""
        return self.profile(
            command=arguments["command"],
            num_replays=arguments.get("num_replays", 3),
            kernel_filter=arguments.get("kernel_filter"),
            auto_select=arguments.get("auto_select", False),
            quick=arguments.get("quick", False),
        )
