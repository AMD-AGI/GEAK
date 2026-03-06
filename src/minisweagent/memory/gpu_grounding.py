"""GPU Hardware Grounding Module.

Auto-detects GPU architecture and specs at runtime, injecting them into
the agent's context so it can make architecture-aware decisions from step 1.

Also provides the correct gpu_architecture string for memory DB records,
fixing the gfx942 hardcode bug.

Inspired by GPU Kernel Scientist (2506.20807) which emphasizes hardware
grounding as the first step in kernel optimization.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from functools import lru_cache


@dataclass
class GPUSpecs:
    """Detected GPU hardware specifications."""
    architecture: str  # e.g., "gfx942", "gfx950", "gfx90a"
    arch_family: str   # e.g., "gfx94x", "gfx90x" (for memory scoping)
    model_name: str    # e.g., "MI300X", "MI325X", "MI250X"
    compute_units: int
    peak_hbm_bandwidth_gbs: float
    lds_size_per_cu_kb: int
    wavefront_size: int
    fp32_tflops: float
    num_gpus: int

    def format_for_prompt(self) -> str:
        """Format GPU specs for injection into agent task prompt."""
        return (
            f"--- GPU Hardware (auto-detected) ---\n"
            f"Architecture: {self.architecture} ({self.model_name})\n"
            f"Compute Units: {self.compute_units}\n"
            f"Peak HBM Bandwidth: {self.peak_hbm_bandwidth_gbs:.0f} GB/s\n"
            f"LDS per CU: {self.lds_size_per_cu_kb} KB\n"
            f"Wavefront Size: {self.wavefront_size}\n"
            f"FP32 Peak: {self.fp32_tflops:.1f} TFLOPS\n"
            f"GPUs Available: {self.num_gpus}\n"
            f"---"
        )

    def format_for_memory_tag(self) -> str:
        """Return architecture string for memory DB records."""
        return self.architecture


# Known GPU specs database
_GPU_SPECS_DB = {
    "gfx942": GPUSpecs(
        architecture="gfx942", arch_family="gfx94x", model_name="MI300X",
        compute_units=304, peak_hbm_bandwidth_gbs=5300.0,
        lds_size_per_cu_kb=64, wavefront_size=64, fp32_tflops=163.4, num_gpus=0,
    ),
    "gfx950": GPUSpecs(
        architecture="gfx950", arch_family="gfx94x", model_name="MI325X",
        compute_units=256, peak_hbm_bandwidth_gbs=6000.0,
        lds_size_per_cu_kb=64, wavefront_size=64, fp32_tflops=163.4, num_gpus=0,
    ),
    "gfx90a": GPUSpecs(
        architecture="gfx90a", arch_family="gfx90x", model_name="MI250X",
        compute_units=220, peak_hbm_bandwidth_gbs=3200.0,
        lds_size_per_cu_kb=64, wavefront_size=64, fp32_tflops=95.7, num_gpus=0,
    ),
    "gfx908": GPUSpecs(
        architecture="gfx908", arch_family="gfx90x", model_name="MI100",
        compute_units=120, peak_hbm_bandwidth_gbs=1200.0,
        lds_size_per_cu_kb=64, wavefront_size=64, fp32_tflops=23.1, num_gpus=0,
    ),
}


@lru_cache(maxsize=1)
def detect_gpu_specs() -> GPUSpecs:
    """Auto-detect GPU architecture and specs via rocm-smi.

    Returns GPUSpecs with detected values, falling back to known specs DB.
    """
    arch = "unknown"
    num_gpus = 0

    try:
        result = subprocess.run(
            ["rocm-smi", "--showid"],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode == 0:
            # Count GPUs
            gpu_lines = re.findall(r'GPU\[\d+\]', result.stdout)
            num_gpus = len(set(gpu_lines))

            # Extract GFX version
            gfx_match = re.search(r'GFX Version:\s*(gfx\w+)', result.stdout)
            if gfx_match:
                arch = gfx_match.group(1)
            else:
                # Try Device ID mapping
                devid_match = re.search(r'Device ID:\s*0x(\w+)', result.stdout)
                if devid_match:
                    devid = devid_match.group(1).lower()
                    # Known device ID mappings
                    if devid in ("74a1", "74a0"):
                        arch = "gfx942"
                    elif devid in ("75a3", "75a0", "75a1"):
                        arch = "gfx950"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # Try alternative detection via torch
    if arch == "unknown":
        try:
            import torch
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                gcn_arch = getattr(props, 'gcnArchName', '')
                if gcn_arch:
                    arch = gcn_arch
                num_gpus = torch.cuda.device_count()
        except Exception:
            pass

    # Look up known specs or create minimal entry
    if arch in _GPU_SPECS_DB:
        specs = GPUSpecs(**{**_GPU_SPECS_DB[arch].__dict__})
        specs.num_gpus = num_gpus
        return specs

    # Unknown arch -- return minimal specs
    arch_family = arch[:5] + "x" if len(arch) >= 5 else "unknown"
    return GPUSpecs(
        architecture=arch, arch_family=arch_family, model_name="Unknown",
        compute_units=0, peak_hbm_bandwidth_gbs=0, lds_size_per_cu_kb=64,
        wavefront_size=64, fp32_tflops=0, num_gpus=num_gpus,
    )


def get_architecture_for_memory() -> str:
    """Get the GPU architecture string for memory DB records."""
    return detect_gpu_specs().architecture


def get_arch_family_for_retrieval() -> str:
    """Get the architecture family for memory retrieval scoping."""
    return detect_gpu_specs().arch_family
