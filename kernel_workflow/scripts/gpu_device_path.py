#!/usr/bin/env python3
"""Resolve HIP device zero under the caller's selected mask to exposed DRM sysfs.

The lock wrapper sets HIP_VISIBLE_DEVICES exactly as for its eventual command,
preserving ROCR and other runtime settings. No kernel or tensor is launched.
"""

import ctypes
import os
from pathlib import Path
import re
import stat
import sys


def normalize_bdf(value):
    match = re.fullmatch(r"([0-9a-fA-F]{4,8}):([0-9a-fA-F]{2}):([0-9a-fA-F]{2})\.([0-7])", value)
    if match is None:
        raise ValueError(f"invalid HIP PCI bus ID: {value!r}")
    domain, bus, device, function = (int(part, 16) for part in match.groups())
    return f"{domain:04x}:{bus:02x}:{device:02x}.{function}"


def selected_bdf():
    try:
        hip = ctypes.CDLL("libamdhip64.so")
    except OSError:
        hip = ctypes.CDLL("/opt/rocm/lib/libamdhip64.so")
    hip.hipGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
    hip.hipDeviceGetPCIBusId.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
    count = ctypes.c_int()
    status = hip.hipGetDeviceCount(ctypes.byref(count))
    if status != 0 or count.value != 1:
        raise RuntimeError(f"selected mask must expose exactly one HIP device: status={status}, count={count.value}")
    value = ctypes.create_string_buffer(64)
    status = hip.hipDeviceGetPCIBusId(value, len(value), 0)
    if status != 0:
        raise RuntimeError(f"hipDeviceGetPCIBusId failed: {status}")
    return normalize_bdf(value.value.decode("ascii"))


def device_path(bdf, drm_root=Path("/sys/class/drm"), dev_root=Path("/dev/dri")):
    matches = []
    for render in drm_root.glob("renderD*"):
        target = (render / "device").resolve(strict=True)
        if target.name.lower() != bdf:
            continue
        node = (dev_root / render.name).stat()
        major, minor = (int(value) for value in (render / "dev").read_text().strip().split(":"))
        if not stat.S_ISCHR(node.st_mode) or node.st_rdev != os.makedev(major, minor):
            raise RuntimeError(f"exposed DRM node does not match sysfs: {render.name}")
        if not all((target / name).is_file() for name in ("gpu_busy_percent", "mem_info_vram_used")):
            raise RuntimeError(f"idle counters unavailable for {bdf}")
        matches.append(target)
    if len(matches) != 1:
        raise RuntimeError(f"expected one exposed DRM device for {bdf}, found {len(matches)}")
    return matches[0]


if __name__ == "__main__":
    try:
        print(device_path(selected_bdf()))
    except (OSError, ValueError, RuntimeError) as error:
        print(f"GPU identity error: {error}", file=sys.stderr)
        sys.exit(1)
