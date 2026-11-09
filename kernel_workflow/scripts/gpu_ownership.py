#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check whether inherited GEAK ownership matches a requested GPU spec.

This checks the caller contract exported by gpu_lock.sh, gpu_lease.py and the
e2e gpu_lock_set.py wrapper. It does not acquire locks or prove ownership of a
physical device. Consumers run under the existing outer lock and its source
provenance guard; they must not use these markers to bypass that entry point.

Exit 0: matching ownership, run directly. Exit 1: no ownership, acquire through
gpu_lock.sh. Exit 2: inconsistent or mismatched ownership, refuse execution.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Mapping, Optional, Tuple

from gpu_lease import GpuRequest

# gpu_lease.py supports Python 3.8; keep this adjacent helper compatible too.
# ruff: noqa: UP006, UP045


def _ids(value: str) -> Tuple[int, ...]:
    # Use the existing allocator's ID parser and validation, without acquiring a lease.
    return GpuRequest.from_pool(value, count=1).pool_ids


def _request(spec: str) -> GpuRequest:
    if spec.startswith("group:"):
        value = spec[len("group:"):]
        return GpuRequest.from_fixed_ids(value, count=len(_ids(value)))
    if spec.startswith("pool:"):
        parts = spec.split(":", 2)
        if len(parts) != 3 or not parts[1].isascii() or not parts[1].isdigit():
            raise ValueError("expected pool:<count>:<ids>")
        return GpuRequest.from_pool(parts[2], count=int(parts[1]))
    return GpuRequest.from_pool(spec, count=1)


def held_ids(environ: Mapping[str, str]) -> Optional[Tuple[int, ...]]:
    """Resolve complete group markers before the last single-card lock marker."""
    active = environ.get("GEAK_GPU_LEASE_ACTIVE", "0")
    leased = environ.get("GEAK_GPU_LEASE_IDS", "")
    locked_set = environ.get("GEAK_GPU_LOCKED_SET", "")
    single = environ.get("GEAK_GPU_LOCK_HELD", "")
    if not leased and not locked_set and not single and active in ("", "0"):
        return None
    if active not in ("", "0", "1"):
        raise ValueError("invalid GEAK_GPU_LEASE_ACTIVE")
    if bool(leased) != (active == "1"):
        raise ValueError("GEAK_GPU_LEASE_ACTIVE=1 requires GEAK_GPU_LEASE_IDS, and vice versa")
    ids = _ids(leased) if leased else (_ids(locked_set) if locked_set else _ids(single))
    if locked_set and _ids(locked_set) != ids:
        raise ValueError("GEAK_GPU_LOCKED_SET conflicts with the active lease")
    if single:
        single_ids = _ids(single)
        if len(single_ids) != 1 or single_ids[0] not in ids:
            raise ValueError("GEAK_GPU_LOCK_HELD conflicts with the held GPU set")
    visible = environ.get("HIP_VISIBLE_DEVICES", "")
    if not visible or _ids(visible) != ids:
        raise ValueError("HIP_VISIBLE_DEVICES does not match the held GPU IDs and order")
    return ids


def check_ownership(spec: str = "", environ: Optional[Mapping[str, str]] = None) -> bool:
    """Return False only when no ownership is inherited; raise on an unsafe mismatch."""
    env = os.environ if environ is None else environ
    ids = held_ids(env)
    if ids is None:
        return False
    requested = _request(spec) if spec else None
    allowed = env.get("GEAK_GPU_ALLOWED", "")
    if allowed:
        allocation = set(_ids(allowed))
        if not set(ids).issubset(allocation):
            raise ValueError("held GPU IDs are outside GEAK_GPU_ALLOWED")
        if requested is not None and not set(requested.pool_ids).issubset(allocation):
            raise ValueError("requested GPU spec is outside GEAK_GPU_ALLOWED")
    if requested is not None:
        if requested.count != len(ids) or not set(ids).issubset(requested.pool_ids):
            raise ValueError("requested GPU spec does not match the held GPU set")
        if requested.fixed_ids is not None and requested.fixed_ids != ids:
            raise ValueError("requested fixed GPU group order differs from the held group")
    return True


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-spec", default="", help="single id, comma pool, group:<ids>, or pool:<n>:<ids>")
    args = parser.parse_args(argv)
    try:
        return 0 if check_ownership(args.gpu_spec) else 1
    except ValueError as exc:
        print(f"ERROR: GPU ownership: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
