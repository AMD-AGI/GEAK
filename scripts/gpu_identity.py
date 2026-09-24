#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Detect AMD GPU ISA and product identity from one structured rocminfo agent.

`gfx1201` is an ISA shared by multiple products.  This helper returns target
`r9700` only when the same GPU agent reports the exact marketing name
`AMD Radeon AI PRO R9700`; every other product remains `unknown`.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

R9700_MARKETING_NAME = "AMD Radeon AI PRO R9700"
_AGENT_RE = re.compile(r"^\s*Agent\s+\d+\s*$")
_FIELD_RE = re.compile(r"^\s*([^:]+):\s*(.*?)\s*$")
_GFX_RE = re.compile(r"^gfx[0-9a-f]+$", re.IGNORECASE)


class IdentityError(RuntimeError):
    """Raised when visible GPU agents cannot produce one safe identity."""


def _agent_fields(lines: list[str]) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in lines:
        match = _FIELD_RE.match(line)
        if match:
            # rocminfo may contain nested ISA/cache sections with their own
            # `Name:` fields later in the same Agent block. The top-level agent
            # fields occur first and are the identity source.
            fields.setdefault(
                match.group(1).strip().lower(),
                match.group(2).strip(),
            )
    return fields


def parse_rocminfo(text: str) -> dict[str, Any]:
    """Return one homogeneous visible-GPU identity from rocminfo text."""
    blocks: list[list[str]] = []
    current: list[str] | None = None
    for line in text.splitlines():
        if _AGENT_RE.match(line):
            if current is not None:
                blocks.append(current)
            current = []
        elif current is not None:
            current.append(line)
    if current is not None:
        blocks.append(current)

    gpu_agents: list[dict[str, Any]] = []
    for block in blocks:
        fields = _agent_fields(block)
        gfx = fields.get("name", "").lower()
        if not _GFX_RE.fullmatch(gfx) or gfx == "gfx000":
            continue
        marketing_name = fields.get("marketing name", "").strip()
        raw_cu = fields.get("compute unit", "")
        try:
            physical_cu_count = int(raw_cu)
        except (TypeError, ValueError):
            physical_cu_count = 0
        if physical_cu_count <= 0:
            raise IdentityError(
                f"GPU agent {gfx} has no positive physical Compute Unit count"
            )
        gpu_agents.append(
            {
                "gfx": gfx,
                "marketing_name": marketing_name,
                "target": (
                    "r9700" if marketing_name == R9700_MARKETING_NAME else "unknown"
                ),
                "physical_cu_count": physical_cu_count,
            }
        )

    if not gpu_agents:
        raise IdentityError("rocminfo reported no non-gfx000 GPU agent")

    identities = {
        (
            agent["gfx"],
            agent["target"],
            agent["marketing_name"],
            agent["physical_cu_count"],
        )
        for agent in gpu_agents
    }
    if len(identities) != 1:
        summary = ", ".join(
            f"{agent['gfx']}:{agent['target']}:{agent['marketing_name'] or '<unnamed>'}"
            for agent in gpu_agents
        )
        raise IdentityError(f"visible GPU agents have mixed identities: {summary}")

    identity = dict(gpu_agents[0])
    identity["visible_gpu_agents"] = len(gpu_agents)
    return identity


def _read_rocminfo(path: str | None) -> str:
    if path:
        proc = subprocess.run(
            [path],
            check=False,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise IdentityError(f"{path} exited {proc.returncode}")
        return proc.stdout
    binary = shutil.which("rocminfo")
    if binary is None and Path("/opt/rocm/bin/rocminfo").is_file():
        binary = "/opt/rocm/bin/rocminfo"
    if binary is None:
        raise IdentityError("rocminfo is not installed")
    return _read_rocminfo(binary)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--input", help="read captured rocminfo output from this file")
    source.add_argument("--rocminfo-bin", help="execute this rocminfo binary")
    source.add_argument("--stdin", action="store_true", help="read rocminfo output from stdin")
    args = parser.parse_args(argv)
    try:
        if args.input:
            text = Path(args.input).read_text(encoding="utf-8")
        elif args.stdin:
            text = sys.stdin.read()
        else:
            text = _read_rocminfo(args.rocminfo_bin)
        print(json.dumps(parse_rocminfo(text), sort_keys=True))
        return 0
    except (IdentityError, OSError, UnicodeError) as error:
        print(f"E_GPU_IDENTITY: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
