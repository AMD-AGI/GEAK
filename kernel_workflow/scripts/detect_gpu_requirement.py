#!/usr/bin/env python3
"""Detect how many GPUs one run of a task needs, from its test/benchmark sources.

Usage: detect_gpu_requirement.py <file-or-dir> [...]

Prints JSON: {"kind", "gpus_per_job", "evidence"} where kind is
  single    no multi-process / collective code found: 1 GPU
  fixed     an explicit rank count (largest one found) is set in the sources
  scalable  the rank count follows the visible GPUs (device_count, nproc=gpu): use the whole pool
  unknown   distributed code without a readable rank count: the caller must decide
This is a static hint: the Director confirms it against the code (e.g. a rank count passed as a
positional argument) before the workflow sizes its GPU leases. Directories are scanned two levels
deep; pass the task's own test/bench files for a large repo.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

MAX_FILES = 400
MAX_BYTES = 1 << 20
SKIP_DIRS = {".git", "__pycache__", "build", "node_modules", "third_party", "3rdparty", ".torch_ext"}
SUFFIXES = {".py", ".sh", ".json", ".yaml", ".yml"}

_RANK_KEYS = r"(?:world[_-]?size|nproc(?:s|[_-]per[_-]node)?|num[_-]?gpus|n[_-]?gpus|gpus[_-]per[_-]job)"
FIXED = [
    re.compile(r"--nproc[_-]per[_-]node[= ]+(\d+)"),
    re.compile(r"\bnprocs\s*=\s*(\d+)"),
    re.compile(r"\bprocesses\s*=\s*(\d+)"),
    re.compile(rf"""\b{_RANK_KEYS}\b["']?\s*[:=]\s*(\d+)\b""", re.IGNORECASE),
    re.compile(r"""["']WORLD_SIZE["']\s*[,:=]\s*["']?(\d+)"""),
    re.compile(rf"""add_argument\(\s*["']--{_RANK_KEYS}["'][^)]*?\bdefault\s*=\s*(\d+)""", re.IGNORECASE),
]
SCALABLE = [
    re.compile(r"--nproc[_-]per[_-]node[= ]+(?:gpu|auto)\b"),
    re.compile(r"\bnprocs\s*=\s*torch\.cuda\.device_count\(\)"),
    re.compile(rf"\b{_RANK_KEYS}\b\s*=\s*torch\.cuda\.device_count\(\)", re.IGNORECASE),
]
DISTRIBUTED = re.compile(
    r"\b(?:init_process_group|torchrun|mp\.spawn|multiprocessing\.spawn|all_reduce|all_gather|"
    r"reduce_scatter|all_to_all|shmem_init|mori\.shmem|torch\.distributed|init_dist\w*|"
    r"get_distributed_init_method|parallel_state)\b"
)


def _files(paths):
    out = []
    for raw in paths:
        p = Path(raw)
        if p.is_file():
            out.append(p)
            continue
        if not p.is_dir():
            continue
        base = len(p.parts)
        for root, dirs, files in os.walk(p):
            dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and len(Path(root).parts) - base < 2)
            out.extend(Path(root) / f for f in sorted(files) if Path(f).suffix in SUFFIXES)
    return out[:MAX_FILES]


def detect(paths):
    fixed, scalable, markers = [], [], []
    for f in _files(paths):
        try:
            if f.stat().st_size > MAX_BYTES:
                continue
            lines = f.read_text(errors="replace").splitlines()
        except OSError:
            continue
        # A number only counts as a rank count in a file that is itself multi-process
        # (a collective, a launcher) or in task metadata; elsewhere it is unrelated.
        ranked = f.suffix in {".json", ".yaml", ".yml"} or any(DISTRIBUTED.search(line) for line in lines)
        for no, line in enumerate(lines, 1):
            where = f"{f}:{no}: {line.strip()[:120]}"
            if DISTRIBUTED.search(line):
                markers.append(where)
            if not ranked:
                continue
            for pat in FIXED:
                for m in pat.finditer(line):
                    if int(m.group(1)) > 1:
                        fixed.append((int(m.group(1)), where))
            if any(p.search(line) for p in SCALABLE):
                scalable.append(where)
    if fixed:
        n = max(v for v, _ in fixed)
        return {"kind": "fixed", "gpus_per_job": n,
                "evidence": list(dict.fromkeys(w for v, w in fixed if v == n))[:5]}
    if scalable:
        return {"kind": "scalable", "gpus_per_job": 0, "evidence": scalable[:5]}
    if markers:
        return {"kind": "unknown", "gpus_per_job": 0, "evidence": markers[:5]}
    return {"kind": "single", "gpus_per_job": 1, "evidence": []}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    print(json.dumps(detect(sys.argv[1:]), indent=1))
