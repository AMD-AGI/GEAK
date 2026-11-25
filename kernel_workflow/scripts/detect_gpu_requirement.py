#!/usr/bin/env python3
"""Detect how many GPUs one run of a task needs, from its test/benchmark sources.

Usage: detect_gpu_requirement.py <file-or-dir> [...]
       detect_gpu_requirement.py --declaration <task dir>

Prints JSON: {"kind", "gpus_per_job", "evidence"} where kind is
  single    no multi-process / collective code found: 1 GPU
  fixed     an explicit rank count (largest one found) is set in the sources
  scalable  the rank count follows the visible GPUs (device_count, nproc=gpu): use the whole pool
  unknown   distributed code without a readable rank count: the caller must decide
This is a static hint: the Director confirms it against the code (e.g. a rank count passed as a
positional argument) before the workflow sizes its GPU leases. Directories are scanned two levels
deep; pass the task's own test/bench files for a large repo. What an e2e extraction captured from
the serving process (selection traces, shape captures) is skipped: its rank counts are the
server's (e.g. a TP8 world_size), not the test's. So is the runtime GEAK copies into a task
(harness_lib.py, leg_runner.py, overlay_setup.py, ut_runner.py): its own process-group code says
nothing about the task's.

A GPU count the task itself declares comes first: `required_gpu_count` in run/requested_kernel.json
or run/RETURN.json (kernel extraction), `capabilities.multi_rank.world_size` in meta.json (a generated
UT). The task dir is each directory argument, or the dir holding a file argument. A declared count
above 1 is returned as `fixed` (the largest if they disagree), since dispatch sizes the job by that
declaration; a declared 1 only settles a scan that found distributed code but no count. Everything
else is the source scan, as before.

--declaration reads only what the task declares and prints deterministic JSON
  {"required_gpu_count": <largest declared count, or null>, "ut_total_s": <seconds>, "evidence": [...]}
required_gpu_count comes from the three declarations above (the largest if they disagree, as in the
scan). ut_total_s is the whole-UT time limit, computed like ut_runner.py `timeouts(meta, world)["total_s"]`
with world = the declared count; when that is not a positive number (meta.timeouts.total_s absent, null,
not a number, or not above 0, e.g. ut_runner's 0 = "no limit" of a single-GPU task) it is the multi-rank
default 3600, and the evidence says why. (A lease run timeout of 0 would stop the command at once.)
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
from pathlib import Path

MAX_FILES = 400
MAX_BYTES = 1 << 20
SKIP_DIRS = {".git", "__pycache__", "build", "node_modules", "third_party", "3rdparty", ".torch_ext",
             "_capture"}
SUFFIXES = {".py", ".sh", ".json", ".yaml", ".yml"}
# e2e capture output: selection_trace*.json, capture_*.json, capture.pid-<pid>.rank-<rank>/
CAPTURE_PREFIXES = ("selection_trace", "capture_", "capture.")
# The runtime every UT task carries (e2e_workflow/scripts/ut_runner.py RUNTIME_FILES)
RUNTIME_FILES = frozenset(("harness_lib.py", "leg_runner.py", "overlay_setup.py", "ut_runner.py"))
# Task metadata: its numbers count without a collective in the same file. Other JSON is data.
METADATA_NAMES = {"meta.json"}
METADATA_SUFFIXES = {".yaml", ".yml"}

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
# (path under the task dir, key path) of each GPU count a task declares; absent means 1.
DECLARATIONS = (
    (("run", "requested_kernel.json"), ("required_gpu_count",)),
    (("run", "RETURN.json"), ("required_gpu_count",)),
    (("meta.json",), ("capabilities", "multi_rank", "world_size")),
)


def _task_dirs(paths):
    dirs = []
    for raw in paths:
        p = Path(raw)
        if p.is_file():  # a file names its task dir; run/<record>.json lies one level below it
            p = p.parent.parent if p.parent.name == "run" else p.parent
        if p.is_dir() and p not in dirs:
            dirs.append(p)
    return dirs


def _declared_values(task):
    """[(file, key path, value)] for each declaration `task` writes (absent or null ones are skipped)."""
    out = []
    for rel, keys in DECLARATIONS:
        f = task.joinpath(*rel)
        try:
            if f.stat().st_size > MAX_BYTES:
                continue
            value = json.loads(f.read_text(errors="replace"))
        except (OSError, ValueError):
            continue
        for key in keys:
            value = value.get(key) if isinstance(value, dict) else None
        if value is not None:
            out.append((f, keys, value))
    return out


def _gpu_count(value):
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def declared(paths):
    """[(count, evidence)] for every GPU count the task dirs of `paths` declare."""
    return [(value, f"{f}: {'.'.join(keys)} = {value} (declared)")
            for task in _task_dirs(paths) for f, keys, value in _declared_values(task) if _gpu_count(value)]


# ut_runner.py DEFAULT_TIMEOUTS_SINGLE / DEFAULT_TIMEOUTS_MULTI["total_s"]: the whole-UT limit without
# meta.timeouts (0 = none). A limit that is not a positive number becomes the multi-rank default.
UT_TOTAL_S_DEFAULT = {False: 0, True: 3600}
MULTI_RANK_UT_TOTAL_S = 3600


def _seconds(v):
    return int(v) if float(v).is_integer() else v


def ut_total_s(meta_path, world):
    """(seconds, reason): ut_runner.timeouts(meta, world)["total_s"] when it is a positive number,
    else the multi-rank default MULTI_RANK_UT_TOTAL_S."""
    fallback = f"the multi-rank default {MULTI_RANK_UT_TOTAL_S} is used"
    try:
        meta = json.loads(meta_path.read_text(errors="replace"))
    except (OSError, ValueError) as exc:
        return MULTI_RANK_UT_TOTAL_S, f"{meta_path} unreadable ({type(exc).__name__}); {fallback}"
    if not isinstance(meta, dict):
        return MULTI_RANK_UT_TOTAL_S, f"{meta_path} is not a JSON object; {fallback}"
    tmo = meta.get("timeouts") or {}
    where = f"{meta_path}: meta.timeouts.total_s"
    if not isinstance(tmo, dict):
        return MULTI_RANK_UT_TOTAL_S, f"{meta_path}: meta.timeouts is not an object ({json.dumps(tmo)}); {fallback}"
    if tmo.get("total_s") is None:
        total = UT_TOTAL_S_DEFAULT[world > 1]
        why = f"{where} {'is null' if 'total_s' in tmo else 'absent'}: ut_runner's default for world_size {world} is {total}"
    else:
        value = tmo["total_s"]
        try:
            if isinstance(value, bool):
                raise TypeError(value)
            total = float(value)
        except (TypeError, ValueError):
            return MULTI_RANK_UT_TOTAL_S, f"{where} is not a number ({json.dumps(value)}); {fallback}"
        why = f"{where} = {json.dumps(value)} (ut_runner timeouts, world_size {world})"
    if not (math.isfinite(total) and total > 0):
        return MULTI_RANK_UT_TOTAL_S, f"{why}, not a positive number of seconds; {fallback}"
    return _seconds(total), why


def declaration(path):
    """The GPU count and whole-UT time limit the task at `path` declares (see --declaration)."""
    tasks = _task_dirs([path])
    if not tasks:
        raise FileNotFoundError(f"no task dir at {path}")
    task = tasks[0]
    values = _declared_values(task)
    counts = sorted({v for _, _, v in values if _gpu_count(v)})
    evidence = [f"{f}: {'.'.join(keys)} = {json.dumps(v)} "
                f"{'(declared)' if _gpu_count(v) else '(not a positive integer: ignored)'}" for f, keys, v in values]
    if len(counts) > 1:
        evidence.insert(0, f"declared GPU counts disagree ({', '.join(map(str, counts))}); the largest is used")
    if not counts:
        evidence.append(f"{task}: no GPU count declared (run/requested_kernel.json or run/RETURN.json "
                        "required_gpu_count, meta.json capabilities.multi_rank.world_size)")
    count = counts[-1] if counts else None
    total, why = ut_total_s(task / "meta.json", count or 1)
    evidence.append(f"ut_total_s = {total}: {why}")
    return {"required_gpu_count": count, "ut_total_s": total, "evidence": evidence}


def _files(paths):
    out = []
    for raw in paths:
        p = Path(raw)
        if p.is_file():
            if not p.name.startswith(CAPTURE_PREFIXES) and p.name not in RUNTIME_FILES:
                out.append(p)
            continue
        if not p.is_dir():
            continue
        base = len(p.parts)
        for root, dirs, files in os.walk(p):
            dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and not d.startswith(CAPTURE_PREFIXES)
                             and len(Path(root).parts) - base < 2)
            out.extend(Path(root) / f for f in sorted(files)
                       if Path(f).suffix in SUFFIXES and not f.startswith(CAPTURE_PREFIXES) and f not in RUNTIME_FILES)
    return out[:MAX_FILES]


def detect(paths):
    decl = declared(paths)
    counts = sorted({v for v, _ in decl})
    if counts and counts[-1] > 1:
        evidence = [w for _, w in decl]
        if len(counts) > 1:
            evidence.insert(0, f"declared GPU counts disagree ({', '.join(map(str, counts))}); the largest is used")
        return {"kind": "fixed", "gpus_per_job": counts[-1], "evidence": evidence[:5]}
    found = _scan(paths)
    if decl and found["kind"] == "unknown":  # distributed code, no count in it: the declared 1 settles it
        return {"kind": "single", "gpus_per_job": 1, "evidence": ([w for _, w in decl] + found["evidence"])[:5]}
    return found


def _scan(paths):
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
        ranked = (f.name in METADATA_NAMES or f.suffix in METADATA_SUFFIXES
                  or any(DISTRIBUTED.search(line) for line in lines))
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
    if sys.argv[1] == "--declaration":
        if len(sys.argv) != 3:
            sys.exit(__doc__)
        try:
            print(json.dumps(declaration(sys.argv[2]), indent=1))
        except FileNotFoundError as exc:
            sys.exit(f"detect_gpu_requirement.py --declaration: {exc}")
    else:
        print(json.dumps(detect(sys.argv[1:]), indent=1))
