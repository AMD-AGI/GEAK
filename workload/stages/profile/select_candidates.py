#!/usr/bin/env python3
"""Pass the slot's configured discovery scope to an unchanged skill selector."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


def selector_args(params):
    top_k = params.get("selection_top_k", 10)
    threshold = params.get("selection_overall_pct_gt", 0.5)
    phases = params.get("selection_phases", ["PREFILL", "DECODE"])
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("selection_top_k must be a positive integer")
    if (isinstance(threshold, bool) or not isinstance(threshold, (int, float))
            or not math.isfinite(threshold) or not 0 <= threshold <= 100):
        raise ValueError("selection_overall_pct_gt must be finite and in [0, 100]")
    if (not isinstance(phases, list) or not phases
            or any(not isinstance(p, str) or not p.strip() for p in phases)
            or len(set(phases)) != len(phases)):
        raise ValueError("selection_phases must be a nonempty list of unique phase names")
    return ["--top-k", str(top_k), "--overall-pct-gt", str(threshold), "--phases", *phases]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranking", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source-map", type=Path,
                        help="reviewed operator attribution from kernel-extract")
    args = parser.parse_args()
    try:
        out = Path(os.environ["OUT"])
        skill = Path(os.environ["SKILL_ROOT"])
        params = json.loads(Path(os.environ["PARAMS"]).read_text())
        ranking = args.ranking or out / "kernel_ranking.json"
        output = args.output or out / "selection_candidates.json"
        if ranking.resolve() == output.resolve():
            raise ValueError("candidate selection must not overwrite the ranking")
        selector = skill / "scripts/select_targets.py"
        arguments = selector_args(params)
        invocation = {"ranking_sha256": hashlib.sha256(ranking.read_bytes()).hexdigest(),
                      "selector_sha256": hashlib.sha256(selector.read_bytes()).hexdigest(),
                      "arguments": arguments}
        if args.source_map is not None:
            invocation['source_map_sha256'] = hashlib.sha256(args.source_map.read_bytes()).hexdigest()
            arguments += ['--source-map', str(args.source_map.resolve())]
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            previous = json.loads(output.read_text())
            if previous.get("selection_invocation") == invocation:
                print("Reusing compatible candidate selection:", output)
                return 0
            parser.error("candidate output already exists; reuse it or choose a new --output")
        pending = output.with_name(output.name + ".%d.pending" % time.time_ns())
        command = [sys.executable, str(selector), str(ranking), "--output", str(pending), *arguments]
        result = subprocess.run(command)
        if result.returncode == 0:
            document = json.loads(pending.read_text())
            document["selection_invocation"] = invocation
            pending.write_text(json.dumps(document, indent=2) + "\n")
            pending.replace(output)
        return result.returncode
    except (OSError, KeyError, ValueError, TypeError) as exc:
        parser.exit(1, "Candidate selection: %s\n" % exc)


if __name__ == "__main__":
    raise SystemExit(main())
