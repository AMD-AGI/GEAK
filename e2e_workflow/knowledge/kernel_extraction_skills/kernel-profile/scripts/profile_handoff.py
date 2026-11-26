#!/usr/bin/env python3
"""Export or verify the profile -> UT handoff (`profile_handoff.json`).

kernel-ut reads only this file and the artifacts it hashes. It never re-profiles, so the
handoff must say which kernels were selected, what each one's ranking row recorded (shares,
phases, shapes, source frames, gaps) and which evidence those claims rest on.

  export  EVAL_DIR/{kernel_workload,kernel_ranking,selection_targets}.json -> profile_handoff.json
  verify  every hashed artifact is unchanged and the handoff still re-derives from it
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ARTIFACTS = ("kernel_workload.json", "kernel_ranking.json", "selection_targets.json")
INVENTORY = ("selection_scope", "candidate_inventory", "coverage_summary")


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load(path):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        raise ValueError(f"missing profile artifact {path}") from None
    except json.JSONDecodeError as error:
        raise ValueError(f"{path} is not JSON: {error}") from None


def build(eval_dir):
    eval_dir = Path(eval_dir).resolve()
    docs = {name: _load(eval_dir / name) for name in ARTIFACTS}
    ranking = docs["kernel_ranking.json"]
    missing = [key for key in INVENTORY if key not in ranking]
    if missing:
        raise ValueError("kernel_ranking.json lacks %s; see selection-coverage.md "
                         "'Inventory and completion evidence'" % ", ".join(missing))
    rows = {}
    for row in ranking.get("ranking") or []:
        rows[row.get("id", row.get("name"))] = row
    targets = docs["selection_targets.json"].get("targets")
    if not isinstance(targets, list):
        raise ValueError("selection_targets.json has no targets list (run select_targets.py)")
    kernels = []
    for target in targets:
        row = rows.get(target.get("id"))
        if row is None:
            raise ValueError(f"selected target {target.get('id')!r} is not a kernel_ranking.json row")
        kernels.append({"id": target["id"], "selected_phases": target.get("selected_phases", []),
                        "phase_ranks": target.get("phase_ranks", {}),
                        "overall_pct": target.get("overall_pct"), "ranking_row": row})
    return {
        "schema": "kernel_profile_handoff/v1",
        "eval_dir": str(eval_dir),
        "artifacts": {name: {"path": name, "sha256": _sha256(eval_dir / name)} for name in ARTIFACTS},
        "policy": docs["selection_targets.json"].get("policy", {}),
        "denominators": ranking.get("denominators", {}),
        "coverage_summary": ranking["coverage_summary"],
        "kernels": kernels,
    }


def verify(handoff_path):
    handoff_path = Path(handoff_path).resolve()
    handoff = _load(handoff_path)
    eval_dir = handoff_path.parent
    for name, record in (handoff.get("artifacts") or {}).items():
        actual = _sha256(eval_dir / record["path"])
        if actual != record["sha256"]:
            raise ValueError(f"{name} changed since export ({actual} != {record['sha256']})")
    fresh = build(eval_dir)
    fresh["eval_dir"] = handoff.get("eval_dir")
    if fresh != handoff:
        raise ValueError("profile_handoff.json does not re-derive from its artifacts; re-export it")
    return handoff


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    export = sub.add_parser("export")
    export.add_argument("--eval-dir", required=True, type=Path)
    export.add_argument("--out", type=Path, help="default: EVAL_DIR/profile_handoff.json")
    check = sub.add_parser("verify")
    check.add_argument("--handoff", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.cmd == "export":
            out = args.out or args.eval_dir / "profile_handoff.json"
            if out.resolve().parent != args.eval_dir.resolve():
                parser.error("the handoff must sit in EVAL_DIR next to the artifacts it hashes")
            handoff = build(args.eval_dir)
            out.write_text(json.dumps(handoff, indent=2, allow_nan=False) + "\n")
            print(json.dumps({"handoff": str(out), "kernels": len(handoff["kernels"])}))
        else:
            handoff = verify(args.handoff)
            print(json.dumps({"ok": True, "kernels": len(handoff["kernels"])}))
    except ValueError as error:
        print(f"profile_handoff: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
