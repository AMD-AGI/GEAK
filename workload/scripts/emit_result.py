#!/usr/bin/env python3
"""Write `$OUT/result.json` — a stage's only outward interface (see ../stages/CONTRACT.md §3).

It is both a library and a CLI, so Python implementations can `from emit_result import
emit` while shell implementations simply call it:

    python3 "$WORKLOAD_ROOT/scripts/emit_result.py" --status ok \\
        --artifact baseline=baseline.json --metric throughput=446.87889 \\
        --notes "adopted the 2026-09-16 receipt"

It validates locally before writing, so a result.json that stage.py would certainly
reject never reaches disk.
"""

import argparse
import json
import os
import sys

VALID_STATUS = ("ok", "blocked", "fail")


def read_json(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except (IOError, ValueError):
        if default is None:
            return {}
        return default


def emit(status, artifacts=None, metrics=None, notes="", out=None,
         stage=None, skill=None):
    """Write result.json. Returns a process exit code: ok -> 0, anything else -> 1."""
    out = out or os.environ.get("OUT")
    stage = stage or os.environ.get("STAGE")
    skill = skill or os.environ.get("STAGE_SKILL")
    if not out or not stage or not skill:
        print("missing OUT/STAGE/STAGE_SKILL; a stage must run inside the environment "
              "prepared by stage.py prepare", file=sys.stderr)
        return 2
    if status not in VALID_STATUS:
        print("status must be one of %s" % list(VALID_STATUS), file=sys.stderr)
        return 2

    artifacts = dict(artifacts or {})
    metrics = dict(metrics or {})
    notes = (notes or "").strip()

    problems = []
    for name, rel in artifacts.items():
        if os.path.isabs(rel):
            problems.append("artifacts.%s is the absolute path %r; the contract requires "
                            "a path relative to $OUT" % (name, rel))
        elif not os.path.exists(os.path.join(out, rel)):
            problems.append("artifacts.%s points at %r, which does not exist" % (name, rel))
    if status == "ok" and not artifacts:
        problems.append("status=ok but there are no artifacts")
    if status in ("blocked", "fail") and not notes:
        problems.append("status=%s requires notes" % status)
    if problems:
        for p in problems:
            print("ERROR %s" % p, file=sys.stderr)
        return 2

    doc = {"stage": stage, "skill": skill, "status": status,
           "artifacts": artifacts, "metrics": metrics}
    if notes:
        doc["notes"] = notes

    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "result.json")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)
    print("[result] %s/%s -> %s" % (stage, skill, status))
    if notes:
        print("[result] %s" % notes)
    return 0 if status == "ok" else 1


def _kv(pairs, numeric=False):
    out = {}
    for p in pairs or []:
        if "=" not in p:
            raise SystemExit("expected key=value, got %r" % p)
        k, v = p.split("=", 1)
        if numeric:
            try:
                v = json.loads(v)
            except ValueError:
                pass
        out[k] = v
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--status", required=True, choices=VALID_STATUS)
    ap.add_argument("--artifact", action="append",
                    help="name=path-relative-to-$OUT, repeatable")
    ap.add_argument("--metric", action="append", help="name=value, repeatable")
    ap.add_argument("--notes", default="")
    args = ap.parse_args(argv)
    return emit(args.status, _kv(args.artifact), _kv(args.metric, numeric=True),
                args.notes)


if __name__ == "__main__":
    sys.exit(main())
