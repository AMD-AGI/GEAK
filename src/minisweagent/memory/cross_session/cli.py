"""CLI for cross-session memory management.

Usage:
  geak-memory list [--limit N]
  geak-memory stats
  geak-memory consolidate
  geak-memory export [--output FILE]
  geak-memory import FILE
  geak-memory serve [--port PORT]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _get_backend():
    from minisweagent.memory.cross_session import get_backend

    backend = get_backend()
    if backend is None:
        print("Cross-session memory is disabled. Set GEAK_CROSS_SESSION_MEMORY_URL or unset GEAK_MEMORY_NO_CROSS_SESSION.", file=sys.stderr)
        sys.exit(1)
    return backend


def cmd_list(args):
    backend = _get_backend()
    exps = backend.list_experiences(limit=args.limit)
    if not exps:
        print("No experiences stored yet.")
        return
    print(f"{'ID':<18} {'Kernel':<25} {'Category':<15} {'Bottleneck':<10} {'Speedup':<8} {'Success'}")
    print("-" * 100)
    for e in exps:
        print(
            f"{e.record_id:<18} {e.kernel_name[:24]:<25} {e.kernel_category:<15} "
            f"{e.bottleneck_type:<10} {e.best_speedup:<8.2f} {'yes' if e.success else 'no'}"
        )
    print(f"\nTotal: {len(exps)} experiences")

    skills = backend.list_skills()
    if skills:
        print(f"\n{'Skill':<50} {'Category':<15} {'Evidence':<10} {'Success Rate'}")
        print("-" * 90)
        for s in skills:
            cats = ",".join(s.kernel_categories[:2])
            print(f"{s.title[:49]:<50} {cats:<15} {s.evidence_count:<10} {s.success_rate:.0%}")


def cmd_stats(args):
    backend = _get_backend()
    stats = backend.get_stats()
    print(f"Experiences: {stats.get('experience_count', 0)}")
    print(f"Skills:      {stats.get('skill_count', 0)}")
    print(f"Success rate: {stats.get('success_rate', 0):.1%}")

    cats = stats.get("categories", {})
    if cats:
        print("\nBy category:")
        for cat, cnt in sorted(cats.items(), key=lambda x: -x[1]):
            print(f"  {cat}: {cnt}")

    bns = stats.get("bottlenecks", {})
    if bns:
        print("\nBy bottleneck:")
        for bn, cnt in sorted(bns.items(), key=lambda x: -x[1]):
            print(f"  {bn}: {cnt}")


def cmd_consolidate(args):
    backend = _get_backend()
    from minisweagent.memory.cross_session.consolidation import consolidate

    skills = consolidate(backend)
    print(f"Consolidation complete: {len(skills)} skills produced")
    for s in skills:
        print(f"  {s.title} (evidence={s.evidence_count}, success={s.success_rate:.0%})")


def cmd_export(args):
    backend = _get_backend()
    output = args.output or "geak_memory_export.jsonl"

    exps = backend.list_experiences(limit=100000)
    skills = backend.list_skills()

    with open(output, "w", encoding="utf-8") as f:
        for e in exps:
            f.write(json.dumps({"type": "experience", **e.to_dict()}, default=str) + "\n")
        for s in skills:
            f.write(json.dumps({"type": "skill", **s.to_dict()}, default=str) + "\n")

    print(f"Exported {len(exps)} experiences and {len(skills)} skills to {output}")


def cmd_import(args):
    backend = _get_backend()
    from minisweagent.memory.cross_session.schemas import ExperienceRecord, StrategySkill

    path = Path(args.file)
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(1)

    exp_count = 0
    skill_count = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
            record_type = d.pop("type", "experience")
            if record_type == "skill":
                backend.store_skill(StrategySkill.from_dict(d))
                skill_count += 1
            else:
                backend.store_experience(ExperienceRecord.from_dict(d))
                exp_count += 1
        except (json.JSONDecodeError, Exception) as exc:
            print(f"Skipping invalid line: {exc}", file=sys.stderr)

    print(f"Imported {exp_count} experiences and {skill_count} skills")


def cmd_serve(args):
    try:
        import uvicorn
    except ImportError:
        print("uvicorn is required for serve mode: pip install uvicorn fastapi", file=sys.stderr)
        sys.exit(1)

    print(f"Starting GEAK memory server on port {args.port}...")
    uvicorn.run(
        "minisweagent.memory.cross_session.server.app:app",
        host="0.0.0.0",
        port=args.port,
        log_level="info",
    )


def main():
    parser = argparse.ArgumentParser(
        prog="geak-memory",
        description="GEAK cross-session optimization memory management",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="List stored experiences and skills")
    p_list.add_argument("--limit", type=int, default=50, help="Max experiences to show")

    sub.add_parser("stats", help="Show memory statistics")
    sub.add_parser("consolidate", help="Consolidate experiences into strategy skills")

    p_export = sub.add_parser("export", help="Export memory to JSONL file")
    p_export.add_argument("--output", "-o", default=None, help="Output file path")

    p_import = sub.add_parser("import", help="Import memory from JSONL file")
    p_import.add_argument("file", help="JSONL file to import")

    p_serve = sub.add_parser("serve", help="Start shared memory server")
    p_serve.add_argument("--port", type=int, default=8642, help="Server port")

    args = parser.parse_args()

    handlers = {
        "list": cmd_list,
        "stats": cmd_stats,
        "consolidate": cmd_consolidate,
        "export": cmd_export,
        "import": cmd_import,
        "serve": cmd_serve,
    }
    handlers[args.command](args)


if __name__ == "__main__":
    main()
