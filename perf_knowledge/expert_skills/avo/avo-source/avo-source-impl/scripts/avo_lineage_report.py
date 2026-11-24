"""Export an AVO lineage (``lineage.json``) to a readable version table.

Each committed version (``avo-vN``) is one row: speedup, latency (+ delta vs
baseline), strategy, parent, and per-shape speedups. Useful for comparing all
versions of a run at a glance without walking git history.

Usage:
    python scripts/avo_lineage_report.py <path>
      <path> may be a lineage.json file OR a directory containing one
             (searched recursively, e.g. an avo state dir or output dir).

    # formats (default: pretty table to stdout)
    python scripts/avo_lineage_report.py <path> --format md   > lineage.md
    python scripts/avo_lineage_report.py <path> --format csv  > lineage.csv

    --best-only   show only the row with the highest speedup

    # strategy ledger (every strategy TRIED, incl. failures) from attempts.jsonl
    python scripts/avo_lineage_report.py <state_dir> --strategies --format md > strategy_ledger.md
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path


def _find_lineage(path: Path) -> Path | None:
    """Resolve a lineage.json from a file or (recursively) a directory."""
    if path.is_file():
        return path
    if path.is_dir():
        direct = path / "lineage.json"
        if direct.is_file():
            return direct
        matches = sorted(path.rglob("lineage.json"))
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            print(f"error: multiple lineage.json under {path}; pass an exact one:", file=sys.stderr)
            for m in matches:
                print(f"  {m}", file=sys.stderr)
            raise SystemExit(2)
    return None


def _find_attempts(path: Path) -> Path | None:
    """Resolve an attempts.jsonl from a file or (recursively) a directory."""
    if path.is_file():
        if path.name.endswith(".jsonl"):
            return path
        sib = path.parent / "attempts.jsonl"
        return sib if sib.is_file() else None
    if path.is_dir():
        for cand in (path / "attempts.jsonl", path / "avo_state" / "attempts.jsonl"):
            if cand.is_file():
                return cand
        matches = sorted(path.rglob("attempts.jsonl"))
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            print(f"error: multiple attempts.jsonl under {path}; pass an exact one:", file=sys.stderr)
            for m in matches:
                print(f"  {m}", file=sys.stderr)
            raise SystemExit(2)
    return None


def _load_attempts(p: Path) -> list[dict]:
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _aggregate_strategies(records: list[dict]) -> list[dict]:
    """Roll up attempts.jsonl into one row per strategy (the strategy ledger).

    Captures the FULL search: how many times each strategy was tried, whether any
    attempt committed, the best verified/self speedup it reached, and a status
    breakdown — so a human can see which directions paid off and which kept failing.
    """
    agg: dict[str, dict] = {}
    for r in records:
        s = (r.get("strategy") or "").replace("\n", " ").strip() or "(unspecified)"
        a = agg.setdefault(s, {"strategy": s, "attempts": 0, "committed": 0, "statuses": {},
                               "best_verified": None, "best_self": None, "steps": set()})
        # An attempt and its later commit are TWO lines for the SAME try: count the
        # attempt (committed:false) under `attempts` and the commit (committed:true)
        # under `committed`, so a committed strategy reads "1 attempt, 1 committed".
        if r.get("committed"):
            a["committed"] += 1
        else:
            a["attempts"] += 1
        st = r.get("status")
        if st:
            a["statuses"][st] = a["statuses"].get(st, 0) + 1
        for key, field in (("verified_geomean", "best_verified"), ("self_speedup", "best_self")):
            v = r.get(key)
            if isinstance(v, (int, float)) and (a[field] is None or v > a[field]):
                a[field] = v
        si = r.get("step_index") if r.get("step_index") is not None else r.get("step")
        if si is not None:
            a["steps"].add(si)
    rows = list(agg.values())
    for a in rows:
        a["n_steps"] = len(a.pop("steps"))
        # Legacy commit-only runs (pre attempt-ledger) have no committed:false lines:
        # fall back so the attempt count is never a misleading zero.
        if a["attempts"] == 0 and a["committed"] > 0:
            a["attempts"] = a["committed"]
    rows.sort(key=lambda a: (a["committed"] > 0, a["attempts"], a["best_verified"] or 0.0), reverse=True)
    return rows


def _strategy_outcome(a: dict) -> str:
    if a["committed"]:
        return f"committed x{a['committed']}"
    dead = sum(n for s, n in a["statuses"].items() if s in ("failed", "no_change", "null", "infra_failure"))
    if dead:
        return f"dead ({dead}/{a['attempts']} no commit)"
    return f"tried x{a['attempts']} (uncommitted)"


def _render_strategies(rows: list[dict], *, fmt: str, src: Path) -> str:
    def best(v):
        return f"{v:.4f}x" if isinstance(v, (int, float)) else "-"

    def statuses(d):
        return ", ".join(f"{k}:{v}" for k, v in sorted(d.items())) if d else "-"

    headers = ["Strategy", "Attempts", "Committed", "Best verified", "Best self", "Statuses", "Outcome"]
    cells = [[a["strategy"], str(a["attempts"]), str(a["committed"]),
              best(a["best_verified"]), best(a["best_self"]), statuses(a["statuses"]),
              _strategy_outcome(a)] for a in rows]

    if fmt == "csv":
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["strategy", "attempts", "committed", "best_verified", "best_self", "statuses", "outcome"])
        for a in rows:
            w.writerow([a["strategy"], a["attempts"], a["committed"],
                        a["best_verified"] if isinstance(a["best_verified"], (int, float)) else "",
                        a["best_self"] if isinstance(a["best_self"], (int, float)) else "",
                        statuses(a["statuses"]), _strategy_outcome(a)])
        return buf.getvalue().rstrip("\n")

    n_attempts = sum(a["attempts"] for a in rows)
    n_committed = sum(a["committed"] for a in rows)
    if fmt == "md":
        out = [f"# AVO strategy ledger", "",
               f"_{len(rows)} distinct strategies · {n_attempts} attempts · "
               f"{n_committed} committed · source: `{src}`_", "",
               "| " + " | ".join(headers) + " |",
               "|" + "|".join(["---"] * len(headers)) + "|"]
        out += ["| " + " | ".join(c) + " |" for c in cells]
        return "\n".join(out)

    widths = [len(h) for h in headers]
    for row in cells:
        for i, c in enumerate(row):
            widths[i] = max(widths[i], len(c))

    def line(r):
        return "  ".join(c.ljust(widths[i]) for i, c in enumerate(r))

    out = [f"# AVO strategy ledger: {src}",
           f"# {len(rows)} strategies   {n_attempts} attempts   {n_committed} committed", "",
           line(headers), line(["-" * w for w in widths])]
    out += [line(r) for r in cells]
    return "\n".join(out)


def _load_rows(lineage_path: Path) -> list[dict]:
    data = json.loads(lineage_path.read_text(encoding="utf-8"))
    rows = []
    for node in data.get("committed", []):
        score = node.get("score", {}) or {}
        rows.append(
            {
                "id": node.get("id", "?"),
                "git_ref": node.get("git_ref") or "",
                "speedup": score.get("speedup"),
                "latency_ms": score.get("latency_ms"),
                "strategy": (node.get("strategy") or "").replace("\n", " ").strip(),
                "parent_id": node.get("parent_id"),
                "per_shape": score.get("per_shape") or {},
                "committed_at": node.get("committed_at", ""),
            }
        )
    return rows


def _baseline_latency(rows: list[dict]) -> float | None:
    for r in rows:
        if r["id"] == "v0" and isinstance(r["latency_ms"], (int, float)):
            return r["latency_ms"]
    return None


def _fmt_speedup(v) -> str:
    return f"{v:.4f}x" if isinstance(v, (int, float)) else "n/a"


def _fmt_latency(v) -> str:
    return f"{v:.4f}" if isinstance(v, (int, float)) else "n/a"


def _fmt_per_shape(d: dict) -> str:
    if not d:
        return ""
    return " ".join(f"{k}={v:.3f}x" for k, v in sorted(d.items()))


def _render_table(rows: list[dict], baseline_ms: float | None) -> str:
    headers = ["version", "speedup", "latency_ms", "vs_base", "strategy", "parent", "per_shape"]
    table = []
    for r in rows:
        lat = r["latency_ms"]
        if isinstance(lat, (int, float)) and isinstance(baseline_ms, (int, float)) and baseline_ms > 0:
            vs_base = f"{baseline_ms / lat:.3f}x" if lat > 0 else "n/a"
        else:
            vs_base = ""
        table.append(
            [
                r["git_ref"] or r["id"],
                _fmt_speedup(r["speedup"]),
                _fmt_latency(lat),
                vs_base,
                r["strategy"] or "-",
                (("avo-" + r["parent_id"]) if r["parent_id"] else "-"),
                _fmt_per_shape(r["per_shape"]) or "-",
            ]
        )
    widths = [len(h) for h in headers]
    for row in table:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def line(cells):
        return "  ".join(c.ljust(widths[i]) for i, c in enumerate(cells))

    out = [line(headers), line(["-" * w for w in widths])]
    out.extend(line(row) for row in table)
    return "\n".join(out)


def _render_md(rows: list[dict], baseline_ms: float | None) -> str:
    headers = ["Version", "Speedup", "Latency (ms)", "vs baseline", "Strategy", "Parent", "Per-shape"]
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        lat = r["latency_ms"]
        if isinstance(lat, (int, float)) and isinstance(baseline_ms, (int, float)) and baseline_ms > 0 and lat > 0:
            vs_base = f"{baseline_ms / lat:.3f}x"
        else:
            vs_base = ""
        cells = [
            r["git_ref"] or r["id"],
            _fmt_speedup(r["speedup"]),
            _fmt_latency(lat),
            vs_base,
            r["strategy"] or "",
            ("avo-" + r["parent_id"]) if r["parent_id"] else "",
            _fmt_per_shape(r["per_shape"]),
        ]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def _render_csv(rows: list[dict], baseline_ms: float | None) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["version", "speedup", "latency_ms", "vs_baseline", "strategy", "parent", "per_shape", "committed_at"])
    for r in rows:
        lat = r["latency_ms"]
        if isinstance(lat, (int, float)) and isinstance(baseline_ms, (int, float)) and baseline_ms > 0 and lat > 0:
            vs_base = round(baseline_ms / lat, 4)
        else:
            vs_base = ""
        writer.writerow(
            [
                r["git_ref"] or r["id"],
                r["speedup"] if isinstance(r["speedup"], (int, float)) else "",
                lat if isinstance(lat, (int, float)) else "",
                vs_base,
                r["strategy"],
                ("avo-" + r["parent_id"]) if r["parent_id"] else "",
                _fmt_per_shape(r["per_shape"]),
                r["committed_at"],
            ]
        )
    return buf.getvalue().rstrip("\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Export an AVO lineage.json to a version table.")
    ap.add_argument("path", type=Path, help="lineage.json / attempts.jsonl file, or a directory containing one")
    ap.add_argument("--format", choices=["table", "md", "csv"], default="table")
    ap.add_argument("--best-only", action="store_true", help="show only the highest-speedup version")
    ap.add_argument("--strategies", action="store_true",
                    help="render the STRATEGY LEDGER from attempts.jsonl (every strategy tried, "
                         "committed?, best speedup, status breakdown) instead of the committed lineage")
    args = ap.parse_args(argv)

    if args.strategies:
        ap_path = _find_attempts(args.path)
        if ap_path is None:
            print(f"error: no attempts.jsonl found at {args.path}", file=sys.stderr)
            return 2
        rows = _aggregate_strategies(_load_attempts(ap_path))
        if not rows:
            print(f"(empty: no attempts in {ap_path})", file=sys.stderr)
            return 0
        print(_render_strategies(rows, fmt=args.format, src=ap_path))
        return 0

    lineage_path = _find_lineage(args.path)
    if lineage_path is None:
        print(f"error: no lineage.json found at {args.path}", file=sys.stderr)
        return 2

    rows = _load_rows(lineage_path)
    if not rows:
        print(f"(empty lineage: no committed versions in {lineage_path})", file=sys.stderr)
        return 0

    baseline_ms = _baseline_latency(rows)

    if args.best_only:
        scored = [r for r in rows if isinstance(r["speedup"], (int, float))]
        rows = [max(scored, key=lambda r: r["speedup"])] if scored else rows[:1]

    if args.format == "md":
        print(_render_md(rows, baseline_ms))
    elif args.format == "csv":
        print(_render_csv(rows, baseline_ms))
    else:
        print(f"# AVO lineage: {lineage_path}")
        best = max((r for r in rows if isinstance(r["speedup"], (int, float))), key=lambda r: r["speedup"], default=None)
        if best is not None:
            print(f"# versions: {len(rows)}   best: {best['git_ref'] or best['id']} @ {_fmt_speedup(best['speedup'])}")
        print()
        print(_render_table(rows, baseline_ms))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
