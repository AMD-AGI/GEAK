#!/usr/bin/env python3
"""Design-independent score cap: per-case floors, the weighted cap, and how much of the gap is closed.

Usage: roofline.py <cases.json> [--md out.md] [--json out.json]
cases.json = {
  "peaks": {"read_tbs": 7.4, "write_tbs": 6.0, "tflops": 2300},   # measured (hbm_stream.py, MFMA probe)
  "cases": [{"name": "...", "weight": 0.12,          # share of the score (the harness's case weight)
             "base_us": 17.4, "cand_us": 15.2,        # measured in the eval's own regime
             "read_bytes": 1.6e6, "write_bytes": 0.4e6, "flops": 0,  # the op's unavoidable traffic and work
             "fixed_us": 13.5}]}                       # what an EMPTY kernel costs in the same window
floor_us = fixed_us + max(read/read_bw + write/write_bw, flops/peak). Bytes and flops must be the
operator's (inputs read once, outputs written once), not one implementation's. Score = sum(w) / sum(w/s),
the GEAK weighted ratio-of-sums. gap_closed = (score - 1) / (cap - 1).
"""
import argparse, json, sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cases")
    ap.add_argument("--md", default="")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    d = json.load(open(a.cases))
    pk, rows = d["peaks"], []
    for c in d["cases"]:
        t_mem = c.get("read_bytes", 0) / (pk["read_tbs"] * 1e6) + c.get("write_bytes", 0) / (pk.get("write_tbs", pk["read_tbs"]) * 1e6)
        t_cmp = c.get("flops", 0) / (pk.get("tflops", 1e9) * 1e6)
        floor = c.get("fixed_us", 0.0) + max(t_mem, t_cmp)
        bound = "memory" if t_mem >= t_cmp else "compute"
        if floor > c["base_us"]:
            sys.exit(f"{c['name']}: floor {floor:.2f} us > baseline {c['base_us']} us: bytes, peaks or fixed_us are wrong")
        rows.append({**c, "floor_us": round(floor, 3), "bound": bound, "kernel_floor_us": round(max(t_mem, t_cmp), 3),
                     "cap": c["base_us"] / floor, "speedup": c["base_us"] / c["cand_us"] if c.get("cand_us") else None})
    w = sum(r["weight"] for r in rows)
    cap = w / sum(r["weight"] / r["cap"] for r in rows)
    have = all(r["speedup"] for r in rows)
    score = w / sum(r["weight"] / r["speedup"] for r in rows) if have else None
    closed = (score - 1) / (cap - 1) if have and cap > 1 else None
    lines = ["| case | weight | base µs | cand µs | floor µs (fixed + kernel) | bound | speedup | cap |", "|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: -r["weight"] * (1 - (r["speedup"] or 1) / r["cap"])):
        sp = f"{r['speedup']:.3f}" if r["speedup"] else "-"
        lines.append(f"| {r['name']} | {r['weight']:.3f} | {r['base_us']:.2f} | {r.get('cand_us', '-')} | "
                     f"{r['floor_us']:.2f} ({r.get('fixed_us', 0):.2f} + {r['kernel_floor_us']:.2f}) | {r['bound']} | {sp} | {r['cap']:.3f} |")
    lines.append(f"\nWeighted cap {cap:.4f}" + (f"; score {score:.4f}; gap closed {closed:.0%}" if have else ""))
    lines.append("Rows are sorted by the score they still leave on the table (weight x distance to their cap).")
    out = "\n".join(lines)
    print(out)
    if a.md:
        open(a.md, "w").write(out + "\n")
    if a.json:
        json.dump({"cap": cap, "score": score, "gap_closed": closed, "rows": rows}, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
