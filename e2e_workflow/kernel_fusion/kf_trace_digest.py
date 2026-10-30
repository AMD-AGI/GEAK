#!/usr/bin/env python3
"""Fusion-oriented digest of torch-profiler traces (the KernelFusion TopK step's deterministic half).

parse_profile.py answers "which kernels are hot". Fusion needs a different view of the same trace:
which kernels run back to back inside one step, how short they are, how much of the step is idle
launch gap, how often the per-layer block repeats, and which Python frame launched each kernel.
This script produces that view; the analyst agent applies the fusion prompt to it.

    kf_trace_digest.py --trace-dir DIR [--rank 0] --out OUT_DIR [--top 30]
    kf_trace_digest.py --trace FILE [--trace FILE ...] --out OUT_DIR

DIR may hold traces at any depth. Traces are grouped by their parent directory (one group = one
sampling point / one capture) and one rank is kept per group (``TP-<rank>`` / ``rank<rank>`` in the
file name; the first file when no name carries a rank).

Per trace and per step kind (the label inside sglang's ``step[...]`` GPU annotation, e.g. DECODE,
EXTEND, VERIFY, DRAFT; the whole window when the trace has no step annotations):
  steps, device-busy us/step, annotated span us/step, idle fraction, kernels/step, short-kernel
  share, the top kernels per step, the repeating block length (layer period) of the median step, and
  that step's full ordered kernel sequence with gaps and launching frames (``seq_*.tsv``).

Writes OUT_DIR/digest.json, OUT_DIR/digest.md and OUT_DIR/seq_<group>_<kind>.tsv. Stdlib only.
"""
from __future__ import annotations

import argparse
import bisect
import gzip
import json
import os
import re
import statistics
import sys
from collections import Counter, defaultdict

KERNEL_CATS = {"kernel", "gpu_memcpy", "gpu_memset"}
STEP_RE = re.compile(r"^step\[([A-Za-z_]+)")
RANK_RE = re.compile(r"(?:TP-|rank|_tp)(\d+)")
SHORT_US = 5.0
FRAME_HINT = re.compile(r"(sglang|vllm|aiter|atom|flydsl|triton_kernels)[/.]")


def short_name(name: str) -> str:
    if name.startswith("void "):
        return name[5:95]           # demangled C++: template args carry '(', keep them
    n = name.split("(")[0] if not name.startswith("_ZN") else name
    m = re.match(r"_ZN(?:\d+\w+?)*?(\d+)([A-Za-z_]\w*)", n)
    if n.startswith("_ZN") and m:
        # Itanium mangling: take the innermost readable identifier of the nested name.
        parts, i = [], 3
        while i < len(n) and n[i].isdigit():
            j = i
            while j < len(n) and n[j].isdigit():
                j += 1
            ln = int(n[i:j])
            parts.append(n[j:j + ln])
            i = j + ln
        n = "::".join(parts) or n
    return n[:90]


def load_events(path: str) -> list[dict]:
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", encoding="utf-8", errors="replace") as f:
        doc = json.load(f)
    return doc.get("traceEvents", []) if isinstance(doc, dict) else doc


def discover(trace_dir: str, rank: int) -> dict[str, list[str]]:
    """Per group, the candidate files to try: the wanted rank first, newest first (a re-capture
    lands next to the files of a failed one)."""
    groups: dict[str, list[str]] = defaultdict(list)
    for root, _dirs, files in os.walk(trace_dir, followlinks=True):
        if "capture_traces" in root.split(os.sep):
            continue    # CUDA-graph capture warm-up, not serving (same rule as roles/profiler.md)
        for fn in files:
            if fn.endswith((".trace.json.gz", ".trace.json", ".pt.trace.json.gz", ".json.gz")):
                groups[os.path.relpath(root, trace_dir) or "."].append(os.path.join(root, fn))
    picked = {}
    for g, files in sorted(groups.items()):
        files.sort(key=lambda f: (-os.path.getmtime(f), f))
        ranked = [f for f in files if (m := RANK_RE.search(os.path.basename(f))) and int(m.group(1)) == rank]
        picked[g] = ranked + [f for f in files if f not in ranked]
    return picked


def frame_index(events: list[dict]):
    """Per-tid sorted CPU intervals (cpu_op + python_function) for 'who launched this kernel'."""
    by_tid: dict = defaultdict(list)
    for e in events:
        if e.get("ph") == "X" and e.get("cat") in ("cpu_op", "python_function"):
            by_tid[(e.get("pid"), e.get("tid"))].append((e["ts"], e["ts"] + e.get("dur", 0), e))
    for v in by_tid.values():
        v.sort(key=lambda t: (t[0], -t[1]))
    return by_tid


def launch_frames(rt: dict, by_tid) -> tuple[str, str]:
    """(innermost cpu_op, innermost framework python frame) enclosing a runtime launch event."""
    ivs = by_tid.get((rt.get("pid"), rt.get("tid")), [])
    ts = rt["ts"]
    hi = bisect.bisect_right(ivs, (ts, float("inf")))
    op = frame = ""
    best_op = best_fr = -1.0
    # Walk back over intervals that start before ts; the enclosing ones end after ts. Bounded scan.
    for k in range(hi - 1, max(-1, hi - 4000), -1):
        s, en, e = ivs[k]
        if en < ts:
            continue
        if e["cat"] == "cpu_op" and s > best_op:
            best_op, op = s, e["name"]
        elif e["cat"] == "python_function" and s > best_fr and FRAME_HINT.search(e["name"]):
            best_fr, frame = s, e["name"]
    return op, frame


def covered_us(ks: list[dict]) -> float:
    """Union of kernel intervals: device time with at least one kernel running (streams overlap)."""
    tot, end = 0.0, None
    for k in sorted(ks, key=lambda e: e["ts"]):
        s, e = k["ts"], k["ts"] + k.get("dur", 0)
        if end is None or s > end:
            tot += e - s
            end = e
        elif e > end:
            tot += e - end
            end = e
    return tot


def layer_period(names: list[str]) -> dict:
    """Smallest shift p at which the step's kernel-name sequence best repeats itself."""
    n = len(names)
    if n < 8:
        return {"period": 0, "repeats": 0, "match": 0.0}
    best = (0, 0.0)
    for p in range(3, min(400, n // 2) + 1):
        hits = sum(1 for i in range(n - p) if names[i] == names[i + p])
        frac = hits / (n - p)
        if frac > best[1] + 0.02:
            best = (p, frac)
        if frac >= 0.97:
            break
    p, frac = best
    return {"period": p, "repeats": round(n / p, 1) if p else 0, "match": round(frac, 3)}


def digest_trace(path: str, top: int) -> dict:
    ev = load_events(path)
    kernels = sorted((e for e in ev if e.get("ph") == "X" and e.get("cat") in KERNEL_CATS),
                     key=lambda e: e["ts"])
    steps = sorted((e for e in ev if e.get("cat") == "gpu_user_annotation" and STEP_RE.match(e.get("name", ""))),
                   key=lambda e: e["ts"])
    rt_by_corr = {e["args"]["correlation"]: e for e in ev
                  if e.get("cat") == "cuda_runtime" and isinstance(e.get("args"), dict) and "correlation" in e["args"]}
    by_tid = frame_index(ev)

    # Assign kernels to the step annotation whose device-side window contains the kernel start.
    windows: list[tuple[str, float, float]] = []
    if steps:
        windows = [(STEP_RE.match(s["name"]).group(1).upper(), s["ts"], s["ts"] + s.get("dur", 0)) for s in steps]
    elif kernels:
        windows = [("WINDOW", kernels[0]["ts"], kernels[-1]["ts"] + kernels[-1].get("dur", 0))]
    starts = [w[1] for w in windows]
    per_step: list[list[dict]] = [[] for _ in windows]
    for k in kernels:
        i = bisect.bisect_right(starts, k["ts"]) - 1
        if i >= 0 and k["ts"] <= windows[i][2]:
            per_step[i].append(k)

    kinds: dict[str, list[int]] = defaultdict(list)
    for i, w in enumerate(windows):
        if per_step[i]:
            kinds[w[0]].append(i)

    out = {"trace": path, "num_kernels": len(kernels), "num_steps": len(windows), "step_kinds": {}}
    for kind, idxs in sorted(kinds.items()):
        busy = [sum(k.get("dur", 0) for k in per_step[i]) for i in idxs]
        span = [windows[i][2] - windows[i][1] for i in idxs]
        cover = [covered_us(per_step[i]) for i in idxs]
        nk = [len(per_step[i]) for i in idxs]
        agg: dict[str, list[float]] = defaultdict(list)
        for i in idxs:
            for k in per_step[i]:
                agg[k["name"]].append(k.get("dur", 0))
        tot = sum(busy) or 1.0
        rows = sorted(({"kernel": short_name(n), "calls_per_step": round(len(d) / len(idxs), 2),
                        "us_per_step": round(sum(d) / len(idxs), 2), "avg_us": round(sum(d) / len(d), 2),
                        "pct_step": round(100 * sum(d) / tot, 2)} for n, d in agg.items()),
                      key=lambda r: -r["us_per_step"])
        all_d = [d for v in agg.values() for d in v]
        short = [d for d in all_d if d < SHORT_US]
        # Median-busy step = the representative sequence the analyst reads.
        rep = idxs[sorted(range(len(idxs)), key=lambda j: busy[j])[len(idxs) // 2]]
        seq, prev_end = [], None
        for k in per_step[rep]:
            corr = (k.get("args") or {}).get("correlation")
            rt = rt_by_corr.get(corr)
            op, frame = launch_frames(rt, by_tid) if rt else ("", "")
            launch = (rt or {}).get("name", "")
            seq.append({"kernel": short_name(k["name"]), "dur_us": round(k.get("dur", 0), 2),
                        "gap_us": round(k["ts"] - prev_end, 2) if prev_end is not None else 0.0,
                        "stream": (k.get("args") or {}).get("stream"), "launch": launch,
                        "cpu_op": op, "frame": frame})
            prev_end = max(prev_end or 0, k["ts"] + k.get("dur", 0))
        out["step_kinds"][kind] = {
            "steps": len(idxs),
            "busy_us_per_step": round(statistics.mean(busy), 1),   # summed kernel time (all streams)
            "span_us_per_step": round(statistics.mean(span), 1),
            "idle_frac": round(max(0.0, 1 - sum(cover) / (sum(span) or 1.0)), 3),
            "kernels_per_step": round(statistics.mean(nk), 1),
            "distinct_kernels": len(agg),
            "short_kernel_share": {"count_frac": round(len(short) / max(1, len(all_d)), 3),
                                   "time_frac": round(sum(short) / max(1e-9, sum(all_d)), 3),
                                   "threshold_us": SHORT_US},
            "graph_replayed": bool(seq) and sum(1 for s in seq if "Graph" in s["launch"]) > len(seq) / 2,
            "layer_period": layer_period([s["kernel"] for s in seq]),
            "top_kernels": rows[:top],
            "_seq": seq,
        }
    return out


def to_md(groups: dict[str, dict]) -> str:
    L = ["# KernelFusion trace digest", ""]
    for g, d in groups.items():
        L += [f"## `{g}` — {os.path.basename(d['trace'])}",
              f"{d['num_kernels']} kernels, {d['num_steps']} step windows", ""]
        for kind, s in d["step_kinds"].items():
            lp = s["layer_period"]
            L += [f"### {kind}: {s['steps']} steps, busy {s['busy_us_per_step']} us/step "
                  f"(span {s['span_us_per_step']} us, idle {100 * s['idle_frac']:.1f}%), "
                  f"{s['kernels_per_step']} kernels/step",
                  f"- kernels < {s['short_kernel_share']['threshold_us']} us: "
                  f"{100 * s['short_kernel_share']['count_frac']:.1f}% of launches, "
                  f"{100 * s['short_kernel_share']['time_frac']:.1f}% of device time; "
                  f"graph-replayed: {s['graph_replayed']}"
                  + (" (launch frames unavailable: map kernels to source by name)" if s["graph_replayed"] else ""),
                  f"- repeating block: {lp['period']} kernels x {lp['repeats']} (match {lp['match']}); "
                  f"sequence: `{s['seq_file']}`", "",
                  "| kernel | calls/step | us/step | avg us | % step |", "|---|---|---|---|---|"]
            L += [f"| `{r['kernel']}` | {r['calls_per_step']} | {r['us_per_step']} | {r['avg_us']} | {r['pct_step']} |"
                  for r in s["top_kernels"]]
            L.append("")
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--trace", action="append", default=[])
    ap.add_argument("--trace-dir", default="")
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    picked = dict(discover(a.trace_dir, a.rank)) if a.trace_dir else {}
    for t in a.trace:
        picked[os.path.basename(os.path.dirname(os.path.abspath(t))) + "/" + os.path.basename(t)] = [t]
    if not picked:
        sys.stderr.write("no trace files found\n")
        return 2
    os.makedirs(a.out, exist_ok=True)
    groups = {}
    for g, cands in picked.items():
        d = None
        for path in cands:
            try:
                d = digest_trace(path, a.top)
            except (OSError, EOFError, ValueError) as e:   # truncated .gz, cut-off JSON
                sys.stderr.write(f"skip unreadable trace {path}: {type(e).__name__}: {e}\n")
                continue
            if d["num_kernels"]:
                break
            sys.stderr.write(f"skip trace without GPU kernels {path}\n")
            d = None
        if d is None:
            sys.stderr.write(f"group {g}: no usable trace\n")
            continue
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", g).strip("_") or "trace"
        for kind, s in d["step_kinds"].items():
            s["seq_file"] = os.path.join(a.out, f"seq_{safe}_{kind}.tsv")
            with open(s["seq_file"], "w") as f:
                f.write("idx\tkernel\tdur_us\tgap_us\tstream\tlaunch\tcpu_op\tframe\n")
                for i, r in enumerate(s.pop("_seq")):
                    f.write(f"{i}\t{r['kernel']}\t{r['dur_us']}\t{r['gap_us']}\t{r['stream']}\t"
                            f"{r['launch']}\t{r['cpu_op']}\t{r['frame']}\n")
        groups[g] = d
    if not groups:
        sys.stderr.write("no usable trace (all unreadable or without GPU kernels)\n")
        return 3
    with open(os.path.join(a.out, "digest.json"), "w") as f:
        json.dump({"schema": "geak.kf_trace_digest/1", "groups": groups}, f, indent=1)
    with open(os.path.join(a.out, "digest.md"), "w") as f:
        f.write(to_md(groups))
    print(f"KF_DIGEST groups={len(groups)} out={a.out}/digest.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
