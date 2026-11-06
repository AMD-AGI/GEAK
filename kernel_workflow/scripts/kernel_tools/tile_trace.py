#!/usr/bin/env python3
"""tile_trace — a tiny, unified GPU-kernel profile trace format + visualizer.

One compact JSON that is BOTH:
  (a) directly loadable by Perfetto / chrome://tracing (it IS the Chrome Trace Event
      Format — a strict subset of what torch.profiler exports), and
  (b) renderable by matplotlib into a multi-stream / single-stream timeline (bars
      positioned by start time, WIDTH = duration) plus an op-class / per-unit share
      breakdown.

Format (standard Chrome "traceEvents", kept minimal):

  {
    "traceEvents": [
      {"ph":"M","pid":0,"name":"process_name","args":{"name":"kernelA"}},
      {"ph":"M","pid":0,"tid":0,"name":"thread_name","args":{"name":"MFMA"}},
      {"ph":"X","pid":0,"tid":0,"name":"mfma","cat":"compute","ts":0.0,"dur":D,"args":{...}},
      {"ph":"C","pid":0,"name":"occupancy","ts":0,"args":{...}}
    ],
    "displayTimeUnit": "ms",
    "meta": {...}   # our extension; Perfetto ignores it
  }

ph used: "X" complete slice (draws a bar), "M" metadata (lane labels), "C" counter,
"i" instant. Tracks: pid = a kernel/config; tid = a stream / hardware-unit / stage
lane. Concurrent bars on parallel tids visualize OVERLAP; end-to-end bars on one tid
visualize a sequential (stage-decomposition) view.

The ATT exporters (att_to_perfetto.py / att_merge_perfetto.py) build TileTrace from
real per-instruction cycles; this file is the format + renderer + a synthetic --demo.

GEAK shared kernel tool (kernel_workflow/scripts/kernel_tools/); the Gluon pack's scripts/ copy is a shim.

Usage:
  python3 tile_trace.py trace.json --timeline t.png --breakdown b.png
  python3 tile_trace.py --demo --outdir out/       # synthetic example (no real data)
  python3 tile_trace.py --selftest
  # programmatic:
  from tile_trace import TileTrace
  t = TileTrace(); k = t.process("kernelA"); t.slice(k, "MFMA", "mfma", ts=0, dur=1, cat="mfma")
  t.save("trace.json"); t.render_timeline("timeline.png")
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from typing import Optional


# ------------------------------------------------------------------ builder ----
class TileTrace:
    """Accumulates Chrome-trace events with a small kernel/unit-lane helper API."""

    def __init__(self, display_time_unit: str = "ms", meta: Optional[dict] = None):
        self.events: list[dict] = []
        self.display_time_unit = display_time_unit
        self.meta = meta or {}
        self._pids: dict[str, int] = {}
        self._tids: dict[tuple, int] = {}

    def process(self, name: str) -> int:
        if name not in self._pids:
            pid = len(self._pids)
            self._pids[name] = pid
            self.events.append({"ph": "M", "pid": pid, "name": "process_name",
                                "args": {"name": name}})
        return self._pids[name]

    def thread(self, pid: int, name: str) -> int:
        key = (pid, name)
        if key not in self._tids:
            tid = len([k for k in self._tids if k[0] == pid])
            self._tids[key] = tid
            self.events.append({"ph": "M", "pid": pid, "tid": tid,
                                "name": "thread_name", "args": {"name": name}})
        return self._tids[key]

    def slice(self, pid: int, lane: str, name: str, ts: float, dur: float,
              cat: str = "", **args) -> None:
        tid = self.thread(pid, lane)
        self.events.append({"ph": "X", "pid": pid, "tid": tid, "name": name,
                            "cat": cat, "ts": float(ts), "dur": float(dur), "args": args})

    def counter(self, pid: int, name: str, ts: float, **values) -> None:
        self.events.append({"ph": "C", "pid": pid, "name": name, "ts": float(ts), "args": values})

    def instant(self, pid: int, lane: str, name: str, ts: float, **args) -> None:
        tid = self.thread(pid, lane)
        self.events.append({"ph": "i", "pid": pid, "tid": tid, "name": name,
                            "ts": float(ts), "s": "t", "args": args})

    def to_dict(self) -> dict:
        return {"traceEvents": self.events, "displayTimeUnit": self.display_time_unit,
                "meta": self.meta}

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=1)

    @staticmethod
    def load(path: str) -> "LoadedTrace":
        with open(path) as f:
            return LoadedTrace(json.load(f))

    def render_timeline(self, out: str, **kw) -> None:
        LoadedTrace(self.to_dict()).render_timeline(out, **kw)

    def render_breakdown(self, out: str, **kw) -> None:
        LoadedTrace(self.to_dict()).render_breakdown(out, **kw)


# ------------------------------------------------------------------- reader ----
@dataclass
class _Lane:
    pid: int
    tid: int
    proc: str
    name: str
    slices: list = field(default_factory=list)


class LoadedTrace:
    """Parses a Chrome-trace dict into ordered lanes for rendering."""

    def __init__(self, data: dict):
        self.data = data
        self.unit = data.get("displayTimeUnit", "ms")
        ev = data.get("traceEvents", [])
        procname: dict = {}
        threadname: dict = {}
        for e in ev:
            if e.get("ph") == "M" and e.get("name") == "process_name":
                procname[e["pid"]] = e["args"].get("name", f"pid{e['pid']}")
            elif e.get("ph") == "M" and e.get("name") == "thread_name":
                threadname[(e["pid"], e.get("tid", 0))] = e["args"].get("name", f"tid{e.get('tid',0)}")
        lanes: dict = {}
        for e in ev:
            if e.get("ph") != "X":
                continue
            key = (e["pid"], e.get("tid", 0))
            if key not in lanes:
                lanes[key] = _Lane(key[0], key[1], procname.get(key[0], f"pid{key[0]}"),
                                   threadname.get(key, f"tid{key[1]}"))
            lanes[key].slices.append(e)
        self.lanes = [lanes[k] for k in sorted(lanes.keys())]
        self.procname = procname

    @staticmethod
    def _color_map(cats):
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors
        base = {
            "compute": "#d62728", "mfma": "#d62728", "matmul": "#d62728",
            "valu": "#1f77b4", "alu": "#1f77b4", "exp": "#9467bd", "softmax": "#9467bd",
            "lds": "#2ca02c", "lds_read": "#2ca02c", "lds_write": "#98df8a",
            "mem": "#ff7f0e", "gld": "#ff7f0e", "gst": "#ffbb78",
            "scalar": "#7f7f7f", "waitcnt": "#c7c7c7", "nop": "#e377c2",
            "barrier": "#8c564b", "bubble": "#eeeeee", "wall": "#bbbbbb",
            "qk": "#1f77b4", "pv": "#d62728", "other": "#cccccc",
        }
        cats2 = [c for c in cats if c not in base]
        if cats2:
            pal = cm.get_cmap("tab20", max(len(cats2), 1))
            for i, c in enumerate(cats2):
                base[c] = mcolors.to_hex(pal(i))
        return base

    def render_timeline(self, out: str, title: str = "tile_trace timeline",
                        width: float = 13.0, row_h: float = 0.42) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt
        lanes = self.lanes
        if not lanes:
            raise SystemExit("no 'X' slice events to plot")
        cats = sorted({s.get("cat") or s["name"] for L in lanes for s in L.slices})
        cmap = self._color_map(cats)
        fig, ax = plt.subplots(figsize=(width, max(2.0, row_h * len(lanes) + 1.2)))
        ylabels = []
        last_proc = None
        for row, L in enumerate(lanes):
            y = len(lanes) - 1 - row
            lbl = L.name if L.proc == last_proc else f"{L.proc} | {L.name}"
            last_proc = L.proc
            ylabels.append((y, lbl))
            for s in L.slices:
                c = cmap.get(s.get("cat") or s["name"], "#cccccc")
                ax.barh(y, s["dur"], left=s["ts"], height=0.7, color=c,
                        edgecolor="black", linewidth=0.4, zorder=3)
                if s["dur"] > 0:
                    pct = s.get("args", {}).get("pct")
                    txt = s["name"] + (f" {pct:.0f}%" if isinstance(pct, (int, float)) else "")
                    ax.text(s["ts"] + s["dur"] / 2, y, txt, ha="center", va="center",
                            fontsize=6.5, zorder=4,
                            color="white" if c not in ("#eeeeee", "#bbbbbb", "#c7c7c7") else "black")
        ax.set_yticks([y for y, _ in ylabels])
        ax.set_yticklabels([l for _, l in ylabels], fontsize=8)
        ax.set_xlabel(f"time ({self.unit})")
        ax.set_title(title)
        ax.grid(axis="x", linestyle=":", alpha=0.5, zorder=0)
        handles = [mpatches.Patch(color=cmap[c], label=c) for c in cats]
        ax.legend(handles=handles, ncol=min(len(cats), 6), fontsize=7,
                  loc="upper center", bbox_to_anchor=(0.5, -0.12 / max(1, len(lanes) / 6)))
        fig.tight_layout()
        fig.savefig(out, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out}")

    def render_breakdown(self, out: str, title: str = "tile_trace op-class share",
                        value: str = "pct", lane: Optional[str] = None) -> None:
        """100%-stacked share per process. If `lane` is given, only aggregate
        slices on lanes whose name contains it (e.g. lane='OPCLASS')."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt
        procs: dict = {}
        for L in self.lanes:
            if lane is not None and lane not in L.name:
                continue
            for s in L.slices:
                v = s.get("args", {}).get(value, s.get("dur", 0.0))
                cat = s.get("cat") or s["name"]
                procs.setdefault(L.proc, {}).setdefault(cat, 0.0)
                procs[L.proc][cat] += float(v)
        names = list(procs)
        cats = sorted({c for d in procs.values() for c in d})
        cmap = self._color_map(cats)
        fig, ax = plt.subplots(figsize=(10, max(2.0, 0.6 * len(names) + 1.5)))
        for i, n in enumerate(names):
            left = 0.0
            tot = sum(procs[n].values()) or 1.0
            for c in cats:
                v = procs[n].get(c, 0.0)
                if v <= 0:
                    continue
                frac = 100.0 * v / tot
                ax.barh(i, frac, left=left, color=cmap[c], edgecolor="white", linewidth=0.5)
                if frac > 4:
                    ax.text(left + frac / 2, i, f"{c}\n{frac:.0f}%", ha="center", va="center",
                            fontsize=6.5, color="white" if cmap[c] not in ("#eeeeee", "#c7c7c7") else "black")
                left += frac
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel("share (%)")
        ax.set_xlim(0, 100)
        ax.set_title(title)
        handles = [mpatches.Patch(color=cmap[c], label=c) for c in cats]
        ax.legend(handles=handles, ncol=min(len(cats), 6), fontsize=7,
                  loc="upper center", bbox_to_anchor=(0.5, -0.18))
        fig.tight_layout()
        fig.savefig(out, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out}")


# --------------------------------------------------------------------- demo ----
def build_demo() -> TileTrace:
    """Synthetic (NON-measured) example: two kernels, each with hardware-unit busy
    lanes (concurrent), an op-class lane, a bubble lane and a wall lane. Numbers are
    illustrative placeholders only — the real tool builds this from ATT cycles."""
    t = TileTrace(display_time_unit="cyc",
                  meta={"source": "synthetic demo (illustrative, not measured)"})
    cat_of = {"MFMA": "mfma", "VALU": "valu", "LDS": "lds", "MEM": "mem"}
    kernels = {
        "kernelA_well_fed": dict(wall=100.0,
                                 busy={"MFMA": 80, "VALU": 70, "LDS": 10, "MEM": 5},
                                 opclass={"mfma": 10, "valu": 60, "lds_read": 8, "waitcnt": 6, "scalar": 16}),
        "kernelB_starved": dict(wall=100.0,
                                busy={"MFMA": 45, "VALU": 55, "LDS": 20, "MEM": 5},
                                opclass={"mfma": 8, "valu": 55, "lds_read": 12, "waitcnt": 18, "scalar": 7}),
    }
    for kname, d in kernels.items():
        pid = t.process(kname)
        wall = d["wall"]
        for unit in ("MFMA", "VALU", "LDS", "MEM"):
            b = d["busy"][unit]
            t.slice(pid, unit, unit.lower(), ts=0.0, dur=b / 100.0 * wall, cat=cat_of[unit], pct=b)
        x0 = 0.0
        for cls, share in sorted(d["opclass"].items(), key=lambda kv: -kv[1]):
            t.slice(pid, "OPCLASS", cls, ts=x0, dur=share / 100.0 * wall, cat=cls, pct=share)
            x0 += share / 100.0 * wall
        mfma = d["busy"]["MFMA"]
        t.slice(pid, "BUBBLE(MFMA-idle)", "bubble", ts=0.0, dur=(1 - mfma / 100.0) * wall,
                cat="bubble", pct=100 - mfma)
        t.slice(pid, "WALL", "wall", ts=0.0, dur=wall, cat="wall", pct=100.0)
    return t


def _demo(outdir: str) -> None:
    import os
    os.makedirs(outdir, exist_ok=True)
    t = build_demo()
    tj = os.path.join(outdir, "demo_trace.json")
    t.save(tj)
    print(f"wrote {tj}  (load in https://ui.perfetto.dev or chrome://tracing)")
    lt = TileTrace.load(tj)
    lt.render_timeline(os.path.join(outdir, "demo_timeline.png"),
                       title="tile_trace synthetic demo — per-unit busy (concurrent) + bubble")
    lt.render_breakdown(os.path.join(outdir, "demo_breakdown.png"), lane="OPCLASS",
                        title="tile_trace synthetic demo — op-class share")


# ---------------------------------------------------------------------- cli ----
def _selftest() -> int:
    import os
    import tempfile
    t = TileTrace(display_time_unit="ns", meta={"k": 1})
    p = t.process("kA")
    t.slice(p, "MFMA", "mfma", ts=0, dur=10, cat="mfma")
    t.slice(p, "VALU", "valu", ts=5, dur=3, cat="valu")
    t.counter(p, "occ", ts=0, waves=4)
    t.instant(p, "MFMA", "mark", ts=2)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "t.json")
        t.save(path)
        d = json.load(open(path))
        lt = TileTrace.load(path)
        ok = (d["displayTimeUnit"] == "ns" and d["meta"] == {"k": 1}
              and sum(1 for e in d["traceEvents"] if e["ph"] == "X") == 2
              and isinstance(lt, LoadedTrace))
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            print("  (matplotlib absent: render step skipped -- format path only)")
        else:
            lt.render_timeline(os.path.join(td, "tl.png"))
            ok = ok and os.path.getsize(os.path.join(td, "tl.png")) > 0
    print("[tile_trace] SELFTEST " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def main() -> None:
    import sys
    if "--selftest" in sys.argv[1:]:
        raise SystemExit(_selftest())
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace", nargs="?", help="Chrome-trace JSON to render")
    ap.add_argument("--timeline", help="output PNG for the timeline (Gantt) view")
    ap.add_argument("--breakdown", help="output PNG for the op-class share view")
    ap.add_argument("--demo", action="store_true", help="build+render a synthetic example")
    ap.add_argument("--outdir", default="out", help="demo output dir")
    a = ap.parse_args()
    if a.demo:
        _demo(a.outdir)
        return
    if not a.trace:
        ap.error("provide a trace.json, or use --demo")
    lt = TileTrace.load(a.trace)
    if a.timeline:
        lt.render_timeline(a.timeline)
    if a.breakdown:
        lt.render_breakdown(a.breakdown)
    if not (a.timeline or a.breakdown):
        base = a.trace.rsplit(".", 1)[0]
        lt.render_timeline(base + "_timeline.png")
        lt.render_breakdown(base + "_breakdown.png")


if __name__ == "__main__":
    main()
