#!/usr/bin/env python3
"""One sampling point's traces (every TP rank) -> a per-rank, per-instance kernel census.

Step 3b of log -> sampling points -> capture -> census -> ledger. `serving_probe.py`
captures a point; this turns its trace directory into the `per_rank` census that
`coverage_ledger.py --census POINT=PATH` consumes. Agents used to write this by hand in
every run, each with its own instance definition and collective rule, which made two
runs against one deployment incomparable and cost ~30 minutes of GPU-idle time per run.

    python3 point_census.py --trace-dir <dir of one point's *.trace.json[.gz]> \
        --phase DECODE --point-id de_4 --out census/de_4.json [--shape 4]

What one INSTANCE is
--------------------
The observed workload counts a prefill BATCH and a decode ITERATION, so a census must
be per one of those, not per span:

* DECODE: the interval between the starts of consecutive target-forward anchors
  (`step[VERIFY|TARGET_VERIFY bs=N]` under speculation, `step[DECODE bs=N]` without).
  It holds exactly one verify, one draft extend and one draft loop, plus the per-step
  copies and broadcasts; N anchors give N-1 instances.
* PREFILL: one target `step[EXTEND ...]` plus the draft extend that follows it. A draft
  named `DRAFT`/`DRAFT_EXTEND*` attaches to the preceding EXTEND. Older sglang names the
  draft extend identically to the target (`step[EXTEND bs=1 toks=T]` twice); pass
  `--draft-shares-name` for a speculative server of that kind and consecutive identical
  spans are paired. Without it they are separate instances, which is right for a
  server without speculation and wrong by 2x for one with it.

A decode window whose anchors exist only on the CPU timeline was captured with the decode
CUDA graph enabled: the replayed kernels are not traced at all, so any census of it is an
incomplete inventory (measured: zero launches of the model's own MoE GEMM). It is refused.
Capture decode points on a server started with `--disable-cuda-graph`; kernel durations
do not depend on who launched them, and the collective spin that eager launch adds is
removed by the split below.

Collective spin, per LAUNCH
---------------------------
A rank that reaches a collective early spins inside it, and the profiler bills the spin
as device time. Under eager decode it is most of the step (`cross_device_reduce_1stage`
median 640 us against a 5.5 us floor). The j-th launch of a collective inside an
instance is the same collective on every rank, so

    transfer_j = min over ranks of duration_j      -> the kernel
    wait_r     = sum_j (duration_r,j - transfer_j) -> __collective_wait__

Rank-mean total mass is unchanged. If ranks disagree on a collective's launch count
inside an instance, that instance falls back to min over rank totals and is counted in
`collective_split.unaligned_instances`.

Output: {"per_rank": {rank: {kernel: total us over the selected instances}},
         "instances": N, ...provenance}. coverage_ledger divides by `instances`.

Stdlib only.
"""
import argparse
import bisect
import glob
import json
import multiprocessing
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trace_scan import scan, DEVICE_CATS  # noqa: E402

STEP_RE = re.compile(r"step\[(?P<mode>[A-Z_0-9]+) bs=(?P<bs>\d+)(?: toks=(?P<toks>\d+))?")
RANK_RE = re.compile(r"TP-(\d+)")
PREFILL_TARGET = {"EXTEND"}
DRAFT_MODES = {"DRAFT", "DRAFT_EXTEND", "DRAFT_EXTEND_V2"}
DECODE_ANCHOR_MODES = ("TARGET_VERIFY", "VERIFY", "DECODE")   # most specific first
# Cross-device operators only. A bare `reduce`/`gather` would bill local compute
# (`_sparse_mla_reduce_kernel`, `gemm2_..._reduce_...`, `vectorized_gather_kernel`).
COLLECTIVE_PAT = re.compile(
    r"cross_device_reduce|quickreduce|allreduce|all_reduce|allgather_vec|all_gather"
    r"|reduce_scatter|alltoall|all_to_all|nccl|rccl", re.I)
WAIT_KERNEL = "__collective_wait__"


def kernels_in(dev, starts, ts, te):
    lo, hi = bisect.bisect_left(starts, ts), bisect.bisect_left(starts, te)
    out, coll = defaultdict(float), defaultdict(list)
    for _t, dur, nm in dev[lo:hi]:
        out[nm] += dur
        if COLLECTIVE_PAT.search(nm):
            coll[nm].append(dur)
    return dict(out), hi - lo, dict(coll)


def prefill_steps(spans, draft_shares_name):
    """Target EXTEND spans, each with the draft extend that belongs to it."""
    steps = []
    for s in spans:
        mode = s[3]
        cur = steps[-1] if steps else None
        if mode in DRAFT_MODES:
            if cur is not None and not cur["has_draft"]:
                cur["end"], cur["has_draft"] = s[1], True
            continue
        if mode not in PREFILL_TARGET:
            continue
        if (draft_shares_name and cur is not None and not cur["has_draft"]
                and cur["name"] == s[2]):
            cur["end"], cur["has_draft"] = s[1], True
            continue
        steps.append({"name": s[2], "start": s[0], "end": s[1], "shape": s[5],
                      "mode": mode, "has_draft": False})
    return steps


def step_spans(events):
    """(gpu step spans, cpu step spans) from scan() events: (start, end, name, mode, bs, toks)."""
    gpu, cpu = [], []
    for cat, name, ts, dur, *_ in events:
        if cat not in ("gpu_user_annotation", "user_annotation"):
            continue
        m = STEP_RE.match(name)
        if m:
            rec = (ts, ts + dur, name, m.group("mode"), int(m.group("bs")),
                   int(m.group("toks") or 0))
            (gpu if cat == "gpu_user_annotation" else cpu).append(rec)
    gpu.sort()
    cpu.sort()
    return gpu, cpu


def instance_windows(gpu, cpu, phase, draft_shares_name=False):
    """-> (windows, anchor, error). A window is {start, end, shape, mode, paired}.

    The one definition of an instance (see the module docstring); kernel_shapes.py uses
    it too, so a shape manifest and a census always talk about the same steps.
    """
    if phase == "PREFILL":
        return ([{"start": st["start"], "end": st["end"], "shape": st["shape"],
                  "mode": st["mode"], "paired": st["has_draft"]}
                 for st in prefill_steps(gpu, draft_shares_name)],
                "gpu_user_annotation step[EXTEND]", None)
    anchors, anchor = [], None
    for mode in DECODE_ANCHOR_MODES:
        anchors = [s for s in gpu if s[3] == mode]
        if anchors:
            anchor = "gpu_user_annotation step[%s]" % mode
            break
    if len(anchors) < 2:
        hidden = any(s[3] in DECODE_ANCHOR_MODES for s in cpu)
        return [], None, (
            "fewer than 2 decode anchors on the GPU timeline (GPU step modes %s, CPU step "
            "modes %s). %s" % (sorted({s[3] for s in gpu}), sorted({s[3] for s in cpu}),
            "The target forward is on the CPU timeline only: this window was captured "
            "with the decode CUDA graph enabled and the replayed kernels are not traced. "
            "Re-capture with --enable-profile-cuda-graph, or on a server started with "
            "--disable-cuda-graph." if hidden else "The window holds no complete decode step."))
    return ([{"start": a[0], "end": nxt[0], "shape": a[4], "mode": a[3], "paired": True}
             for a, nxt in zip(anchors, anchors[1:])], anchor, None)


def scan_rank(job):
    """One rank's trace -> {shape: [instance, ...]} in time order. Runs in a worker."""
    path, phase, draft_shares_name = job
    m = RANK_RE.search(os.path.basename(path))
    rank = m.group(1) if m else os.path.basename(path)
    s = scan(path)
    gpu, cpu = step_spans(s.events)
    dev = [(ts, dur, name) for cat, name, ts, dur, *_ in s.events if cat in DEVICE_CATS]
    starts = [d[0] for d in dev]
    wins, anchor, err = instance_windows(gpu, cpu, phase, draft_shares_name)
    if err:
        return {"rank": rank, "error": err}
    by_shape = defaultdict(list)
    for w in wins:
        pk, n, coll = kernels_in(dev, starts, w["start"], w["end"])
        by_shape[w["shape"]].append({"shape": w["shape"], "mode": w["mode"], "per_kernel": pk,
                                     "n": n, "coll": coll, "paired": w["paired"]})
    return {"rank": rank, "by_shape": dict(by_shape), "anchor": anchor,
            "scan_method": s.method, "device_rows": len(dev)}


def bill(selected, instances):
    """{rank: [instance...]} -> ({rank: {kernel: us}}, collective report)."""
    ranks = sorted(selected)
    acc = {r: defaultdict(float) for r in ranks}
    rep = {"rule": "per-launch min over ranks -> kernel, excess -> " + WAIT_KERNEL,
           "aligned_launches": 0, "unaligned_instances": 0, "per_kernel": {}}
    stat = defaultdict(lambda: {"launches_per_instance": None, "billed_us": 0.0,
                                "measured_us_rank_mean": 0.0, "unaligned_instances": 0})
    for i in range(instances):
        gs = {r: selected[r][i] for r in ranks}
        names = set()
        for r in ranks:
            for k, v in gs[r]["per_kernel"].items():
                if not COLLECTIVE_PAT.search(k):
                    acc[r][k] += v
            names.update(gs[r]["coll"])
        for k in names:
            runs = [gs[r]["coll"].get(k, []) for r in ranks]
            st = stat[k]
            st["measured_us_rank_mean"] += sum(sum(x) for x in runs) / len(ranks)
            lens = {len(x) for x in runs}
            if len(lens) == 1 and lens != {0}:
                n = lens.pop()
                billed = sum(min(run[j] for run in runs) for j in range(n))
                st["launches_per_instance"] = n
                rep["aligned_launches"] += n
            else:
                billed = min(sum(x) for x in runs)
                st["unaligned_instances"] += 1
                rep["unaligned_instances"] += 1
            st["billed_us"] += billed
            for r, run in zip(ranks, runs):
                acc[r][k] += billed
                acc[r][WAIT_KERNEL] += sum(run) - billed
    for k, st in stat.items():
        m = st["measured_us_rank_mean"]
        rep["per_kernel"][k] = {
            "launches_per_instance": st["launches_per_instance"],
            "measured_us_per_instance_rank_mean": round(m / instances, 2),
            "billed_us_per_instance": round(st["billed_us"] / instances, 2),
            "spin_removed_pct": round(100 * (1 - st["billed_us"] / m), 2) if m else None,
            "unaligned_instances": st["unaligned_instances"],
        }
    return ({r: {k: round(v, 3) for k, v in acc[r].items() if v} for r in ranks}, rep)


def census(trace_dir, phase, point_id=None, shape="auto", drop_first=0,
           draft_shares_name=False, workers=None, planned_shape=None):
    files = sorted(glob.glob(os.path.join(trace_dir, "*.trace.json*")))
    if not files:
        raise SystemExit("point_census: no *.trace.json[.gz] in %s" % trace_dir)
    jobs = [(p, phase, draft_shares_name) for p in files]
    workers = workers or min(len(jobs), os.cpu_count() or 1)
    if workers > 1:
        with multiprocessing.Pool(workers) as pool:
            scans = pool.map(scan_rank, jobs)
    else:
        scans = [scan_rank(j) for j in jobs]
    bad = [s for s in scans if s.get("error")]
    if bad:
        raise SystemExit("point_census: %s rank %s: %s"
                         % (trace_dir, bad[0]["rank"], bad[0]["error"]))

    if shape == "auto":
        mass = defaultdict(float)
        for s in scans:
            for k, gs in s["by_shape"].items():
                mass[k] += sum(sum(g["per_kernel"].values()) for g in gs)
        if not mass:
            raise SystemExit("point_census: no %s instance in %s" % (phase, trace_dir))
        if planned_shape:
            # The point stands for its planned shape: a window also holds whatever else
            # the scheduler ran (a merged two-sequence extend, a prefix residue), and the
            # most expensive shape in it is not necessarily the one that was asked for.
            import math
            chosen = min(mass, key=lambda k: (abs(math.log(max(k, 1) / planned_shape)),
                                              -mass[k]))
        else:
            chosen = max(mass, key=mass.get)
    else:
        chosen = int(shape)

    selected, inventory, steps = {}, {}, {}
    for s in scans:
        r = s["rank"]
        inventory[r] = {str(k): {"instances": len(v),
                                 "us": round(sum(sum(g["per_kernel"].values()) for g in v), 1),
                                 "modes": sorted({g["mode"] for g in v})}
                        for k, v in sorted(s["by_shape"].items())}
        selected[r] = s["by_shape"].get(chosen, [])[drop_first:]
        steps[r] = len(selected[r])
    counts = sorted(set(steps.values()))
    if len(counts) != 1:
        raise SystemExit("point_census: ranks disagree on the number of %s instances at "
                         "shape %s (%s); they did not run the same forwards."
                         % (phase, chosen, steps))
    instances = counts[0]
    if instances == 0:
        raise SystemExit("point_census: no %s instance at shape %s in %s. Inventory: %s"
                         % (phase, chosen, trace_dir, json.dumps(inventory)))
    unpaired = sum(1 for r in selected for g in selected[r] if not g["paired"])

    per_rank, coll = bill(selected, instances)
    return {
        "contract": "kernel_extract/per_point_census/v2",
        "point_id": point_id or os.path.basename(trace_dir.rstrip("/")),
        "phase": phase,
        "achieved_shape": int(chosen),
        "shape_meaning": ("extend_num_tokens of the scheduler step (toks=)"
                          if phase == "PREFILL" else "scheduler step batch size (bs=)"),
        "instances": instances,
        "values_are": "device microseconds summed over `instances` instances, per rank; "
                      "coverage_ledger divides by instances",
        "anchor": next((s["anchor"] for s in scans if s.get("anchor")), None),
        "draft_shares_name": draft_shares_name,
        "prefill_steps_without_draft": unpaired if phase == "PREFILL" else None,
        "ranks": len(per_rank),
        "dropped_first_instances": drop_first,
        "scan_methods": sorted({s["scan_method"] for s in scans}),
        "device_rows_per_rank": {s["rank"]: s["device_rows"] for s in scans},
        "collective_split": coll,
        "window_inventory_by_shape": inventory,
        "trace_dir": os.path.abspath(trace_dir),
        "per_rank": per_rank,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--trace-dir", required=True)
    ap.add_argument("--phase", required=True, choices=["PREFILL", "DECODE"])
    ap.add_argument("--shape", default="auto",
                    help="PREFILL toks / DECODE bs to census; 'auto' = the shape holding "
                         "the most device time in the window")
    ap.add_argument("--planned-shape", type=float, default=None,
                    help="with --shape auto: take the window's shape nearest to this one "
                         "(the point's planned new_tokens / batch_size)")
    ap.add_argument("--drop-first", type=int, default=0,
                    help="drop the first N selected instances on each rank")
    ap.add_argument("--draft-shares-name", action="store_true",
                    help="PREFILL on a speculative server whose draft extend repeats the "
                         "target's step[EXTEND ...] name: pair consecutive identical spans")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--point-id", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = census(a.trace_dir, a.phase, a.point_id, a.shape, a.drop_first,
                 a.draft_shares_name, a.workers, a.planned_shape)
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)
    pr = out["per_rank"]
    nr, inst = max(1, len(pr)), out["instances"]
    tot = sum(sum(v.values()) for v in pr.values()) / nr / inst
    wait = sum(v.get(WAIT_KERNEL, 0) for v in pr.values()) / nr / inst
    print("%s: %s shape=%s instances=%d ranks=%d  per-instance rank-mean %.1f us "
          "(%.1f%% %s) -> %s" % (out["point_id"], a.phase, out["achieved_shape"], inst,
                                 len(pr), tot, 100 * wait / tot if tot else 0,
                                 WAIT_KERNEL, a.out))
    if out["collective_split"]["unaligned_instances"]:
        print("  NOTE: %d collective rows fell back to instance-granularity min"
              % out["collective_split"]["unaligned_instances"])
    if out.get("prefill_steps_without_draft"):
        print("  NOTE: %d prefill steps had no draft extend attached; on a speculative "
              "server with identically named draft spans pass --draft-shares-name"
              % out["prefill_steps_without_draft"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
