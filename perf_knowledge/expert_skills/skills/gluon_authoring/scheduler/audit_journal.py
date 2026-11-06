#!/usr/bin/env python3
"""Audit a broker journal: prove no lane was ever double-booked, and price the scheduler.

    python3 audit_journal.py [~/.cache/tile-runtime/gpu-broker/state/journal.ndjson] [--json]

TWO QUESTIONS, and they are the two nobody can answer from the artifacts alone.

**Was the arbitration sound?** Every timed number this fleet produced rests on the claim that no two
exclusive leases overlapped on one lane. That claim is checkable, and this checks it -- by replaying
the journal in `seq` order and watching for a grant on a lane that is already held.

Order by `seq`, never by `t`. Timestamps tie: a real 16-kernel run had 88 of 238 events sharing a
timestamp, and a reader that sorts by time and breaks ties grant-before-release reports dozens of
double-bookings that never happened. `seq` is assigned under the scheduler's own lock, so it is the
true order of state changes. (This tool exists because that false alarm happened during development
and cost a real investigation.)

**What did the sharing cost?** `wait_s` per acquisition is the price of a shared pool over a
partitioned one. If it is near zero the box has spare cards; if it dominates, the fleet is
scheduler-bound and wants either more GPUs or less concurrency. Nothing else in the run records it.
"""
from __future__ import annotations

import collections
import json
import os
import sys


def load(path):
    recs = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    recs.append(json.loads(line))
                except ValueError:
                    pass          # a torn last line from a killed daemon is normal
    return recs


def audit(recs, epoch=None):
    """`epoch` selects one daemon's records; None means the LAST one in the file.

    THE JOURNAL IS APPEND-ONLY AND SHARED ACROSS DAEMON LIFETIMES. A restarted broker writes to the
    same path, and its lease ids start again at 1 -- so replaying the whole file mixes two daemons'
    id spaces and reports lease 2 being granted while "lease 2" (a different lease, from a dead
    daemon) is still open. Observed on a box where five daemons had appended over a session: the
    auditor reported 2 double-bookings, both phantom.

    Each `broker_start` opens a new epoch. Auditing the last one by default is the useful reading:
    "was the run I just did sound". `--epoch all` is available for spelunking, and honestly reports
    that its exclusivity check spans id spaces.
    """
    # Records written before `seq` existed fall back to file order, which is still the write order.
    for i, r in enumerate(recs):
        if r.get("seq") is None:
            r["seq"] = i

    epochs = [i for i, r in enumerate(recs) if r.get("event") == "broker_start"]
    if epoch != "all" and epochs:
        start = epochs[-1] if epoch is None else epochs[min(int(epoch), len(epochs) - 1)]
        scoped = recs[start:]
    else:
        scoped = recs
    ordered = sorted(scoped, key=lambda r: r["seq"])
    recs = scoped

    held = {}                      # lane -> lease_id, for EXCLUSIVE only
    lease_owner = {}               # lease_id -> owner, so a reclaim can find what it dropped
    lease_group = {}               # lease_id -> group; a release does not always carry it
    overlaps = []
    grants = releases = 0
    waits, holds = [], []
    long_holds = []                # every completed hold, kept whole so the longest can be NAMED
    per_group = collections.Counter()
    per_lane = collections.Counter()
    residents = collections.Counter()

    for r in ordered:
        ev = r.get("event")
        if ev == "grant":
            grants += 1
            per_group[r.get("group") or "?"] += 1
            per_lane[r.get("lane")] += 1
            lease_owner[r.get("lease")] = r.get("owner")
            lease_group[r.get("lease")] = r.get("group")
            if r.get("kind") == "exclusive":
                lane = r.get("lane")
                if lane in held:
                    overlaps.append({"lane": lane, "incumbent": held[lane],
                                     "intruder": r.get("lease"), "seq": r["seq"]})
                held[lane] = r.get("lease")
            else:
                residents[r.get("lane")] += 1
        elif ev == "release":
            releases += 1
            if r.get("held_s") is not None:
                holds.append(r["held_s"])
                # Keep the record, not just the number: naming the holder is the whole difference
                # between "something blocked the pool" and a next step. The group and owner come from
                # the release itself, so this works on journals written before `lease_overrun` existed.
                long_holds.append({"held_s": r["held_s"], "lane": r.get("lane"),
                                   "group": r.get("group") or lease_group.get(r.get("lease")),
                                   "owner": r.get("owner")})
            for lane, lid in list(held.items()):
                if lid == r.get("lease"):
                    del held[lane]
        elif ev == "disconnect_reclaim":
            # A RECLAIM ENDS A LEASE, exactly as a release does -- the client's socket closed and the
            # daemon took its lanes back. Counting it only as an incident and never clearing the lane
            # made the NEXT grant on that lane look like a double-booking: the auditor still believed
            # the dead client held it. Measured on one run, six of seven reported exclusivity failures
            # were this phantom, each sitting one or two records after a reclaim, and the seventh was
            # the same shape a few records further out. A false FAIL here is expensive in a specific
            # way -- it says "every number from that lane is suspect", which invalidates real results.
            #
            # The per-lease detail this needs is now in the record (lane, kind, held_s per reclaimed
            # lease). Older journals carry only a count, so fall back to clearing by owner: the reclaim
            # names the owner whose leases were dropped, which is enough to find them.
            for d in (r.get("reclaimed") or []):
                releases += 1
                if d.get("held_s") is not None:
                    holds.append(d["held_s"])
                for lane, lid in list(held.items()):
                    if lid == d.get("lease"):
                        del held[lane]
            if not r.get("reclaimed"):
                # Pre-detail journal: we know an owner lost its leases but not which. Clearing every
                # lane this owner holds is the closest honest reading; leaving them held is not, since
                # the daemon has definitively released them.
                owner = r.get("owner")
                for lane, lid in list(held.items()):
                    if owner and lease_owner.get(lid) == owner:
                        del held[lane]
        elif ev == "enqueue":
            pass
        if r.get("waited_s") is not None:
            waits.append(r["waited_s"])

    def pct(xs, p):
        if not xs:
            return None
        s = sorted(xs)
        return round(s[min(len(s) - 1, int(len(s) * p))], 3)

    return {
        "records": len(recs),
        "epochs_in_file": len(epochs),
        "audited_epoch": "all" if epoch == "all" else (len(epochs) - 1 if epochs else 0),
        "grants": grants,
        "releases": releases,
        "still_held_at_end": len(held),
        "double_bookings": len(overlaps),
        "overlap_detail": overlaps[:10],
        "queue_timeouts": sum(1 for r in recs if r.get("event") == "queue_timeout"),
        "crash_reclaims": sum(1 for r in recs if r.get("event") == "disconnect_reclaim"),
        "lease_reclaims_ttl": sum(1 for r in recs if r.get("event") == "lease_expired"),
        "foreign_events": sum(1 for r in recs if r.get("event") == "lane_foreign"),
        "groups": len([g for g in per_group if g != "?"]),
        "grants_per_group": dict(per_group.most_common(30)),
        "grants_per_lane": dict(sorted(per_lane.items(), key=lambda kv: (kv[0] is None, kv[0]))),
        "resident_grants_per_lane": dict(sorted(residents.items(),
                                                key=lambda kv: (kv[0] is None, kv[0]))),
        "hold_s": {"p50": pct(holds, 0.5), "p90": pct(holds, 0.9),
                   "max": round(max(holds), 3) if holds else None,
                   "total": round(sum(holds), 1) if holds else 0},
        "wait_s": {"p50": pct(waits, 0.5), "p90": pct(waits, 0.9),
                   "max": round(max(waits), 3) if waits else None,
                   "total": round(sum(waits), 1) if waits else 0},
        "lease_overruns": sum(1 for r in recs if r.get("event") == "lease_overrun"),
        # Only worth printing when a hold is long enough to have blocked somebody; a run whose
        # longest hold is a second has no blocker to name.
        "longest_holds": [x for x in sorted(long_holds, key=lambda d: -d["held_s"])[:5]
                          if x["held_s"] >= 60],
    }


def render(a):
    out = []
    out.append(f"records={a['records']}  grants={a['grants']}  releases={a['releases']}"
               f"  still_held={a['still_held_at_end']}")
    if a.get("epochs_in_file", 1) > 1:
        out.append(f"note: {a['epochs_in_file']} daemon lifetimes in this file; audited epoch "
                   f"{a['audited_epoch']}"
                   + ("  (lease ids repeat across daemons, so `all` mixes id spaces)"
                      if a["audited_epoch"] == "all" else " (the most recent)"))
    verdict = "PASS" if a["double_bookings"] == 0 else f"FAIL ({a['double_bookings']})"
    out.append(f"exclusivity: {verdict}"
               + ("" if a["double_bookings"] == 0
                  else "  <-- two leases on one lane; every number from that lane is suspect"))
    for o in a["overlap_detail"]:
        out.append(f"    lane {o['lane']}: lease {o['incumbent']} still held when "
                   f"{o['intruder']} was granted (seq {o['seq']})")
    out.append(f"queue_timeouts={a['queue_timeouts']}  crash_reclaims={a['crash_reclaims']}"
               f"  foreign_lane_events={a['foreign_events']}")
    out.append(f"groups={a['groups']}  per-lane grants: {a['grants_per_lane']}")
    h, w = a["hold_s"], a["wait_s"]
    out.append(f"hold_s  p50={h['p50']} p90={h['p90']} max={h['max']} total={h['total']}")
    out.append(f"wait_s  p50={w['p50']} p90={w['p90']} max={w['max']} total={w['total']}")
    if h["total"] and w["total"] is not None:
        share = 100.0 * w["total"] / (w["total"] + h["total"])
        out.append(f"scheduler cost: {share:.1f}% of lane-seconds were spent QUEUEING")
        # READ THE SHAPE BEFORE RECOMMENDING ANYTHING. A total says how much waiting there was, never
        # whether it was SPREAD (every acquisition pays a little -- genuinely too little GPU for the
        # concurrency) or CONCENTRATED (nearly everything is instant, and a few long holds block
        # everyone -- too much GPU held by too few). The cures are opposite, and this line printed
        # "add cards or cut concurrency" for both. On a measured 4-GPU run it said exactly that while
        # the median wait was 0.00 s and 87% of acquisitions were instant: the real finding was that
        # 0.58% of leases held 65.5% of all GPU-seconds, which adding cards would not have touched.
        if share > 50:
            spread = (w["p50"] or 0) > 0.5 or (w["p90"] or 0) > 5.0
            if spread:
                out.append("  -> SPREAD: the median acquisition waits too, so the pool really is "
                           "oversubscribed. Add cards or cut concurrency.")
            else:
                out.append(f"  -> CONCENTRATED: median wait {w['p50']}s, p90 {w['p90']}s -- the pool "
                           f"is NOT generally short. A few long holds are blocking everyone (max wait "
                           f"{w['max']}s). Adding cards would not help; look at the longest holds "
                           f"below and at `lease_overrun`.")
    # WHO HELD THE CARDS. In the concentrated case the blocker is one or two leases, and nothing here
    # named them: `held_s` rides on every release and was only ever summed into a percentile.
    if a.get("longest_holds"):
        out.append("longest EXCLUSIVE holds (what everything else queued behind):")
        for x in a["longest_holds"]:
            out.append(f"  {x['held_s']:9.1f}s lane={x['lane']} group={x.get('group') or '?'} "
                       f"owner={str(x.get('owner'))[:44]}")
    if a.get("lease_overruns"):
        out.append(f"lease_overrun events: {a['lease_overruns']}  (an EXCLUSIVE lease crossed the "
                   f"broker's reporting floor -- see --lease-overrun-s)")
    top = sorted(a["grants_per_group"].items(), key=lambda kv: -kv[1])[:8]
    if top:
        out.append("busiest groups: " + ", ".join(f"{k}={v}" for k, v in top))
    return "\n".join(out)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    as_json = "--json" in argv
    epoch = None
    if "--epoch" in argv:
        i = argv.index("--epoch")
        epoch = argv[i + 1] if i + 1 < len(argv) else None
        del argv[i:i + 2]
    argv = [a for a in argv if a != "--json"]
    path = argv[0] if argv else os.path.join(
        os.environ.get("GEAK_GPU_RUNTIME_ROOT", os.path.expanduser("~/.cache/tile-runtime/gpu-broker")),
        "state", "journal.ndjson")
    if not os.path.exists(path):
        print(f"no journal at {path}", file=sys.stderr)
        return 2
    a = audit(load(path), epoch=epoch)
    print(json.dumps(a, indent=2) if as_json else render(a))
    # Non-zero ONLY on a real exclusivity violation. Queue timeouts and reclaims are facts about a
    # busy box, not failures, and exiting non-zero on them would train everyone to ignore this.
    return 1 if a["double_bookings"] else 0


if __name__ == "__main__":
    sys.exit(main())
