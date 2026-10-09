#!/usr/bin/env python3
"""Policy tests for scheduler/core.py. Fake clock, no GPU, no socket, no daemon. Milliseconds.

Every test here is a property the scheduler CLAIMS, written so that it fails if the claim stops
holding. The reason they can exist at all is that `core.Scheduler` takes `now` as an argument
instead of reading a clock -- fairness, aging and TTL reclaim are assertions about time, and with a
real clock they would be either slow or flaky, which in practice means deleted.

    python3 scheduler/tests/test_scheduler_core.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import (EXCLUSIVE, PRIO_BULK, PRIO_MEASURE, PRIO_VERIFY, RESIDENT,  # noqa: E402
                  Scheduler, parse_priority, runtime_class)

FAILS = []


def ok(cond, msg):
    print(("  ok: " if cond else "  FAIL: ") + msg)
    if not cond:
        FAILS.append(msg)


def section(t):
    print(f"\n# {t}")


def fresh(n_lanes=2, **kw):
    s = Scheduler(range(n_lanes), vram_total_mb=262144, **kw)
    for ln in s.lanes.values():
        ln.foreign_busy = False
    return s


# ---------------------------------------------------------------------------
def test_exclusivity():
    section("one EXCLUSIVE per lane, never two")
    s = fresh(2)
    for i in range(4):
        s.submit(0, EXCLUSIVE, owner=f"o{i}", group=f"g{i}")
    granted, _ = s.tick(0)
    ok(len(granted) == 2, f"2 lanes -> exactly 2 grants, got {len(granted)}")
    ok(len({g.lane for g in granted}) == 2, "the two grants are on different lanes")
    ok(len(s.queue) == 2, "the other two stay queued")

    # The invariant, checked directly rather than inferred from counts.
    for lane in s.lanes:
        holders = [lv for lv in s.leases.values()
                   if lv.lane == lane and lv.kind == EXCLUSIVE]
        ok(len(holders) <= 1, f"lane {lane} has at most one exclusive holder")

    s.release(granted[0].lease_id)
    g2, _ = s.tick(1)
    ok(len(g2) == 1 and g2[0].lane == granted[0].lane,
       "releasing a lane immediately hands it to the next in line")


def test_resident_does_not_block_exclusive():
    section("a RESIDENT never blocks an EXCLUSIVE -- the whole point of the split")
    s = fresh(1)
    s.submit(0, RESIDENT, owner="parked", group="k1", vram_mb=8000)
    s.tick(0)
    ok(len(s.residents_on(0)) == 1, "the resident is parked on lane 0")

    s.submit(1, EXCLUSIVE, owner="timer", group="k2")
    g, _ = s.tick(1)
    ok(len(g) == 1 and g[0].lane == 0,
       "an exclusive is still granted on the lane holding a parked resident")
    ok(s.resident_vram_mb(0) == 8000, "and the resident keeps its VRAM accounted")

    # ...but two exclusives still cannot coexist there.
    s.submit(2, EXCLUSIVE, owner="other", group="k3")
    g2, _ = s.tick(2)
    ok(len(g2) == 0, "a second exclusive on that lane is still refused")


def test_wake_returns_to_its_own_lane():
    section("a wake goes to the lane its process is parked on, or waits")
    s = fresh(3)
    s.submit(0, RESIDENT, owner="w", group="k", vram_mb=1000)
    s.tick(0)
    home = s.residents_on(0) or s.residents_on(1) or s.residents_on(2)
    lane = home[0].lane

    # Occupy the parked lane, leaving OTHER lanes free.
    s.submit(1, EXCLUSIVE, owner="squatter", group="other", lanes=(lane,))
    s.tick(1)

    s.submit(2, EXCLUSIVE, owner="w", group="k", want_lane=lane, lanes=(lane,))
    g, _ = s.tick(2)
    ok(len(g) == 0,
       "the wake WAITS for its own lane rather than taking a free one -- its VRAM is there")

    s.release([lv for lv in s.leases.values() if lv.owner == "squatter"][0].lease_id)
    g2, _ = s.tick(3)
    ok(len(g2) == 1 and g2[0].lane == lane, "and gets it the moment that lane frees")


def test_priority_order():
    section("higher priority is served first")
    s = fresh(1)
    s.submit(0, EXCLUSIVE, owner="bulk", group="a", priority=PRIO_BULK)
    s.submit(0, EXCLUSIVE, owner="verify", group="b", priority=PRIO_VERIFY)
    s.submit(0, EXCLUSIVE, owner="measure", group="c", priority=PRIO_MEASURE)
    g, _ = s.tick(0)
    ok(len(g) == 1 and g[0].owner == "verify",
       f"verify (30) beats measure (40) and bulk (60); got {g[0].owner}")


def test_runtime_short_first_and_aging():
    section("runtime metadata: short work passes unstarted long work without starving it")
    s = fresh(1, aging_s=10.0)
    s.submit(0, EXCLUSIVE, owner="long", group="long", priority=PRIO_MEASURE,
             meta={"command_kind": "profile", "estimated_runtime_s": 600,
                   "safe_yield_kind": "profile_pass_boundary"})
    s.submit(0, EXCLUSIVE, owner="short", group="short", priority=PRIO_MEASURE,
             meta={"command_kind": "verify", "estimated_runtime_s": 30})
    g, _ = s.tick(0)
    ok(g and g[0].owner == "short",
       "a short request passes an unstarted long request in its priority band")
    ok(runtime_class(g[0].meta) == "short", "short metadata is classified correctly")
    s.release(g[0].lease_id)
    g2, _ = s.tick(1)
    ok(g2 and g2[0].owner == "long", "the deferred long request runs next, not discarded")
    ok(runtime_class(g2[0].meta) == "long", "long metadata is classified correctly")
    ok(runtime_class({"estimated_runtime_s": 120}) == "medium",
       "the 120-second boundary is medium")
    ok(runtime_class({"estimated_runtime_s": 301}) == "long",
       "work above 300 seconds is long")


def test_aging_prevents_starvation():
    section("aging: a low-priority request cannot be starved forever")
    s = fresh(1, aging_s=10.0)
    s.submit(0, EXCLUSIVE, owner="bulk", group="bulkgroup", priority=PRIO_BULK)

    # A relentless stream of higher-priority work, exactly like a sweep.
    now = 0.0
    served_bulk_at = None
    for i in range(60):
        now += 5.0
        s.submit(now, EXCLUSIVE, owner=f"sweep{i}", group="sweepgroup", priority=PRIO_MEASURE)
        for lv in list(s.leases.values()):        # each holder finishes immediately
            s.release(lv.lease_id)
        g, _ = s.tick(now)
        if any(x.owner == "bulk" for x in g):
            served_bulk_at = now
            break
    ok(served_bulk_at is not None,
       f"the bulk request eventually runs (at t={served_bulk_at}) despite a permanent "
       "stream of higher-priority work")

    # And without aging it starves -- which is what makes the test above meaningful.
    s2 = fresh(1, aging_s=0)
    s2.submit(0, EXCLUSIVE, owner="bulk", group="bulkgroup", priority=PRIO_BULK)
    starved = True
    now = 0.0
    for i in range(60):
        now += 5.0
        s2.submit(now, EXCLUSIVE, owner=f"sweep{i}", group="sweepgroup", priority=PRIO_MEASURE)
        for lv in list(s2.leases.values()):
            s2.release(lv.lease_id)
        g, _ = s2.tick(now)
        if any(x.owner == "bulk" for x in g):
            starved = False
            break
    ok(starved, "with aging OFF the same request starves (so the test above proves aging works)")


def test_fair_share_across_kernels():
    section("fair share: one kernel's backlog cannot lock out a sibling")
    s = fresh(4)
    # Kernel A queues a big same-priority backlog FIRST -- as a sweep does.
    for i in range(20):
        s.submit(0, EXCLUSIVE, owner=f"A{i}", group="kernelA", priority=PRIO_MEASURE)
    # Kernel B arrives a moment later with one request.
    s.submit(0.1, EXCLUSIVE, owner="B0", group="kernelB", priority=PRIO_MEASURE)

    g, _ = s.tick(1)
    groups = [x.group for x in g]
    ok("kernelB" in groups,
       f"kernel B gets a lane in the first pass despite queueing 20th; grants={groups}")
    ok(groups.count("kernelA") <= 3,
       f"and kernel A does not take all 4 lanes ({groups.count('kernelA')} of 4)")


def test_fifo_within_a_group():
    section("FIFO within one group -- fairness must not scramble a kernel's own order")
    s = fresh(1)
    for i in range(5):
        s.submit(i, EXCLUSIVE, owner=f"r{i}", group="same", priority=PRIO_MEASURE)
    seen = []
    now = 10.0
    for _ in range(5):
        g, _ = s.tick(now)
        if g:
            seen.append(g[0].owner)
            s.release(g[0].lease_id)
        now += 1
    ok(seen == [f"r{i}" for i in range(5)], f"submission order preserved: {seen}")


def test_lease_ttl_reclaim():
    section("crash recovery: a lease that stops being renewed is reclaimed")
    s = fresh(1)
    s.submit(0, EXCLUSIVE, owner="crasher", group="k")
    g, _ = s.tick(0, lease_ttl_s=100)
    ok(len(g) == 1, "granted")
    s.submit(1, EXCLUSIVE, owner="waiter", group="k2")
    g2, _ = s.tick(50, lease_ttl_s=100)
    ok(len(g2) == 0, "while the holder's TTL is alive, the waiter waits")
    g3, _ = s.tick(101, lease_ttl_s=100)
    ok(len(g3) == 1 and g3[0].owner == "waiter",
       "past the TTL the dead holder is reclaimed and the lane reassigned")
    ok(s.stats["reclaimed_leases"] == 1, "and the reclaim is counted, not silent")

    # A LIVE holder that renews must not be reclaimed -- the mirror of the above, and the one that
    # matters for a legitimate 40-minute climb.
    s2 = fresh(1)
    s2.submit(0, EXCLUSIVE, owner="slow", group="k")
    g, _ = s2.tick(0, lease_ttl_s=100)
    lid = g[0].lease_id
    for t in range(30, 300, 30):
        s2.renew(t, lid, 100)
        s2.tick(t, lease_ttl_s=100)
    ok(lid in s2.leases, "a holder that keeps renewing keeps its lane for as long as it needs")


def test_disconnect_releases_everything():
    section("a closed connection releases every lease it held")
    s = fresh(3)
    for i in range(3):
        s.submit(0, EXCLUSIVE, owner="dead", group="k")
    s.tick(0)
    ok(len(s.leases) == 3, "one owner holds three lanes")
    gone = s.release_owner("dead")
    ok(len(gone) == 3 and len(s.leases) == 0, f"disconnect released all {len(gone)}")
    # It must hand back the LEASES, not a tally: the daemon needs granted_at to journal how long
    # each reclaimed lease was held, and a count makes that time unrecoverable.
    ok(all(hasattr(lv, "granted_at") and hasattr(lv, "lane") for lv in gone),
       "and returns the leases themselves, so their held time can still be computed")


def test_queue_deadline():
    section("a request past its deadline is FAILED, not granted late")
    s = fresh(1)
    s.submit(0, EXCLUSIVE, owner="holder", group="k")
    s.tick(0)
    s.submit(0, EXCLUSIVE, owner="impatient", group="k2", queue_timeout_s=10)
    _, failed = s.tick(5)
    ok(not failed, "still inside its deadline at t=5")
    _, failed = s.tick(11)
    ok(len(failed) == 1 and failed[0].owner == "impatient",
       "past the deadline it is failed so the caller can decide, not silently held")
    ok(s.stats["expired_requests"] == 1, "and counted")


def test_foreign_and_drain():
    section("a foreign-busy or drained lane is never handed out")
    s = fresh(2)
    s.lanes[0].foreign_busy = True
    s.submit(0, EXCLUSIVE, owner="a", group="k")
    s.submit(0, EXCLUSIVE, owner="b", group="k2")
    g, _ = s.tick(0)
    ok(len(g) == 1 and g[0].lane == 1, "only the clean lane is granted")

    s2 = fresh(2)
    s2.lanes[1].disabled = True
    s2.submit(0, EXCLUSIVE, owner="a", group="k")
    s2.submit(0, EXCLUSIVE, owner="b", group="k2")
    g2, _ = s2.tick(0)
    ok(len(g2) == 1 and g2[0].lane == 0, "a drained lane is skipped")


def test_vram_admission_and_eviction():
    section("residents are admitted against VRAM, and evicted oldest-first")
    s = Scheduler([0], vram_total_mb=10000, aging_s=0)
    s.lanes[0].foreign_busy = False
    # cap = 80% of 10000 = 8000
    s.submit(0, RESIDENT, owner="r1", group="k1", vram_mb=5000)
    s.submit(0, RESIDENT, owner="r2", group="k2", vram_mb=2000)
    s.tick(0)
    ok(s.resident_vram_mb(0) == 7000, "two residents fit under the cap")

    s.submit(1, RESIDENT, owner="r3", group="k3", vram_mb=5000)
    g, _ = s.tick(1)
    ok(len(g) == 0, "a third that would exceed the cap is NOT admitted")

    plan = s.evict_candidates(0, need_mb=4000)
    ok([lv.owner for lv in plan] == ["r1"],
       f"evicting the oldest alone frees enough; plan={[lv.owner for lv in plan]}")
    plan2 = s.evict_candidates(0, need_mb=6000)
    ok([lv.owner for lv in plan2] == ["r1", "r2"], "a bigger need evicts oldest-first, in order")


def test_no_head_of_line_blocking():
    section("a request pinned to a busy lane does not block the rest of the queue")
    s = fresh(2)
    s.submit(0, EXCLUSIVE, owner="occupier", group="k0", lanes=(0,))
    s.tick(0)
    # This one can ONLY use lane 0, which is taken. It is also the highest priority.
    s.submit(1, EXCLUSIVE, owner="pinned", group="k1", lanes=(0,), priority=PRIO_VERIFY)
    s.submit(1, EXCLUSIVE, owner="anylane", group="k2", priority=PRIO_MEASURE)
    g, _ = s.tick(2)
    ok(any(x.owner == "anylane" for x in g),
       "the lower-priority request that CAN be placed still runs")
    ok(not any(x.owner == "pinned" for x in g), "and the blocked one stays queued")


def test_parse_priority():
    section("priority parsing is forgiving")
    ok(parse_priority("verify") == PRIO_VERIFY, "a band name resolves")
    ok(parse_priority("VERIFY") == PRIO_VERIFY, "case-insensitively")
    ok(parse_priority(35) == 35, "a raw int passes through")
    ok(parse_priority("nonsense") == PRIO_MEASURE, "garbage falls back rather than raising")
    ok(parse_priority(None) == PRIO_MEASURE, "so does None")
    ok(parse_priority(-5) == 0 and parse_priority(999) == 99, "out-of-range is clamped")


def test_snapshot_is_plain_data():
    section("the snapshot is serialisable -- it is what the CLI and the tests both read")
    import json
    s = fresh(2)
    s.submit(0, EXCLUSIVE, owner="a", group="k")
    s.submit(0, RESIDENT, owner="b", group="k", vram_mb=100)
    s.tick(0)
    snap = s.snapshot(1.0)
    ok(json.loads(json.dumps(snap)) == snap, "snapshot round-trips through JSON")
    ok(len(snap["lanes"]) == 2 and "stats" in snap, "and carries lanes + stats")


def main():
    print("=" * 72)
    print("scheduler/core.py — policy tests (fake clock, no GPU)")
    print("=" * 72)
    for fn in [
        test_exclusivity, test_resident_does_not_block_exclusive,
        test_wake_returns_to_its_own_lane, test_priority_order,
        test_runtime_short_first_and_aging, test_aging_prevents_starvation,
        test_fair_share_across_kernels,
        test_fifo_within_a_group, test_lease_ttl_reclaim,
        test_disconnect_releases_everything, test_queue_deadline,
        test_foreign_and_drain, test_vram_admission_and_eviction,
        test_no_head_of_line_blocking, test_parse_priority,
        test_snapshot_is_plain_data,
    ]:
        fn()
    print()
    if FAILS:
        print(f"FAIL: {len(FAILS)} check(s) failed")
        for f in FAILS:
            print("  - " + f)
        return 1
    print("SELFTEST PASS: the scheduler's policy claims hold")
    return 0


if __name__ == "__main__":
    sys.exit(main())
