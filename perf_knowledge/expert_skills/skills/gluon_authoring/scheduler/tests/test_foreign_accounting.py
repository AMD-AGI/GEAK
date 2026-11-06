#!/usr/bin/env python3
"""Who owns the VRAM on a lane: the broker's own work, or a stranger?

Getting this wrong is expensive in a way that hides itself. If our own footprint is not credited to
us, the broker fences the card it is currently using, and the run shows long queues and idle cards at
the same time -- a contradiction that reads as "the box is contended" rather than as a bug. If a
stranger's footprint IS credited to us, the broker hands out an occupied card and the measurement
taken there is silently wrong. So both directions are asserted here, plus the boundary between them.

No GPU and no socket: `_probe_lanes` is driven directly with an injected sysfs reader and an injected
clock, which is what makes the grace window testable at all -- it is a duration, and waiting one out
in real time would put a minute of sleep in the suite.

    python3 scheduler/tests/test_foreign_accounting.py
"""
from __future__ import annotations

import os
import sys
import tempfile

SCHED = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCHED)

import gpu_broker  # noqa: E402
from core import EXCLUSIVE, RESIDENT  # noqa: E402

_fail = 0


def ok(cond, msg):
    """Print, count -- and RAISE, so pytest sees a failure too.

    These read as a script (`python3 tests/test_foreign_accounting.py`) but check.sh runs the
    directory under pytest, which decides pass/fail from exceptions, not from a counter. With a bare
    counter a broken implementation reported "7 passed" while every check printed FAIL -- verified
    by reverting the accounting fix and running both ways. A test file that cannot fail in CI is
    worse than no test file: it is a green light nobody is holding.
    """
    global _fail
    if cond:
        print(f"  ok: {msg}")
        return
    _fail += 1
    print(f"  FAIL: {msg}")
    raise AssertionError(msg)


def section(t):
    print(f"\n# {t}")


class Box:
    """A fake card. `used` is what sysfs would report right now."""

    def __init__(self):
        self.busy = 0
        self.used = 0
        self.total = 196608

    def install(self, monkey_lanes=(0,)):
        gpu_broker.probe_lane = lambda lane: (self.busy, self.used, self.total)


def fresh(box, **kw):
    """A broker with hardware probing ON but pointed at the fake card."""
    gpu_broker.probe_lane = lambda lane: (box.busy, box.used, box.total)
    kw.setdefault("state_dir", tempfile.mkdtemp(prefix="fa-"))
    return gpu_broker.Broker([0], probe_hardware=True, **kw)


def at(broker, t):
    """Run one maintenance probe as if the wall clock read `t`."""
    real = gpu_broker.time.time
    gpu_broker.time.time = lambda: t
    try:
        broker._probe_lanes()
    finally:
        gpu_broker.time.time = real


def last_fence(broker):
    """The most recent lane_foreign/lane_clear record this broker journalled."""
    recs = [r for r in getattr(broker, "_test_journal", [])
            if r[0] in ("lane_foreign", "lane_clear")]
    return recs[-1] if recs else None


def capture_journal(broker):
    broker._test_journal = []
    orig = broker.journal

    def spy(event, **kw):
        broker._test_journal.append((event, kw))
        return orig(event, **kw)
    broker.journal = spy


# ---------------------------------------------------------------------------


def test_our_own_exclusive_footprint_is_not_foreign():
    section("VRAM we allocate under our OWN lease is ours, even though we never declared it")
    box = Box()
    b = fresh(box)
    capture_journal(b)
    # Take an exclusive lease the way nearly all real work does: no vram_mb declared, because the
    # caller does not know how much its benchmark will allocate.
    b.sched.submit(0, EXCLUSIVE, owner="bench", group="k", lanes=[0])
    b.sched.tick(0.0)
    ok(b.sched.exclusive_on(0) is not None, "an exclusive lease is held on the lane")

    box.used = 40000                      # our benchmark allocates 40 GB
    at(b, 100.0)
    ok(not b.sched.lanes[0].foreign_busy,
       "a lane we hold is never fenced -- we are the tenant")
    ok(b.sched.lanes[0].ours_vram_mb == 40000,
       "and the broker records that footprint as its own, without being told")

    # Release. The memory does not vanish the instant the lease ends: the process is still tearing
    # down, and the broker cannot see that across a container boundary.
    for lv in list(b.sched.leases.values()):
        b.sched.release(lv.lease_id)
    at(b, 110.0)
    ok(not b.sched.lanes[0].foreign_busy,
       "just after release the SAME memory is still explained -- this is the bug that fenced our own cards")


def test_grace_expires_so_a_real_tenant_is_still_caught():
    section("the self-credit is bounded: past the grace, unexplained memory is foreign again")
    box = Box()
    b = fresh(box, ours_grace_s=30.0)
    capture_journal(b)
    b.sched.submit(0, EXCLUSIVE, owner="bench", group="k", lanes=[0])
    b.sched.tick(0.0)
    box.used = 40000
    at(b, 100.0)
    for lv in list(b.sched.leases.values()):
        b.sched.release(lv.lease_id)

    at(b, 120.0)          # 20 s after last seen: inside the grace
    ok(not b.sched.lanes[0].foreign_busy, "inside the grace the residue is still ours")

    at(b, 140.0)          # 40 s after: grace spent, and the memory is STILL there
    ok(b.sched.lanes[0].foreign_busy,
       "past the grace it is unexplained again -- a leak of ours or a stranger, both worth fencing")
    ev = last_fence(b)
    ok(ev and ev[0] == "lane_foreign" and ev[1].get("why") == "vram",
       "and the journal says WHY it was fenced, so an operator need not guess")


def test_a_stranger_arriving_on_an_idle_lane_is_fenced_immediately():
    section("a lane we never held, holding somebody else's model, is foreign at once")
    box = Box()
    b = fresh(box)
    capture_journal(b)
    box.used = 60000
    at(b, 100.0)
    ok(b.sched.lanes[0].foreign_busy,
       "we hold no lease and never did: this memory was never ours to credit")
    ok(b.sched.lanes[0].ours_vram_mb == 0, "nothing is credited to us on a lane we never held")


def test_a_stranger_who_arrives_while_we_hold_is_absorbed_and_then_caught():
    section("the honest limit of observation: a tenant arriving under our lease is credited to us")
    # This is the cost of learning the footprint by watching instead of by declaration, and it is
    # worth stating rather than hiding. While we hold the lane we assume we are the only tenant --
    # which is what the lease MEANS -- so a stranger landing inside that window is absorbed. It is
    # caught one grace period after we let go, and exclusivity was already violated by then.
    box = Box()
    b = fresh(box, ours_grace_s=30.0)
    b.sched.submit(0, EXCLUSIVE, owner="bench", group="k", lanes=[0])
    b.sched.tick(0.0)
    box.used = 50000                       # not ours, but it arrived while we held the lane
    at(b, 100.0)
    ok(not b.sched.lanes[0].foreign_busy, "absorbed while the lease is held (documented limitation)")
    for lv in list(b.sched.leases.values()):
        b.sched.release(lv.lease_id)
    at(b, 200.0)
    ok(b.sched.lanes[0].foreign_busy, "but the grace expires and the stranger is then fenced")


def test_declared_resident_vram_still_counts():
    section("a declared resident is still explained -- the old contract is not broken")
    box = Box()
    b = fresh(box)
    b.sched.submit(0, RESIDENT, owner="park", group="k", lanes=[0], vram_mb=8000)
    b.sched.tick(0.0)
    box.used = 7000                        # under what the resident declared
    at(b, 100.0)
    ok(not b.sched.lanes[0].foreign_busy, "usage below the declared residency is explained")
    ok(b.sched.resident_vram_mb(0) == 8000, "and the declaration is still the source for it")


def test_busy_and_vram_are_reported_separately():
    section("the journal distinguishes a computing stranger from a memory-holding one")
    box = Box()
    b = fresh(box)
    capture_journal(b)
    box.busy, box.used = 90, 0
    at(b, 100.0)
    ev = last_fence(b)
    ok(ev and ev[1].get("why") == "busy",
       "a card computing for somebody else is fenced as `busy`, not as `vram`")
    ok(b.sched.lanes[0].foreign_busy, "and it is fenced")


def test_unreadable_counters_do_not_fence():
    section("a card we cannot read is not assumed guilty")
    b = gpu_broker.Broker([0], state_dir=tempfile.mkdtemp(prefix="fa-"), probe_hardware=True)
    gpu_broker.probe_lane = lambda lane: (None, None, None)
    at(b, 100.0)
    ok(not b.sched.lanes[0].foreign_busy,
       "unreadable is 'cannot tell', never 'foreign' -- fencing on it would idle a whole pool")


def test_a_reclaim_ends_a_lease_for_the_auditor_too():
    section("a reclaimed lane is free -- the auditor must not call the next grant a double-booking")
    # The daemon releases a dead client's lanes immediately, but audit_journal.py only cleared a lane
    # on `release`, so the very next grant on it looked like two leases at once. On a real run six of
    # seven reported exclusivity FAILURES were this phantom. A false FAIL is not a harmless nit here:
    # the report says "every number from that lane is suspect", which throws away good measurements.
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "aj", os.path.join(os.path.dirname(SCHED), "kernel_workflow", "scheduler", "audit_journal.py")
        if False else os.path.join(SCHED, "audit_journal.py"))
    aj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(aj)

    recs = [
        {"seq": 1, "event": "broker_start", "lanes": [0]},
        {"seq": 2, "event": "grant", "lane": 0, "lease": 1, "kind": "exclusive",
         "owner": "dead", "group": "k"},
        {"seq": 3, "event": "disconnect_reclaim", "owner": "dead", "leases": 1, "queued": 0,
         "reclaimed": [{"lease": 1, "lane": 0, "kind": "exclusive", "group": "k", "held_s": 2.0}]},
        {"seq": 4, "event": "grant", "lane": 0, "lease": 2, "kind": "exclusive",
         "owner": "next", "group": "k"},
        {"seq": 5, "event": "release", "lane": 0, "lease": 2, "owner": "next", "held_s": 1.0},
    ]
    a = aj.audit(recs)
    ok(a["double_bookings"] == 0,
       f"the grant after a reclaim is not a double-booking (got {a['double_bookings']})")
    ok(a["releases"] == a["grants"],
       f"a reclaimed lease counts as ended, so releases match grants ({a['releases']}/{a['grants']})")

    # And the check must still FAIL when exclusivity is genuinely broken -- no reclaim in sight.
    bad = [
        {"seq": 1, "event": "broker_start", "lanes": [0]},
        {"seq": 2, "event": "grant", "lane": 0, "lease": 1, "kind": "exclusive", "owner": "a", "group": "k"},
        {"seq": 3, "event": "grant", "lane": 0, "lease": 2, "kind": "exclusive", "owner": "b", "group": "k"},
    ]
    ok(aj.audit(bad)["double_bookings"] == 1,
       "and a real double-booking is still reported")


def main():
    tests = [
        test_our_own_exclusive_footprint_is_not_foreign,
        test_grace_expires_so_a_real_tenant_is_still_caught,
        test_a_stranger_arriving_on_an_idle_lane_is_fenced_immediately,
        test_a_stranger_who_arrives_while_we_hold_is_absorbed_and_then_caught,
        test_declared_resident_vram_still_counts,
        test_busy_and_vram_are_reported_separately,
        test_unreadable_counters_do_not_fence,
        test_a_reclaim_ends_a_lease_for_the_auditor_too,
    ]
    real_probe = gpu_broker.probe_lane
    try:
        for t in tests:
            t()
    finally:
        gpu_broker.probe_lane = real_probe
    print("\nSELFTEST PASS: our own footprint is ours, a stranger's is not, and the journal says which."
          if not _fail else f"\nFAIL: {_fail} check(s) failed.")
    return 1 if _fail else 0


if __name__ == "__main__":
    sys.exit(main())
