#!/usr/bin/env python3
"""The GPU broker: one daemon per box, a Unix socket, NDJSON in and out.

    python3 gpu_broker.py --serve [--gpus 0,1,2,3,4,5,6,7]   # run it (started deliberately;
                                                              # client autostart is OFF by default)
    python3 gpu_broker.py --status                            # what is happening right now
    python3 gpu_broker.py --drain 3                           # stop granting lane 3, let it finish
    python3 gpu_broker.py --undrain 3
    python3 gpu_broker.py --shutdown
    python3 gpu_broker.py --selftest                          # no GPU, no socket

WHAT THIS IS FOR. Eight cards, sixteen kernels, each kernel an IC Tech Lead that profiles, sweeps,
fans out arms and climbs -- and every one of those steps needs a GPU only at the moment it measures.
Today those moments are arbitrated by `flock` in `gpu_lock.sh`, which can answer "is this lane free"
but not "who should get it next", and which cannot represent a parked process holding VRAM at all.
This daemon owns both questions for the whole box.

THE ARCHITECTURE IS DELIBERATELY BORING. All policy is in `core.Scheduler`, which has no I/O and
takes `now` as a parameter. This file is the shell: accept connections, parse a line, call the state
machine, write a line back. That split is why the fairness and aging properties can be tested in
milliseconds with a fake clock (`tests/test_scheduler_core.py`) instead of by running the fleet.

CONNECTION-SCOPED LEASES ARE THE CRASH-RECOVERY MECHANISM. A lease belongs to the connection that
took it. When the socket closes -- normal exit, crash, `kill -9`, container teardown -- the kernel
tells us, and every lease on it is released immediately. That covers the overwhelming majority of
failures without any liveness probing, and it works across PID namespaces, which matters because the
clients here may be inside a different container from the daemon. The TTL + heartbeat is the backstop
for the one case the socket cannot see: a client still connected but wedged.

WHY NOT JUST KEEP flock. Three things it cannot express, all of which this workload has:
  1. A QUEUE with priorities. With 16 kernels on 8 cards, flock hands the lane to whoever happens to
     call at the right microsecond. A kernel needing one 3-second verify can sit behind a kernel
     running a 400-point sweep, indefinitely, and nothing records that it happened.
  2. RESIDENCY. `sweep_driver --serve` parks in VRAM between readings, which is worth 14-48x per
     reading. Under flock a parked worker either holds the lock (blocking the card for minutes while
     computing nothing) or drops it (and any sibling may then land a timed run on top of it). Today
     it does the latter and nothing notices -- see the `--serve` bypass note in scheduler/README.md.
  3. FAIRNESS ACROSS KERNELS. flock has no notion of "this kernel already holds three lanes".

OPT-IN IN GEAK. GEAK's kernel_workflow/scripts/gpu_lock.sh (flock on /tmp/team_gpu_locks) is the
box's single GPU lock; it consults this daemon only when GEAK_GPU_BROKER=1 and a socket is listening,
and even then the granted lane still takes that same flock, so brokered and non-brokered processes
interlock. Socket: $GEAK_GPU_BROKER_SOCK, else $GEAK_GPU_RUNTIME_ROOT/gpu.sock (default
~/.cache/tile-runtime/gpu-broker/gpu.sock); journal under $GEAK_GPU_RUNTIME_ROOT/state/.

FAILURE POSTURE: DEGRADE, NEVER BLOCK THE FLEET. If this daemon is not running, `gpu_lock.sh` falls
back to its flock path and every existing call site behaves exactly as before. If it dies
mid-run, clients see their socket close, and the client library falls back to flock rather than
failing the command. A scheduler that can take down a 12-hour fleet run when it crashes is worse than
no scheduler; this one is designed so its own absence is a performance regression, not an outage.
"""
from __future__ import annotations

import argparse
import errno
import json
import os
import socket
import socketserver
import select
import sys
import threading
import time
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from core import (EXCLUSIVE, RESIDENT, Scheduler, parse_priority, public_metadata,
                  runtime_class)  # noqa: E402

RUNTIME_ROOT = os.environ.get(
    "GEAK_GPU_RUNTIME_ROOT",
    os.path.join(os.path.expanduser("~"), ".cache", "tile-runtime", "gpu-broker"),
)
DEFAULT_SOCK = os.environ.get("GEAK_GPU_BROKER_SOCK", os.path.join(RUNTIME_ROOT, "gpu.sock"))
DEFAULT_STATE = os.environ.get("GEAK_GPU_BROKER_STATE", os.path.join(RUNTIME_ROOT, "state"))
PROTO_VERSION = 1


# ---------------------------------------------------------------------------
# Hardware probing. Isolated here so every other layer is testable without a GPU.
# ---------------------------------------------------------------------------
def detect_lanes():
    """Visible GPU ids. HIP_VISIBLE_DEVICES wins when set -- inside a container that is the truth,
    and asking sysfs there reports the host's cards, which is how a run gets 'allocated' a device
    number that does not exist in its own namespace.

    LOGICAL vs PHYSICAL. A lane id is handed to the command as HIP_VISIBLE_DEVICES and to
    gpu_lock.sh as its GPU id, i.e. it is a HIP ordinal of the CALLER's process. It equals the
    physical card index (and the renderD(128+8*id) sysfs node the idleness probes read, and the
    /tmp/team_gpu_locks/gpu_<id>.lock file every GEAK process flocks) only when no
    ROCR_VISIBLE_DEVICES / container device remap is in effect. Ids taken from HIP_VISIBLE_DEVICES
    here are therefore only meaningful when every tenant of the box uses the same numbering; under
    a ROCR mask pass --gpus explicitly in the numbering the clients use, and expect the sysfs probe
    (probe_lane) to describe the physical card with that index, which may be a different one."""
    env = os.environ.get("GEAK_GPU_BROKER_GPUS") or os.environ.get("HIP_VISIBLE_DEVICES")
    if env:
        out = []
        for tok in env.split(","):
            tok = tok.strip()
            if tok.isdigit():
                out.append(int(tok))
        if out:
            return out
    n = 0
    while os.path.exists(f"/sys/class/drm/renderD{128 + 8 * n}"):
        n += 1
    return list(range(n)) if n else [0]


def _sysfs_dev(lane):
    try:
        return os.path.realpath(f"/sys/class/drm/renderD{128 + 8 * int(lane)}/device")
    except Exception:                                            # noqa: BLE001
        return None


def probe_lane(lane):
    """(busy_pct, vram_used_mb, vram_total_mb). Any unreadable field returns None for that field,
    and a None is treated by callers as 'cannot tell', never as zero -- the same rule the evidence
    layer uses. A scheduler that reads an unreadable counter as 0 will happily place a timed run on
    a card it cannot see."""
    dev = _sysfs_dev(lane)
    if not dev or not os.path.isdir(dev):
        return None, None, None

    def rd(name):
        try:
            with open(os.path.join(dev, name)) as fh:
                return int(fh.read().strip())
        except Exception:                                        # noqa: BLE001
            return None

    busy = rd("gpu_busy_percent")
    used = rd("mem_info_vram_used")
    total = rd("mem_info_vram_total")
    return (busy,
            None if used is None else used // (1024 * 1024),
            None if total is None else total // (1024 * 1024))


# HOW LONG AN EXCLUSIVE LEASE MAY BE HELD BEFORE IT IS WORTH SAYING SO. Not a limit: nothing is
# revoked here, because a long hold is often exactly right (one `plain_autotune` pass is one command
# and owns its card for the duration). It is a REPORTING floor, and the reason it needs one is that
# the two cases are indistinguishable in the accounting today.
#
# Measured over a real 4-GPU fleet, 1380 leases: p50 held 0.6 s, p95 5.0 s -- and 8 leases (0.58%)
# held more than 60 s, between them 65.5% of ALL GPU-seconds the fleet ever held. The single longest
# ran 1294 s. Those few holds are what produced every wait over a minute in the run, including a
# 1010 s one, and all 57 queue timeouts (every one of them a resident sweep worker giving up on its
# 120 s measurement window while a whole-command lease sat on the lane).
#
# TTL does not catch this: the client heartbeat renews for as long as the process lives, so a lease
# may exceed `lease_ttl_s` indefinitely and never expire. `lease_expired` was 0 across that entire
# run while a 1294 s hold was in flight.
LEASE_OVERRUN_S = 120.0


class Broker:
    def __init__(self, lanes, state_dir=DEFAULT_STATE, aging_s=30.0,
                 lease_ttl_s=900.0, foreign_busy_pct=5, foreign_vram_mb=1024,
                 probe_hardware=True, ours_grace_s=90.0,
                 lease_overrun_s=LEASE_OVERRUN_S):
        self.lock = threading.RLock()
        self.sched = Scheduler(lanes, aging_s=aging_s)
        self.state_dir = state_dir
        self.lease_ttl_s = float(lease_ttl_s)
        self.lease_overrun_s = float(lease_overrun_s)
        # Lease ids already reported as overrunning, so the 2 s maintenance tick says it ONCE per
        # lease rather than 600 times across a 20-minute hold.
        self._overrun_said = set()
        # SYNTHETIC LANES. With this off, no sysfs is consulted and no lane is ever marked
        # foreign-busy. It exists for tests: the arbitration logic does not care whether a lane is a
        # real GPU, but the FOREIGN check does -- and on a box where GPU0 legitimately holds 231 GB
        # of somebody's model, every test that allocates "lane 0" fails for a reason that has
        # nothing to do with what it is testing. Draining/undraining cannot substitute, because the
        # 2-second maintenance probe re-marks the lane immediately.
        self.probe_hardware = bool(probe_hardware)
        self.foreign_busy_pct = int(foreign_busy_pct)
        self.foreign_vram_mb = int(foreign_vram_mb)
        # How long a released lane keeps being credited for the VRAM we were just using. Covers the
        # gap between "our lease ended" and "our process actually freed its memory", which the broker
        # cannot observe directly across a container boundary. Short by default: this window is
        # exactly how long a genuinely foreign tenant could arrive unnoticed.
        self.ours_grace_s = float(ours_grace_s)
        self.started_at = time.time()
        self.sock_path = None
        self.waiters = {}          # rid -> threading.Event, signalled when that rid is granted
        self.granted = {}          # rid -> Lease
        self.failed = {}           # rid -> reason
        self.conn_owners = {}      # owner -> connection id
        self._jlock = threading.Lock()
        self._jseq = 0
        os.makedirs(self.state_dir, exist_ok=True)
        self.journal_path = os.path.join(self.state_dir, "journal.ndjson")
        self._probe_lanes(initial=True)

    # -- journal ------------------------------------------------------------
    def journal(self, event, **kw):
        """Append-only. This is the ONLY durable record of who held what and for how long, and it is
        what makes a slow fleet run diagnosable after the fact: `wait_s` per acquisition tells you
        whether you are GPU-bound or scheduler-bound, and nothing else in the system can tell you
        that. Best-effort -- a full disk must not stop the fleet.

        `t` is MICROSECOND resolution and every record carries a monotonic `seq`, because the
        headline property an auditor wants to reconstruct from this file is "was a lane ever held
        exclusively by two leases at once". At the original millisecond rounding a real 16-kernel
        run put 88 of 238 events into ties, and a reader that breaks a tie the wrong way (grant
        before release) reports 77 double-bookings that did not happen. An audit record whose
        resolution is coarser than the thing it records cannot answer the question it exists for --
        and here it answered it alarmingly and wrongly.

        `seq` is the tie-break of record: it is assigned under the same lock that mutates the
        scheduler, so it is the true order of state changes regardless of clock granularity.
        """
        with self._jlock:
            self._jseq += 1
            seq = self._jseq
        rec = {"t": round(time.time(), 6), "seq": seq, "event": event}
        rec.update(kw)
        try:
            with open(self.journal_path, "a") as fh:
                fh.write(json.dumps(rec, default=str) + "\n")
        except Exception:                                        # noqa: BLE001
            pass

    # -- hardware -----------------------------------------------------------
    def _probe_lanes(self, initial=False):
        """Mark lanes that a FOREIGN tenant is using, so we never hand one out.

        The subtlety that makes this worth its own method: our OWN work legitimately holds VRAM, so a
        naive `vram_used > threshold => busy` test (which is what gpu_lock.sh does today) would
        permanently fence off every card we are successfully using. Observed on a box here: a card at
        231 GB used and 0% busy, which the old test calls 'foreign' forever.

        WHAT COUNTS AS "OURS" IS OBSERVED, NOT DECLARED. This used to subtract only the VRAM that
        RESIDENT leases had declared. That is the entire footprint only if every GPU consumer parks
        as a resident and states its size up front -- and the exclusive path, which is how most work
        actually acquires, declares nothing. So on any run made of exclusive leases the explained
        total is structurally zero, every megabyte our own benchmark allocates reads as a stranger's,
        and the scheduler fences the card it is itself using. The symptom is self-contradictory and
        therefore easy to misread: long queues and idle cards at the same time.

        A declaration cannot be the mechanism, because it requires every caller to know and report a
        number that only the allocator knows. Observation can: while we hold a lease on a lane, we
        are the only tenant that should be running, so whatever VRAM is present is BY DEFINITION
        ours. Record that, and it explains the same memory after release -- for a bounded grace
        period, because a process that exited frees its memory and a stranger who arrives later must
        still be caught. Past the grace the residue is unexplained again, which is the correct answer
        whether it is a leak of ours or somebody else's job.

        The grace is the whole safety argument, so it is short by default and configurable: too long
        and a real tenant is invisible for that window; too short and our own allocator's teardown
        looks foreign. It is time-bounded rather than event-bounded on purpose -- an event ("the
        process exited") is exactly what a scheduler on the other side of a container boundary cannot
        observe.
        """
        if not self.probe_hardware:
            return
        now = time.time()
        for lane_id, lane in self.sched.lanes.items():
            busy, used, total = probe_lane(lane_id)
            if total:
                lane.vram_total_mb = total
            if busy is None and used is None:
                lane.foreign_busy = False        # cannot tell -> do not fence
                continue
            we_hold = (self.sched.exclusive_on(lane_id) is not None
                       or bool(self.sched.residents_on(lane_id)))
            declared = self.sched.resident_vram_mb(lane_id)
            if we_hold:
                # We are the tenant, so this reading is our own footprint. Keep the high-water mark:
                # a sweep's usage moves with the config under test and the peak is what has to be
                # explained after we let go.
                if used is not None:
                    lane.ours_vram_mb = max(lane.ours_vram_mb, used)
                lane.ours_seen_at = now
            elif lane.ours_seen_at and (now - lane.ours_seen_at) > self.ours_grace_s:
                lane.ours_vram_mb = 0            # grace spent: stop crediting ourselves for it
                lane.ours_seen_at = 0.0
            ours_vram = max(declared, lane.ours_vram_mb)
            unexplained = None if used is None else max(0, used - ours_vram)
            # WHY, not just whether. A bare boolean makes the journal unable to answer the only
            # question an operator has when a pool sits fenced -- is this someone else's job, or our
            # own residue? Both readings lead to opposite actions, and the audit could not tell them
            # apart after the fact.
            reasons = []
            if not we_hold:
                if busy is not None and busy > self.foreign_busy_pct:
                    reasons.append("busy")
                if unexplained is not None and unexplained > self.foreign_vram_mb:
                    reasons.append("vram")
            foreign = bool(reasons)
            if foreign != lane.foreign_busy:
                self.journal("lane_foreign" if foreign else "lane_clear",
                             lane=lane_id, busy=busy, used_mb=used, ours_mb=ours_vram,
                             ours_declared_mb=declared, ours_observed_mb=lane.ours_vram_mb,
                             unexplained_mb=unexplained, why=",".join(reasons) or None)
            lane.foreign_busy = foreign

    def _overrun_limit(self, lv):
        estimate = public_metadata(lv.meta).get("estimated_runtime_s")
        try:
            return float(estimate) if float(estimate) > 0 else self.lease_overrun_s
        except (TypeError, ValueError):
            return self.lease_overrun_s

    def _finish_lease_locked(self, lv, owner, end_reason="release", record_pause=True):
        """End one lease and journal the cooperative pause outcome, if any."""
        pause = self.sched.clear_pause(lv.lease_id)
        if not self.sched.release(lv.lease_id):
            return False
        held = time.time() - lv.granted_at
        overrun_limit = self._overrun_limit(lv)
        self.journal("release", lease=lv.lease_id, lane=lv.lane, owner=owner,
                     group=lv.group, held_s=round(held, 3), end_reason=end_reason,
                     metadata=public_metadata(lv.meta),
                     overrun_limit_s=overrun_limit,
                     overran=bool(overrun_limit > 0 and lv.kind == EXCLUSIVE and held >= overrun_limit))
        self._overrun_said.discard(lv.lease_id)
        if pause and record_pause:
            self.journal("pause_yield", lease=lv.lease_id, lane=lv.lane, owner=owner,
                         group=lv.group, action=end_reason, reason=pause["reason"],
                         safe_yield_kind=pause["safe_yield_kind"])
        if pause:
            self.sched.stats["cooperative_yields"] += 1
        return True

    def _cancel_request_locked(self, rid, owner, reason):
        """Remove a queued (or just-granted) request before a dead client can orphan it."""
        queued = next((r for r in self.sched.queue if r.rid == rid), None)
        if queued is not None:
            self.sched.queue.remove(queued)
        self.waiters.pop(rid, None)
        self.failed.pop(rid, None)
        granted = self.granted.pop(rid, None)
        now = time.time()
        if granted is not None:
            self._finish_lease_locked(granted, owner, end_reason="cancel_after_grant")
        self.journal("cancel", rid=rid, owner=owner, reason=reason,
                     state="granted" if granted else "queued",
                     group=(granted.group if granted else (queued.group if queued else None)),
                     waited_s=round(now - (queued.enqueued_at if queued else now), 3),
                     lease=(granted.lease_id if granted else None),
                     lane=(granted.lane if granted else None))
        return queued is not None or granted is not None

    def _cancel_owner_queue_locked(self, owner, reason):
        """Cancel each queued request individually so the journal preserves the reason."""
        rids = [r.rid for r in self.sched.queue if r.owner == owner]
        for rid in rids:
            self._cancel_request_locked(rid, owner, reason)
        return len(rids)

    def _request_cooperative_pauses_locked(self, now):
        """Ask safely-yieldable holders to cooperate; never suspend a process."""
        for req in self.sched._fair_order(now):
            for lane_id in req.lanes:
                holder = self.sched.exclusive_on(lane_id)
                if holder is None or holder.owner == req.owner:
                    continue
                request_key = (int(self.sched._effective_priority(req, now)),
                               self.sched._runtime_rank(req))
                holder_key = (holder.priority, runtime_class(holder.meta))
                is_higher_priority = request_key[0] < holder_key[0]
                is_short_ahead = (request_key[0] == holder_key[0]
                                  and request_key[1] == 0
                                  and holder_key[1] == "long")
                if not (is_higher_priority or is_short_ahead):
                    continue
                reason = "higher_priority_waiter" if is_higher_priority else "short_waiter"
                pause = self.sched.request_pause(holder.lease_id, now, reason,
                                                 requested_by=req.rid)
                if pause and pause["requested_at"] == now:
                    self.journal("pause_requested", lease=holder.lease_id, lane=holder.lane,
                                 owner=holder.owner, group=holder.group, reason=reason,
                                 waiting_rid=req.rid, waiting_group=req.group,
                                 safe_yield_kind=pause["safe_yield_kind"])

    # -- the pump -----------------------------------------------------------
    def _pump(self):
        """Run the state machine and wake whoever it granted. Caller must hold self.lock."""
        now = time.time()
        # Snapshot enqueue times BEFORE tick(), which removes granted requests from the queue.
        # `waited_s` on the grant record is the scheduler's own price -- the wall clock a caller
        # spent queueing rather than measuring -- and it is the one number that says whether a slow
        # fleet is GPU-bound or scheduler-bound. It was missing from the journal at first, which
        # made audit_journal.py's cost line permanently blank: the client knew the wait and the
        # daemon knew the wait, and neither wrote it down where an auditor could see it.
        enq_at = {r.rid: r.enqueued_at for r in self.sched.queue}
        granted, failed = self.sched.tick(now, lease_ttl_s=self.lease_ttl_s)
        # TTL reclaim: a holder that stopped renewing is presumed dead and loses its lane. This is
        # rarer than a disconnect (the socket usually closes first) and correspondingly more
        # alarming when it happens -- it means a client was alive enough to hold a connection and
        # too wedged to renew. It must be in the journal for `audit_journal.py` to see it.
        for lv in getattr(self.sched, "last_reclaimed", []):
            self.journal("lease_expired", lease=lv.lease_id, lane=lv.lane, kind=lv.kind,
                         owner=lv.owner, group=lv.group,
                         held_s=round(now - lv.granted_at, 3))
        for lv in granted:
            self.granted[lv.rid] = lv
            self.journal("grant", rid=lv.rid, lease=lv.lease_id, lane=lv.lane, kind=lv.kind,
                         owner=lv.owner, group=lv.group,
                         priority=lv.priority, runtime_class=runtime_class(lv.meta),
                         metadata=public_metadata(lv.meta),
                         waited_s=round(now - enq_at.get(lv.rid, now), 3))
            resident_event = lv.meta.get("resident_event")
            if lv.kind == RESIDENT or resident_event == "wake":
                self.journal("resident_park" if lv.kind == RESIDENT else "resident_wake",
                             rid=lv.rid, lease=lv.lease_id, lane=lv.lane, owner=lv.owner,
                             group=lv.group, reason=lv.meta.get("reason") or resident_event,
                             exclusive=lv.kind == EXCLUSIVE,
                             waited_s=round(now - enq_at.get(lv.rid, now), 3),
                             metadata=public_metadata(lv.meta))
            ev = self.waiters.get(lv.rid)
            if ev:
                ev.set()
        for r in failed:
            self.failed[r.rid] = "queue timeout"
            self.journal("queue_timeout", rid=r.rid, owner=r.owner,
                         waited_s=round(now - r.enqueued_at, 1))
            ev = self.waiters.get(r.rid)
            if ev:
                ev.set()
        self._request_cooperative_pauses_locked(now)

    def _note_overruns(self, now):
        """Say, ONCE per lease, that a hold has crossed the reporting floor.

        Deliberately observation-only. Revoking here would be wrong: the holder is usually a whole
        command that is legitimately still measuring, and killing it mid-flight destroys the work the
        lease exists to protect. What is wrong today is that nobody can SEE it -- `held_s` is
        journalled at release, so a 20-minute hold is invisible for 20 minutes, which is exactly the
        window in which someone might have acted on it."""
        if self.lease_overrun_s <= 0:
            return
        live = set()
        for lid, lv in list(self.sched.leases.items()):
            live.add(lid)
            if lid in self._overrun_said or lv.kind != EXCLUSIVE:
                continue
            held = now - lv.granted_at
            threshold = self._overrun_limit(lv)
            if threshold > 0 and held >= threshold:
                self._overrun_said.add(lid)
                waiting = len(getattr(self.sched, "queue", ()) or ())
                self.journal("lease_overrun", lease=lid, lane=lv.lane, owner=lv.owner,
                             group=lv.group, held_s=round(held, 1),
                             threshold_s=threshold, queued_behind=waiting)
        self._overrun_said &= live          # a released lease cannot overrun again

    def maintenance(self):
        with self.lock:
            self._probe_lanes()
            self._note_overruns(time.time())
            self._pump()

    # -- request handlers ---------------------------------------------------
    def handle(self, req, owner, peer_alive=None):
        op = req.get("op")
        fn = getattr(self, f"op_{op}", None) if op else None
        if fn is None:
            return {"ok": False, "error": f"unknown op {op!r}"}
        try:
            if op == "acquire":
                return fn(req, owner, peer_alive=peer_alive)
            return fn(req, owner)
        except Exception as exc:                                 # noqa: BLE001
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    def op_ping(self, req, owner):
        return {"ok": True, "proto": PROTO_VERSION, "pid": os.getpid(),
                "uptime_s": round(time.time() - self.started_at, 1),
                "lanes": sorted(self.sched.lanes)}

    def op_acquire(self, req, owner, peer_alive=None):
        """Block until granted, or until the queue deadline. Returns the lease.

        The blocking happens HERE, in the connection's own thread, so the client just does a
        request/response and does not implement a polling loop -- which is what stops every client
        inventing its own retry cadence and hammering the socket.
        """
        kind = req.get("kind", EXCLUSIVE)
        timeout = float(req.get("timeout_s", 1800))
        group = req.get("group")
        if req.get("require_group") and not str(group or "").strip():
            return {"ok": False, "error": "group is required for this managed brokered run",
                    "waited_s": 0.0}
        meta = dict(req.get("meta") or {})
        for key in ("command_kind", "estimated_runtime_s", "safe_yield_kind"):
            if req.get(key) not in (None, ""):
                meta[key] = req[key]
        # UNSATISFIABLE ASKS FAIL IMMEDIATELY. A request fenced to lanes this broker does not own can
        # never be granted, so queueing it means blocking the caller for its full timeout and then
        # refusing -- an outcome indistinguishable from a busy box, arriving fifteen minutes late.
        # It is a configuration error and it should read as one.
        #
        # Not hypothetical: sweep_pool's selftest asks for lane 0 while a broker started with
        # `--gpus 1,2` was running, and hung for the entire 900 s window.
        want = req.get("want_lane")
        fence = [int(x) for x in (req.get("lanes") or []) if str(x).lstrip("-").isdigit()]
        unknown = [x for x in fence if x not in self.sched.lanes]
        if fence and len(unknown) == len(fence):
            return {"ok": False, "error":
                    f"no such lane(s) {sorted(set(unknown))}; this broker owns "
                    f"{sorted(self.sched.lanes)}", "waited_s": 0.0}
        if want is not None and int(want) not in self.sched.lanes:
            return {"ok": False, "error":
                    f"no such lane {want}; this broker owns {sorted(self.sched.lanes)}",
                    "waited_s": 0.0}
        with self.lock:
            r = self.sched.submit(
                time.time(), kind=kind, owner=owner,
                group=group, priority=parse_priority(req.get("priority")),
                lanes=tuple(req.get("lanes") or ()) or tuple(self.sched.lanes),
                vram_mb=int(req.get("vram_mb") or 0),
                want_lane=req.get("want_lane"),
                queue_timeout_s=timeout, meta=meta,
            )
            ev = threading.Event()
            self.waiters[r.rid] = ev
            t_queued = time.time()
            self.journal("enqueue", rid=r.rid, kind=kind, owner=owner, group=r.group,
                         priority=r.priority, want_lane=r.want_lane, vram_mb=r.vram_mb,
                         runtime_class=runtime_class(r.meta),
                         metadata=public_metadata(r.meta))
            self._pump()

        # Wait outside the lock, but inspect the peer every 100 ms. A blocking Event.wait() cannot
        # see a queued client's disconnect because this handler owns the only reader for its socket;
        # it would leave a zombie request until the queue timeout. The peer check removes it on the
        # first readable EOF and journals `client_disconnected`.
        deadline = time.monotonic() + timeout + 5.0
        disconnected = False
        while not ev.is_set():
            if peer_alive is not None and not peer_alive():
                disconnected = True
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            ev.wait(timeout=min(0.1, remaining))

        with self.lock:
            if disconnected or (peer_alive is not None and not peer_alive()):
                self._cancel_request_locked(r.rid, owner, "client_disconnected")
                self._pump()
                return {"ok": False, "error": "client disconnected while queued",
                        "cancelled": True, "waited_s": round(time.time() - t_queued, 3)}
            self.waiters.pop(r.rid, None)
            lv = self.granted.pop(r.rid, None)
            why = self.failed.pop(r.rid, None)
            if lv is None:
                # Not granted: make sure it is not still sitting in the queue.
                self.sched.queue = [q for q in self.sched.queue if q.rid != r.rid]
                if why is None:
                    self.journal("cancel", rid=r.rid, owner=owner, reason="client_wait_timeout",
                                 state="queued", group=r.group,
                                 waited_s=round(time.time() - t_queued, 3))
                self._pump()
                return {"ok": False, "error": why or "timeout waiting for a GPU",
                        "waited_s": round(time.time() - t_queued, 1)}
            # The renew token, minted once and returned only to the actor that won the lease. It
            # lets a DEDICATED heartbeat socket renew without being the owning connection -- see
            # op_renew for why a second socket is required rather than merely convenient.
            tok = lv.meta.get("_token")
            if not tok:
                tok = uuid.uuid4().hex
                lv.meta["_token"] = tok
            return {"ok": True, "lease": lv.lease_id, "lane": lv.lane, "kind": lv.kind,
                    "ttl_s": self.lease_ttl_s, "token": tok,
                    "waited_s": round(time.time() - t_queued, 1)}

    def op_release(self, req, owner):
        with self.lock:
            lid = int(req.get("lease", 0))
            lv = self.sched.leases.get(lid)
            if lv and lv.owner != owner:
                return {"ok": False, "error": "not your lease"}
            ok = self._finish_lease_locked(lv, owner, req.get("reason", "release")) if lv else False
            self._pump()
            return {"ok": ok}

    def _authorise_lease(self, req, owner):
        lid = int(req.get("lease", 0))
        lv = self.sched.leases.get(lid)
        if lv is None:
            return None, {"ok": False, "error": "no such lease (expired or released)", "gone": True}
        tok = req.get("token")
        if lv.owner != owner and (not tok or tok != lv.meta.get("_token")):
            return None, {"ok": False, "error": "not your lease"}
        return lv, None

    def op_boundary(self, req, owner):
        """A cooperative safe-point: query, continue, yield, or cancel; never force a process."""
        with self.lock:
            lv, err = self._authorise_lease(req, owner)
            if err:
                return err
            action = str(req.get("action", "status")).lower()
            pause = self.sched.pause_state(lv.lease_id)
            if action == "status":
                return {"ok": True, "lease": lv.lease_id, "pause_requested": pause,
                        "next": "yield|continue|cancel" if pause else "continue"}
            if action == "continue":
                cleared = self.sched.clear_pause(lv.lease_id)
                self.journal("pause_continue", lease=lv.lease_id, lane=lv.lane, owner=owner,
                             group=lv.group, reason=(cleared or {}).get("reason"))
                return {"ok": True, "lease": lv.lease_id, "action": action,
                        "pause_requested": False}
            if action not in ("yield", "cancel"):
                return {"ok": False, "error": "boundary action must be status, continue, yield, or cancel"}
            pause = self.sched.pause_state(lv.lease_id)
            self._finish_lease_locked(lv, owner, f"cooperative_{action}", record_pause=False)
            self.journal(f"pause_{action}", lease=lv.lease_id, lane=lv.lane, owner=owner,
                         group=lv.group, reason=(pause or {}).get("reason") or "owner_choice",
                         safe_yield_kind=(pause or {}).get(
                             "safe_yield_kind", lv.meta.get("safe_yield_kind")))
            self._pump()
            return {"ok": True, "lease": lv.lease_id, "action": action,
                    "pause_requested": bool(pause)}

    def op_renew(self, req, owner):
        """Renew a lease. Authorised by the owning CONNECTION, or by the lease's token.

        The token exists because a heartbeat must not share the caller's socket. `Conn.call` holds a
        per-connection mutex for the whole round-trip, so a renew queued behind the caller's own
        BLOCKING acquire cannot fire -- which is the steady state here: a resident parks, then waits
        for a measurement window, and its TTL lapses while it waits. Measured: 4 s TTL, 14 s queue,
        resident evicted while still holding its VRAM.

        A second socket fixes that but is a different `owner`, and owner-identity is what makes
        crash recovery work (a closed socket releases its leases), so it cannot simply be relaxed.
        The token is the narrow grant: it authorises RENEW only, it is handed out once at grant time
        to whoever got the lease, and it confers no power to release or to acquire.
        """
        with self.lock:
            lid = int(req.get("lease", 0))
            lv = self.sched.leases.get(lid)
            if lv is None:
                # Say WHICH, because "unknown" and "not yours" call for opposite responses: the
                # first means our lease was reclaimed and we must stop, the second is a bug.
                return {"ok": False, "error": "no such lease (expired or released)",
                        "gone": True}
            tok = req.get("token")
            if lv.owner != owner and (not tok or tok != lv.meta.get("_token")):
                return {"ok": False, "error": "not your lease"}
            ttl = float(req.get("ttl_s", self.lease_ttl_s))
            return {"ok": self.sched.renew(time.time(), lid, ttl), "ttl_s": ttl}

    def op_status(self, req, owner):
        with self.lock:
            self._probe_lanes()
            snap = self.sched.snapshot(time.time())
            snap["pid"] = os.getpid()
            snap["uptime_s"] = round(time.time() - self.started_at, 1)
            snap["journal"] = self.journal_path
            snap["socket"] = {"path": self.sock_path, "listening": bool(self.sock_path)}
            return {"ok": True, "status": snap}

    def op_drain(self, req, owner):
        """Stop granting on a lane without killing what is on it. The operator's tool for taking a
        card out of service -- a suspected-bad GPU, or one wanted for something else -- without
        aborting a 40-minute climb that is mid-flight on it."""
        with self.lock:
            lane = int(req.get("lane"))
            if lane not in self.sched.lanes:
                return {"ok": False, "error": f"no lane {lane}"}
            self.sched.lanes[lane].disabled = bool(req.get("disabled", True))
            self.sched.lanes[lane].disabled_reason = str(req.get("reason", "drained by operator"))
            self.journal("drain" if req.get("disabled", True) else "undrain", lane=lane)
            self._pump()
            return {"ok": True, "lane": lane,
                    "disabled": self.sched.lanes[lane].disabled}

    def op_shutdown(self, req, owner):
        """Ask the daemon to stop. The connection handler sets the server flag after we reply.

        This op exists as a real handler because without it `handle()` fell through to
        `unknown op 'shutdown'` -- the daemon still stopped (the handler checks the op name
        separately), so it worked while reporting failure, and every scripted teardown logged an
        error it was right to ignore. A control path whose success looks like a failure trains
        people to ignore its output.
        """
        held = len(self.sched.leases)
        self.journal("shutdown_requested", owner=owner, leases_held=held)
        return {"ok": True, "stopping": True, "leases_held": held}

    def op_bye(self, req, owner):
        with self.lock:
            n = len(self.sched.release_owner(owner))
            c = self._cancel_owner_queue_locked(owner, "client_bye")
            self._pump()
            return {"ok": True, "released": n, "cancelled": c}

    def on_disconnect(self, owner):
        """THE crash-recovery path. The socket closing is the signal; nothing needs to poll."""
        with self.lock:
            gone = self.sched.release_owner(owner)
            c = self._cancel_owner_queue_locked(owner, "client_disconnected")
            if gone or c:
                # Record WHICH leases and for HOW LONG, not just how many. A reclaim ends a lease
                # exactly as `release` does, so an auditor summing held time must be able to include
                # it -- with only a count it cannot, and every crash-reclaimed lease silently drops
                # out of the utilisation figure. That turns a lower bound into something that reads
                # like a measurement, and the gap is not small: a run here reclaimed 43 leases whose
                # duration was, in consequence, unknowable after the fact.
                now = time.time()
                self.journal("disconnect_reclaim", owner=owner, leases=len(gone), queued=c,
                             reclaimed=[{"lease": lv.lease_id, "lane": lv.lane, "kind": lv.kind,
                                         "group": lv.group,
                                         "held_s": round(now - lv.granted_at, 3)} for lv in gone])
            self._pump()


class _Handler(socketserver.StreamRequestHandler):
    def _peer_alive(self):
        """Non-consuming EOF probe used while this handler is blocked in `acquire`."""
        try:
            ready, _, _ = select.select([self.request], [], [], 0)
            if not ready:
                return True
            data = self.request.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT)
            return bool(data)
        except BlockingIOError:
            return True
        except OSError:
            return False

    def handle(self):
        broker = self.server.broker
        owner = f"conn{next(self.server.conn_ids)}@{os.getpid()}"
        label = None
        try:
            for raw in self.rfile:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    req = json.loads(raw)
                except Exception as exc:                         # noqa: BLE001
                    self._send({"ok": False, "error": f"bad json: {exc}"})
                    continue
                # A client may LABEL itself, which only affects logging and fair-share grouping.
                if req.get("label") and label is None:
                    label = str(req["label"])[:120]
                    owner = f"{label}#{owner}"
                resp = broker.handle(req, owner, peer_alive=self._peer_alive)
                self._send(resp)
                if req.get("op") == "shutdown":
                    self.server.want_shutdown = True
                    return
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            broker.on_disconnect(owner)

    def _send(self, obj):
        try:
            self.wfile.write((json.dumps(obj, default=str) + "\n").encode())
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass


class _Server(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True
    allow_reuse_address = True
    # socketserver's default is FIVE. That is the pending-connection backlog, and exceeding it makes
    # connect() fail outright -- at which point the client degrades to UNBROKERED and runs its
    # benchmark on an unarbitrated card, which is precisely the contamination this daemon exists to
    # prevent, arriving silently and only under load.
    #
    # Measured: 32 simultaneous clients against the default lost 8 of them. A fleet of 16 kernels,
    # each with a trunk plus arms, reaches that easily -- and the failure would have looked like
    # "the scheduler does not help much", not like an error.
    request_queue_size = 512

    def __init__(self, path, broker):
        self.broker = broker
        self.want_shutdown = False
        import itertools
        self.conn_ids = itertools.count(1)
        super().__init__(path, _Handler)


def serve(sock_path, lanes, **kw):
    # A stale socket from a crashed daemon would make bind() fail with EADDRINUSE forever. Probe it:
    # if nobody answers, it is a corpse and we remove it. If somebody does, we are the redundant one
    # and exit quietly -- which is what makes auto-start races harmless.
    if os.path.exists(sock_path):
        probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        probe.settimeout(1.0)
        try:
            probe.connect(sock_path)
            probe.close()
            print(f"[broker] already running at {sock_path}", file=sys.stderr)
            return 0
        except OSError:
            try:
                os.unlink(sock_path)
            except OSError as exc:
                if exc.errno != errno.ENOENT:
                    raise

    broker = Broker(lanes, **kw)
    broker.sock_path = sock_path
    srv = _Server(sock_path, broker)
    try:
        os.chmod(sock_path, 0o777)   # same-box, cross-container clients
    except OSError:
        pass
    broker.journal("broker_start", pid=os.getpid(), lanes=lanes, sock=sock_path)
    print(f"[broker] pid={os.getpid()} lanes={lanes} sock={sock_path}", file=sys.stderr)

    def maintenance_loop():
        # The timer is what makes TTL reclaim and aging real without a client poking us: an idle
        # fleet still needs expired leases collected and drained lanes noticed.
        while not srv.want_shutdown:
            time.sleep(2.0)
            try:
                broker.maintenance()
            except Exception:                                    # noqa: BLE001
                pass

    threading.Thread(target=maintenance_loop, daemon=True).start()

    def shutdown_watch():
        while not srv.want_shutdown:
            time.sleep(0.25)
        srv.shutdown()

    threading.Thread(target=shutdown_watch, daemon=True).start()
    try:
        srv.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        broker.journal("broker_stop", pid=os.getpid())
        try:
            os.unlink(sock_path)
        except OSError:
            pass
    return 0


def _client_call(sock_path, req, timeout=30):
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect(sock_path)
    except (ConnectionRefusedError, FileNotFoundError) as exc:
        # A socket FILE left behind by a killed daemon refuses connections forever. `--status` is
        # the first thing an operator reaches for when something looks wrong, so it must say "not
        # running" rather than dumping a traceback that looks like a bug in the tool.
        raise SystemExit(
            f"[broker] not running at {sock_path} ({type(exc).__name__}).\n"
            f"[broker] start one:  python3 {os.path.basename(__file__)} --serve --sock {sock_path}"
        ) from exc
    s.sendall((json.dumps(req) + "\n").encode())
    buf = b""
    while not buf.endswith(b"\n"):
        chunk = s.recv(65536)
        if not chunk:
            break
        buf += chunk
    s.close()
    return json.loads(buf.decode() or "{}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--shutdown", action="store_true")
    ap.add_argument("--drain", type=int, default=None)
    ap.add_argument("--undrain", type=int, default=None)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--sock", default=DEFAULT_SOCK)
    ap.add_argument("--gpus", default="")
    ap.add_argument("--state-dir", default=DEFAULT_STATE)
    ap.add_argument("--aging-s", type=float, default=30.0)
    ap.add_argument("--lease-ttl-s", type=float, default=900.0)
    ap.add_argument("--lease-overrun-s", type=float, default=LEASE_OVERRUN_S,
                    help="journal `lease_overrun` when an EXCLUSIVE lease has been held this long "
                         "(0 = off). Observation only -- nothing is revoked. TTL does not cover this: "
                         "the client heartbeat renews for as long as the holder lives, so a lease can "
                         "outlive --lease-ttl-s and never expire")
    # The foreign-tenant thresholds and the self-credit grace. Exposed because the right values are
    # a property of the BOX -- idle VRAM floor, how twitchy gpu_busy_percent is, how long an
    # allocator takes to tear down -- and a box that needs different ones should not need a patch.
    ap.add_argument("--foreign-busy-pct", type=int, default=5,
                    help="fence a lane we hold no lease on above this gpu_busy_percent")
    ap.add_argument("--foreign-vram-mb", type=int, default=1024,
                    help="fence a lane whose UNEXPLAINED VRAM exceeds this (MB)")
    ap.add_argument("--ours-grace-s", type=float, default=90.0,
                    help="after releasing a lane, keep crediting ourselves for the VRAM we were "
                         "using for this long -- covers allocator teardown we cannot observe")
    ap.add_argument("--no-probe", action="store_true",
                    help="synthetic lanes: never consult sysfs, never mark a lane foreign-busy "
                         "(tests only -- on a real box this would hand out an occupied card)")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)

    if a.selftest:
        return selftest()

    if a.serve:
        lanes = ([int(x) for x in a.gpus.split(",") if x.strip().isdigit()]
                 if a.gpus else detect_lanes())
        return serve(a.sock, lanes, state_dir=a.state_dir, aging_s=a.aging_s,
                     lease_ttl_s=a.lease_ttl_s, probe_hardware=not a.no_probe,
                     foreign_busy_pct=a.foreign_busy_pct, foreign_vram_mb=a.foreign_vram_mb,
                     ours_grace_s=a.ours_grace_s, lease_overrun_s=a.lease_overrun_s)

    if a.status:
        r = _client_call(a.sock, {"op": "status", "label": "cli"})
        if a.json:
            print(json.dumps(r, indent=2))
        else:
            print(render_status(r.get("status") or {}))
        return 0 if r.get("ok") else 1

    if a.shutdown:
        try:
            print(json.dumps(_client_call(a.sock, {"op": "shutdown", "label": "cli"})))
        except OSError as exc:
            print(f"not running: {exc}", file=sys.stderr)
            return 1
        return 0

    if a.drain is not None or a.undrain is not None:
        lane = a.drain if a.drain is not None else a.undrain
        r = _client_call(a.sock, {"op": "drain", "lane": lane,
                                  "disabled": a.drain is not None, "label": "cli"})
        print(json.dumps(r))
        return 0 if r.get("ok") else 1

    ap.print_help()
    return 2


def render_status(s):
    if not s:
        return "(broker not running)"
    out = [f"broker pid={s.get('pid')} up={s.get('uptime_s')}s"]
    out.append("")
    out.append("LANE  STATE      EXCLUSIVE-HOLDER                 RESIDENTS  VRAM(ours/total)")
    for ln in s.get("lanes", []):
        state = "DRAIN" if ln["disabled"] else ("FOREIGN" if ln["foreign_busy"] else "ok")
        holder = ln["exclusive"] or "-"
        out.append(f"{ln['lane']:>4}  {state:<9}  {holder[:32]:<32} "
                   f"{len(ln['residents']):>9}  {ln['resident_vram_mb']}/{ln['vram_total_mb']}MB")
    q = s.get("queue", [])
    out.append("")
    out.append(f"QUEUE ({len(q)} waiting)")
    for r in q[:15]:
        out.append(f"  rid={r['rid']:<5} {r['kind']:<9} prio={r['priority']:<3}"
                   f" eff={r['effective']:<7} waited={r['waited_s']}s  {r['owner'][:40]}")
    if len(q) > 15:
        out.append(f"  ... and {len(q) - 15} more")
    st = s.get("stats", {})
    out.append("")
    out.append("  ".join(f"{k}={v}" for k, v in st.items()))
    return "\n".join(out)


def selftest():
    """No socket, no GPU. Exercises the state machine through the Broker's own handlers."""
    fails = []

    def check(cond, msg):
        print(("  ok: " if cond else "  FAIL: ") + msg)
        if not cond:
            fails.append(msg)

    print("\n# gpu_broker selftest (no GPU, no socket)")
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        b = Broker([0, 1], state_dir=td)
        # Nothing real is on these fake lanes; the probe would mark them however the box looks, so
        # pin them clear -- this selftest is about the protocol, not the hardware.
        for ln in b.sched.lanes.values():
            ln.foreign_busy = False

        r = b.handle({"op": "ping"}, "t1")
        check(r.get("ok") and r.get("proto") == PROTO_VERSION, "ping returns the protocol version")

        r1 = b.handle({"op": "acquire", "kind": EXCLUSIVE, "timeout_s": 5}, "t1")
        check(r1.get("ok") and r1.get("lane") in (0, 1), f"exclusive acquire granted: {r1}")
        r2 = b.handle({"op": "acquire", "kind": EXCLUSIVE, "timeout_s": 5}, "t2")
        check(r2.get("ok") and r2.get("lane") != r1.get("lane"),
              "a second exclusive lands on the OTHER lane")
        check(b.sched.exclusive_on(r1["lane"]).owner == "t1", "lane 1 is attributed to t1")

        r3 = b.handle({"op": "acquire", "kind": RESIDENT, "vram_mb": 1024,
                       "lanes": [r1["lane"]], "timeout_s": 5}, "t3")
        check(r3.get("ok"), "a RESIDENT is granted on a lane that already has an exclusive holder")
        check(b.sched.resident_vram_mb(r1["lane"]) == 1024, "its VRAM is accounted on that lane")

        check(b.handle({"op": "renew", "lease": r1["lease"], "ttl_s": 60}, "t1").get("ok"),
              "the holder can renew")
        check(not b.handle({"op": "renew", "lease": r1["lease"]}, "somebody_else").get("ok"),
              "a stranger cannot renew someone else's lease")
        check(not b.handle({"op": "release", "lease": r1["lease"]}, "somebody_else").get("ok"),
              "a stranger cannot release someone else's lease")

        b.on_disconnect("t1")
        check(b.sched.exclusive_on(r1["lane"]) is None,
              "a disconnect reclaims the dead client's lease")

        r = b.handle({"op": "drain", "lane": 0}, "cli")
        check(r.get("ok") and b.sched.lanes[0].disabled, "a lane can be drained")
        b.handle({"op": "drain", "lane": 0, "disabled": False}, "cli")
        check(not b.sched.lanes[0].disabled, "and undrained")

        r = b.handle({"op": "status"}, "cli")
        check(r.get("ok") and "lanes" in r["status"], "status renders")
        check(isinstance(render_status(r["status"]), str), "the human renderer works")
        check(b.handle({"op": "nonsense"}, "cli").get("ok") is False, "unknown ops are refused")

    print("\nSELFTEST PASS: broker protocol behaves" if not fails else f"\nFAIL: {len(fails)} check(s)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
