#!/usr/bin/env python3
"""The scheduling decision, as a pure function of state. No sockets, no GPUs, no clock.

WHY THIS FILE HAS NO I/O. A GPU scheduler's hard parts are fairness, aging, eviction and crash
reclaim -- and every one of them is a decision about *time* and *ordering*, which is exactly what is
untestable once it is tangled with a socket and a real clock. So the policy lives here as a state
machine that takes `now` as an argument, and the daemon (`gpu_broker.py`) is the thin shell that owns
the socket, calls `tick(now)`, and performs whatever the machine returned. Every property the
scheduler claims -- FIFO within a priority, no starvation, no double-grant, a dead client's lease
reclaimed -- is then a unit test with a fake clock and no hardware. `tests/test_scheduler_core.py`
runs in milliseconds and needs no GPU.

THE MODEL, in one paragraph. A GPU is a lane. A lane holds at most one EXCLUSIVE lease at a time,
plus any number of RESIDENT leases. An EXCLUSIVE lease is a timed measurement or a compile: it owns
the card's compute and nothing else may run there. A RESIDENT lease is a parked worker holding VRAM
and no compute -- `sweep_driver.py --serve` between readings. Residents do NOT block an exclusive
grant, because a parked process is not using the SMs; what they cost is VRAM, which is tracked
separately and is what eviction is about. A resident that wants to *measure* takes a short exclusive
lease on the lane it already lives on (`wake`), which is the whole point of the split: the process
stays warm across readings, but the reading itself is still exclusive, so the number is clean.

WHAT THIS BUYS OVER `gpu_lock.sh`, which already does spot allocation. flock answers "is this lane
free right now"; it cannot answer "who should get it next". With 16 kernels racing for 8 cards that
distinction is the whole game: flock hands the lane to whichever process happens to call at the
right microsecond, so a kernel that needs one 3-second timing can sit behind a kernel that wants
forty of them, indefinitely and invisibly. A queue with priorities and aging is what turns that into
a bounded wait. flock also cannot express residency at all -- a parked worker either holds the lock
(and blocks everyone) or drops it (and loses its VRAM).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Lease kinds.
# ---------------------------------------------------------------------------
# EXCLUSIVE  the lane's compute is yours. Timed benchmarks, profiles, compiles, correctness runs.
#            Granted to exactly one holder per lane.
# RESIDENT   you hold VRAM on this lane and promise not to compute. A parked --serve worker.
#            Many per lane, bounded by VRAM. Never blocks an EXCLUSIVE grant.
#
# There is deliberately NO "shared compute" kind. It was considered and rejected: the moment a task
# that produces a NUMBER can share a lane, every number in the system becomes conditional on what
# else happened to be running, and the contamination is invisible in the artifact. Compiles could
# safely share -- but a compile is also the thing most likely to be mis-tagged by a caller, and a
# mis-tagged timing is unrecoverable. The throughput left on the table is real and it is the right
# trade: this system's output is measurements, not utilisation.
EXCLUSIVE = "exclusive"
RESIDENT = "resident"
KINDS = (EXCLUSIVE, RESIDENT)

# Priority bands. LOWER SORTS FIRST. Named rather than numeric at the call site so a caller cannot
# invent priority 0 and jump the whole fleet.
#
# The ordering encodes one judgement: work that UNBLOCKS other work outranks work that merely
# produces a result. A wake is a resident that already paid for its VRAM and has a process parked
# waiting -- making it queue behind a fresh 40-point sweep wastes the residency the system exists to
# provide. Interactive is for a human at a terminal, who is a scarcer resource than a GPU.
PRIO_INTERACTIVE = 10
PRIO_WAKE = 20        # a parked resident wants its measurement window
PRIO_VERIFY = 30      # a verify/close gates a whole kernel's completion
PRIO_MEASURE = 40     # the ordinary timed reading
PRIO_SWEEP = 50       # bulk config search: high volume, individually cheap, interruptible
PRIO_BULK = 60        # compiles, warmups, anything speculative
PRIORITIES = {
    "interactive": PRIO_INTERACTIVE, "wake": PRIO_WAKE, "verify": PRIO_VERIFY,
    "measure": PRIO_MEASURE, "sweep": PRIO_SWEEP, "bulk": PRIO_BULK,
}

RUNTIME_SHORT = "short"
RUNTIME_MEDIUM = "medium"
RUNTIME_LONG = "long"
RUNTIME_CLASSES = (RUNTIME_SHORT, RUNTIME_MEDIUM, RUNTIME_LONG)
SAFE_YIELD_NONE = "none"


def parse_priority(v, default=PRIO_MEASURE) -> int:
    """Accept a band name or a raw int; anything unrecognised becomes the default rather than an
    error. A scheduler that refuses work because of a typo in a priority label has converted a
    cosmetic mistake into a lost run."""
    if v is None or v == "":
        return default
    if isinstance(v, int):
        return max(0, min(99, v))
    s = str(v).strip().lower()
    if s in PRIORITIES:
        return PRIORITIES[s]
    try:
        return max(0, min(99, int(s)))
    except ValueError:
        return default


def normalize_metadata(meta=None) -> dict:
    """Normalise optional runtime metadata without turning a bad annotation into an outage."""
    out = dict(meta or {})
    if out.get("command_kind") is not None:
        out["command_kind"] = str(out["command_kind"])[:80]
    if out.get("safe_yield_kind") is not None:
        out["safe_yield_kind"] = str(out["safe_yield_kind"])[:80]
    if out.get("estimated_runtime_s") not in (None, ""):
        try:
            out["estimated_runtime_s"] = max(0.0, float(out["estimated_runtime_s"]))
        except (TypeError, ValueError):
            out.pop("estimated_runtime_s", None)
    return out


def runtime_class(meta) -> str:
    """Classify an estimate: short <120 s, medium <=300 s, long >300 s."""
    try:
        estimate = float((meta or {}).get("estimated_runtime_s"))
    except (TypeError, ValueError):
        # Unknown work must not claim the latency privilege reserved for measured short work.
        return RUNTIME_MEDIUM
    if estimate < 120.0:
        return RUNTIME_SHORT
    if estimate <= 300.0:
        return RUNTIME_MEDIUM
    return RUNTIME_LONG


def public_metadata(meta) -> dict:
    """Return metadata safe for status and journals; private tokens never leave the broker."""
    return {str(k): v for k, v in dict(meta or {}).items() if not str(k).startswith("_")}


@dataclass
class Request:
    """One queued ask. Immutable except for `deadline` bookkeeping the daemon owns."""
    rid: int
    kind: str                     # EXCLUSIVE | RESIDENT
    owner: str                    # free-text: "kernel=name/arm=direction". Fair-share key below.
    group: str                    # the FAIR-SHARE key -- normally the kernel. See _fair_order.
    priority: int
    enqueued_at: float
    lanes: Tuple[int, ...] = ()   # the lanes this request may be placed on (its allocation fence)
    vram_mb: int = 0              # RESIDENT only: how much VRAM it intends to hold
    want_lane: Optional[int] = None   # a wake naming the lane its process already lives on
    expires_at: Optional[float] = None  # queue deadline; past it the request is failed, not granted
    meta: dict = field(default_factory=dict)


@dataclass
class Lease:
    """A granted request. `expires_at` is a LEASE deadline: past it, the holder is presumed dead."""
    lease_id: int
    rid: int
    kind: str
    lane: int
    owner: str
    group: str
    granted_at: float
    expires_at: float
    priority: int
    vram_mb: int = 0
    meta: dict = field(default_factory=dict)


@dataclass
class Lane:
    """One GPU."""
    lane_id: int
    vram_total_mb: int = 0
    # Set by the daemon from sysfs. A lane with foreign work is not handed out -- see `usable`.
    foreign_busy: bool = False
    # OUR OWN FOOTPRINT, LEARNED BY WATCHING. The broker credits itself for the VRAM present while it
    # holds a lease on this lane, and keeps crediting it for a bounded grace after release. Without
    # this, only DECLARED resident VRAM is explained -- and the exclusive path declares none, so a
    # lane holding our own benchmark's memory reads as a foreign tenant and fences the card we are
    # using. High-water rather than last-seen: a sweep's usage tracks the config under test, and the
    # peak is what must still be explained once we let go.
    ours_vram_mb: int = 0
    ours_seen_at: float = 0.0
    disabled: bool = False
    disabled_reason: str = ""

    def usable(self) -> bool:
        return not self.disabled and not self.foreign_busy


class Scheduler:
    """The state machine. Every mutator takes `now` explicitly; nothing here reads a clock."""

    def __init__(self, lane_ids, vram_total_mb=0, aging_s=30.0, max_resident_vram_pct=80):
        self.lanes: Dict[int, Lane] = {
            int(i): Lane(int(i), vram_total_mb=vram_total_mb) for i in lane_ids
        }
        self.queue: List[Request] = []
        self.leases: Dict[int, Lease] = {}
        # A pause is advisory state only. The broker records it; the owner yields at its own safe
        # boundary. No scheduler path sends signals to, suspends, or kills a GPU process.
        self.pause_requests: Dict[int, dict] = {}
        self._rid = itertools.count(1)
        self._lid = itertools.count(1)
        # AGING. Every `aging_s` a request spends queued promotes it one priority band. This is the
        # anti-starvation guarantee and it is the reason a bulk compile behind an endless stream of
        # sweeps eventually runs: its effective priority decreases without bound, so it cannot be
        # overtaken forever. Without it, strict priority + a saturated fleet = permanent starvation
        # for the lowest band, which is not a theoretical concern here (sweeps are exactly such a
        # stream). Tested in test_scheduler_core.py::test_aging_prevents_starvation.
        self.aging_s = float(aging_s)
        self.max_resident_vram_pct = int(max_resident_vram_pct)
        # Leases reclaimed by the most recent tick(), for the daemon to journal. See tick() step 1.
        self.last_reclaimed: List[Lease] = []
        self.stats = {"granted": 0, "denied": 0, "expired_requests": 0,
                      "reclaimed_leases": 0, "evicted_residents": 0, "preempted": 0,
                      "pause_requested": 0, "cooperative_yields": 0}

    # -- introspection ------------------------------------------------------
    def lane_of(self, lease_id) -> Optional[int]:
        lv = self.leases.get(lease_id)
        return lv.lane if lv else None

    def exclusive_on(self, lane_id) -> Optional[Lease]:
        for lv in self.leases.values():
            if lv.lane == lane_id and lv.kind == EXCLUSIVE:
                return lv
        return None

    def residents_on(self, lane_id) -> List[Lease]:
        return [lv for lv in self.leases.values()
                if lv.lane == lane_id and lv.kind == RESIDENT]

    def resident_vram_mb(self, lane_id) -> int:
        return sum(lv.vram_mb for lv in self.residents_on(lane_id))

    # -- mutation -----------------------------------------------------------
    def submit(self, now, kind, owner, group=None, priority=PRIO_MEASURE, lanes=(),
               vram_mb=0, want_lane=None, queue_timeout_s=None, meta=None) -> Request:
        if kind not in KINDS:
            raise ValueError(f"unknown lease kind {kind!r}")
        allowed = tuple(int(x) for x in (lanes or self.lanes.keys()))
        req = Request(
            rid=next(self._rid), kind=kind, owner=str(owner),
            # Fair share defaults to the OWNER when no group is given, which makes an un-grouped
            # caller its own share rather than silently joining everyone else's.
            group=str(group if group not in (None, "") else owner),
            priority=int(priority), enqueued_at=float(now), lanes=allowed,
            vram_mb=int(vram_mb or 0),
            want_lane=None if want_lane is None else int(want_lane),
            expires_at=(None if not queue_timeout_s else float(now) + float(queue_timeout_s)),
            meta=normalize_metadata(meta),
        )
        self.queue.append(req)
        return req

    def _effective_priority(self, req: Request, now: float) -> float:
        if self.aging_s <= 0:
            return req.priority
        waited = max(0.0, now - req.enqueued_at)
        # Promote only after a complete aging interval. Truncating a fractional float at sort time
        # would promote a request after a few microseconds (40 - 0.001 -> int(39.999) == 39),
        # silently making every later request "higher priority" than a running holder.
        return req.priority - int(waited // self.aging_s)

    def _runtime_rank(self, req: Request) -> int:
        return RUNTIME_CLASSES.index(runtime_class(req.meta))

    def _fair_order(self, now: float) -> List[Request]:
        """Order the queue by aged priority, runtime class, fair share, then FIFO.

        FAIR SHARE is per `group`, and the group is normally the KERNEL. Without it, priority alone
        is not enough on this workload: one kernel's sweep can enqueue hundreds of same-priority
        requests, and pure (priority, FIFO) would then serve that kernel's entire backlog before a
        sibling kernel's first request -- which is precisely the 23x-imbalance failure the per-kernel
        pinning was meant to fix, reintroduced one level up.

        THE BAND IS ROUNDED TO AN INTEGER, AND THAT IS LOAD-BEARING. Aging produces a continuous
        float, so two requests submitted a tenth of a second apart have *different* effective
        priorities -- and a strict float comparison then decides the whole ordering before the
        fair-share term is ever consulted. Result: the kernel that queued first takes every free
        lane, and fair share silently never engages. That is exactly what happened; it is caught by
        tests/test_scheduler_core.py::test_fair_share_across_kernels, where kernel A took all 4
        lanes with kernel B's lone request stuck behind a 20-deep backlog it had beaten by 0.1s.

        Rounding to a band restores the intended precedence: priority first, then fairness, then
        arrival. Requests age into whole bands (one per `aging_s`), and *within* a band the group
        holding fewest lanes wins. Sub-band arrival differences no longer outrank fairness -- but
        they still break ties inside a group, via `rid`, which is what keeps one kernel's own
        requests in submission order (test_fifo_within_a_group).
        """
        held: Dict[str, int] = {}
        for lv in self.leases.values():
            held[lv.group] = held.get(lv.group, 0) + 1
        # Runtime is deliberately considered only after aged priority. A fresh short request can
        # overtake an unstarted long request in the same priority band, while a long request that
        # waits through enough aging bands eventually outranks fresh short work. Within each class,
        # re-sorting after every grant preserves per-group fair sharing and FIFO.
        return sorted(
            self.queue,
            key=lambda r: (int(self._effective_priority(r, now)), self._runtime_rank(r),
                           held.get(r.group, 0), r.rid),
        )

    def _placeable(self, req: Request, now: float) -> Optional[int]:
        """Which lane, if any, can take this request right now. None = keep it queued."""
        cands = [self.lanes[i] for i in req.lanes if i in self.lanes]
        cands = [ln for ln in cands if ln.usable()]
        if not cands:
            return None

        if req.kind == RESIDENT:
            # Residents care about VRAM, not about who holds compute. Pack onto the lane with the
            # most headroom so residents spread out rather than stacking on lane 0.
            best, best_free = None, -1
            for ln in cands:
                cap = ln.vram_total_mb * self.max_resident_vram_pct // 100 if ln.vram_total_mb else 0
                free = cap - self.resident_vram_mb(ln.lane_id) if cap else (1 << 30)
                if free >= req.vram_mb and free > best_free:
                    best, best_free = ln.lane_id, free
            return best

        # EXCLUSIVE.
        # A wake names the lane its process is parked on. That is not a preference: the process's
        # VRAM is THERE, so granting it a different lane would mean it cannot use the grant at all.
        # Either its own lane is free or it waits.
        if req.want_lane is not None:
            if req.want_lane not in req.lanes or req.want_lane not in self.lanes:
                return None
            ln = self.lanes[req.want_lane]
            if not ln.usable():
                return None
            return req.want_lane if self.exclusive_on(req.want_lane) is None else None

        # Otherwise: any free lane. Prefer the one with the FEWEST residents, so a timed run lands
        # on the quietest card. Residents are parked and should not perturb a measurement, but
        # "should not" is a claim about their behaviour, and a free choice costs nothing.
        free = [ln for ln in cands if self.exclusive_on(ln.lane_id) is None]
        if not free:
            return None
        free.sort(key=lambda ln: (len(self.residents_on(ln.lane_id)), ln.lane_id))
        return free[0].lane_id

    def tick(self, now, lease_ttl_s=900.0) -> Tuple[List[Lease], List[Request]]:
        """Advance the machine. Returns (newly granted, failed-by-deadline).

        The daemon calls this on every event and on a timer. It is idempotent in the sense that
        calling it twice with no state change grants nothing the second time.
        """
        # 1. Reclaim expired leases FIRST, so a dead holder's lane is available in this same pass.
        #    This is the crash-recovery path: a client that dies without releasing stops renewing,
        #    its lease ages out, and the lane returns to the pool. There is no other mechanism --
        #    process liveness is deliberately not consulted here, because the daemon may not share a
        #    PID namespace with its clients (containers), and a lease that outlives its holder for
        #    one TTL is a bounded cost while a lane lost forever is not.
        #
        #    Reclaimed leases are parked on `self.last_reclaimed` for the daemon to journal. They
        #    are NOT added to tick()'s return tuple: every caller and test unpacks two values, and
        #    widening the signature to carry a rare event would churn all of them. Without this the
        #    TTL path was completely invisible to the auditor -- `audit_journal.py` counted a
        #    `lease_expired` event that no writer ever emitted, so `lease_reclaims_ttl` read 0 on
        #    every run, including runs where a wedged client really had lost its lane. A silent
        #    reclaim looks exactly like a clean release, which is the one distinction an operator
        #    chasing "who took my GPU" actually needs.
        self.last_reclaimed = []
        for lid, lv in list(self.leases.items()):
            if lv.expires_at is not None and now >= lv.expires_at:
                del self.leases[lid]
                self.pause_requests.pop(lid, None)
                self.stats["reclaimed_leases"] += 1
                self.last_reclaimed.append(lv)

        # 2. Fail requests past their queue deadline. A caller that said "I need this within 60s or
        #    not at all" gets told NO rather than being handed a lane at minute nine, when whatever
        #    it wanted the measurement for has already moved on.
        failed: List[Request] = []
        keep: List[Request] = []
        for r in self.queue:
            if r.expires_at is not None and now >= r.expires_at:
                failed.append(r)
                self.stats["expired_requests"] += 1
            else:
                keep.append(r)
        self.queue = keep

        # 3. Grant in fair order. One pass: a request that cannot be placed stays queued and does
        #    NOT block the rest of the queue (no head-of-line blocking) -- with heterogeneous lane
        #    fences, a request waiting on a specific busy lane would otherwise stall everyone.
        #
        #    RE-SORTING AFTER EVERY GRANT is what makes fair share actually fair. `_fair_order`
        #    ranks partly on how many leases a group already holds, and if that count is computed
        #    once before the loop it is stale the moment the first grant lands -- so a group that
        #    queued a 20-deep backlog takes EVERY free lane in the pass, which is precisely the
        #    lockout fair share exists to prevent. Caught by
        #    tests/test_scheduler_core.py::test_fair_share_across_kernels, which had kernel A taking
        #    all 4 lanes with kernel B's single request behind them.
        #
        #    The cost is a re-sort per grant, i.e. O(lanes x queue log queue). Lanes are ~8 and the
        #    queue is at most a few hundred, so this is microseconds; the alternative (an
        #    incrementally maintained heap) buys nothing measurable and is a great deal easier to
        #    get subtly wrong.
        granted: List[Lease] = []
        while True:
            for req in self._fair_order(now):
                lane = self._placeable(req, now)
                if lane is not None:
                    break
            else:
                break               # nothing left that can be placed
            lv = Lease(
                lease_id=next(self._lid), rid=req.rid, kind=req.kind, lane=lane,
                owner=req.owner, group=req.group, granted_at=float(now),
                expires_at=float(now) + float(lease_ttl_s), priority=req.priority,
                vram_mb=req.vram_mb, meta=dict(req.meta),
            )
            self.leases[lv.lease_id] = lv
            self.queue.remove(req)
            granted.append(lv)
            self.stats["granted"] += 1
        return granted, failed

    def renew(self, now, lease_id, ttl_s) -> bool:
        """The heartbeat. A holder must renew or be reclaimed -- see tick() step 1."""
        lv = self.leases.get(lease_id)
        if not lv:
            return False
        lv.expires_at = float(now) + float(ttl_s)
        return True

    def release(self, lease_id) -> bool:
        self.pause_requests.pop(lease_id, None)
        return self.leases.pop(lease_id, None) is not None

    def release_owner(self, owner) -> List[Lease]:
        """Drop every lease held by one owner and RETURN THEM. The daemon calls this when a
        connection closes, which is the fast path for crash recovery: a client that dies takes its
        socket with it, and the kernel closes it for us. The TTL above is the backstop for the case
        where the socket stays open but the client is wedged.

        Returning the leases rather than a count is what lets the journal record how long each one
        was held. A bare count cannot: the Lease carries `granted_at` and `lane`, and dropping it
        here made that time unrecoverable downstream -- an auditor summing `held_s` over `release`
        events silently omits every crash-reclaimed lease, so utilisation comes out as a lower
        bound that reads like a measurement. Callers wanting the old number take len()."""
        gone = [lv for lid, lv in list(self.leases.items()) if lv.owner == owner]
        for lv in gone:
            del self.leases[lv.lease_id]
            self.pause_requests.pop(lv.lease_id, None)
        return gone

    def cancel_owner(self, owner) -> int:
        n = len(self.queue)
        self.queue = [r for r in self.queue if r.owner != owner]
        return n - len(self.queue)

    def evict_candidates(self, lane_id, need_mb) -> List[Lease]:
        """Which residents to evict to free `need_mb` on a lane. Oldest first (LRU by grant time).

        Returns the plan; it does not apply it. Eviction kills a process, so the decision and the
        act are kept separate -- the daemon tells the owner to go away and only drops the lease when
        it confirms, which is what stops a resident being 'evicted' while it is mid-measurement.
        """
        res = sorted(self.residents_on(lane_id), key=lambda lv: lv.granted_at)
        out, freed = [], 0
        for lv in res:
            if freed >= need_mb:
                break
            out.append(lv)
            freed += lv.vram_mb
        return out

    def request_pause(self, lease_id, now, reason, requested_by=None) -> Optional[dict]:
        """Request, but never enforce, a safe-boundary yield from an EXCLUSIVE holder."""
        lv = self.leases.get(lease_id)
        if lv is None or lv.kind != EXCLUSIVE:
            return None
        safe_yield = str(lv.meta.get("safe_yield_kind") or SAFE_YIELD_NONE)
        if safe_yield in ("", SAFE_YIELD_NONE):
            return None
        old = self.pause_requests.get(lease_id)
        if old is not None:
            return old
        pause = {
            "requested_at": float(now),
            "reason": str(reason),
            "requested_by": requested_by,
            "safe_yield_kind": safe_yield,
        }
        self.pause_requests[lease_id] = pause
        self.stats["pause_requested"] += 1
        return pause

    def pause_state(self, lease_id) -> Optional[dict]:
        pause = self.pause_requests.get(lease_id)
        return dict(pause) if pause else None

    def clear_pause(self, lease_id) -> Optional[dict]:
        pause = self.pause_requests.pop(lease_id, None)
        return dict(pause) if pause else None

    def snapshot(self, now) -> dict:
        """Everything an operator or a test needs, as plain data."""
        return {
            "now": now,
            "lanes": [
                {
                    "lane": ln.lane_id,
                    "usable": ln.usable(),
                    "disabled": ln.disabled,
                    "disabled_reason": ln.disabled_reason,
                    "foreign_busy": ln.foreign_busy,
                    "vram_total_mb": ln.vram_total_mb,
                    "resident_vram_mb": self.resident_vram_mb(ln.lane_id),
                    "exclusive": (lambda x: x.owner if x else None)(self.exclusive_on(ln.lane_id)),
                    "exclusive_lease": (lambda x: None if x is None else {
                        "lease_id": x.lease_id, "owner": x.owner, "group": x.group,
                        "priority": x.priority,
                        "ttl_left_s": round(x.expires_at - now, 1),
                        "metadata": public_metadata(x.meta),
                        "pause_requested": self.pause_state(x.lease_id),
                    })(self.exclusive_on(ln.lane_id)),
                    "residents": [lv.owner for lv in self.residents_on(ln.lane_id)],
                    "resident_leases": [
                        {"lease_id": lv.lease_id, "owner": lv.owner, "group": lv.group,
                         "priority": lv.priority, "vram_mb": lv.vram_mb,
                         "ttl_left_s": round(lv.expires_at - now, 1),
                         "metadata": public_metadata(lv.meta)}
                        for lv in self.residents_on(ln.lane_id)
                    ],
                }
                for ln in sorted(self.lanes.values(), key=lambda l: l.lane_id)
            ],
            "queue": [
                {"rid": r.rid, "kind": r.kind, "owner": r.owner, "group": r.group,
                 "priority": r.priority, "effective": round(self._effective_priority(r, now), 3),
                 "waited_s": round(now - r.enqueued_at, 1), "want_lane": r.want_lane,
                 "vram_mb": r.vram_mb, "runtime_class": runtime_class(r.meta),
                 "metadata": public_metadata(r.meta)}
                for r in self._fair_order(now)
            ],
            "leases": [
                {"lease_id": lv.lease_id, "kind": lv.kind, "lane": lv.lane, "owner": lv.owner,
                 "group": lv.group, "priority": lv.priority,
                 "held_s": round(now - lv.granted_at, 1),
                 "vram_mb": lv.vram_mb,
                 "ttl_left_s": (None if lv.expires_at is None
                                else round(lv.expires_at - now, 1)),
                 "runtime_class": runtime_class(lv.meta),
                 "metadata": public_metadata(lv.meta),
                 "pause_requested": self.pause_state(lv.lease_id)}
                for lv in sorted(self.leases.values(), key=lambda l: l.lease_id)
            ],
            "stats": dict(self.stats),
        }
