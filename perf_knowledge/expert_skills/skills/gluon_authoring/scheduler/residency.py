#!/usr/bin/env python3
"""Residency: park a warm worker in VRAM, take the GPU only while it measures.

    from residency import ResidentWorker
    w = ResidentWorker(argv_builder, lane_hint=None, group="kernel_name", vram_mb=8000)
    w.start()                       # takes a RESIDENT lease, spawns the driver on the granted lane
    r = w.measure({"BLOCK_M": 128}, budget_ms=100)   # takes a short EXCLUSIVE window, then releases
    w.stop()

WHY THIS EXISTS. `sweep_driver.py --serve` is worth 14-48x per reading because it keeps torch, the
CUDA context and the harness's input tensors alive between measurements -- one reading drops from
~3.6 s of process to ~0.10 s. But a warm process is also a process sitting on a GPU, and the two
facts pull in opposite directions:

  * If the worker holds an exclusive lock for its whole life, it blocks a card for minutes while
    computing almost nothing. On 8 cards with 16 kernels that is the entire fleet, idle.
  * If it holds nothing -- which is what happens TODAY, because sweep_pool.py never calls
    gpu_lock.sh at all -- then its VRAM and its CUDA context are invisible to every other actor on
    the box, and a sibling's timed benchmark lands on the same card. The number that comes back is
    contaminated and nothing in the artifact says so.

The split lease is the resolution. A RESIDENT lease says "my VRAM lives here, I am not computing";
it is visible to the scheduler (so nobody is told the lane is empty), it is accounted against the
lane's VRAM (so residents cannot pile up past capacity), and it does NOT block an exclusive grant.
When the worker actually measures it takes a short EXCLUSIVE window on the lane it is already parked
on -- so the reading is exclusive, which is the only thing that makes it a measurement, while the
process stays warm across readings, which is what makes it fast.

WITHOUT A BROKER this degrades to exactly today's behaviour: the lease calls return unbrokered, the
lane comes from `lane_hint`, and nothing is coordinated. That is a bug-for-bug fallback on purpose --
it is the behaviour every number recorded before the broker existed was measured against.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from gpu_client import BrokerUnavailable, connect, lease  # noqa: E402

# A measurement that has not answered in this long is not slow, it is wedged. `sweep_pool.Worker.ask`
# accepts a `timeout` argument and never uses it, so a hung driver blocks `readline()` forever and
# takes the sweep -- and, once the broker is in play, its lane -- with it. Here the timeout is real.
DEFAULT_REQUEST_TIMEOUT_S = float(os.environ.get("GEAK_RESIDENT_REQ_TIMEOUT_S", "900"))


class ResidentWorker:
    """One warm `--serve` driver, holding a RESIDENT lease, waking per measurement."""

    def __init__(self, argv_builder, group=None, vram_mb=8000, lane_hint=None,
                 lanes=None, label=None, request_timeout_s=DEFAULT_REQUEST_TIMEOUT_S,
                 max_restarts=8):
        self._build = argv_builder
        self.group = group or "resident"
        self.vram_mb = int(vram_mb)
        self.lane_hint = lane_hint
        self.lanes = list(lanes) if lanes else None
        self.label = label or f"resident:{self.group}"
        self.request_timeout_s = float(request_timeout_s)
        # A worker that dies on every config would otherwise respawn forever, turning a bad search
        # space into an infinite loop of process starts. sweep_pool has no such cap.
        self.max_restarts = int(max_restarts)

        self.conn = None
        self.park = None            # the RESIDENT lease
        self.park_token = None
        self.lane = None
        self.proc = None
        self.ready = None
        self.served = 0
        self.restarts = 0
        self.woke = 0
        self.wake_wait_s = 0.0
        self.dead = False
        # Set when the broker reclaimed our RESIDENT lease while we were parked -- the
        # lane may already belong to another tenant, so every later reading is suspect.
        self.park_lost = False
        self._lk = threading.Lock()

    # -- lifecycle ----------------------------------------------------------
    def start(self):
        """Take the resident lease, then spawn the driver ON the granted lane.

        Order matters: the lane must be known before the process starts, because the driver's GPU is
        fixed by HIP_VISIBLE_DEVICES at spawn and its VRAM lands wherever it lands. Asking the broker
        afterwards would be asking it to bless a decision already made.
        """
        try:
            self.conn = connect(label=self.label)
            r = self.conn.call({"op": "acquire", "kind": "resident", "group": self.group,
                                "vram_mb": self.vram_mb, "priority": "bulk",
                                "lanes": self.lanes, "timeout_s": 1800,
                                "meta": {"command_kind": "resident_sweep",
                                         "estimated_runtime_s": 900,
                                         "safe_yield_kind": "resident_wake_boundary",
                                         "resident_event": "park",
                                         "reason": "resident_sweep_park"}}, timeout=1830)
            if r.get("ok"):
                self.lane = r["lane"]
                self.park = r["lease"]
                self.park_token = r.get("token")
                self._start_heartbeat()
            else:
                self.lane = self.lane_hint          # broker said no; fall back
        except BrokerUnavailable:
            self.conn = None
            self.lane = self.lane_hint              # no broker: today's behaviour exactly

        if self.lane is None:
            self.lane = 0
        self._spawn()
        return self

    def _start_heartbeat(self):
        """Renew the RESIDENT lease on its own connection, never `self.conn`.

        This class is the worst case for a shared socket. `measure()` issues a BLOCKING acquire on
        `self.conn` for its measurement window, and `Conn.call` holds a per-connection mutex for the
        whole round-trip -- so on a busy box the renew thread is locked out for exactly as long as
        the wake queues. The resident then loses the lease covering the VRAM it is still holding,
        the broker hands its lane to somebody else, and two processes compute on one card.

        The failure is silent from in here: nothing tells a worker its lease lapsed, which is why
        `_died`/`stats` now surface `park_lost`.
        """
        def beat():
            hb = None
            period = 120.0
            while self.park is not None and not self.dead:
                time.sleep(period)
                if self.park is None or self.dead:
                    break
                try:
                    if hb is None:
                        hb = connect(label=f"hb:resident:{self.group}")
                    r = hb.call({"op": "renew", "lease": self.park, "ttl_s": 900,
                                 "token": self.park_token})
                    if isinstance(r, dict) and r.get("ok") is False:
                        self.park_lost = True     # reclaimed; our lane may not be ours any more
                        break
                except Exception:                                # noqa: BLE001
                    try:
                        if hb is not None:
                            hb.close()
                    except Exception:                            # noqa: BLE001
                        pass
                    hb = None                     # retry on the next beat; a restart is survivable
            try:
                if hb is not None:
                    hb.close()
            except Exception:                                    # noqa: BLE001
                pass
        threading.Thread(target=beat, daemon=True).start()

    def _spawn(self):
        argv = self._build(self.lane)
        self.proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, text=True, bufsize=1)
        line = self._readline_with_timeout(self.proc, 300)
        if not line:
            err = self._drain_stderr()
            raise RuntimeError(f"{self.label}: worker died before ready\n{err}")
        self.ready = json.loads(line)
        if not self.ready.get("ready"):
            raise RuntimeError(f"{self.label}: first line was not a ready banner: {line[:200]}")

    def _readline_with_timeout(self, proc, timeout_s):
        """A bounded readline. THE reason this module does not just reuse sweep_pool.Worker: a
        blocking readline on a wedged driver holds the lane forever and there is no way out of it
        from inside the process."""
        out = {"line": None}

        def rd():
            try:
                out["line"] = proc.stdout.readline()
            except Exception:                                    # noqa: BLE001
                out["line"] = None

        t = threading.Thread(target=rd, daemon=True)
        t.start()
        t.join(timeout_s)
        if t.is_alive():
            return None                 # caller treats None as death and respawns
        return out["line"]

    def _drain_stderr(self):
        """Best-effort stderr tail. KILL FIRST, then read.

        `stderr.read()` blocks until EOF, and EOF only arrives when the process exits. On the
        wedge path the process is very much alive -- that is the whole problem -- so reading first
        hangs forever, which is the exact failure this class exists to prevent, reintroduced in the
        error handler. Found by faulthandler: nine checks passed and the tenth stalled here.

        Killing first makes EOF arrive, so the read returns whatever the worker managed to say
        before it died, which is the part worth keeping.
        """
        try:
            if self.proc and self.proc.poll() is None:
                self.proc.kill()
                try:
                    self.proc.wait(timeout=5)
                except Exception:                                # noqa: BLE001
                    pass
        except Exception:                                        # noqa: BLE001
            pass
        try:
            if self.proc and self.proc.stderr:
                return (self.proc.stderr.read() or "")[-800:]
        except Exception:                                        # noqa: BLE001
            pass
        return ""

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    # -- measuring ----------------------------------------------------------
    def measure(self, cfg, budget_ms, cache="hot", abort_above_ms=None, rid=0,
                priority="wake"):
        """Wake, measure, release. The exclusive window covers ONLY the reading."""
        with self._lk:
            req = {"id": rid, "cfg": cfg, "budget_ms": budget_ms, "cache": cache,
                   "abort_above_ms": abort_above_ms}
            if self.conn is not None and self.park is not None:
                t0 = time.time()
                # want_lane: our VRAM is on this lane, so a grant anywhere else is unusable. The
                # broker enforces that; passing `lanes` too keeps it true even if want_lane is
                # dropped by an older broker.
                g = self.conn.call({"op": "acquire", "kind": "exclusive",
                                    "want_lane": self.lane, "lanes": [self.lane],
                                    "group": self.group, "priority": priority,
                                    "timeout_s": 900,
                                    "meta": {"command_kind": "resident_wake",
                                             "estimated_runtime_s": 30,
                                             "safe_yield_kind": "resident_wake_boundary",
                                             "resident_event": "wake",
                                             "reason": "resident_measurement"}}, timeout=930)
                self.wake_wait_s += time.time() - t0
                self.woke += 1
                if g.get("ok"):
                    try:
                        return self._ask(req)
                    finally:
                        try:
                            # A resident wake is the natural cooperative boundary: it never
                            # suspends the worker mid-reading; it yields immediately after that
                            # reading and records the boundary outcome with the broker.
                            self.conn.call({"op": "boundary", "lease": g["lease"], "action": "yield"})
                        except Exception:                        # noqa: BLE001
                            pass
                # Could not get the window. Measure anyway rather than lose the sweep -- and SAY SO,
                # so a reading taken on a possibly-shared card is distinguishable in the record from
                # one taken under an exclusive window. A silent degrade here would put contaminated
                # numbers in the envelope with nothing to flag them.
                r = self._ask(req)
                if isinstance(r, dict):
                    r["exclusive_window"] = False
                    r.setdefault("detail", "measured WITHOUT an exclusive window: " +
                                 str(g.get("error", "broker refused")))
                return r
            return self._ask(req)

    def _ask(self, req):
        if not self.alive():
            return self._died(req, "worker was not running")
        try:
            self.proc.stdin.write(json.dumps(req) + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            return self._died(req, f"stdin closed: {exc}")
        line = self._readline_with_timeout(self.proc, self.request_timeout_s)
        if line is None:
            return self._died(req, f"no response within {self.request_timeout_s}s (wedged)")
        if not line:
            return self._died(req, "worker produced no response")
        try:
            r = json.loads(line)
        except Exception as exc:                                 # noqa: BLE001
            return self._died(req, f"unparseable response {line[:160]!r}: {exc}")
        self.served += 1
        r["lane"] = self.lane
        r.setdefault("exclusive_window", self.park is not None)
        return r

    def _died(self, req, why):
        """A config that killed the backend is RECORDED, not retried -- it will kill it again."""
        err = self._drain_stderr()
        try:
            if self.proc:
                self.proc.kill()
        except Exception:                                        # noqa: BLE001
            pass
        resp = {"status": "rejected", "ms": None, "err": 1.0, "lane": self.lane,
                "effective": dict(req.get("cfg") or {}), "meter": {"state": "not-reached"},
                "id": req.get("id"), "exclusive_window": False,
                "detail": f"worker-died: {why}{(' | ' + err) if err else ''}"[:600]}
        if self.restarts >= self.max_restarts:
            self.dead = True
            resp["detail"] += f" | GIVING UP after {self.restarts} restarts"
            return resp
        try:
            self._spawn()
            self.restarts += 1
        except Exception as exc:                                 # noqa: BLE001
            self.dead = True
            resp["detail"] += f" | respawn FAILED: {exc}"
        return resp

    def stop(self):
        self.dead = True
        try:
            if self.alive():
                self.proc.stdin.write(json.dumps({"stop": True}) + "\n")
                self.proc.stdin.flush()
                self.proc.wait(timeout=10)
        except Exception:                                        # noqa: BLE001
            try:
                self.proc.kill()
            except Exception:                                    # noqa: BLE001
                pass
        if self.conn is not None and self.park is not None:
            try:
                self.conn.call({"op": "release", "lease": self.park, "reason": "resident_stop"})
            except Exception:                                    # noqa: BLE001
                pass
            self.park = None
            self.park_token = None
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    def stats(self):
        return {"lane": self.lane, "served": self.served, "restarts": self.restarts,
                "park_lost": self.park_lost,
                "woke": self.woke, "wake_wait_s": round(self.wake_wait_s, 2),
                "brokered": self.park is not None, "dead": self.dead,
                "vram_mb": self.vram_mb}

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
        return False


def selftest():
    """Uses a FAKE driver -- a tiny python that speaks the NDJSON protocol. No GPU, no torch."""
    import tempfile
    fails = []

    def ok(c, m):
        print(("  ok: " if c else "  FAIL: ") + m)
        if not c:
            fails.append(m)

    print("\n# residency selftest (fake driver, no GPU)")
    fake = r'''
import json, sys, os
print(json.dumps({"ready": True, "adapter": "fake", "owns_meter": False,
                  "anchor": {}}), flush=True)
for line in sys.stdin:
    line = line.strip()
    if not line: continue
    q = json.loads(line)
    if q.get("stop"): break
    cfg = q.get("cfg") or {}
    if cfg.get("boom"):        # a config that kills the backend
        sys.exit(7)
    if cfg.get("hang"):        # a config that wedges
        import time; time.sleep(3600)
    print(json.dumps({"id": q.get("id"), "status": "ok", "ms": 0.5,
                      "err": 0.0, "effective": cfg,
                      "gpu": os.environ.get("HIP_VISIBLE_DEVICES")}), flush=True)
print(json.dumps({"bye": True}), flush=True)
'''
    with tempfile.TemporaryDirectory() as td:
        drv = os.path.join(td, "fake_driver.py")
        with open(drv, "w") as fh:
            fh.write(fake)

        def build(lane):
            env_prefix = [sys.executable, drv]
            os.environ["HIP_VISIBLE_DEVICES"] = str(lane)
            return env_prefix

        # Force the UNBROKERED path. GEAK_GPU_BROKER=0 is the honest way to do it: it is the same
        # switch operators use to A/B against flock, and it is read at call time.
        #
        # The obvious alternative -- point GEAK_GPU_BROKER_SOCK at a nonexistent file -- does NOT
        # work, and failing to notice that cost a debugging round: gpu_client binds DEFAULT_SOCK at
        # import, so setting the variable afterwards changes nothing, autostart sees the canonical
        # path, and a real daemon gets spawned. Worse, it inherits the fake driver's stdout pipe and
        # never exits, so the selftest hangs with no output at all rather than failing.
        import importlib

        import gpu_client
        os.environ["GEAK_GPU_BROKER"] = "0"
        importlib.reload(gpu_client)
        globals()["connect"] = gpu_client.connect
        globals()["BrokerUnavailable"] = gpu_client.BrokerUnavailable

        w = ResidentWorker(build, group="t", vram_mb=100, lane_hint=3,
                           request_timeout_s=5, max_restarts=2)
        w.start()
        ok(w.ready and w.ready.get("ready"), "the fake driver came up and sent its banner")
        ok(w.lane == 3, f"unbrokered: it used the lane hint (lane={w.lane})")

        r = w.measure({"BLOCK_M": 64}, budget_ms=10, rid=1)
        ok(r.get("status") == "ok" and r.get("ms") == 0.5, f"a measurement round-trips: {r}")
        ok(r.get("lane") == 3, "the reading is tagged with its lane")
        ok(w.served == 1, "served counter advanced")

        r2 = w.measure({"boom": 1}, budget_ms=10, rid=2)
        ok(r2.get("status") == "rejected", "a backend-killing config comes back rejected")
        ok("worker-died" in (r2.get("detail") or ""), "and is attributed to that config")
        ok(w.restarts == 1, "the worker respawned once")

        r3 = w.measure({"BLOCK_M": 128}, budget_ms=10, rid=3)
        ok(r3.get("status") == "ok", "the respawned worker serves the NEXT config fine")

        r4 = w.measure({"hang": 1}, budget_ms=10, rid=4)
        ok(r4.get("status") == "rejected" and "wedged" in (r4.get("detail") or ""),
           "a WEDGED worker is timed out rather than blocking forever (sweep_pool cannot do this)")

        st = w.stats()
        ok(st["served"] >= 2 and st["restarts"] >= 1, f"stats are reported: {st}")
        w.stop()
        ok(not w.alive(), "stop() shuts the driver down")

    print("\nSELFTEST PASS: residency works and degrades" if not fails else f"\nFAIL: {len(fails)}")
    return 1 if fails else 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    print(__doc__)
