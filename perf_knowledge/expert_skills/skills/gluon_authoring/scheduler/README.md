# GPU scheduler — one broker for the whole box

Eight cards. Sixteen kernels. Each kernel is an IC Tech Lead that profiles, sweeps, fans structural
arms and climbs — and every one of those steps wants a GPU only at the instant it measures. This
directory is the thing that decides who gets which card, when.

It is **opt-in, off by default, and inert when absent**. GEAK's single GPU lock is
`kernel_workflow/scripts/gpu_lock.sh` (flock on `/tmp/team_gpu_locks/gpu_<id>.lock`; the pack's
`scripts/gpu_lock.sh` is a shim that execs it). That wrapper consults this broker only when
`GEAK_GPU_BROKER=1` **and** a broker socket is listening; otherwise it takes the flock path and every
call site behaves exactly as before. Even on the brokered path the granted lane still takes the same
flock, so brokered and non-brokered processes on one box interlock. Nothing in GEAK starts a broker
or enables it -- a campaign that wants queueing/fair share does both explicitly (see *Using it*).

```
core.py         the scheduling decision as a pure state machine. No I/O, no clock, no GPU.
gpu_broker.py   the daemon. Unix socket, NDJSON, one per box.
gpu_client.py   library + CLI. The context manager, the retry policy, and the fallback.
residency.py    a warm --serve worker that parks in VRAM and wakes per reading.
tests/          policy tests (fake clock) + integration tests (real daemon, killed clients)
```

## Why not just keep `gpu_lock.sh`

`flock` answers *"is this lane free right now"*. It cannot answer *"who should get it next"*, and
with 16 kernels on 8 cards that difference is the whole problem.

| | `gpu_lock.sh` (flock) | this broker |
|---|---|---|
| mutual exclusion | yes | yes |
| who goes next | whoever calls at the right microsecond | priority, then fair share, then FIFO |
| starvation | unbounded | bounded by aging |
| fair share across kernels | none | per-kernel, re-evaluated per grant |
| a parked resident | invisible, or blocks the card | a first-class lease |
| crash recovery | lock dies with the process | same, plus TTL for wedged holders |
| why was it slow | nothing recorded | `wait_s` per acquisition, in the journal |

Three specific things it could not express, all of which this workload has:

1. **A queue.** A kernel needing one 3-second verify can sit behind a kernel running a 400-point
   sweep, indefinitely, with nothing recording that it happened.
2. **Residency.** `sweep_driver.py --serve` parks in VRAM between readings, worth 14–48× per reading.
   Under flock a parked worker either holds the lock (blocking the card for minutes while computing
   nothing) or drops it (and a sibling lands a timed run on top of it) — and it did the latter,
   because `sweep_pool.py` spawns its workers with a bare `HIP_VISIBLE_DEVICES` and never went
   through `gpu_lock.sh`, so a resident's context was invisible to every other actor on the box.
   It now takes a short `wake`-priority exclusive window per reading through this broker
   (`sweep_pool._lease_window`), and falls back to the old unbrokered reading when no broker is
   listening. `plain_best_config.json` records the window rate, so an unarbitrated resident run is
   visible in the artifact instead of only in the timings it contaminated.
3. **Fair share.** flock has no notion of "this kernel already holds three lanes".

## The model

A GPU is a **lane**. A lane holds at most one **EXCLUSIVE** lease, plus any number of **RESIDENT**
leases.

- **EXCLUSIVE** — the lane's compute is yours. Timed benchmarks, profiles, compiles, correctness
  runs. One per lane.
- **RESIDENT** — you hold VRAM here and promise not to compute. A parked `--serve` worker. Many per
  lane, bounded by VRAM, and it **never blocks an exclusive grant** — a parked process is not using
  the SMs.

A resident that wants to measure calls `wake()`, which takes a **short exclusive window on the lane
it is already parked on**. That is the whole design in one sentence: the process stays warm across
readings, but each reading is still exclusive, so the number is clean.

There is deliberately **no shared-compute lease kind**. It was considered and rejected: the moment a
task that produces a number can share a lane, every number becomes conditional on what else happened
to be running, and the contamination does not appear in the artifact. Compiles could safely share —
but a compile is also the thing most likely to be mis-tagged by a caller, and a mis-tagged timing is
unrecoverable. The throughput left on the table is real; this system's output is measurements.

### Priority bands

Lower sorts first. Work that **unblocks other work** outranks work that merely produces a result.

| band | value | for |
|---|---|---|
| `interactive` | 10 | a human waiting at a terminal |
| `wake` | 20 | a parked resident wants its measurement window |
| `verify` | 30 | a verify/close gates a whole kernel's completion |
| `measure` | 40 | the ordinary timed reading (default) |
| `sweep` | 50 | bulk config search — high volume, individually cheap |
| `bulk` | 60 | compiles, warmups, anything speculative |

**Aging** promotes a request one band per `--aging-s` (default 30 s) of waiting, which is the
anti-starvation guarantee: a bulk compile behind an endless stream of sweeps eventually runs, because
its effective priority decreases without bound. `test_scheduler_core.py` asserts both halves — that
it is served with aging on, and that it starves with aging off.

**Fair share** is per `group`, normally the kernel. Within a band, the group holding fewest lanes
wins. Without it, one kernel's 400-point sweep serves its entire backlog before a sibling's first
request — the 23× imbalance that per-kernel pinning was meant to fix, reintroduced one level up.

## Using it

```bash
# start it deliberately (client autostart is OFF unless GEAK_GPU_BROKER_AUTOSTART=1)
python3 scheduler/gpu_broker.py --serve --gpus 0,1,2,3,4,5,6,7

python3 scheduler/gpu_broker.py --status      # lanes, queue, leases, stats
python3 scheduler/gpu_broker.py --status --json
python3 scheduler/gpu_client.py status --json
python3 scheduler/gpu_broker.py --drain 3     # stop granting lane 3; running work is untouched
python3 scheduler/gpu_broker.py --undrain 3
python3 scheduler/gpu_broker.py --shutdown
```

Every existing call site works unchanged once the campaign opts in — `gpu_lock.sh` delegates when
`GEAK_GPU_BROKER=1` and a broker is listening:

```bash
export GEAK_GPU_BROKER=1                       # opt in (default 0 = flock only)
cd $WORKSPACE && bash $GEAK/kernel_workflow/scripts/gpu_lock.sh $GPU_ID <cmd>
# (or the pack shim $SKILL_DIR/scripts/gpu_lock.sh -- it execs the same file)
```

From Python:

```python
from gpu_client import lease

with lease(kind="exclusive", group="kernel_name", priority="measure") as g:
    bench(gpu=g.lane)              # g.lane is None if no broker -> caller carries on

with lease(kind="resident", vram_mb=8000, group="kernel_name") as park:
    for cfg in configs:
        with park.wake():          # short exclusive window on the SAME lane
            measure(cfg)
```

GEAK ships no fleet driver that starts the broker (the upstream tile-programming `kernel_fleet.js`
did; it is not part of GEAK). Start it yourself, export `GEAK_GPU_BROKER=1` for the campaign, and
read the scheduler's cost back with `audit_journal.py`.

### Lane ids: logical, not physical

A lane id is handed to the command as `HIP_VISIBLE_DEVICES` and to `gpu_lock.sh` as its GPU id, so it
is a HIP ordinal **of the caller's process**. `detect_lanes()` takes `GEAK_GPU_BROKER_GPUS`, else the
broker's own `HIP_VISIBLE_DEVICES`, else counts `/sys/class/drm/renderD*` nodes. Those agree with the
physical card index -- and with the sysfs node the idleness probes read (`renderD(128+8*id)`) and the
`/tmp/team_gpu_locks/gpu_<id>.lock` file every GEAK process flocks -- only when no
`ROCR_VISIBLE_DEVICES` mask or container device remap is in effect. Under a mask, every tenant of the
box must use the same numbering, pass `--gpus` in that numbering, and treat the sysfs idleness
reading as describing physical card `<id>`, which may be a different card.

## Environment

| variable | default | meaning |
|---|---|---|
| `GEAK_GPU_BROKER` | `0` | `1` opts in: `gpu_lock.sh` and `gpu_client` use a listening broker. Unset/`0` = flock only |
| `GEAK_GPU_RUNTIME_ROOT` | `~/.cache/tile-runtime/gpu-broker` | broker runtime root (socket, `state/journal.ndjson`, `broker.log`) |
| `GEAK_GPU_BROKER_SOCK` | `$GEAK_GPU_RUNTIME_ROOT/gpu.sock` | socket path (else a `.geak_gpu_sock` marker at a parent of `$PWD`) |
| `GEAK_GPU_REQUIRE_MARKER` | `1` | refuse to adopt the canonical socket unless env or a marker named it |
| `GEAK_GPU_BROKER_AUTOSTART` | `0` | `1` spawns a daemon on first use (canonical socket only) |
| `GEAK_GPU_GROUP` | inferred from `$PWD` | the fair-share key |
| `GEAK_GPU_PRIORITY` | `measure` | band for this acquisition |
| `GEAK_GPU_POOL_WAIT` | `1200` | queue deadline, seconds |
| `GEAK_GPU_REQUIRE_GROUP` | `0` | when `1`, a brokered wrapper run fails if it cannot resolve a stable group |

### How the fair-share key gets there

Three sources, in descending order of trust:

1. **`GEAK_GPU_GROUP`** — explicit; nothing to infer.
2. **A `.geak_gpu_group` marker file** at the campaign/eval-dir root, found by walking up from
   `$PWD`. It is **exact** when present -- but nothing in GEAK writes it (`kernel_lane.js` does not);
   a campaign that wants per-kernel fair share writes it itself (or exports `GEAK_GPU_GROUP`).
3. **A `$PWD`-shape heuristic**, for trees the lane did not create (a hand-run experiment).

The Python client and shell wrapper call the same resolver. Its fallback ignores transient
direction leaves such as `dir_<n>`, `direction_<n>`, `round_<n>`, and `workspace`; those are work
units of a kernel, never separate fair-share tenants. Managed runs may set
`GEAK_GPU_REQUIRE_GROUP=1` (or pass `gpu_client.py run --require-group`) to fail before execution
rather than silently use a connection-specific fallback identity.

### Runtime metadata and cooperative boundaries

Each request can carry `command_kind`, `estimated_runtime_s`, and `safe_yield_kind`. Estimates are
classified as short (`<120s`), medium (`120–300s`), or long (`>300s`). Within an aged priority band,
short work can pass an unstarted long request; normal per-kernel fair sharing still applies, and
aging eventually promotes the long request so it cannot starve.

The broker never sends a signal to, suspends, or kills an active GPU process. It may journal a
`pause_requested` record for a holder that declared a safe boundary. At that boundary the holder
uses `Lease.boundary()` / broker `boundary` to select `continue`, `yield`, or `cancel`. Resident
workers park as `resident`; every timed point requests a short exclusive `resident_wake` window and
journals its park, wake, wait, and cooperative-boundary reason.

`--status --json` and `gpu_client.py status --json` expose the broker socket, lane state, queued
requests, leases, TTLs, fair-share group, runtime metadata, and any advisory pause request. Private
renewal tokens are deliberately excluded.

The indirection through a file is not incidental. The acquisitions are made by *agents*, in shell
commands they compose themselves from prose in the role files — so no environment variable the
orchestrator sets would ever reach them, and threading it through the prompts would mean editing
every role file. Whoever runs the campaign writes the marker; `gpu_lock.sh` reads it.

For one review cycle this was the heuristic *alone*, and it was wrong on the documented default
layout: for `<exp_root>/team_<kernel>_<ts>/<kernel>/…` it matched the `team_` wrapper and returned
`team_<kernel>_<ts>` — a per-**run** string, so two runs of one kernel counted as two tenants. Every
log looked plausible because one run's trunk and arms all agreed with each other.

## Failure posture: degrade, never block the fleet

- **No daemon** → clients return an unbrokered lease and run anyway.
- **Daemon dies mid-run** → the socket closes, clients fall back rather than failing the command.
- **A client dies** → its socket closes and every lease on it is released immediately. This works
  across PID namespaces, which matters because clients may be in a different container.
- **A client wedges** (connected but stuck) → the lease TTL expires and the lane is reclaimed. The
  holder renews at ⅓ TTL in a background thread **on its own socket**, so a legitimate 40-minute
  climb keeps its lane. The dedicated socket is load-bearing — see below.
- **A lease is reclaimed anyway** → the holder finds out. `gpu_client run` warns on stderr naming
  the lane; a sweep reading carries `exclusive_window: false` and `window_reclaimed: true`. Losing
  a lane silently is what turns a scheduling hiccup into an unexplainable number.
- **Connection refused under load** → retried with jittered backoff. See below.

A scheduler that can take down a 12-hour fleet run when it crashes is worse than no scheduler.

## Every reading says whether it was arbitrated

`exclusive_window: true|false` travels on each sweep reading, plus `window_reclaimed` when a lease
died mid-measurement. A reading taken without a window is not wrong — the pool measures anyway
rather than abandoning the sweep on a busy box — but it is weaker evidence, and by the time a config
is ranked nobody can reconstruct which kind it was. A contaminated reading is otherwise
indistinguishable from a merely slow one.

## Bugs this work found

All found by tests, all invisible in production.

**The listen backlog.** `socketserver`'s default `request_queue_size` is **5**. With 32 simultaneous
clients, 8 were refused, silently degraded to UNBROKERED, and ran their benchmarks on unarbitrated
cards — precisely the contamination the daemon exists to prevent, appearing only under load and
leaving no trace in the artifact. Fixed by raising the backlog to 512 *and* by retrying a refused
connection (a refusal is transient; an absence is not, and conflating them is what caused the silent
degrade). Asserted by `test_broker_integration.py`.

**The foreign-tenant check cannot see its own residents.** `gpu_lock.sh` marks a lane busy at
`vram_used > 1024 MB`. Residency means deliberately holding tens of GB, so that test would fence off
every card the system is successfully using. Observed on this box: GPU 0 at 231 GB used, 0% busy —
permanently "foreign" by the old rule. The broker subtracts what it knows it holds and judges only
the remainder.

**The heartbeat starved exactly when residency needed it.** `Conn.call` holds a per-connection mutex
for the whole round-trip, so a renew sharing the caller's socket could not fire while any request
was in flight — and the request it normally waits behind is the blocking `acquire` for a measurement
window, which is *the* residency pattern. Measured: 4 s TTL, 14 s queue, resident evicted while
still holding its VRAM, its lane handed to another tenant. Renewals now use a dedicated socket,
authorised by a per-lease **renew token** (a second socket is a different `owner`, and
owner-identity is what makes connection-close reclaim work, so it could not just be relaxed). The
token grants renew and nothing else.

**Stamping outside the `with` inverted its own answer.** `Lease.release()` clears `brokered`, so
reading the flag after the block reported every arbitrated reading as unarbitrated — the opposite of
the truth, and exactly as silent as the bug the stamp exists to expose.

## The sweep path (upstream tooling, not shipped in GEAK)

`sweep_pool.py`, `plain_autotune` and `sweep_driver.py --serve` belong to the upstream
tile-programming fleet tooling; none of them ships in GEAK, so this section describes the contract a
resident sweep must meet rather than code in this repo (the integration test that exercises it skips
here). In that tooling, `tools/sweep_pool.py` is where resident workers live, and it had three defects that this layer had to
close rather than schedule around:

- **It bypassed both allocators.** Workers were spawned with a bare `HIP_VISIBLE_DEVICES` and never
  touched `gpu_lock.sh`, so a resident's CUDA context was invisible: the lockfile read *free* and a
  sibling's timed benchmark landed on the same card. Fixed — `Pool.measure` and `Pool.measure_batch`
  now take a short exclusive window per reading via `_lease_window()`, which is a no-op when no
  broker is running, so the default path is unchanged.
- **A config's repeated readings scattered across GPUs.** `plain_autotune._pool_read` never passed
  `lane_index`, so the reps ladder and the outlier re-read hit different cards — and their spread
  was then taken as run-to-run noise, folding *cross-lane* variance into the significance test that
  decides whether a delta is real. Fixed with `_lane_for(cfg)`, a stable hash of the config key, so
  one config always returns to one lane.
- **`Worker.ask(timeout=...)` accepted a timeout and never used it**; a hung driver blocks
  `readline()` forever. `residency.ResidentWorker` implements the bounded read, and is the path to
  use for new resident work.

The fair-share key reaches the sweep the same way it reaches everything else: `GEAK_GPU_GROUP`, else
an (optional) `.geak_gpu_group` marker walked up from `$PWD`.

Set `GEAK_SWEEP_NO_LEASE=1` for a pool whose workers touch no GPU (the selftest does this) — leasing
on their behalf would ask a real broker for a real card and wait out the window for nothing.

## Auditing a run

```bash
python3 scheduler/audit_journal.py [~/.cache/tile-runtime/gpu-broker/state/journal.ndjson]
```

Answers the two questions nothing else can:

**Was the arbitration sound?** It replays the journal and checks that no lane ever held two exclusive
leases at once. Every timed number the fleet produced rests on that.

**What did sharing cost?** `wait_s` versus `hold_s` is the price of a shared pool. Measured here on 7
real MI325X cards:

```
16 kernels / 7 GPUs      exclusivity: PASS   per-lane grants 16..19   wait 0%     (not GPU-bound)
 8 kernels / 2 GPUs      exclusivity: PASS   per-lane grants 12/12    wait 70.4%  (GPU-bound)
```

The second is not a failure — it is the scheduler correctly reporting that eight kernels on two cards
want more cards. Under the partitioned model that run produces the same contention with no record of
it, and the only symptom is that the fleet is mysteriously slow.

Order records by `seq`, never by `t` — see the note at the top of `audit_journal.py` for the false
alarm that motivated it.

## Testing

```bash
python3 scheduler/tests/test_scheduler_core.py       # policy, fake clock, milliseconds
python3 scheduler/tests/test_broker_integration.py   # real daemon, real sockets, killed clients
python3 scheduler/gpu_broker.py --selftest
python3 scheduler/gpu_client.py selftest
python3 scheduler/residency.py --selftest
```

All five run on a CPU-only runner (no GPU, no broker needed beyond the temp daemons they start).
They are not yet wired into GEAK's `ci-l0-checks.yml`; run them directly. The integration tests opt
in with `GEAK_GPU_BROKER=1` and point `gpu_lock.sh` at a private lock namespace
(`GEAK_GPU_LOCK_DIR`, test-only) so they never queue behind a real tenant of `/tmp/team_gpu_locks`.
The policy tests are fast and deterministic because `core.py` takes `now` as an argument — fairness
and aging are assertions about time, and with a real clock they would be slow or flaky, which in
practice means deleted.
