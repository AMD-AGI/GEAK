# Clocks and power: the drift underneath Rule 6b, and why you probably cannot fix it

`measurement.md` Rule 6b says to interleave A/B timings because clocks drift on
gfx950. That rule prescribes a *software* workaround. The obvious question it
does not answer is why we are working around the problem instead of removing it:
GPUs have a clock-pinning mechanism, and if pinning worked, interleaving would be
unnecessary and every back-to-back number in the literature would be fine.

This file is the answer. Short version: the drift is real and reproducible, the
pinning mechanism exists, and **in a container it silently does nothing**.

## The drift, reproduced in isolation

No A/B, no tuning, no two kernels. One bf16 GEMM at 4096³, warmed up 50
iterations, then timed in twelve consecutive rounds of 30 iterations each. Every
round does identical work.

```
per-round TFLOPS: 1239 1291 1284 1342 1366 1376 1393 1397 1398 1396 1393 1399
best=1399  worst=1239  spread=1.129x  first_vs_last=0.885x
```

This is not noise. It is a **monotonic ramp** that flattens around round 7, and
it is the entire mechanism behind Rule 6b: the part is ~13% slower at the start
of a measurement session than in its steady state. Whatever you time first looks
bad. In a sweep, "first" is the baseline, so every early candidate inherits an
advantage it did not earn.

Note that this survives a 50-iteration warmup. Warmup is the usual answer to this
problem and here it is not sufficient — the ramp is still climbing 7 rounds and
~2000 GEMMs later.

## The fix that exists

ROCm exposes performance determinism, which pins SCLK to a fixed ceiling instead
of letting the governor range:

```bash
rocm-smi -d 3 --setperfdeterminism 1900   # pin
rocm-smi -d 3 --resetperfdeterminism      # release
```

That is the correct tool for this problem and it is what you should reach for.

## The fix that silently does not work

Run inside the vLLM container on this machine:

```
$ id -u
0
$ rocm-smi -d 3 --setperfdeterminism 1900 >/dev/null 2>&1; echo "exit=$?"
exit=0
$ rocm-smi -d 3 --setperfdeterminism 1900 2>&1 | grep -ciE "error|fail|permission|denied"
0
$ rocm-smi -d 3 --showperflevel
GPU[3]  : Performance Level: auto
```

**Root user. Exit code 0. Zero error, failure, permission or denial text
anywhere in the output. And the performance level is unchanged.**

Re-running the drift measurement after that apparently-successful call:

```
per-round TFLOPS: 1194 1272 1324 1346 1382 1390 1399 1401 1398 1400 1399 1387
best=1401  worst=1194  spread=1.173x  first_vs_last=0.861x
```

Unchanged — 17.3% spread, same monotonic ramp. Nothing was pinned.

The cause is one level down. `rocm-smi` writes through sysfs, and in the
container that filesystem is mounted read-only:

```
$ echo manual > /sys/class/drm/card3/device/power_dpm_force_performance_level
bash: .../power_dpm_force_performance_level: Read-only file system
```

Being UID 0 is not enough; the mount is read-only regardless of capability, and
the container is not started `--privileged`. `rocm-smi` swallows the write
failure and reports success.

## Why this matters more than an ordinary missing feature

The failure mode is the dangerous kind. A reasonable engineer pins the clocks,
sees no error, concludes the machine is now deterministic, and **stops
interleaving** — at which point they are taking back-to-back measurements on a
part that ramps 13-17%, while believing they have eliminated exactly that
problem. The silent success is worse than a hard failure would be, because a
hard failure would have sent them back to Rule 6b.

So the operational rules are:

1. **Never infer that clocks are pinned from `rocm-smi` exiting 0.** Verify with
   `--showperflevel` and require it to read something other than `auto`. That
   check is cheap and it is the only trustworthy signal.
2. **In an unprivileged or read-only-sysfs container — which is how essentially
   all of this work is done — clock pinning is unavailable.** Interleaving is
   not a convenience, it is the only mechanism you have.
3. If you do control the host and can pin clocks, Rule 6b becomes less critical
   but is still correct: interleaving costs nothing and removes thermal and
   neighbour effects that pinning does not.

## What was not tested

Whether pinning actually removes the ramp on this hardware. It could not be
tested here because it could not be enabled. On a privileged host the experiment
is exactly the script above, run before and after `--setperfdeterminism`, and it
is worth doing — if pinning flattens the ramp, a privileged benchmark host is a
meaningful investment for anyone doing this at scale.

Also untested: power capping (`--setpoweroverdrive`), which is the other lever on
the same phenomenon, and which is blocked by the same read-only mount.
