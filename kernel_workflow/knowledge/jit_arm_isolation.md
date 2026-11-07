# JIT Arm Isolation — making an A/B actually be two binaries

## When this file applies

The candidate is gated by an env var, a config flag, or a module-level constant, **and** the kernel
is produced by a caching JIT (FlyDSL, Triton, `torch.compile`, or any framework with a compile cache
on disk). If your patch is plain source that is always compiled, this file does not apply — skip it.

This is an **authoring** obligation, not a verification one. Verify will catch a one-binary A/B and
void your direction; catching it there costs the round. Catching it here costs one hash.

## The failure

Two arms print two different markers from Python and then execute **one identical cached binary**,
because the switch never entered the cache key. The measurement reads ~1.000, and every gate passes:
the patch is real, the marker fires, the null arm behaves, the numbers are internally consistent.
There is no symptom. It is indistinguishable from "the idea did not work" — which is precisely why it
is dangerous: it does not produce a wrong number, it produces a **null that closes an axis**.

A retroactive audit on this workflow found an all-ON arm, an all-OFF arm **and** the canonical
unpatched tree all resolving to one `disk_key`. Three arms, two separate checkouts, one binary. The
"well-powered null" it had produced was void and the axis had to be reopened.

## Why separate checkouts do not save you

The compile cache root is typically **per-machine, not per-tree** (FlyDSL: `~/.flydsl/cache`). Two
workspaces with different source therefore hit the *same* cache. Building the arms in separate
checkouts isolates the source and not the artifact — the intuition that "different tree ⇒ different
build" is false here, and it is the intuition most people bring.

## What enters the key and what does not

Worked example, FlyDSL (`jit_function.py`, `_walk` at :247, same at :438/:534): the tracer collects
**only the outer function's `co_names`**. Consequences:

| construct | enters the key? |
|---|---|
| a closure scalar the tracer captures directly (e.g. an `aStages` int) | **yes** |
| a module-level constant read in the **outer** traced function | **yes** |
| an env var / global read inside a **nested or inner-JIT body** | **no** |
| a change that only renames the emitted kernel symbol | **no** |

The middle two are the trap: the switch is visibly in the source, visibly read at run time, and
invisible to the cache. A knob can also survive *by accident* — one on this workflow worked only
because its arm A moved a closure scalar (which does enter the key) while its arm B, which read a
global in an inner body, was never exercised.

Other frameworks have the same shape with different details. Triton keys on the JITFunction source
plus specialization constants — a Python-level global read inside a helper does not key. Check your
framework's key construction rather than assuming; the question to answer is always *"is the value my
switch changes an input to the key, or only to the runtime behaviour of an already-keyed body?"*

## The remedy: anchor the key in the outer function

Read the switch **once, at module level or in the outer traced function**, into a name the tracer
sees, and let the inner body read that:

```python
_key_anchor = int(os.environ.get("MY_SWITCH", "0"))   # module level, read by the OUTER traced fn

def outer(...):
    mode = _key_anchor        # now `mode` is a closure scalar → it enters the key
    ...
    inner(mode)               # inner reads the parameter, never the env var
```

Confirmed working on this workflow: anchoring a three-valued knob this way produced three distinct
keys (`7cf14e9c` / `af58afb4` / `40aee6de`) with the canonical tree proved identical to arm 0 at IR
`sha be49c3b728f4e9e5` — which is the *other* half of the proof and is easy to forget: the null arm
must **match** canonical as surely as the candidate must **differ** from base.

## Even a correct anchor collapses to one binary in a warm process

The anchor above makes the switch *keyable*; it does not make a **dose ladder** actually vary it.
Two mechanisms freeze the value inside a single interpreter, and both bit this workflow — a synthetic
MegaMoE control read a monotone dose ladder as a flat 0.00%, "proved" the harness blind,
and aborted the whole run at round 1:

- **Import-time reads.** `x = int(os.environ.get("SWITCH","0"))` at module top runs **once**, at first
  import. Setting `os.environ["SWITCH"]` later in the same process changes nothing — the module object
  is cached. Read the switch **at compile time** (inside the factory that builds the kernel), not at
  module import, so a fresh call re-reads it.
- **In-memory compile memos.** A get-or-compile cache keyed on the *declared* compile params (FlyDSL
  MegaMoE: `_G2_LAUNCH_CACHE`, a dict keyed on the compile-kwarg tuple; plus the framework's own
  in-process cache) returns the **first** dose's compiled launcher for every later dose, because your
  switch is not one of those declared params. The on-disk key would differ but is never consulted —
  the process already has an answer in RAM. (Symptom seen: `NEW_LAUNCH_DIRS=8` at the first dose then
  `=0` for all the rest.)

The remedy is not more anchoring, it is **process isolation**: run **every dose and every A/B arm as
its own fresh process** (a fresh `torchrun`), with the switch exported in that process's environment.
One process = one binary, no matter how many doses you sweep inside it. A warm interpreter reusing the
first dose's kernel is the single most common way a correct anchor still produces a flat null.

## Prove it before you spend a lease: the two-process key dump

Cheap, and not optional when your control is JIT-gated. In **two separate processes**, build the
launcher and print its resolved cache key, for `switch=0` and for `switch=N`:

```python
launch = compile_the_kernel(**params)      # needs only the arch probe, no GPU lease
launch._ensure_cache_manager()             # FlyDSL: resolves the on-disk manager_key
print(launch.manager_key)
```

Require **`switch=0 != switch=N`** (distinct binaries) *and* **`switch=0 == switch=0`** across two
processes (null identity + proof the key is deterministic). Only then spend the GPU pairs. This dump
resolves the key *before* launch, so it costs no lease; skipping it costs the whole positive-control
step and aborts the run on a false "harness is blind".

## Confirmed recipe on the MegaMoE stage2 kernel (validated on-box)

The exact positive control that passed keying + magnitude on this operator, reproduce it:

1. **Build the control tree by copying the WHOLE workspace WITH `csrc` present** — `cp -a <workspace>
   /tmp/<uniq>` (or copy + symlink `csrc`). A fresh from-scratch tree missing `csrc` crashes every
   dose with `FileNotFoundError: aiter_enum.h not found` (`aiter/utility/aiter_types.py` resolves
   `csrc/include/aiter_enum.h`). Separate checkouts do NOT isolate the artifact (§ above) — the copy
   is only to keep the spin knob out of the task tree, so still build under `/tmp` and move it aside
   **out of `/tmp`** when done.
2. **Inject** into `aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage2.py`: inside
   `compile_mega_moe_stage2`, a factory-LOCAL `_spin_count = int(os.environ.get("GEAK_S2_SPIN","0"))`
   (compile-time read, `import os` at top), referenced inside the nested `@flyc.kernel kernel_epilog_v2`
   at kernel entry as `if const_expr(_spin_count > 0):\n    for _ in range_constexpr(_spin_count):
   rocdl.s_sleep(2)`. `rocdl.s_sleep` is a side-effecting ROCDL op the compiler will not DCE; the
   factory-local referenced in the nested kernel becomes a closure freevar → enters the key. Output-
   neutral (correctness untouched).
3. **Size**: `s_sleep` overlaps across co-resident workgroups, so ~0.3µs/unit — **N must be in the
   hundreds**. Confirmed ladder on 8192_uniform (base ≈ 4.68 ms rank-max): N=256→+1.7%, N=512→+3.9%,
   **N=640→+5.5%** (mid-band), N=896→+8.9%. Anything ≤256 is buried in the ~1.09% worst-pair noise.
4. **Confirmed keys**: `GEAK_S2_SPIN=0 → 71978067946bbdad08574dcf8b040b7f`,
   `=640 → a9326b9b588953c0ab797e8e0b51be8b`, spin0 identical across two processes. (The absolute hash
   depends on the full compile-param set, so a standalone probe yields a different hash than the on-arm
   bench config — what matters is that flipping the spin flips the key.)

## What you owe in your report

Whenever your candidate is JIT-gated, put in `activation_evidence`:

- the **base** arm's cache key / name-normalised ISA hash / resolved binary path,
- the **candidate** arm's same quantity, shown to differ,
- and where a null arm is supposed to be byte-identical work, its hash shown to match canonical.

Name-normalise the ISA hash (strip symbol names) so a kernel rename cannot fake a difference. The
verify role has three dedicated fields for these and the script compares them itself; a hash written
only into prose does not bind.

**Cheap ways to get the hash:** dump the framework's cache key for the compiled callable; hash the
emitted ISA/IR text after stripping symbol names; or compare the resolved cache directory per arm.
Any one of the three is sufficient. Doing none of them makes your result provisional at best and void
at worst.

## The same instrument, run backwards: proving a guard *cannot* move

Everything above uses ISA identity as a **failure** detector — two arms that hash the same did not
really run two arms. The identical hash is also a **result**, and it is the cheapest result in this
workflow, because it settles a guard without spending a lease on it.

The situation: your change helps the shapes it targets and appears to cost a few percent on a
small guard. The instinct is to buy more reps until the noise resolves. Do not — first ask whether
the compiled code for that shape differs at all between the arms. If both arms emit **byte-identical
ISA** for the guard's shape, the guard's measured delta is noise *by construction*: the same binary
cannot be slower than itself. Reproduce the hash once per arm, put both in `activation_evidence`,
and the guard is closed at zero GPU cost. Confirmed on this workflow: a candidate that read −2.5% on
a small guard was resolved this way — both arms at `sha 9b3da4e5b1181d5a` for that shape — instead of
by ten more paired reps.

This inverts the usual burden and it is worth stating plainly: **a measurement can only tell you a
guard did not move to within its noise floor; identical ISA tells you it could not have moved.** When
both are available, the hash is the stronger claim and the cheaper one. Reach for it whenever a
change is gated on a compile-time predicate (a tile size, a block-count threshold, a dtype branch)
and some guard's shape falls outside the predicate.

### A shape that compiles a different kernel config is a different experiment

The precondition for the above, and a trap on its own. In a JIT'd kernel with shape-driven tuning,
each guard shape may select a **different compiled configuration** — different tile shape, different
persistence mode, different scratch budget — under the same source. So a change written against one
configuration is *structurally incapable* of touching a guard that compiles another one, and any
delta you measure there is attributable to something else.

Concretely, on this workflow the large guards compiled a persistent `BM=128` config while the small
guards compiled `t64x512x256 … dcu128 … pc0 ptr0` — `BM=64`, non-persistent, zero scratch. A
candidate gated on `BM=128` could not reach the small guards at all, which is both why their ISA was
identical and why a per-guard attribution written without that fact would have been wrong.

So: **enumerate the compiled config per guard before attributing any per-guard result.** Dump the
kernel name / cache key that each guard's shape resolves to and put the table in the report next to
the per-guard numbers. It costs one run of the harness with the JIT's naming visible, and it turns
"this guard is inexplicably flat" into "this guard compiles a config the change does not touch",
which is an explanation rather than an anomaly.

## Do not confuse this with activation

They fail at different layers and have different remedies:

| | symptom | remedy |
|---|---|---|
| **inactive** | the host path never ran | fix the predicate; check the marker |
| **one binary** | the host path ran on both arms, and both dispatched the same code | anchor the key |

A marker proves the first and says nothing about the second.
