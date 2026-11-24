# Gluon pipeline — authored overlap: register prefetch, what is on offer, and the gfx942 downgrade set

## Authored overlap (no compiler patch)

On the Gluon path overlap is written by hand: no pipeliner pass runs in `gluon_to_ttgir` (3.8.0 source).
Mechanism overview: `../../tile-programming/pipeline.md`. Hand staging and re-injection do not combine in
one loop — a hand barrier makes that pass skip the loop and a mutable `local_alloc` makes it error
(`../../tile-programming/pipeline.md ## Authoring the overlap yourself (the climb default on the Gluon path)`).
Whether the loop has latency to hide at all: `../../tile-programming/pipeline.md ### First: does this loop want a pipeline at all?`.

### Register-level prefetch — no LDS, no async surface, no marker

Keep `depth` operand bundles in flight in registers and rotate with a statically unrolled loop (load
`i + depth`, MMA `i`, rotate). Same on gfx950 and gfx942, no version gate. `gl.static_assert` that the
trip count divides by `depth` (no remainder loop); make `depth` a launch parameter (it trades against
VGPRs). It suits register-resident tiles and loops with few MFMAs per iteration (decode GEMM, chunk
walks), where a register double-buffer that fits without spilling has measured ahead of a deeper staged
pipeline (`../../pitfalls/negative-patterns.md ## gfx942 / CDNA3 negative-result signatures (mechanism + error text)`, item 2);
AMD's gfx950 tutorial GEMMs, which are compute-bound and LDS-staged, are built on an async LDS ring
instead, with register prefetch layered on top. A/B the two on the loop at hand.

### What is actually on offer, and which architecture has it

**A — data movement** (gfx942 cells: compile-and-run probes)

| # | mechanism | gfx950 | gfx942 (downgrade) | gfx1250 / RDNA |
| --- | --- | --- | --- | --- |
| A1 | async global→LDS multi-buffer (`commit_group` / `wait_group`) | `cdna4.async_copy`, 128- or 32-bit | **✓ at 32-bit/thread, clean tiling, shared** `order=[1,0]` — all three; measured slower than sync staging on the one kernel tried | out of scope |
| A2 | per-tensor pipeline depth | over A1 or A5 | over A5 or A1 | — |
| A3 | several chains at staggered depths | over A1 | over A5 or A1 | — |
| A4 | sub-buffer splitting (`.index(i)` / `.slice(...)`) | ✓ | over A5 | — |
| A5 | sync staging: `allocate_shared_memory` + `.store()` / `.load()` + barrier | ✓ (control for the C cases) | **✓ — the usual form** | — |

**B — scheduling hints** (move no data; their effect is known only after measuring)

| # | mechanism | gfx950 | gfx942 | availability |
| --- | --- | --- | --- | --- |
| B1 | `warp_pipeline_stage` markers | ✓ — C5 in `scripts/pipeline_examples_cdna4.py` (`num_warps=8`, `warpsPerCTA=[8,1]` on `[64, 32]`, no `gl.barrier()` in the loop; `s_setprio` in the asm proves it lowered) | ✓ — emits `s_setprio` | symbol 3.7.0+; race-free GPU-verified on 3.8.0 / 3.9.0 only |
| B2 | `sched_barrier` / `sched_group_barrier` / `set_prio` | **✗ symbol absent** | **✗ symbol absent** | absent from `gl.amd.cdna3`/`.cdna4` on 3.6.0–3.8.0 |
| B3 | `warp_specialize` | **✗** `PassManager::run failed` (`TritonAMDGPUConvertWarpSpecializeToLLVM`; gfx1250-only) | **✗** same | symbol present in core `gl` |

- **B2:** importing these from `gl.amd.cdna3` inside `try/except` with **no-op stubs** compiles, but on
  3.6.0–3.8.0 the symbols are absent (3.8.0 re-checked by `hasattr`), so the stubs are what runs and the
  iglp/priority hints do nothing (pattern seen in aiter `aiter/ops/triton/gluon/pa_decode_gluon.py`, if
  available). The reachable
  form is one layer down: B1 markers lower to `s_setprio` + `sched_barrier` region boundaries in
  `make_llir`, which an LLVM-level pass can use (`../../tile-programming/llir-codesign.md ## Declaring the interleave`).
- **B1 is a hint**, sometimes performance-negative (`../../pitfalls/platform-known-issues.md`); below
  `num_warps=8` it creates no inter-wave offset on 3.8.0 (the group split keys on thread id < 256, so 4
  waves form one group — `marker-and-version-gates.md`, Gate 0).
- **A symbol that imports may still mean something else:** a serving stack had replaced
  `triton.next_power_of_2` process-wide with a plain Python function — fine on the host, wrong inside a
  traced body that needs a `constexpr_function`. Probe by compile-and-run, not `hasattr`
  (`../../tile-programming/layout-recipes.md ## Compute the layout instead of tabulating it (constexpr_function)`).

**S — ordering** (decide when a read is allowed; rows are worked compile-and-run cases, not counts)

| # | mechanism | gfx950 | gfx942 (downgrade) |
| --- | --- | --- | --- |
| S1 | `commit_group` / `wait_group` — groups allowed in flight at the read | ✓ | with A1, at 32-bit |
| S2 | `load_shared_relaxed` — assert the fill already retired; after a `wait_group` it only adds alias-scope info (no barrier removed), after a bare `s_waitcnt` it drops the read's barrier | ✓ | with A1 |
| S3 | hand-issued `s_waitcnt vmcnt(N)` via `gl.inline_asm_elementwise` — drain finer than one group | door present (not count-probed) | door present (not count-probed) |
| S4 | `gl.barrier()` (3.8.0 `gl.barrier(*, cluster=False)`; 3.6.0 `thread_barrier`) — CTA rendezvous + LDS-write visibility | ✓ | ✓ (with A5 the only pairing) |

Failure modes differ per row:

- **S1:** a missing `commit_group` is a silent race on 3.8.0 (no asyncmark, so no `wait_group` covers the copy); a stale depth either loses overlap or reads
  before the fill — same edit, so a depth change is a correctness change (`async-ordering.md ## Getting wait_group right`).
- **S2:** missing = a perf remark; wrongly asserted = removes a barrier you needed and still compiles
  (`../../tile-programming/pipeline.md ### Vetted double-buffer skeleton (copy, then specialize)`).
- **S3:** a wrong `s_waitcnt` is a wrong answer at an unrun shape and forfeits S2's soundness
  (`../inline-asm-reference.md ## Class 4 — synchronization`).
- **S4:** a barrier *inside* a `warp_pipeline_stage` fails conversion; one left in a loop you wanted
  re-injected makes that pass skip it **silently**. Its *absence* is the correctness bug.
- **Barrier after `wait_group`** (per-wave retire): needed when a wave reads LDS another wave filled. On
  3.8.0 the compiler already
  emits `s_barrier` after every `wait_group` and before refills
  (`../../tile-programming/pipeline.md ### What the compiler inserts after wait_group: barriers and load_shared_relaxed`).
  The overwrite-before-read barrier is a separate decision.
- `amdg.cond_barrier` is not callable — only stage markers (or the ping-pong pass, which stock
  `gluon_to_ttgir` lacks; the re-injection route adds `add_block_pingpong`) emit it. `fence_async_shared`
  is NVIDIA-only (`../appendix-api.md ## Spellings that do not exist on the CDNA Gluon path`).

**Summary:** gfx950 = A1 (128/32-bit, C cases) + A5 control + A2/A3/A4 built from A1 + B1 hint, ordered by
S1/S2/S4. gfx942 = A5 default, A1 at 32-bit only, B1 hint.

### Worked examples — every one compiled and run on gfx942

gfx942 downgrade set (and the A5 control on either arch), runnable as `scripts/pipeline_examples_cdna3.py`
(prints `ds_read` / `ds_write` / `s_setprio` census + numerics — read your own counts). On gfx950 run
`scripts/pipeline_examples_cdna4.py` (cases A5, C1–C5; C1–C4 in `async-ordering.md ## The async forms — gfx950 only, and the shape differs from A5`).
C1–C5 are script labels, not the profiling calibration chain C0–C4.

> The cdna3 script's `correct` column is a smoke test: every iteration reads the **same** tile, so a
> wrong buffer index still sums correctly. Its `BLK` covers `[64, 16]` on a `[32, 32]` tile — harmless for
> sync staging, **fatal on an async path** (two accesses per lane ⇒ `failed to translate module to LLVM IR`,
> the same text as a genuinely absent op).

**A5 — two LDS buffers, barrier between phases** (gfx942 default; gfx950 control):

```python
BLK: gl.constexpr = gl.BlockedLayout([1, 4], [16, 4], [4, 1], [1, 0], [])
SH: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, order=[1, 0])
_barrier = getattr(gl, "thread_barrier", None) or gl.barrier   # 3.6.0 spelling shim

@gluon.jit
def a5_sync_double_buffer(out, inp, M: gl.constexpr, N: gl.constexpr, ITERS: gl.constexpr):
    s = gl.allocate_shared_memory(gl.float32, [2, M, N], SH)
    o = (gl.arange(0, M, layout=gl.SliceLayout(1, BLK))[:, None] * N
         + gl.arange(0, N, layout=gl.SliceLayout(0, BLK))[None, :])
    acc = gl.full([M, N], 0.0, gl.float32, layout=BLK)
    s.index(0).store(gl.load(inp + o))                 # prologue fills buffer 0
    for i in range(ITERS):
        cur = i % 2
        nxt = (i + 1) % 2
        _barrier()
        if i + 1 < ITERS:
            s.index(nxt).store(gl.load(inp + o))       # stage i+1 while consuming i
        acc += s.index(cur).load(BLK)
        _barrier()
    gl.store(out + o, acc)
```

Runtime `i % 2` buffer indices: over sync staging they can emit ISA identical to constant indices, and
shipped Gluon indexes async main loops with a runtime modulo — unroll only when the ISA shows a gain
(`../../tile-programming/pipeline.md ### Hand-built buffering rules (correctness + scheduling footguns)`).

**A2 — two tensors at different lead distances** (plain's auto-pipeliner takes one `num_stages` per
loop, so it does not build this). Same skeleton with `sb` double-buffered and led by one iteration, `ss` single-buffered with no
lead, both stored between the two barriers, consumed as `sb.index(cur).load(BLK) * ss.index(0).load(BLK)`.
Production form: a block-scaled GEMM leading operands by two K-iterations and scales by one (keeps scale
latency off the critical path and saves `ds_read` bandwidth). A3 = the same with more chains (e.g. a
page-index chain beside a KV chain, drained separately).

**A4 — one buffer consumed as half-slices:** `s.slice(0, M // 2, 0).load(HALF)` /
`s.slice(M // 2, M // 2, 0).load(HALF)` — `slice(offset, length, dim)`; the consuming layout must match the
sliced shape, not the parent's (the usual first failure).

**B1 — cluster markers reorder, they stage nothing:**

```python
    for _i in range(ITERS):
        with gl.amd.warp_pipeline_stage("load", priority=1):    # probe values: BACKWARDS from the
            v = gl.load(inp + o)                                 # shipped GEMM convention (memory stage
        with gl.amd.warp_pipeline_stage("compute", priority=3): # higher) -- a probe, not a recipe
            acc += v * 2.0
```

Census: `s_setprio` appears, `ds_read`/`ds_write` stay 0. With LDS staging the same marker creates the
two-group phase offset (`../../tile-programming/warp-pipeline.md`).
