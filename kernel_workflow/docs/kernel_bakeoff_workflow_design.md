# Design: `kernel_bakeoff_workflow` — standalone multi-backend bake-off

> **Goal.** Give one kernel (e.g. a HIP kernel) and **explore HIP / Triton / FlyDSL in parallel** —
> reuse an existing implementation per language where one exists, author one from scratch where it does
> not — then **pick the fastest** verified result. This already exists inside `e2e_workflow`'s HeadKernel
> track; this design exposes it as a **standalone entry-point workflow** for users who run only the
> kernel layer.
>
> **Design principle: minimal change to the current repo.** `kernel_workflow.js` is **not touched at
> all**. `e2e_workflow.js` is **not touched**. The only edit to an existing file is one **additive**
> line in `op_benchmarker.md`. Everything else is **new files** under a new `kernel_bakeoff_workflow/`
> directory.

Status: **IMPLEMENTED (unified-entry variant)** · Owner: kernel layer · Last updated: 2026-07-27

> **⚠️ What shipped differs from the original proposal below.** Instead of a *separate*
> `kernel_bakeoff_workflow.js` entry, the capability shipped as a **single entry point with `mode`
> dispatch** (chosen for a cleaner UX — one interface for users):
>
> - `kernel_workflow/kernel_workflow.js` is now a **dispatcher**: `mode=optimize|author` passes straight
>   through to the single-language worker; `mode=bakeoff` **is** the bake-off orchestrator (it plays
>   the role the proposed `kernel_bakeoff_workflow.js` would have — freeze → discover → fan-out → report).
> - The unchanged single-language pipeline (this file's "worker") was renamed
>   `kernel_workflow.js` → **`kernel_lane.js`** (content unchanged besides `meta.name`).
> - `oracle_freezer.md` lives at `kernel_workflow/roles/` (not a new `kernel_bakeoff_workflow/` dir).
> - **Nesting stays 0→1 for the bake-off** because the dispatcher fans out lanes directly
>   (`workflow(kernel_lane.js)`), and every existing nested caller (`e2e_workflow`,
>   `kernel_workflow_bmk`) was repointed to call the **worker `kernel_lane.js` directly** — so nobody
>   routes through the dispatcher and hits 3 levels. This is why the design's separate-top-level-entry
>   requirement (§1.1) is still satisfied without an `_nested` guard.
> - The one additive `op_benchmarker.md` edit (`ENABLE_SERVING_TUNE`, §4.2) shipped as designed.
>
> The rest of this document (the invariants §3/§3.5, the `oracle_freezer` contract §4.1, the
> reference-in-place strategy §4.2, the edge cases §8) describes the shipped behavior accurately; only
> the packaging (one entry + `mode` vs. a separate script) changed.

---

## 1. Decision: a separate entry-point workflow, not a new `mode`

Two ways to add the capability were considered:

- **A — `mode=bakeoff` inside `kernel_workflow.js`.** Rejected: bloats the single-language script, needs
  an `_nested` runtime guard to prevent 3-level nesting, and mixes orchestration into the worker.
- **B — a separate `kernel_bakeoff_workflow.js` (chosen).** A small orchestrator that calls the
  **unchanged** single-language `kernel_workflow` once per language and picks the winner.

Design B is chosen because it mirrors the architecture that already works: `e2e_workflow` is *"an
orchestrator layer wrapping the unchanged `kernel_workflow`"* (`e2e_workflow.js:3-4`, `:955-963`).
`kernel_bakeoff_workflow` is a **smaller sibling** of that same shape — cross-language bake-off only, no
serving / no config sweep / no e2e gate.

### 1.1 Why this satisfies the one-level-nesting rule for free

The Workflow runtime nests `workflow()` **one level only**. Design B never risks violating it:

| Call stack | Nesting | Legal? |
|---|---|---|
| `kernel_bakeoff_workflow` → per-language `kernel_workflow` | 0 → 1 | ✅ |
| `e2e_workflow` → `kernel_workflow` (single language, today) | 0 → 1 | ✅ unchanged |
| `e2e_workflow` → `kernel_bakeoff_workflow` | **never happens** | e2e has its own head bake-off; it does not call the bakeoff entry |

Because e2e never calls the bakeoff entry, no `_nested` guard is needed at all (unlike the rejected
option A). The two orchestrators are simply peers that both drive the same single-language worker.

---

## 2. What is reused, re-implemented, and newly written

"Reuse e2e's bake-off logic" is real but **not a literal import** — e2e's bake-off is three parts with
different reuse profiles:

| e2e piece | Reuse? | How `kernel_bakeoff_workflow` handles it |
|---|---|---|
| `kernel_extractor.extract_op` (freeze oracle + baseline) | ❌ **serving-coupled** — captures shapes/oracle from a *running server* via monkeypatch overlays (`kernel_extractor.md:98-114`) | **New, serving-agnostic** role `oracle_freezer` (§4.1) freezes directly from the kernel dir |
| `op_benchmarker` role + `scripts/op_bench.py` + `scripts/harness_lib.py` (per-language discover / measure / `author_plan`) | ✅ operate on an op-task-dir, server-independent | **Reference in place** at `e2e_workflow/` by path — no move, no copy (§4.2). Tier-B serving tune gated off. |
| lane fan-out orchestration (build lanes → run → harvest → pick best) | ⚠️ inline JS, entangled with deep-mode + e2e gate (`e2e_workflow.js:1152-1300`) | **Re-implement the pattern** compactly in the new script (§4.3) — ~the `runBakeoff()` sketch |

Net: **1 new orchestrator script + 1 new role**, everything else referenced in place.

---

## 3. The anti-cheating invariant (carried over verbatim)

> **The speedup denominator is ALWAYS the frozen original kernel** (`baseline_src/` /
> `meta.baseline_callable`), never an authored same-language seed.

This is the exact fake-win the e2e harness exists to prevent (`op_benchmarker.md`, red block: optimized-
HIP vs naive-HIP = fake 15.7× isolated / ~0% real). In bakeoff:

- `oracle_freezer` freezes the input kernel into `baseline_src/` **once**, up front.
- Every lane — the input-language `optimize` lane and the Triton/FlyDSL `author` lanes alike — is scored
  by the **same immutable `unittest.py`** against that one frozen baseline.
- `pickBest` therefore compares `baseline_ms / lane_ms` across lanes on a **common denominator**, so the
  winner is genuinely the fastest (not "whoever beat their own naive seed").

---

## 3.5 Where the oracle is created — upstream, before any lane

Generating the oracle (frozen `baseline_src/` + immutable `unittest.py` + `meta.json`) is **never the
worker's job** — it belongs to the **orchestrator layer**, and it happens **before** any per-language
`kernel_workflow` lane starts. Both orchestrators follow the identical shape:

| | Orchestrator | Oracle-creation step | Source of the truth | Then per lane |
|---|---|---|---|---|
| **e2e** | `e2e_workflow` | `kernel_extractor.extract_op` (`e2e_workflow.js:1153`) | live sglang/vllm **server** (monkeypatch capture) | `workflow(kernel_workflow, {kernel_path: ext.task_dir, mode, target_language})` (`:1468`, `:1596`) |
| **standalone** | `kernel_bakeoff_workflow` | `oracle_freezer` (§4.1) | the **input kernel**, frozen into `baseline_src/` and re-run live per parity draw (no server, no stored golden) | `workflow(kernel_workflow, {kernel_path: oracle.task_dir, mode, target_language})` (§4.3) |

Both record **no tensors at all**. `kernel_extractor` captures shapes/dtypes/regimes off a live server
and synthesizes operands from that spec; `oracle_freezer` synthesizes from recorded seeds. Storing a
golden in either lane would be redundant (the frozen baseline is already a runnable reference and must
exist anyway as the timing denominator), would cost hundreds of MB–GB that every lane and every engineer
workspace then tar-copies, and would add a failure mode of its own (a recorded golden is only valid while
the operands reproduce bit-for-bit, so a box or torch-build change becomes a hard failure).
The `reference_io.pt` `kernel_extractor` used to ship for operands thought unsynthesizable (MoE routing
tables, paged-KV metadata) is retired. The residual cost is PERFORMANCE realism only: MoE routing skew
is synthesized from a prior (`h.skewed_topk_ids`) and flagged in `notes` — correctness is unaffected,
since both legs get identical routing and the baseline leg is the reference.

Sequence in both cases: **create oracle → `op_benchmarker` bake-off (on `task_dir`) → fan out lanes.**

When `kernel_workflow` runs as a lane, the op task dir it receives **already contains the immutable
oracle**, so its `benchmark_engineer` **detects and reuses it verbatim** — it does **not** generate a new
one (`benchmark_engineer.md:54-59`: *"if the workspace holds an IMMUTABLE `unittest.py` + `meta.json` …
THAT is the runner — reuse it verbatim … do NOT write a new harness"*).

`oracle_freezer` is therefore precisely the **standalone counterpart of `extract_op`**: same op-task-dir
contract, different input (kernel dir vs live server). Both build the oracle up front; the worker only
ever consumes it.

> **Corollary — `kernel_workflow`'s own baseline generation stays (do NOT delete it).**
> `benchmark_engineer`'s harness/baseline building is the required path for a **standalone
> `mode=optimize`** run on a plain kernel dir, where **no** upstream oracle exists — that is the original,
> primary use of `kernel_workflow` (README: "many users only use the kernel workflow"). The logic is
> **already conditional**: it *generates* only when no runner/oracle is present, and *reuses + just
> re-measures* when an immutable oracle is present (the e2e/bakeoff lane path). So it is neither dead nor
> duplicated by `oracle_freezer` — in a bakeoff lane it simply takes the reuse branch. Deleting it would
> break every standalone optimize run.

---

## 4. Components

### 4.1 NEW — `kernel_bakeoff_workflow/roles/oracle_freezer.md` (serving-agnostic)

Turns an already-runnable kernel dir into a standard **op task dir**, with **no server**.

- **Inputs:** `KERNEL_PATH` (input kernel dir), `EVAL_DIR`, `GPU_ID`, optional `OP_SPEC` hints, optional
  `WORKLOAD_SPEC_PATH`.
- **Does:**
  1. Find or synthesize a minimal runnable driver for the input kernel. If the dir ships a driver/test,
     use it; else reuse `kernel_workflow`'s `benchmark_engineer` COMMANDMENT-building capability to
     produce setup/correctness/bench commands.
  2. Record the case manifest into `meta.cases[]` — `{sig, dims…, seed, regime}` per case, over
     small/medium/large (or the workload cases if `WORKLOAD_SPEC_PATH` is given). Shapes are pinned so
     rounds stay comparable; values are regenerated from the seed at run time. **Nothing is executed for
     the record and no tensors are written to disk.**
     > ⚠️ **The correctness truth MUST be the input kernel's own behavior.** Do **not** let this fall
     > through to `benchmark_engineer`'s naive-PyTorch correctness fallback (`benchmark_engineer.md:99-103`):
     > the whole point is that the Triton/FlyDSL ports are checked for parity against the **real HIP
     > kernel**, and the speedup denominator is the **real HIP kernel**. A naive reference here would
     > validate ports against the wrong behavior and bench against the wrong baseline.
     > `benchmark_engineer` is reused only for the *driver plumbing*, never as the correctness source.
  3. Freeze the input source into `baseline_src/`; set `meta.baseline_callable` (`baseline_frozen=true`);
     compute the integrity anchors `baseline_src_sha256` / `harness_lib_sha256` / `unittest_sha256`
     (no golden-tensor hash is computed — no such file exists in any lane).
  4. Vendor `harness_lib.py` into the task dir; write immutable `unittest.py` + `meta.json`; detect
     `live_backend` (the input language, e.g. `hip`).
- **Returns** (`FREEZE_SCHEMA`, a subset of e2e's `EXTRACT_OP_SCHEMA` at `e2e_workflow.js:440-448`):

```json
{
  "op_kind": "gemm|attn|elementwise|other",
  "task_dir": "<abs op task dir>",
  "live_backend": "hip",
  "candidate_backends": ["hip","triton","flydsl"],
  "baseline_frozen": true,
  "baseline_callable": "module:attr",
  "op_spec": { "op_kind": "...", "shapes": {}, "dtype": "bf16", "regime": "both" },
  "workload_path": "",
  "smoke": "pass",
  "notes": "..."
}
```
`FREEZE_SCHEMA` carries no golden-tensor hash: the integrity anchors are `baseline_src_sha256` /
`harness_lib_sha256` / `unittest_sha256`.

> It produces the **same op-task-dir contract** as e2e's `kernel_extractor`, but from a kernel dir
> instead of a live server. They stay separate on purpose; only the output contract is shared. (A future
> refactor could factor a common freeze core — out of scope for v1.)

### 4.2 REFERENCE-IN-PLACE — e2e's `op_benchmarker` + bench scripts (no move, no copy)

To keep repo churn near-zero, `kernel_bakeoff_workflow` does **not** move or duplicate these. It points
at e2e's existing copies by path (e2e_workflow is always a sibling in the same repo):

```js
// kernel_bakeoff_workflow.js
const E2E_WF_DIR = String(A.e2e_workflow_dir ||
  (WORKFLOW_DIR.replace(/\/[^/]*$/, '') + '/e2e_workflow')).replace(/\/+$/, '');
// op_benchmarker role read from ${E2E_WF_DIR}/roles/op_benchmarker.md
// SKILL_DIR passed as E2E_WF_DIR so its knowledge/ + scripts/op_bench.py resolve
```

The bakeoff's `roleAgent` reads `op_benchmarker` from `E2E_WF_DIR/roles/`; its own roles (`oracle_freezer`,
`tech_lead` for the final table) from `WORKFLOW_DIR/roles/`.

**The single additive edit to an existing file** — `e2e_workflow/roles/op_benchmarker.md`: add an
`ENABLE_SERVING_TUNE` input (default treated as *on* when server/`MODEL_PATH` inputs are present, so
e2e's behavior is byte-identical). When it is false / no server inputs are supplied (the bakeoff case),
the role **skips Tier-B** (aiter-DB capture, `AITER_TUNE_GEMM`, `--attention-backend`, the CK/fp8 serving
playbooks — all need a live server) and does only **Tier-A discover + `author_plan`**. This is purely
additive: existing e2e calls that don't set it keep full behavior.

> Deferred (not v1): physically moving `op_benchmarker` / `op_bench.py` / `harness_lib.py` into a
> `shared/` dir. Reference-in-place gives single-source reuse *today* with zero moves; the move is a
> later cleanup if the dependency direction (bakeoff → e2e) becomes undesirable.

### 4.3 NEW — `kernel_bakeoff_workflow/kernel_bakeoff_workflow.js` (the orchestrator)

Self-contained (own `roleAgent` / `makeSem` / args, matching the convention where `kernel_workflow` and
`e2e_workflow` each carry their own — copy `makeSem` from `e2e_workflow.js:930-949`).

```js
export const meta = {
  name: 'kernel-bakeoff-workflow',
  description: 'Standalone multi-backend bake-off: given one kernel, explore several backend ' +
    'languages (HIP/Triton/FlyDSL/CK) in parallel — reuse an existing impl or author a fresh one — ' +
    'and pick the fastest verified result against the ONE frozen original baseline. Wraps the ' +
    'unchanged single-language kernel_workflow (one lane per language).',
  whenToUse: 'Optimize one kernel by trying multiple backend languages and keeping the fastest. ' +
    'Pass args.kernel_path (required), args.backends (optional), args.gpu_ids.',
  phases: [
    { title: 'Freeze',   detail: 'oracle_freezer: freeze input kernel -> immutable oracle + baseline_src/ (the denominator)' },
    { title: 'Discover', detail: 'op_benchmarker (Tier-A only): per-language existing-impl probe + measure -> author_plan, best_known_ms' },
    { title: 'Bakeoff',  detail: 'one unchanged kernel_workflow per language (optimize if impl exists, else author), parallel over the GPU pool' },
    { title: 'Report',   detail: 'rank lanes on the SAME frozen baseline; emit table + winner (+ optional apply_to_original)' },
  ],
};

const A = args || {};
if (!A.kernel_path) throw new Error('args.kernel_path is required');
const WORKFLOW_DIR = String(A.workflow_dir || '').replace(/\/+$/, '');
if (!WORKFLOW_DIR) throw new Error('args.workflow_dir is required (dirname of this script)');
const KERNEL_WF_SCRIPT = String(A.kernel_workflow_script ||
  (WORKFLOW_DIR.replace(/\/[^/]*$/, '') + '/kernel_workflow/kernel_workflow.js'));
const E2E_WF_DIR = String(A.e2e_workflow_dir ||
  (WORKFLOW_DIR.replace(/\/[^/]*$/, '') + '/e2e_workflow')).replace(/\/+$/, '');
const EXP_ROOT = String(A.exp_root || (WORKFLOW_DIR.replace(/\/[^/]*$/, '') + '/exp')).replace(/\/+$/, '');
const BACKENDS = (Array.isArray(A.backends) ? A.backends
  : (typeof A.backends === 'string' ? A.backends.split(',') : []))
  .map(s => String(s).trim().toLowerCase()).filter(Boolean);      // empty => auto-discover
const BUDGET = parseInt(A.budget != null ? A.budget : 6, 10);
const GPU_LIST = String(A.gpu_ids != null ? A.gpu_ids : '0').split(',').map(s => s.trim()).filter(Boolean);
const TASK = A.task || '';
const APPLY_TO_ORIGINAL = String(A.apply_to_original != null ? A.apply_to_original : 'false');
const primSpeedup = (o) => {  // same semantics as kernel_workflow.js:96-105
  if (!o) return 0;
  const g = o.final_geomean != null ? o.final_geomean
          : (o.verified_geomean != null ? o.verified_geomean : o.speedup_geomean);
  return Number.isFinite(g) ? g : 0;
};

// --- EVAL_DIR (director-style setup omitted for brevity) ---
const EVAL_DIR = /* director builds an isolated run dir under EXP_ROOT */ '';

// 1) FREEZE — establish the ONE immutable oracle + frozen baseline
phase('Freeze');
const oracle = await agentT(
  roleAgent('oracle_freezer', 'freeze', 'Freeze the input kernel into an immutable op task dir.',
    { KERNEL_PATH: A.kernel_path, EVAL_DIR, GPU_ID: GPU_LIST[0] }),
  { phase: 'Freeze', label: 'freeze', schema: FREEZE_SCHEMA });
if (!oracle || oracle.smoke !== 'pass' || !oracle.task_dir || !oracle.baseline_frozen) {
  log(`freeze FAILED (${oracle ? oracle.notes || oracle.smoke : 'no result'}); aborting.`);
  return { validation_status: 'freeze_failed', winner: null };
}

// 2) DISCOVER — per-language existing-impl probe + measure + author_plan (Tier-A only)
phase('Discover');
const bake = await agentT(
  roleAgentFrom(E2E_WF_DIR, 'op_benchmarker', 'bakeoff', 'DISCOVER impls; DECIDE author_plan. No serving tune.',
    { EVAL_DIR, OP_TASK_DIR: oracle.task_dir, OP_KIND: oracle.op_kind,
      CANDIDATE_BACKENDS: (BACKENDS.length ? BACKENDS : oracle.candidate_backends),
      GPU_ID: GPU_LIST[0], ENABLE_FP8: false, ENABLE_SERVING_TUNE: false, SKILL_DIR: E2E_WF_DIR }),
  { phase: 'Discover', label: 'discover', schema: OPBENCH_SCHEMA });

// resolve lanes: input language -> optimize; else rewrite->optimize / missing->author
const planByLang = Object.fromEntries((bake.author_plan || [])
  .map(a => [String(a.language).toLowerCase(), a.route === 'rewrite' ? 'optimize' : 'author']));
const liveLang = (oracle.live_backend || '').toLowerCase();
const wanted = BACKENDS.length ? BACKENDS
  : [...new Set([liveLang, ...(bake.author_plan || []).map(a => String(a.language).toLowerCase())])].filter(Boolean);
const lanes = wanted.map(lang => ({ lang, key: lang,
  mode: lang === liveLang ? 'optimize' : (planByLang[lang] || 'author') }));
log(`baseline ${bake.best_known_ms || '?'} ms; lanes = ${lanes.map(l => `${l.lang}:${l.mode}`).join(', ')}`);

// 3) BAKEOFF — one UNCHANGED kernel_workflow per language, parallel over the GPU pool (0->1 nesting)
phase('Bakeoff');
const sem = makeSem(GPU_LIST);
const results = await Promise.all(lanes.map(l => sem.with(1, async ([gpu]) => {
  try {
    const r = await workflow({ scriptPath: KERNEL_WF_SCRIPT }, {
      kernel_path: oracle.task_dir, workflow_dir: KERNEL_WF_SCRIPT.replace(/\/[^/]*$/, ''),
      mode: l.mode, target_language: l.lang, op_spec: oracle.op_spec,
      workload_spec_path: oracle.workload_path || '', budget: BUDGET, gpu_ids: gpu, task: TASK,
      exp_root: `${EVAL_DIR}/bakeoff/${l.key}`,
    });
    return { lane: l, r, speedup: primSpeedup(r) };
  } catch (e) { log(`lane ${l.key} failed: ${e && e.message}`); return { lane: l, r: null, speedup: 0 }; }
})));

// 4) REPORT — all lanes share the SAME frozen baseline => directly comparable
phase('Report');
const ranked = results.filter(x => x.r && x.speedup > 0).sort((a, b) => b.speedup - a.speedup);
const winner = ranked[0] || null;
// tech_lead role writes the per-lane comparison table + rationale; APPLY_TO_ORIGINAL applies the winner
return {
  task_dir: oracle.task_dir, baseline_ms: bake.best_known_ms,
  lanes: results.map(x => ({ lang: x.lane.lang, mode: x.lane.mode, speedup: x.speedup,
    eval_dir: x.r && x.r.eval_dir, patch: x.r && x.r.final_patch })),
  winner: winner && { lang: winner.lane.lang, mode: winner.lane.mode, speedup: winner.speedup,
    eval_dir: winner.r.eval_dir, patch: winner.r.final_patch },
  validation_status: winner ? 'ok' : 'no_winner',
};
```

Helpers the new script carries (small, dependency-free):
- `roleAgent(role, …)` — reads `${WORKFLOW_DIR}/roles/${role}.md` (its own roles).
- `roleAgentFrom(dir, role, …)` — same but reads `${dir}/roles/${role}.md` (for e2e's `op_benchmarker`),
  passing `SKILL_DIR=dir` so that role's knowledge/scripts resolve.
- `makeSem(GPU_LIST)` — the GPU semaphore lifted from `e2e_workflow.js:930-949` (1 GPU/lane; `N==1`
  serializes).
- `agentT`, `phase`, `log`, `cfg`, schemas (`FREEZE_SCHEMA`, `OPBENCH_SCHEMA`).

### 4.4 NEW — `kernel_bakeoff_workflow/README.md`

Short: what it does, the `backends` arg, the frozen-baseline invariant, the invocation example (§6), and
a one-line note that it wraps the unchanged `kernel_workflow`.

---

## 5. Change footprint (minimal)

| File | Change |
|---|---|
| `kernel_workflow/kernel_workflow.js` | **none** |
| `e2e_workflow/e2e_workflow.js` | **none** |
| `e2e_workflow/roles/op_benchmarker.md` | **1 additive input** (`ENABLE_SERVING_TUNE`, default preserves e2e behavior) |
| `kernel_bakeoff_workflow/kernel_bakeoff_workflow.js` | **new** |
| `kernel_bakeoff_workflow/roles/oracle_freezer.md` | **new** |
| `kernel_bakeoff_workflow/README.md` | **new** |

No file moves. No copies of shared scripts. e2e's HeadKernel behavior is byte-identical.

---

## 6. Invocation (standalone)

```js
Workflow({
  scriptPath: "/abs/GEAK/kernel_bakeoff_workflow/kernel_bakeoff_workflow.js",
  args: {
    kernel_path: "/abs/path/to/my_hip_kernel",       // REQUIRED — source kernel dir (e.g. HIP)
    workflow_dir: "/abs/GEAK/kernel_bakeoff_workflow",// REQUIRED — dirname of this script
    backends: ["hip", "triton", "flydsl"],            // optional; default = auto-discover available
    budget: 6,                                         // per-lane optimize budget
    gpu_ids: "0,1,2",                                  // 1 GPU/lane; parallel when >1, serial when 1
    // apply_to_original: "true"                       // optional: write the winning patch back
  }
})
```

`kernel_workflow_script` / `e2e_workflow_dir` are auto-derived as siblings of `workflow_dir` and only
need to be passed if the layout differs.

Natural-language front door: `bake off /xxx/my_hip_kernel across hip triton flydsl, pick the fastest`.

Output under `<exp_root>/…/`:
- `bakeoff/<lang>/…` — each lane's full single-language `kernel_workflow` run (patch, timing, report).
- top-level report: per-lane speedup table (all vs the same frozen baseline) + the winner.

---

## 7. Non-conflict guarantee (summary)

| Concern | Resolution |
|---|---|
| Double `workflow()` nesting | bakeoff is a separate top-level entry; e2e never calls it → max depth stays 0→1. No `_nested` guard needed. |
| e2e regressions | e2e JS untouched; the one `op_benchmarker` edit is additive and default-preserving. |
| Role/script drift | `op_benchmarker` / `op_bench.py` referenced **in place** at e2e (single source), not copied. |
| Serving vs standalone extraction | Separate roles (`kernel_extractor` serving / `oracle_freezer` kernel-dir); shared output contract only. |
| Serving-only tuning leaking in | Tier-B gated behind `ENABLE_SERVING_TUNE` (off for bakeoff). |

---

## 8. Edge cases & risks

1. **Freeze needs a runnable input.** The frozen `baseline_src/` IS the correctness truth source (re-run
   live per draw), so `oracle_freezer` can only produce a usable oracle if the kernel runs; if the dir has
   no harness it reuses `benchmark_engineer` to synthesize a minimal driver; if even that fails →
   `validation_status: freeze_failed`, abort (no comparable baseline possible).
1a. **The gap live-parity does NOT cover.** `check_random_vs_baseline` compares one draw at a time and
   never holds two candidate outputs live at once, so a candidate returning the SAME module-level
   persistent buffer across separate calls passes it. Dropping the recorded golden also dropped
   `check_correct_multi`, which was the only caller of `assert_independent_outputs`
   (`harness_lib.py:485`). That check needs **no golden** — just two arg sets — so the generated
   `unittest.py` must call it explicitly and fold the result into `all_ok`, passing two arg sets at the
   **SAME dims with different values** (two different shapes would give a shape-keyed static buffer two
   cache slots and let it pass). Measured on the fused-MoE oracle: a shape-keyed `static_out` candidate
   scores `max_rel_err = 0.0` on all 15 parity draws and is caught **only** by this gate.
2. **Backend availability.** A requested language absent on the image (e.g. `aiter.ops.flydsl`
   `ModuleNotFoundError`) is dropped with an advisory, not a hard fail (mirror `op_benchmarker.md`'s
   `backend_absent[]`). The input's own language is always available.
3. **Single-GPU runs.** `gpu_ids:"0"` serializes lanes; correct, just slower — document it.
4. **Cost multiplier.** N languages ≈ N× a single `kernel_workflow` run (each with `budget`). `log()` the
   multiplier so it's not a surprise.
5. **Dependency direction.** bakeoff → e2e (reads e2e's `op_benchmarker`). Acceptable for v1 (same repo,
   always siblings); if it ever chafes, do the deferred move to `shared/`.
6. **Non-GEMM/attn kernels.** `op_benchmarker` is GEMM/attn-centric; for a generic kernel it still does
   Tier-A discover + author_plan (triton always a valid author target) and naturally skips the GEMM
   tuning rungs when `ENABLE_SERVING_TUNE=false`.

---

## 9. Implementation order

1. Add `ENABLE_SERVING_TUNE` to `op_benchmarker.md` (additive; verify e2e unchanged with the flag
   present-but-on).
2. Write `oracle_freezer.md`; test the freeze contract on one HIP kernel (valid op task dir +
   `unittest.py` smoke passes, `baseline_frozen=true`).
3. Write `kernel_bakeoff_workflow.js` (director setup + the 4 phases above + helpers + schemas + README).
4. End-to-end: standalone bakeoff on a HIP kernel across `[hip, triton, flydsl]` on a multi-GPU box;
   confirm all lanes share the frozen baseline and the winner is the true fastest.
5. Regression: run one e2e HeadKernel campaign; confirm byte-identical behavior (the `op_benchmarker`
   edit didn't change the serving path).

---

## 10. Open questions

- **Winner application default** — report-only (recommended) vs auto `apply_to_original`. Default off.
- **Deferred `shared/` move** — keep reference-in-place (v1) or physically relocate `op_benchmarker` +
  bench scripts to a neutral `shared/` later, flipping the dependency direction to `bakeoff → shared ←
  e2e`. Not needed for v1.
- **Doc home** — this file currently sits under `kernel_workflow/docs/`; move to
  `kernel_bakeoff_workflow/docs/` when the new dir is created.
