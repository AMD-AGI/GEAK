---
key: live-kernel + dispatch-seam resolution · any gfx · any framework · any backend
type: method
confidence: ★★★
effect: the kernel you optimize is the kernel the GPU actually ran, bound to the seam the live path actually calls — instead of a plausible-looking guess that engages 0 times
confirms: 1
last_seen: 2026-06-29
---
# Resolve the REAL live kernel + its dispatch seam (never guess the launcher from a kernel name)

**Why this exists.** A head GEMM authored a real isolated 1.92× kernel on Qwen3-4B/vLLM, then engaged
**0 times** e2e and was gate-rejected. Root cause was 100% seam-resolution, not the apply mechanism:
the extractor returned a *guessed* launcher (`aiter.tuned_gemm:gemm_a16w16`) from the kernel's trace
name. But on that box the live dense GEMM (a) imports a **different module** if it ever went Triton
(`aiter.ops.triton.gemm_a16w16`), (b) that branch is gated by a **hardcoded shape whitelist that the
model didn't match**, and (c) the shapes actually dispatched to **closed hipBLASLt** (the `Cijk_*`
Tensile kernels in the trace). The authored kernel bound a symbol nothing calls → 0 engagement.

**The rule: the seam is whatever the LIVE dispatch actually calls for the CAPTURED shapes under the
LIVE env — discovered by observation + reading the installed source, never assumed from the op type,
the model, the gfx, or a `launcher_hint`.** No backend (aiter/triton/ck/…) is privileged. No module
name is hardcoded. Do this BEFORE authoring, so a closed-library op is routed to tuning instead of
wasting an author cycle.

## The two ends you must connect
- **GPU end (ground truth)**: the profiler already recorded the exact kernel that ran. That is
  authoritative — the kernel that ran is whatever the trace says, not what we expect.
- **Host/Python end (the seam)**: the callable whose body decided to launch that GPU kernel. This is
  what the Integrator rebinds (`.py` overlay) or patches (native recompile). Resolve THIS, don't guess it.

## Step 1 — Classify the GPU kernel's backend FAMILY from the trace (pattern, not a fixed list)
Take the kernel name from `profile_topN.json`; if it's a mangled C++ symbol, `c++filt` it; note which
`.so`/module rocprof attributes it to (the profiler gives the owning module path per kernel — use it).
Map signature → owning backend by what the name/.so reveals, e.g.:
- `Cijk_*`, `Cgemm_*`, Tensile-mangled, or attributed to `librocblas`/`libhipblaslt` → **rocBLAS/hipBLASLt
  (Tensile) = CLOSED**. No shipped source, no python seam → not authorable.
- `*wvSplitK*`, `LLMM*`, other names in an `aiter`/`vllm` `.so` → compiled C++ op (editable only if the
  source ships in a rebuildable tree — see [[native-apply-back]]).
- `triton_*`, `*_fwd_kernel*`, inductor `triton_poi_/red_fused_*` → **Triton (editable `.py`)**.
- `ck_*`, composable_kernel device names → **CK** (editable C++ if source shipped).
This is reasoning from the symbol, not memorizing — any new name: demangle it, find its `.so`, decide
closed-lib vs editable-source by whether source ships up-tree.

## Step 2 — Map that GPU kernel back to its launching Python frame (the seam), by evidence
Prefer (a); fall back to (b). Plug in REAL values — never evaluate a branch in your head from defaults.
- **(a) Profiler call-stack / API correlation.** If the trace carries host stacks (torch profiler
  `with_stack=True`, or rocprofv3 HIP-API + correlation IDs), read the launching frame directly — that
  function IS the seam. Cheapest, most reliable when available.
- **(b) Source walk from the op entry.** Open the **live installed** source at the op family's entry
  (the registered custom op / `Linear.forward` / the attention-backend forward — found by `import pkg;
  os.path.dirname` then grep, NOT from memory) and follow the ACTUAL branch tree, evaluating each
  predicate against the **server's live env vars/flags** and the **captured shapes**. The branch that
  fires for those shapes terminates at the real leaf callable = the seam. Read the code; substitute the
  live values; do not assume which branch wins.

## Step 3 — Confirm the seam with a TEMPORARY throwaway probe (allowed, encouraged)
A disposable one-shot python probe is the cheapest proof — it is a verification tool, NOT part of the
integration. Put it in a fresh `_seam_probe/` overlay, delete after:
- `setattr` a passthrough wrapper on the resolved `module:attr` that prints a one-time
  `[seam-probe] <module:attr> HIT shape=<...>` to stderr, then calls the original. (For a leaf imported
  *inside* a function via `from M import f` each call, patch `M.f` — the per-call import re-fetches it.)
- Run the SAME workload for a few seconds (REPEATS=0 warmup window) and grep the server log:
  - **≥1 HIT per worker** → seam confirmed; record the exact `module:attr`. Proceed to author.
  - **0 HITs but a concurrent re-profile still shows the target GPU kernel** → wrong seam. Either you
    patched the wrong module (Step 1's family was off) or the live path bypasses python entirely
    (closed lib) → go to Step 4 "closed". Re-walk Step 2; do NOT author against an unconfirmed seam.
The probe is generic: it does not name any backend, only the `module:attr` you resolved this run.

## Step 4 — Editability is decided by the RESOLVED LEAF, not the op kind
- Leaf = **closed library** (Tensile/hipBLASLt/rocBLAS, or an opaque `.so` with no shipped source) →
  `editable=false` → route to **Config Tuner** (per-shape tuning DB, or a flag/backend swap), do NOT
  author. This is the case that wasted the 1.92× cycle — catch it here, pre-author.
- Leaf = **editable Triton `.py`** or **compiled source in a rebuildable tree** → `editable=true`;
  record the EXACT confirmed `module:attr` (+ source file for native). Author.
- Leaf is closed NOW but an **editable alternative exists behind a flag** (an env that re-routes the
  SAME op to a Triton/CK path) → this is a **CONFIG lever first**: record the flag to flip AND the seam
  it then exposes. Flip → re-confirm via Step 3 against the new live path → then author. (Flipping a
  flag that re-routes to editable code is the general way to "reach" an otherwise-closed op — but it
  only counts if the flag's predicate actually accepts the model's shapes; verify, don't assume.)

## Step 5 — When per-shape branches diverge, bind the nearest common chokepoint
Decode (skinny-M) and prefill (large-M) can take DIFFERENT branches → DIFFERENT leaves → maybe one
editable and one closed. If you bind a single leaf you cover only part of the shape space (and may
regress the other). Prefer the **nearest common chokepoint every shape passes through** (the op entry /
registered custom op / `Linear.forward`), patched to dispatch to the authored kernel for the target
shapes and **fall through to the original for the rest**. Confirm with Step 3 that the chokepoint sees
both regimes' shapes.

## Step 6 — Engagement is proven by re-OBSERVATION, not an e2e wiggle
This is the post-author gate (see [[method-verify-engagement]]). After integrating, prove the OLD GPU
kernel disappeared and the NEW one appears in a re-profile (or the candidate's own one-shot banner fires
INSIDE the cudagraph-captured region — see [[method-cudagraph-safe-integration]]). A throughput delta
with no kernel-swap evidence is noise — reject it (that is exactly what caught the Qwen3-4B false win).

## Anti-hardcode checklist (the failure mode this card prevents)
- [ ] Did NOT assume a backend from the model/gfx — classified from THIS trace's kernel names + `.so`.
- [ ] Did NOT trust a `launcher_hint`/learned-card seam blindly — confirmed with a live probe or re-profile.
- [ ] Resolved the seam by reading the **installed** source + live env + captured shapes, not from memory.
- [ ] If the leaf is a closed lib → routed to tuning, did NOT author.
- [ ] Confirmed the seam sees the shapes BEFORE authoring; confirmed kernel-swap AFTER integrating.
- source: exp/.../e2e_Qwen3-4B_v1 dense-GEMM 0-engagement reject (2026-06-29); vllm rocm_unquantized_gemm dispatch tree
