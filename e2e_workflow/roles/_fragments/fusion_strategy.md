# Fusion strategy

Read by the System Architect (`PHASE=strategize` step 1e, `PHASE=plan_milestone` step 6) when KernelFusion
is switched on (`kernel_fusion=true`) and this run's fusion profile found regions. It ADDS fusion regions to
the kernel track; single kernels are routed exactly as before.

## What you are given
- `FUSION_CANDIDATES`: the fusion profile's regions (`roles/fusion_analyst.md`), each with `id`, `rank`,
  `level`, `region`, `kernels` (the device kernels it launches today, in order), `phases`, `m_range`,
  `seam.target_callable` (ONE Python callable whose body covers the whole region), `region_pct_device_time`,
  `est_gain_pct_expected`, `difficulty`, `bitwise_exact`, `exclusive_with`, `fallback`.
- `FUSION_RECOMMENDED`: the analyst's non-conflicting picks, best first.
- `FUSION_TOPK_JSON`, `FUSION_REPORT`: the full analysis (each region's fused design, the trace digest's
  numbers, the reasoning). Read them before you decide.
- `IMPLEMENTABLE_LEVELS`, `UNSUPPORTED_PATTERNS`, and `OPERATOR_CONSTRAINTS` when present (it binds you).

## Feasibility Review: which regions go on
You decide which fusion opportunities of the analysis go on to the next step: writing a fused kernel and a
region unit test for each. Judge each candidate yourself from the analysis (its numbers, the report's
reasoning, the trace digest, the source) -- no limit on how many you pick and no minimum gain beyond the
kernel track's own bars; pick what you judge worth the effort and say why in `strategy.md`, and say why for
each one you leave out.
The one hard line: a candidate is authored by rebinding ONE Python callable that covers its whole region
(seam.target_callable) and is the one the live server actually runs (check the trace's launch frames: a
model can have several implementations of the same layer); one without such a seam cannot be authored.
Implementable scope: levels `IMPLEMENTABLE_LEVELS`, except `UNSUPPORTED_PATTERNS`; a candidate outside it
is not authored (a level alone is no reason to skip a candidate inside it).

## Routing a region
A region you pick is ONE entry of `kernel_candidates` (never `head_candidates`): the usual shape, plus its
`fusion_id`:
```
{"short_name": "kf_<id>", "fusion_id": "<id>", "classification": "fusion", "pct_gpu_time": <the region's share>,
 "regime": "prefill|decode|both", "candidate_backends": ["triton"], "amdahl_priority": <pct x plausible speedup>,
 "extract_hint": "<seam.target_callable> and why it is the live one"}
```
- `fusion_id` is the region's `id`, exactly: by it the workflow hands the Kernel Extractor and the kernel
  squad the region's facts (kernels, seam, tensors, fused design), so you do not repeat them. An entry whose
  `fusion_id` names no region of `FUSION_CANDIDATES` is an ordinary kernel candidate.
- `pct_gpu_time` is the region's share of GPU time in the latest `PROFILE_TOPN` (the sum of its kernels'
  rows), not the fusion profile's `region_pct_device_time` once the profile has moved; the kernel track's
  bars (`MILESTONE_MIN_PCT`, Amdahl) apply to that share like to any kernel. Drop a region whose kernels are
  gone from the latest profile.
- Do not also route, in the same plan, a single-kernel `kernel_candidates` entry for one of the region's
  `kernels`: the fused kernel replaces them, and two overlays rebinding the same code cannot both go in.
- A region on a head op (one whose `kernels` include the kernel of a head you route in `head_candidates`,
  typically an L2 region: GEMM plus its bias/activation/quant, attention plus its local pre/post-processing)
  is routed here all the same, and that head op stays in `head_candidates` exactly as it would without the
  region. When HeadKernel reaches the head's A/B, the workflow moves the region into the head's slot as one
  more candidate beside the head's own, all measured against the same reference; the best measured one
  lands. A head HeadKernel never reaches leaves its region to the Milestone loop.
- HISTORY names a region tried in a head's slot `<head short_name>:kf_<id>`; it counts as tried, like a
  region the Milestone loop ran.
- Kernels that rebind the same callable or whose regions overlap (exclusive_with) cannot go in together --
  route at most one of them at a time. When it does not land, its `fallback` (or another of them) is still
  eligible in a later plan.
- A fused kernel reorders floating-point work, so its A/B is held to task accuracy (the workflow adds the
  gate), not byte parity. Nothing else about the kernel track changes: dispatch order, budget, HISTORY.
