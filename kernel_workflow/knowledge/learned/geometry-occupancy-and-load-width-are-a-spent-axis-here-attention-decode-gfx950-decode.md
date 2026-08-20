---
key: paged decode attention on gfx950 that is already grid-limited, co-resident and near its achievable bandwidth — the geometry/occupancy family
type: anti-pattern
confidence: ★★
effect: 0 of 7 geometry arms beat the incumbent: fatter workgroups +7.8% slower on the 32-sequence case, splitting the GQA q-head group across workgroups +8.0/+9.9/+23.1% slower on the 2-sequence case, an occupancy clamp -19.7%, a co-scheduling grid remap exactly neutral (~1.00x), kv-head-major remap 2.6-5.2% slower
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 1
toolchain: unknown
last_seen: 2026-08-12
name: geometry-occupancy-and-load-width-are-a-spent-axis-here-attention-decode-gfx950-decode
description: Closed axis: on a co-resident decode attention kernel near its BW ceiling, WG geometry / occupancy / load-width tuning went 0 for 7 arms, ~1.00x or worse.
keywords: ['decode', 'paged-attention', 'wg-geometry', 'occupancy', 'co-residency', 'anti-pattern', 'prefetch', 'isa-diff']
kernels: ['paged_attention_ll4mi_QKV_mfma16_kernel']
platforms: ['gfx950']
kernel_class: attention_decode
regime: decode
layer: learned
lifecycle: active
origin_kernels: ['paged_attention_decode']
---
# Geometry, occupancy and load width are a spent axis here
- lever: - lever: on a decode attention kernel that is already grid-limited and co-resident, consider the geometry family (workgroup fatness, waves/SIMD, grid order, LDS staggering) closed and spend the round on the per-workgroup dependent chain instead.
- apply: - apply: the cheap pre-check is whether bytes-per-workgroup x workgroup-count is invariant under the sweep; if it is, changing the count only trades ramp/tail against co-residency, and co-residency was worth a net +23% per-CU throughput against a ~10% ramp/tail.
- verify: - verify: price co-residency directly with an occupancy-clamp arm before sweeping geometry; that one arm predicts the sign of every other arm in the family.
- pitfall: - pitfall: an 'occupancy win' read off a profiler VGPR field -> the compiler's own waves/SIMD had gone down, not up -> read the compiler's occupancy and diff device ISA, since gating code away at compile time does not imply identical codegen.
- pitfall: wider and non-temporal global loads, deeper prefetch -> marginal bandwidth was already ~100% of nameplate -> the cache hint alone cost the largest case 9.5% with instruction count unchanged.
- pitfall: a bit-identical ~14% cut in VALU instructions bought nothing -> chain length pays here, issue slots do not.
- caution: - caution: also verify the arm actually rebuilt — a JIT that only checks whether the shared object exists never hashes sources, so a header-only change silently re-measures the previous binary; gate every number on built-header md5 == source md5, per instantiation.
- source: - source: kernel_workflow 16h campaign, run kernel_20_geak_0808_16h, 2026-08-12; 14 rounds, director-validated geomean 3.98x
