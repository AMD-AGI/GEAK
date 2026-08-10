---
key: quantize / cast · gfx950 · mixed
type: lever
confidence: ★★
effect: Full stack director-verified 3.75x geomean: 2.77x on the small launch-bound case, 4.76x and 4.00x on the two large memory-bound streaming cases. The two instruction-level fixes below carried the large cases: in-run the native fp8 emit alone measured 1.14x on the small case vs 3.11x / 3.29x on the large ones, and the amortized reciprocal plus pre-convert fixup added a further +21% geomean; VALU/wave 812 -> 159.
confirms_cited: 0
confirms_blind: 1
losses: 0
attempts: 5
toolchain: rocm 7.2 / triton 3.6.0 / torch 2.11.0
last_seen: 2026-08-08
---
# Narrow-dtype convert and divide can lower to emulation - census the ISA
- lever: In a quantize/cast kernel, disassemble before trusting a profiler's VALU-per-element number: on this stack an fp32 -> fp8-fnuz store convert emits ~23 VALU/element of software emulation with zero convert instructions in the dump, and both `1.0/x` and libdevice rcp lower to the full IEEE divide expansion. Each of those steps has a native-instruction route.
- apply: fnuz e4m3 (bias 8) and OCP e4m3 (bias 7) share byte patterns under an exact factor of two - convert `2*v` to the OCP fp8 type, which does lower to the packed native convert, bitcast the bytes, and fold the x2 into the stored scale (scale*0.5) so the division and its rounding are literally unchanged (it is not a reciprocal-multiply, so an exact-numerics gate still holds). For the divide, keep the reciprocal on the REDUCED-RANK divisor: one inline-asm v_rcp_f32 per tile on the [GROUPS] tensor plus a 3-FMA Markstein refine (q=n*r; rem=fma(-d,q,n); q=fma(rem,r,q)), byte-exact wherever over/underflow is unreachable.
- verify: Re-dump the AMDGCN, grep for the packed convert and for v_rcp, and re-count VALU/wave; then byte-compare the output against golden rather than reading the harness's max-rel/cosine verdict.
- caution: OCP e4m3 carries -0 where fnuz flushes to +0/NaN, so also verify a magnitude-band fixup done in fp32 BEFORE the convert (tl.where(abs(q) > 2^-10, q, 0), ~2 VALU vs ~79 for the same fixup on the converted bytes), and also verify inf/NaN inputs by hand: a max-rel gate scores 0-(-0)=0 and a cosine gate is blind to the whole representation gap.
- source: run kernel_20_geak_0808_4h 2026-08-08
