---
key: attention · gfx950 · decode
type: lever
confidence: ★★
effect: 1.51x geomean director-verified; per-case 1.63x on the smallest decode batch, 1.52x mid, 1.39x on the largest — the ratio decays as device time grows because the saving is a fixed ~15-17 us of host per call, not a device win. It was essentially the run's whole gain (1.47x of the 1.51x); the device-side edits that shipped alongside measured a tie.
confirms_cited: 0
confirms_blind: 1
losses: 0
attempts: 3
toolchain: rocm 7.2.3 / torch 2.11.0 / hip (AOT hipcc, template-codegen op)
last_seen: 2026-08-08
---
# Size the host launch path before the device on decode shapes
- lever: On a decode-regime op whose per-call device time is only tens of microseconds, size the host launch path before the device: a wrapper that re-resolves the compiled callable and re-marshals FFI arguments on every call can own the majority of wall time, and the profile will still label the bottleneck 'overhead' without saying which side of the PCIe boundary it is on.
- apply: Memoize the compiled-callable lookup on exactly the tuple of compile-time/template kwargs that forms the build hash — everything pointer- or grid-shaped is refilled per call and belongs outside the key — install FFI argument types once instead of per call, and cache constant device property queries. Re-read the stream on every call so a caller on a side stream is not silently redirected.
- verify: Numerics stay bit-identical (unchanged SNR) and the compiled object is unchanged (same VGPR/SGPR/LDS/grid) — a host-only change that moved device metrics changed something else too. Then confirm the saving is a fixed per-call cost by checking the speedup ratio shrinks monotonically as the case gets bigger.
- caution: Also verify how much of the residual host cost is editable before budgeting a second host round: here ~20 us/call was a synchronizing pageable H2D inside a caller outside the editable surface, widening the same cache a second time returned +1% (noise), and replaying the identical 2-node launch as a captured graph cost a constant +4.6 us more than the direct launches it replaced. Also verify with interleaved A/B medians over >=10 reps — a fixed per-call saving on a sub-50 us op sits inside single-shot wall-clock spread.
- source: run kernel_20_geak_0808_4h 2026-08-08
