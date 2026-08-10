---
key: dense gemm · gfx950 · compute-bound
type: method
confidence: ★★
effect: director-verified 3.71x geomean over the untuned capture (per-case 3.00x on the small-M shape M~2K, 4.18x and 4.09x on the large-M shapes M>=32K); six prior rounds of in-language tuning in the same run had plateaued near 2.9x self-reported, so the library body was worth ~+25% on top of an already-tuned kernel, 1451 vs 1163 TFLOP/s bf16
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 1
toolchain: rocm 7.2.3 / triton 3.6.0 / torch 2.11.0
last_seen: 2026-08-08
---
# Price the vendor library on the exact shapes before funding kernel-source rounds
- lever: Before spending rounds on kernel-source micro-optimization, time a vendor GEMM (hipBLASLt through torch.addmm/torch.mm, or the aiter equivalent) on the exact shapes at hand. The answer bounds the campaign either way: a slower library says the remaining gap is physics and the kernel lane is worth funding; a faster one says the gap is a code-generation gap whose size you now know before the budget is spent.
- apply: Measure it in-process on the same inputs and in the output form the harness fixes: under a caller-owned output buffer the out= form (torch.addmm(bias, a, b, out=c)) carries the whole margin, because a library entry point that allocates its own result pays a full-size copy back that can exceed the win.
- verify: Interleaved in-process A/B on wall-clock ms against the current kernel plus correctness on every case, and confirm from rocprofv3 which kernel actually dispatched (a Tensile Cijk_* name means the library body is the one running).
- caution: Also verify the task's rules accept a body written outside the kernel language before scoring on it, and report both numbers (library and best in-language) so the reviewer keeps that call; the measurement is informative even where the substitution turns out to be out of bounds.
- source: run kernel_20_geak_0808_4h 2026-08-08
