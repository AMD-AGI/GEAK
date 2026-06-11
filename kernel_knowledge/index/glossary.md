# Glossary

- **MFMA / SMFMAC** — Matrix Fused Multiply-Add (and sparse variant); the Matrix-Core ISA op `D=A*B+C`.
- **XCD** — Accelerator Complex Die (chiplet); MI300X has 8 XCDs × 38 CUs = 304 CUs. Clock varies 3–10% across XCDs.
- **CU / SIMD / wavefront** — Compute Unit; 4 SIMD/CU; wavefront (warp) = 64 lanes.
- **VGPR/SGPR/AGPR** — vector / scalar / accumulation registers; 512 VGPR/EU, allocated in 16-granules; MFMA accumulators live in AGPR.
- **LDS** — Local Data Share (shared memory); watch bank conflicts; CDNA4 = 160 KB, 256 B/clk.
- **mfma_16x16 vs 32x32** — MFMA tile shape; 16x16 usually faster on MI300X even for big tiles.
- **OPTIMIZE_EPILOGUE** — store MFMA result in MFMA layout (skip reblock); usually set 1.
- **Tagram hotspot** — perf cliff when GEMM stride is a multiple of 512 B (TN case) on MI300.
- **FNUZ fp8** — CDNA3 fp8 variant (finite, no ±inf encoding) vs OCP fp8 on CDNA4.
- **MXFP / E8M0** — OCP microscaling: block of 32 elements shares one 8-bit (E8M0) exponent; MXFP4/6/8.
- **FlyDSL** — aiter's Python kernel DSL with instruction-level control (tile/split_k/preshuffle/stages).
- **ck_tile** — Composable Kernel's tile-programming framework (FMHA/GEMM templates).
- **aiter** — AI Tensor Engine for ROCm; the central tuned-kernel engine (attn/MoE/GEMM/norm/quant/comm).
- **roofline** — perf model classifying kernels memory-bound (sloped) vs compute-bound (flat ceiling).
- **Amdahl gate** — accept a kernel only if pct_gpu_time × speedup_fraction moves e2e beyond the noise band.

## Sources
- AMD CDNA3/CDNA4 ISA & whitepaper; ROCm MI300X workload optimization guide; Matrix Core CDNA blog (URLs in sourcing_rules.md).
