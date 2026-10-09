---
title: Triton on AMD — ISA verification workflow
kind: language
gens: [gfx950, gfx942]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [both]
status: competitive
updated: 2026-10-07
sources:
  - https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
  - https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py
  - https://llvm.org/docs/AMDGPUUsage.html
---

# Triton on AMD — verify with the ISA

A tuned config is not trusted until you've **read the AMDGCN**. Autotune timing is necessary but the
ISA tells you *why* and catches silent slow paths (scalar loads, scratch spills, FNUZ mismatch).

## 1. Dump everything
```bash
AMDGCN_ENABLE_DUMP=1 \      # final AMDGCN ISA to stderr
MLIR_ENABLE_DUMP=1 \        # TTGIR / TritonAMDGPU IR after each pass
TRITON_PRINT_AUTOTUNING=1 \ # winning config + timing
TRITON_ALWAYS_COMPILE=1 \   # bypass kernel cache so the dump is for THIS run
python my_kernel.py 2> dump.txt
```
Pull the resource numbers:
```bash
grep ".amdhsa_next_free_vgpr"    dump.txt   # VGPRs/lane the occupancy is computed from (ArchVGPR+AGPR on CDNA)
grep "; Occupancy:"              dump.txt   # LLVM's own register-term waves/SIMD for this subtarget
grep ".vgpr_count\|.agpr_count"  dump.txt   # ArchVGPR / AGPR split -- NOT the occupancy input on its own
grep ".sgpr_count"               dump.txt
grep ".group_segment_fixed_size" dump.txt   # LDS bytes
grep ".private_segment_fixed_size" dump.txt  # scratch — MUST be 0
grep "num-warps"                 dump.txt
grep "triton_gpu.shared"         dump.txt    # LDS bytes per shared layout (from MLIR dump)
```
GEAK's shared kernel tools do the arithmetic for you (offline, no GPU; examples use `--arch gfx950`):
`kernel_workflow/scripts/kernel_tools/dump_ir.sh` (per-variant IR + `.s` + LDS metadata),
`amd_occupancy.py --asm <kernel.s>` (register AND LDS terms), `probe.py measure --dir <ir_dir>`
(both limiters per kernel, in WGs/CU), `asm_loop_audit.py <kernel.s>` (hot-loop op mix, waitcnt
quality, KD register budget). Triton kernels size LDS at launch, so the `.s` says 0 bytes -- take the
LDS bytes from the cache metadata `shared` field (dump_ir.sh copies it as `meta_*.json`).

## 2. What good ISA looks like (GEMM inner loop)
| Look for | Good | Bad → retune |
|---|---|---|
| Global loads | `global_load_dwordx4` / `buffer_load_dwordx4` | `global_load_dword` (scalar) |
| Masked tail | `buffer_load_*` (HW bounds) | `global_load_*` + `v_cmp` predication |
| LDS access | `ds_read_b128` / `ds_write_b128` | `ds_read_b32` |
| MFMA | dense `v_mfma_f32_16x16x16` | sparse, gaps = starved core |
| Accumulator | acc stays in AGPR (`a[0:n]`) | `v_accvgpr_read/write` inside loop |
| Scratch | `.private_segment_fixed_size: 0` | nonzero → spilling to HBM (3–5× slower) |
| Waitcnt | minimal, overlapped | `s_waitcnt vmcnt(0)` after every load = no overlap |

## 3. Occupancy boundary check
CDNA (gfx950 main line; gfx942 identical here): ArchVGPR and AGPR share ONE 512-entry file per SIMD,
allocated in granules of **8**, capped at **8** waves/SIMD.
1. Read the register term. Prefer LLVM's own `; Occupancy: N` from the `.s`. Otherwise take
   `.amdhsa_next_free_vgpr` (already ArchVGPR+AGPR combined) — **not** `.vgpr_count`, which counts
   ArchVGPRs only and silently drops the accumulator AGPRs.
2. `waves = min(8, floor(512 / round_up_8(next_free_vgpr)))`. The ladder is
   64→8, 72→7, 80→6, 96→5, 128→4, 168→3, 256→2 (the LAST VGPR count that still fits that many
   waves). If you're one granule over a boundary (e.g. 176 → 2 waves), set `waves_per_eu = target+1`
   so LLVM shaves VGPRs (e.g. 176→168 → 3 waves). Re-dump.
3. Take `min()` with the LDS term: `WGs/CU <= LDS_per_CU // lds_bytes_per_wg` — **160 KiB/CU on
   gfx950**, 64 KiB/CU on gfx942 (downgrade; a 2.5x different divisor, so never default it). The LDS
   bytes of a Triton kernel come from the cache metadata `shared`, not the `.s`.
4. If setting `waves_per_eu` introduced `.private_segment_fixed_size > 0`, you went too far — back off.

Tool: `python3 kernel_workflow/scripts/kernel_tools/amd_occupancy.py --asm kernel.s
--lds-bytes-per-wg <shared> --workgroup-size <threads>` prints both terms and which binds;
`--vgpr N --arch gfx950` is the register-term lookup (it refuses without an arch). The constants come
from `perf_knowledge/hardware/data/hw_constants.json`. RDNA (gfx11*/gfx12*) is a different file
(1536/SIMD, granule 24, cap 16, no AGPR) — never apply this CDNA formula there.

## 4. MFMA shape & dtype sanity
- fp16/bf16 with `matrix_instr_nonkdim=16`: on gfx950 expect the K-doubled
  `v_mfma_f32_16x16x32_{f16,bf16}` (32x32: `v_mfma_f32_32x32x16_*`); on gfx942 (downgrade)
  `v_mfma_f32_16x16x16_*` / `32x32x8_*`. If you see the 32x32 form, your `nonkdim` is 32 (or auto
  picked it) — compare timings. Look a mnemonic up offline with
  `python3 kernel_workflow/scripts/kernel_tools/gfx950_isa.py facts <instr>`.
- gfx950 block-scaled MXFP → `v_mfma_scale_f32_*_f8f6f4`.
- fp8 on gfx942 (downgrade) → `v_mfma_f32_16x16x32_fp8_fp8` (FNUZ). If the build refused to lower the
  dot, you passed OCP `e4m3fn` — convert to `tl.float8e4b8`. gfx950 uses OCP fp8, not FNUZ.

## 5. LDS layout (kpack) check
`kpack=2` on gfx942 should turn the dot-operand LDS reads into `ds_read_b128`. If they're still
`ds_read_b64`/`b32`, either `BLOCK_K` is too small (<64) or the swizzle didn't apply — bump `BLOCK_K`,
re-check. On gfx950 expect `ds_read_b128` without `kpack` (deprecated there).

## 6. Cross-check vs library
Isolated bench the tuned Triton kernel against the library default to know the real gap:
```bash
ROCBLAS_LAYER=2 HIPBLASLT_LOG_LEVEL=2 python compare.py   # log lib solution + fallbacks
```
Then **e2e-gate** through the actual serving seam (aiter), not just isolated TFLOPS — see
[pitfalls.md](pitfalls.md) integration note.

## 7. Drill to the .s if needed
For the raw object:
```bash
# from a cached HSACO or AOT-compiled object
roc-obj-ls kernel.hsaco
llvm-objdump -d --arch=amdgcn kernel.hsaco | less
```
Counter/instruction semantics (`s_waitcnt vmcnt/lgkmcnt`, buffer descriptors, sched barriers) are in
the LLVM AMDGPU backend user guide.

## Sources
- AMDGCN_ENABLE_DUMP / ds_read_b128 / global_load_dwordx4 / OPTIMIZE_EPILOGUE: https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
- HIPOptions / knobs.amd.dump_amdgcn / use_buffer_ops: https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py
- AMDGPU backend (s_waitcnt, buffer descriptors, resource usage attrs): https://llvm.org/docs/AMDGPUUsage.html
- Occupancy math (512 VGPR/SIMD shared by ArchVGPR+AGPR, granule 8, cap 8; compiler-derived ladder): `kernel_workflow/scripts/kernel_tools/amd_occupancy.py` + `perf_knowledge/hardware/data/hw_constants.json` (the ROCm MI300X workload page's 16-granule figure is superseded by the measured LLVM breakpoints)
- MI300X workload optimization guide: https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
