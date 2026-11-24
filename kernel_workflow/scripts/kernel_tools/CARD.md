# Kernel tools — signal → tool

Pointed to only when `use_expert_skills` is on. Evidence tools for AMD kernels; GEAK's verify step decides.
`$KT` = `$SKILL_DIR/scripts/kernel_tools`, `$LOCK` = `GEAK_KERNEL_TOOLS=1 bash $SKILL_DIR/scripts/gpu_lock.sh $GPU_ID`
(every GPU command). Pass `--arch $GFX`; each tool's `--help` is its interface.

| when you need | run | read |
| --- | --- | --- |
| registers, spills, LDS, occupancy limiter | `$LOCK bash $KT/dump_ir.sh --variant rN --out <ir> --kernel-name <kernel> --all -- <bench>`, then `python3 $KT/probe.py measure --dir <ir>/rN_c<i> --arch $GFX` | `<ir>/rN/ir_index.json`: one row per compilation `<i>` (num_warps, num_stages, shared, constexprs) |
| hot-loop schedule (per-counter drains, MFMA mix, barriers, scratch/spill traffic) after a register/schedule change | `python3 $KT/asm_loop_audit.py <ir>/rN_c<i>/rN.s --meta <ir>/rN_c<i> --arch $GFX [--diff <prev audit>]` | `--diff`: what the change moved |
| whether a tile / `num_warps` change still fits | `python3 $KT/probe.py plan --arch $GFX --warps <W> --acc acc=<M>x<N>` | lower bound on VGPRs |
| distance to the hardware floor, and the bound | once: `python3 $SKILL_DIR/../scripts/hardware_context.py --output $EVAL_DIR/hardware_context.json`; then `python3 $KT/hw_budget.py --hardware-context $EVAL_DIR/hardware_context.json --brief --workload <gemm\|…> --shapes … --dtype … --measured-ms <ms>` (or `--tensors … --flops … --flops-engine mfma\|valu`) | `BOTTOM LINE`. Multiple < 1.25x: little room; ≥ 2x: large room (`thresholds.json`). A datasheet multiple is an upper bound; a measured sustained ceiling (MFMA rate; HBM row matching the `--tensors` r/w mix) is printed beside it and sets the verdict; exit 2 = below the floor or that ceiling. A vendor-GEMM figure is a comparison at its own shape. Small kernels: `--floor-us <empty-kernel µs, same timer>` |
| memory ceiling, when `bound: memory` and stop/continue rests on it | `$LOCK python3 $KT/mem_bw_probe.py --hardware-context … --footprint-mb <ws> [--rw-mix R:W \| --rmw \| --runlen B --stride B]` | pass its `lo,hi` as `hw_budget.py --measured-hbm-tb-s` |
| hardware counters | `$LOCK bash $KT/rocprofv3_safe.sh --kernel '<regex>' --out <d> --pmc "<counters>" --cmd "cd <ws> && <bench>"`, then `python3 $KT/parse_pmc.py <d> '<k>' --arch $GFX`; or `KERNEL_TOOLS_PROFILE=1` on `profile_kernel.sh` for a counter pass | pass one group from `parse_pmc.py --print-groups`; the per-pass limit is per hardware block. rc 134: split the group; rc 7: the image loads two LLVM copies (`$KT/profiling-dependencies.md`, "Known abort") |
| an ISA fact with a page citation or an exact value | `python3 $KT/amd_isa.py --arch $GFX search "<words>"`, then `facts`/`layout`/`lds`/`hazard`/`waitcnt`/`limits` | gfx950 records labelled `measured` come from one MI350X node, 2026-11; `SCOPED TOOL-GAP` = not available for this arch |
| SKU facts (CUs, XCDs, L2/MALL, peaks, ridge, grid tail) | `python3 $KT/extract_sku.py --facts --hardware-context $EVAL_DIR/hardware_context.json` | |
| ranking three or more candidates before a verify run | `$LOCK python3 $KT/ab_bench.py --task <ws> --variant base=source --variant v1=<dir> … --control auto --permute --json <out>` | `SCREEN VALID` first; `NOT RESOLVED` = tie |
| instruction-level stalls the audit does not explain (last resort, slow) | `$LOCK bash $KT/capture.sh rN <out> --kernel <k> --source <kernel.py> --lite --cmd "…"`, then `python3 $KT/hotspot_analyzer.py <out>/att/<ui_output> --kernel <k> --arch $GFX` | `att_opclass.py`, `mfma_efficiency.py`, `deep_mfma_analysis.py` split stalls further |

Notes
- `create_harness.py` only when a task has no runner.
- A refusal, `WITHHELD` or `NOT CALIBRATED` is the answer; report it rather than a guess.
