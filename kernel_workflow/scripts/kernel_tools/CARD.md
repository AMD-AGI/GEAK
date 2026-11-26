# Kernel tools — signal → tool

Pointed to only when `use_expert_skills` is on. Evidence tools for AMD kernels; GEAK's verify step decides.
`$KT` = `$SKILL_DIR/scripts/kernel_tools`. GPU commands, from the workspace:
`bash $SKILL_DIR/scripts/gpu_lock.sh $GPU_ID <cmd>` (`$LOCK` below; tools inside it reuse the held GPU, no `--dev`).
Pass `--arch $GFX`; `--help` is each tool's interface. Keep exit codes visible: no `| tail` without `set -o pipefail`, no `>/dev/null` on setup.

| when you need | run | read |
| --- | --- | --- |
| registers, spills, LDS, occupancy limiter | `$LOCK bash $KT/dump_ir.sh --variant rN --out <ir> --kernel-name <kernel> --all -- <one-launch cmd>`, then `python3 $KT/probe.py measure --dir <ir>/rN_c<i> --arch $GFX` | `<ir>/rN/ir_index.json`: a row per compilation `<i>` (num_warps, shared, constexprs) |
| hot-loop schedule after a register/schedule change | `python3 $KT/asm_loop_audit.py <ir>/rN_c<i>/rN.s --meta <ir>/rN_c<i> --arch $GFX [--diff <prev audit>]` | `--diff`: what the change moved |
| whether a tile / `num_warps` change still fits | `python3 $KT/probe.py plan --arch $GFX --warps <W> --acc acc=<M>x<N>` | lower bound on VGPRs |
| distance to the hardware floor, and the bound | once per run: `python3 $KT/hw_context.py --output $EVAL_DIR/hardware_context.json`; then `python3 $KT/hw_budget.py --hardware-context $EVAL_DIR/hardware_context.json --brief --workload <gemm\|…> --shapes … --dtype … --measured-ms <ms>` (or `--tensors … --flops …`) | `BOTTOM LINE`: < 1.25x little room; ≥ 2x large (`thresholds.json`). A measured ceiling printed beside the datasheet one sets the verdict; exit 2 = below it. Small kernels: `--floor-us <empty-kernel µs>`. Never guess `--sku` (MI350X ≠ MI355X) |
| memory ceiling, when `bound: memory` | `$LOCK python3 $KT/mem_bw_probe.py --hardware-context … --footprint-mb <ws> [--rw-mix R:W]` | its `lo,hi` → `hw_budget.py --measured-hbm-tb-s` |
| hardware counters | `$LOCK bash $KT/rocprofv3_safe.sh --kernel '<regex>' --out <d> --pmc "<counters>" --cmd "cd <ws> && <bench>"`, then `python3 $KT/parse_pmc.py <d> '<k>' --arch $GFX` | one group per pass (`parse_pmc.py --print-groups`). rc 134: split it; rc 7: `$KT/profiling-dependencies.md` |
| an ISA fact with a citation or exact value | `python3 $KT/amd_isa.py --arch $GFX search "<words>"`, then `facts`/`layout`/`lds`/`hazard`/`waitcnt` | `SCOPED TOOL-GAP` = not available for this arch |
| SKU facts (CUs, XCDs, L2/MALL, peaks) | `python3 $KT/extract_sku.py --facts --hardware-context $EVAL_DIR/hardware_context.json` | |
| ranking three or more candidates before verify | `$LOCK python3 $KT/ab_bench.py --task <ws> --variant base=source --variant v1=<dir> … --control auto --permute --json <out>` | `SCREEN VALID` first; `NOT RESOLVED` = tie at this noise. On a shared GPU a second unresolved window means stop adding cells: a counter or GEAK's verify decides |
| stalls the audit does not explain (slow) | `$LOCK bash $KT/capture.sh rN <out> --kernel <k> --source <kernel.py> --cmd "…"` (not `--lite`: it skips ATT), then `python3 $KT/hotspot_analyzer.py <out>/att/<ui_output> --kernel <k> --arch $GFX` | `att_opclass.py`, `mfma_efficiency.py` split stalls further |

Notes
- `create_harness.py` only when a task has no runner.
- A refusal, `WITHHELD` or `NOT CALIBRATED` is the answer; report it rather than a guess.
