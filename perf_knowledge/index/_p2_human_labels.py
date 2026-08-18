#!/usr/bin/env python3
"""One-shot: seed the hand-labelled (`src: human`) rows of kb_labels.yaml (plan P2).

These are the CROSS-CUTTING dirs — `optimization/`, `quantization/`, `hardware/`, `backends/`,
`profiling/`, `workflows/` — that no path rule can label, because what a doc is *about* is not
recoverable from its position in the tree. They are also the docs `--bound` queries most need: P1's
rule pass emits no `bound_type` at all, so without this table the roofline routing key stays at the
2 files it had before and `kb_resolve --bound <x>` keeps returning nothing useful.

Every row cites the sentence it was read off (`evidence`), so the label is auditable rather than
asserted. Run once; after that `kb_labels.yaml` is the record and `_label_kb.py --rules` preserves
these rows untouched. Re-running is idempotent (rows are keyed by path + src).

  python3 index/_p2_human_labels.py            # dry-run diff against kb_labels.yaml
  python3 index/_p2_human_labels.py --write    # merge into kb_labels.yaml
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _label_kb import load_labels, write_labels, validate_row  # noqa: E402

PK = "perf_knowledge/"
KW = "kernel_workflow/knowledge/"

# (path, cost, levers, bound_type, risk, evidence)
# cost "" = deliberately unset: a doc that DESCRIBES hardware or classifies a bottleneck is not a
# lever you can pull, and kb_resolve sorts unset-cost last so it never out-ranks an actionable card.
ROWS = [
    # ---- optimization/ — the technique layer; these are exactly what --bound should return -----
    (PK + "optimization/autotuning_methodology.md", "L1",
     ["config.per-shape-tune", "tile.autotune-db"], ["mfma_compute"], "",
     "TL;DR: capture shapes with AITER_TUNE_GEMM, race with gradlib, deploy via AITER_CONFIG_GEMM_BF16"),
    (PK + "optimization/kernel_fusion_strategy.md", "L3",
     ["fusion.epilogue", "fusion.prologue", "fusion.norm-quant"], ["hbm_bw"], "",
     "TL;DR: Fuse to cut HBM round-trips (a bandwidth-bound producer + consumer)"),
    (PK + "optimization/lds_and_bank_conflicts.md", "L3",
     ["layout.swizzle"], ["lds_bank"], "",
     "TL;DR: two lanes hitting the same bank serialize; the two fixes are padding and XOR swizzle"),
    (PK + "optimization/memory_pipelining.md", "L3",
     ["pipeline.async-lds", "pipeline.software-pipeline"], ["lds_bank", "mfma_compute"], "",
     "TL;DR: global_load_lds + software pipelining; full overlap global_load->ds_write->ds_read->v_mfma"),
    (PK + "optimization/mfma_scheduling.md", "L3",
     ["tile.static-shape", "env.flag"], ["mfma_compute"], "",
     "TL;DR: mfma_16x16x16 usually beats mfma_32x32x8; write-out should use OPTIMIZE_EPILOGUE=1"),
    (PK + "optimization/numerical_stability.md", "", [], [], "numerics-affecting",
     "TL;DR: accumulate matmul/reductions in fp32, online softmax, Welford, the fp8 FNUZ/OCP trap"),
    (PK + "optimization/occupancy_and_registers.md", "L1",
     ["occupancy.vgpr"], ["occupancy"], "",
     "TL;DR: occupancy = floor(512 / round_up(VGPR_used,16)), capped at 8 waves/EU"),
    (PK + "optimization/roofline_and_bottlenecks.md", "",
     [], ["hbm_bw", "mfma_compute"], "",
     "TL;DR: classify the kernel — compute-bound vs bandwidth-bound — before optimizing"),
    (PK + "optimization/vectorization_and_coalescing.md", "L3",
     ["vectorize.coalesce"], ["hbm_bw"], "",
     "TL;DR: widest aligned chunk (128-bit global_load_dwordx4); cheapest big win on memory-bound kernels"),
    (PK + "optimization/wave_and_grid_sizing.md", "L1",
     ["occupancy.vgpr"], ["occupancy"], "",
     "TL;DR: use __launch_bounds__ to bound register use for a target occupancy"),
    (PK + "optimization/xcd_l2_locality.md", "L3",
     ["layout.xcd-l2", "layout.swizzle"], ["l2_locality"], "",
     "TL;DR: >=1024 workgroups, 8-multiple tile counts, XCD-aware / swizzled CTA order"),

    # ---- quantization/ ------------------------------------------------------------------------
    (PK + "quantization/accuracy_evaluation.md", "", [], [], "numerics-affecting",
     "TL;DR: never gate on byte parity; isolated err_ratio<0.05 gate + decisive e2e task-accuracy gate"),
    (PK + "quantization/block_scaling_mxfp.md", "L3",
     ["dtype.microscale"], ["mfma_compute"], "numerics-affecting",
     "TL;DR: a group of 32 low-bit elements shares one E8M0 scale; CDNA4 runs it in HW"),
    (PK + "quantization/calibration_and_quark.md", "L2",
     ["dtype.downcast"], [], "numerics-affecting",
     "TL;DR: AMD Quark produces quantized checkpoints; exports HF-format that vLLM and sglang load"),
    (PK + "quantization/deployment_recipes.md", "L0",
     ["env.flag", "dtype.downcast"], [], "numerics-affecting",
     "TL;DR: turn on the AITER master switch — vLLM: VLLM_ROCM_USE_AITER=1 + quantization"),
    (PK + "quantization/fnuz_vs_ocp.md", "", [], [], "numerics-affecting",
     "TL;DR: same byte layout, different exponent bias; CDNA3 = FNUZ, CDNA4 = OCP"),
    (PK + "quantization/formats_overview.md", "", [], [], "numerics-affecting",
     "TL;DR: this page is the bit-layout reference for the precision ladder"),
    (PK + "quantization/hardware_support_matrix.md", "", [], ["mfma_compute"], "",
     "TL;DR: what the matrix core (MFMA) actually accelerates per generation"),
    (PK + "quantization/kv_cache_quantization.md", "L0",
     ["env.flag", "dtype.downcast"], ["hbm_bw"], "numerics-affecting",
     "TL;DR: the KV cache is memory-bound and grows with sequence x batch; --kv-cache-dtype fp8"),
    (PK + "quantization/scaling_strategies.md", "",
     ["dtype.downcast"], [], "numerics-affecting",
     "TL;DR: granularity (elements per scale) x timing (static vs dynamic) decide accuracy-vs-cost"),

    # ---- hardware/ — bound_type ONLY. A hardware doc describes a ceiling; it is not a lever, so
    # it deliberately carries no cost and must never out-rank an actionable card. -----------------
    (PK + "hardware/cdna1_mi100/arch.md", "", [], ["occupancy"], "", "CU / wavefront resource layout"),
    (PK + "hardware/cdna1_mi100/matrix_core.md", "", [], ["mfma_compute"], "", "MFMA throughput tables"),
    (PK + "hardware/cdna2_mi200/arch.md", "", [], ["occupancy"], "", "CU / wavefront resource layout"),
    (PK + "hardware/cdna2_mi200/matrix_core.md", "", [], ["mfma_compute"], "", "MFMA throughput tables"),
    (PK + "hardware/cdna2_mi200/memory.md", "", [], ["hbm_bw", "lds_bank"], "",
     "HBM/L2/LDS hierarchy and bandwidths"),
    (PK + "hardware/cdna2_mi200/occupancy.md", "", [], ["occupancy"], "", "waves/EU and VGPR budget"),
    (PK + "hardware/cdna3_mi300/arch.md", "", [], ["occupancy", "l2_locality"], "",
     "304 CUs across 8 XCDs, each XCD with its own L2"),
    (PK + "hardware/cdna3_mi300/clocks_power.md", "", [], ["mfma_compute"], "",
     "sustained clock under MFMA load bounds achievable compute"),
    (PK + "hardware/cdna3_mi300/isa_notes.md", "", [], ["mfma_compute"], "", "v_mfma encodings and hazards"),
    (PK + "hardware/cdna3_mi300/matrix_core.md", "", [], ["mfma_compute"], "", "MFMA shapes and rates"),
    (PK + "hardware/cdna3_mi300/memory_hierarchy.md", "", [], ["hbm_bw", "lds_bank"], "",
     "HBM3 / Infinity Cache / L2 / 64 KB LDS per CU"),
    (PK + "hardware/cdna3_mi300/occupancy.md", "", [], ["occupancy"], "", "waves/EU and VGPR budget"),
    (PK + "hardware/cdna3_mi300/peak_tables.md", "", [], ["mfma_compute", "hbm_bw"], "",
     "the two roofline ceilings: per-dtype peak FLOP/s and HBM BW"),
    (PK + "hardware/cdna3_mi300/xcd_chiplet.md", "", [], ["l2_locality"], "",
     "8 XCDs, each with its own L2; cross-XCD traffic crosses Infinity Fabric"),
    (PK + "hardware/cdna4_mi350/arch.md", "", [], ["occupancy", "l2_locality"], "",
     "256 CUs across 8 XCDs"),
    (PK + "hardware/cdna4_mi350/clocks_power.md", "", [], ["mfma_compute"], "",
     "sustained clock under MFMA load bounds achievable compute"),
    (PK + "hardware/cdna4_mi350/fp4_fp6_microscaling.md", "", [], ["mfma_compute"], "numerics-affecting",
     "block-scaled MFMA is CDNA4-only hardware"),
    (PK + "hardware/cdna4_mi350/isa_notes.md", "", [], ["mfma_compute"], "", "v_mfma encodings and hazards"),
    (PK + "hardware/cdna4_mi350/matrix_core_blockscale.md", "", [], ["mfma_compute"], "",
     "block-scaled MFMA shapes and rates"),
    (PK + "hardware/cdna4_mi350/memory.md", "", [], ["hbm_bw", "lds_bank"], "",
     "160 KB/CU LDS with ~2x LDS bandwidth; HBM3E"),
    (PK + "hardware/cdna4_mi350/peak_tables.md", "", [], ["mfma_compute", "hbm_bw"], "",
     "the two roofline ceilings: per-dtype peak FLOP/s and HBM BW"),
    (PK + "hardware/shared/dtype_numerics.md", "", [], [], "numerics-affecting",
     "per-dtype range/precision behaviour shared across generations"),
    (PK + "hardware/shared/hbm_infinity_fabric.md", "", [], ["hbm_bw", "sync"], "",
     "HBM bandwidth and the inter-die / inter-GPU fabric collectives ride on"),
    (PK + "hardware/shared/l2_xcd_swizzle.md", "", [], ["l2_locality"], "",
     "how the round-robin workgroup->XCD mapping decides which L2 you hit"),
    (PK + "hardware/shared/matrix_core_mfma_smfmac.md", "", [], ["mfma_compute"], "",
     "the v_mfma / v_smfmac instruction family"),
    (PK + "hardware/shared/memory_model_lds_bank.md", "", [], ["lds_bank"], "",
     "32 banks of 4 bytes; same-bank different-address lanes serialize"),
    (PK + "hardware/shared/wavefront_simd_vgpr_agpr.md", "", [], ["occupancy"], "",
     "512 x 32-bit registers per SIMD in 16-register granules"),

    # ---- profiling/ — how you FIND each bound, so a --bound query should surface it -------------
    (PK + "profiling/benchmarking_methodology.md", "", [], ["host_bound", "launch_overhead"], "",
     "TL;DR: warm, repeated (REPEATS=7), inside a ~0.5% noise band, clocks controlled"),
    (PK + "profiling/common_pitfalls.md", "", [], ["host_bound", "launch_overhead"], "",
     "TL;DR: cold cache/clock, throttling, host fork-storm starving launches"),
    (PK + "profiling/reading_a_kernel_bottleneck.md", "", [], ["mfma_compute", "hbm_bw"], "",
     "TL;DR: four failure modes, four counter signatures — compute-bound, BW-bound, ..."),
    (PK + "profiling/rocprof_compute_workflow.md", "", [], ["mfma_compute", "hbm_bw"], "",
     "TL;DR: profile with --roof-only for a fast compute-vs-BW verdict"),
    (PK + "profiling/rocprofv3_counters.md", "", [], ["mfma_compute", "lds_bank"], "",
     "TL;DR: SQ (MFMA, waves, LDS), TA/TD/TCP (vector L1), TCC (L2)"),
    (PK + "profiling/trace_analysis.md", "", [], ["launch_overhead", "host_bound"], "",
     "TL;DR: when the problem is between kernels — not inside one — you need a timeline"),

    # ---- workflows/ ---------------------------------------------------------------------------
    (PK + "workflows/attention_backend_selection.md", "L0",
     ["env.flag", "backend.swap"], [], "",
     "TL;DR: the attention backend is a server flag — the cheapest e2e lever (no source)"),
    (PK + "workflows/authoring_a_kernel_with_geak.md", "L3", [], [], "",
     "TL;DR: when no editable backend impl exists, you author one from scratch in a target language"),
    (PK + "workflows/choosing_a_backend.md", "L2", ["backend.swap"], [], "",
     "TL;DR: the one-page prior that summarizes the per-operator overview SOTA tables"),
    (PK + "workflows/gemm_tuning_workflow.md", "L1",
     ["config.per-shape-tune", "tile.autotune-db"], ["mfma_compute"], "",
     "TL;DR: aiter.tuned_gemm picks the fastest per shape from aiter's own per-shape DB"),
    (PK + "workflows/integrating_a_new_kernel.md", "L2", ["backend.swap"], [], "",
     "TL;DR: choose the right seam (env/flag/patch/authored), overlay it reversibly"),
    (PK + "workflows/model_bringup_checklist.md", "L0", ["env.flag"], [], "",
     "TL;DR: pin the stack -> enable AITER -> pick the quant format -> pick the attention backend"),
    (PK + "workflows/optimize_e2e_model.md", "", ["env.flag"], ["host_bound"], "",
     "TL;DR: reason in Amdahl mass, tune the cheap config knobs first"),
    (PK + "workflows/optimize_single_kernel.md", "", [], ["launch_overhead", "host_bound"], "",
     "TL;DR: climb a cheapest-first ladder: bench every backend -> tune -> only then rewrite"),

    # ---- case_studies/ — a war story IS a measured lever; these are the highest-evidence cards in
    # the tree and were reachable by nothing, because no path rule fits them. -----------------------
    (PK + "case_studies/by_kernel/fused_norm_quant_win.md", "L3",
     ["fusion.norm-quant"], ["hbm_bw"], "", "fusing norm+quant removes an HBM round-trip"),
    (PK + "case_studies/by_kernel/gated_delta_backend_swap.md", "L2",
     ["backend.swap"], ["hbm_bw"], "", "swapping the gated-delta backend"),
    (PK + "case_studies/by_kernel/gemm_aiter_db_tuning.md", "L1",
     ["config.per-shape-tune", "tile.autotune-db"], ["mfma_compute"], "",
     "the aiter per-shape GEMM DB tuning win (+2.23% e2e), cited by optimization/autotuning_methodology.md"),
    (PK + "case_studies/by_kernel/mfma_tile_selection.md", "L1",
     ["tile.static-shape"], ["mfma_compute"], "",
     "MFMA tile selection: 16x16 vs 32x32 on MI300X GEMM"),
    (PK + "case_studies/by_model/deepseek_mla_mi300x.md", "L2",
     ["backend.swap"], ["hbm_bw", "mfma_compute"], "", "MLA attention serving on MI300X"),
    (PK + "case_studies/by_model/deepseek_v3v4_attention.md", "L2",
     ["backend.swap"], ["hbm_bw", "mfma_compute"], "",
     "TL;DR table: attention / KV cache / serving on MI300X"),
    (PK + "case_studies/by_model/kimi_k2.6_int4_moe_mi300x.md", "L2",
     ["dtype.downcast", "backend.swap"], ["hbm_bw"], "numerics-affecting",
     "int4 MoE serving on MI300X"),
    (PK + "case_studies/by_model/llama_fp8_serving.md", "L0",
     ["env.flag", "dtype.downcast"], ["hbm_bw"], "numerics-affecting",
     "Llama-class fp8 serving on MI300X"),
    (PK + "case_studies/by_model/qwen3.5-27b_sglang_e2e.md", "L0",
     ["env.flag"], ["host_bound", "hbm_bw"], "",
     "the flagship full e2e run: server flags A/B'd against a 0.5% band"),

    # ---- graph.cudagraph-safe — the lever with no home. Graph capture is an E2E lever (see the T5
    # correction in kernel_workflow/docs/optimization_roadmap.md), so it lives in the host/launch docs,
    # not in a kernel card. Without these rows `views/by_lever/graph.cudagraph-safe.md` stays empty
    # and the lever is undiscoverable. -------------------------------------------------------------
    (PK + "languages/hip_cpp/patterns.md", "", ["graph.cudagraph-safe"], ["launch_overhead"], "",
     "hipStreamBeginCapture -> hipGraphInstantiate -> hipGraphLaunch"),
    (PK + "profiling/benchmarking_methodology.md", "", ["graph.cudagraph-safe"], [], "",
     "step 6: Graphs for launch-bound work — use HIP graphs to replay a launch sequence"),
    (PK + "operators/fused_moe_grouped_gemm/overview.md", "", ["graph.cudagraph-safe"], [], "",
     "CUDA-graph-friendly because the CPU doesn't know per-expert counts"),

    # ---- tile.splitk / tile.streamk — same problem: the operator that IS these levers had them on
    # no file, so both views read `- (none)`. -------------------------------------------------------
    (PK + "operators/splitk_streamk_gemm/overview.md", "",
     ["tile.splitk", "tile.streamk"], [], "", "the operator is the lever pair"),

    # ---- kernel_workflow/knowledge/ — already carry cost+levers; only bound_type was missing,
    # which is exactly what kept them unreachable from a host-bound profile result. ---------------
    (KW + "geomean_levers.md", "", [], ["launch_overhead", "host_bound"], "",
     "# Geomean Levers — How to Beat the Wall-Clock Floor"),
    (KW + "wrapper_optimization.md", "", [], ["launch_overhead", "host_bound"], "",
     "# Python Wrapper Optimization Patterns"),
]

# backends/ — cost + levers. Default is "swap in an off-the-shelf library" (L2); the overrides are
# the docs that are really about a TUNING knob or an env flag, which are strictly cheaper.
BACKEND_DEFAULT = ("L2", ["backend.swap"], [], "library backend card — integration seam, not a rewrite")
BACKEND_OVERRIDE = {
    "aiter/configs_db.md": ("L1", ["tile.autotune-db", "config.per-shape-tune"], [],
                            "the per-shape tuned-config DB, not a code change"),
    "aiter/flydsl_path.md": ("L3", [], [], "authoring the kernel in FlyDSL is a rewrite"),
    "composable_kernel_lib/ckprofiler.md": ("L1", ["config.per-shape-tune"], [],
                                            "ckProfiler races instances — a tuning step"),
    "hipblaslt/env.md": ("L0", ["env.flag"], [], "environment variables only, zero code change"),
    "hipblaslt/offline_tuning.md": ("L1", ["config.per-shape-tune", "tile.autotune-db"], [],
                                    "offline per-shape tuning producing a config file"),
    "hipblaslt/tensilelite.md": ("L1", ["tile.autotune-db"], [],
                                 "TensileLite generates/selects kernels from a solution DB"),
    "mori_rccl/deepep.md": ("L2", ["backend.swap"], ["sync"], "collective/EP dispatch path"),
    "mori_rccl/mori_ep.md": ("L2", ["backend.swap"], ["sync"], "collective/EP dispatch path"),
    "mori_rccl/overview.md": ("L2", ["backend.swap"], ["sync"], "collective communication library"),
    "mori_rccl/rccl_tuning.md": ("L1", ["config.per-shape-tune", "env.flag"], ["sync"],
                                 "RCCL tuning via env/config, no source change"),
    "pytorch_inductor/max_autotune.md": ("L1", ["config.per-shape-tune"], [],
                                         "max-autotune is a compile-time search knob"),
    "rocblas_tunableop/tunableop.md": ("L1", ["config.per-shape-tune", "env.flag"], [],
                                       "TunableOp records/replays a per-shape choice via env"),
    "sglang_kernels/attention_backends.md": ("L0", ["env.flag", "backend.swap"], [],
                                             "selected by a server flag"),
}


def backend_rows():
    root = os.path.join(os.path.dirname(os.path.dirname(HERE)), PK, "backends")
    out = []
    for dp, _dn, fn in os.walk(root):
        for f in sorted(fn):
            if not f.endswith(".md") or f == "README.md":
                continue
            rel = os.path.relpath(os.path.join(dp, f), root).replace(os.sep, "/")
            cost, levers, bound, ev = BACKEND_OVERRIDE.get(rel, BACKEND_DEFAULT)
            out.append((PK + "backends/" + rel, cost, levers, bound, "", ev))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    rows = ROWS + backend_rows()
    new = []
    for path, cost, levers, bound, risk, ev in rows:
        r = {"path": path, "src": "human", "evidence": ev}
        if cost:
            r["cost"] = cost
        if levers:
            r["levers"] = levers
        if bound:
            r["bound_type"] = bound
        if risk:
            r["risk"] = risk
        errs = validate_row(r)
        if errs:
            print(f"INVALID {path}: {', '.join(errs)}", file=sys.stderr)
            return 1
        new.append(r)

    existing = [r for r in load_labels() if not (r.get("src") == "human")]
    print(f"{len(new)} human rows ({len(rows) - len(ROWS)} of them backends/), "
          f"{len(existing)} rule/llm rows preserved.")
    if a.write:
        write_labels(existing + new)
        print("kb_labels.yaml updated.")
    else:
        print("dry-run — pass --write to merge.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
