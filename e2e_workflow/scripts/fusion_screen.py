#!/usr/bin/env python3
"""Screen a semantic kernel table into a fusion Top-K.

Trace capture stays upstream. This script is the policy after that table exists
(GEAK_structure.txt, KernelFusion algorithm, items 0-9):

  constants: per-phase kernel floor c0_p, HBM bandwidth, dtype peaks
  -> per-layer dataflow graph G (explicit tensor ids, else inferred edges)
  -> classify every row (stage -> parent operator -> name tokens), big/small
     ops, per-kernel bytes / ideal time / efficiency / headroom
  -> windows: serial, sibling, parallel, epilogue_forward, fan_out, seam_rewrite
  -> match each window against the (sanitized) fusion-kernel catalog
  -> request-level expected gain = launch + memory + efficiency headroom
  -> gate on expected gain (Top-K) or keep as candidate pool
  -> one seam = one entry across phases and layer patterns; alternatives that
     eliminate the same kernel instance form one alt group (one Top-K slot,
     a ladder ordered by S/D)
  -> pass 1 = S / D; pass 2 = measured priors (positive first, negative last)
  -> adaptive K, positive measured priors forced in beside it (<= K+4 rows)
  -> coverage: fusible regions, code seams with >= 3 kernels, inefficient kernels

Constants. c0_p is the median of the shortest 5% kernel durations of phase p:
even under CUDA graphs every kernel still pays launch / ramp / tail, and the
cheapest kernels of the trace measure exactly that floor. Bandwidth and peaks
are measured values when given (--hbm-tbps / --peak-tflops-*, or the
GEAK_HBM_TBPS / GEAK_PEAK_TFLOPS_BF16 / GEAK_PEAK_TFLOPS_FP8 environment),
otherwise datasheet values per gfx.

Dataflow graph. Nodes are kernels, an edge means "reads the other's output".
Explicit tensor ids are used when the table has them (input_tensor_ids /
output_tensor_ids). Otherwise operands are recovered from the row's kernel-level
shapes (argument names say which operands are written; workspace buffers are
ignored) and every input is joined to the nearest earlier writer with the same
element count and dtype family; such edges are marked inferred. Element counts
rather than exact shapes are compared because kernels routinely read reshaped
views ([M, H] written, [M*H/128, 128] read). A kernel without kernel-level
shapes (module-wrapper or layer-level shapes are not kernel shapes) is assumed
to write only tensors of its own module, so it blocks inference only inside
that module; a collective without operands is assumed to write the layer's
hidden state (it reduces the residual stream).

Classification. One class per row decides barrier, op tags and difficulty:
stage column and parent operator first (the operator wins a disagreement, which
is logged), whole kernel-name tokens last, never substrings. Pure data movement
(copy / contiguous / fill) is timed but never classified, counted or matched.

Barriers. Attention bodies, collectives, linear-attention / SSM bodies, MoE
expert GEMMs and LARGE GEMMs (time share >= 5% and compute share >= 10%; a GEMM
whose FLOPs cannot be computed and that takes >= 5% of the step stays a barrier).
Small GEMMs are not barriers.

Efficiency. For shaped GEMM / elementwise / quant / norm kernels:
bytes = reads + writes (weights, activations, outputs, scales), ideal =
max(bytes / BW, FLOPs / PEAK, c0_p), eff = ideal / measured. eff < 0.3 is
inefficient and its headroom (measured - ideal) is recoverable by a rewrite.

Windows (all enumerated, none exclusive):
  serial          adjacent dependent rows on one stream; <= 4 compute kernels,
                  5 with an existing kernel, 8 inside one code seam; a barrier
                  only at an end and only when an existing kernel covers it all.
  sibling         small GEMMs a few rows apart reading the same activation.
  parallel        overlapping small kernels on different streams.
  epilogue_forward a quant / cast / elementwise / norm consumer folded into the
                  epilogue of the producer G names for its input, when no kernel
                  in between writes that input; the producer may be a barrier.
  fan_out         a producer whose output is read in different forms by >= 2
                  consumers (raw by one, through a quant / cast / norm prologue
                  by another): the producer writes every form at once.
  seam_rewrite    a code seam (module forward) with >= 3 non-barrier kernels in
                  one step, rewritten to its minimum (1 GEMM + 1 fused kernel).
                  Kernels of the sub-module that owns an adjacent barrier are
                  that barrier's own prologue / epilogue and stay with it.
Windows with identical members across kinds are one window (the more specific
kind is kept and the others recorded in also_kinds).

Existing kernel. Covered only when every compute member is tagged, a sanitized
catalog kernel carries the window's tags at a compatible quant dtype and the
shape/semantics check passes (quant granularity, batched, second GEMM, no extra
heavy op). A window containing a kernel that already is a catalog fused kernel
keeps only the residual benefit of the members outside it.

Saving (per step, then x layers, capped at the members' time minus the ideal
time of the fused result):
  launch  (n_before - n_after) x c0_p; n_after = 1 for serial / parallel /
          epilogue_forward / fan_out, 1 GEMM (+1 split unless a split / layout
          kernel already follows) for sibling, 2 (1 without a GEMM) for
          seam_rewrite
  memory  intermediate bytes no longer written / read, / BW
  eff     sum of headroom of inefficient members x rho (0.5 self-authored,
          0.8 with an existing efficient kernel)
S = sum_p w_p x saved_p and expected_e2e_gain_pct = 100 S / R. Without shapes
only launches are counted and the window is flagged lower_bound.

Gates (by expected gain, not share): gain >= 0.3%, or >= 0.15% with an existing
kernel, or a positive measured prior hit; and estimated speedup > 1.1x.
0.1% <= gain below the gate goes to the candidate pool (stacking fill-ins, not
Top-K).

Alternative groups. Entries that eliminate the same kernel instance (a quant
foldable into an all-reduce epilogue, into an adjacent kernel, or into a seam
rewrite) are one alt group = one Top-K slot, with a ladder ordered by S/D
(an entry chosen by a measured prior is rung 0). Apply-back walks the ladder.

Difficulty D (1.0 with an existing kernel, else the hardest member):
  elementwise / quant / cast / RoPE / act_and_mul        1.3
  norm; epilogue_forward / fan_out                       1.5
  layout / KV-cache write                                1.8
  sibling / seam_rewrite / small GEMM member             2.0
  topk / sort / gather, unknown                          2.2
  collective, large GEMM, attention, linear-attn body    5.0
A negative measured prior multiplies D by 3; such an entry is never forced and
sorts last.

Passes and K. Pass 1 = S / D; pass 2 = positive priors by measured gain x seam
match, then pass-1 order; merged by "in both passes" then rank sum. K =
clamp(floor(budget x 0.7 / t_item), 8, 16) with --fusion-budget-s, else --top-k.
Positive priors with effective gain >= the noise band whose seam is present are
forced in without a K slot (<= K+4 rows). On a table blind to a required phase,
a prior that does not name its phases is not forced.

Coverage. Every fusible region, every code seam with >= 3 non-barrier kernels
and every inefficient kernel is in a candidate or carries a per-item reason;
an inefficient kernel no window contains is handed off
(handoff:HeadKernel/TuningSkillset).
"""
import argparse
import json
import math
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fusion_catalog as catalog_lib  # noqa: E402
import fusion_priors as priors_lib  # noqa: E402

ADJACENT_GAP_US = 50.0
LAUNCH_US = 5.0              # c0 fallback when a phase has no timed rows
C0_FRACTION = 0.05           # c0_p = median of the shortest 5% of phase p
LARGE_TIME_SHARE = 0.05
LARGE_COMPUTE_SHARE = 0.10
MIN_SPEEDUP = 1.1
GAIN_GATE_PCT = 0.3
EXISTING_GAIN_GATE_PCT = 0.15
POOL_GAIN_PCT = 0.1
INEFFICIENT_EFF = 0.3
RHO_SELF = 0.5
RHO_EXISTING = 0.8
TINY_US = 20.0
# Descriptive only (reported in `optional`): where a window's time goes.
LAUNCH_FRACTION_MIN = 0.10
MEMORY_FRACTION_MIN = 0.25
TINY_FRACTION_MIN = 0.50
SERIAL_MAX = 4
EXISTING_SERIAL_MAX = SERIAL_MAX + 1
SEAM_SERIAL_MAX = 8
SEAM_REWRITE_MIN = 3
SIBLING_LOOKAHEAD = 3
FINAL_K = 8
K_MIN = 8
K_MAX = 16
APPLYBACK_BUDGET_SHARE = 0.7
ITEM_MINUTES = 45.0
SCREEN_PASSES = 2
FORCED_PRIOR_EXTRA = 4
NOISE_BAND_PCT = 1.0
PRIOR_MATCH_MIN = 0.5
NEGATIVE_PRIOR_FACTOR = 3.0
LADDER_MAX = 6
EXISTING_DIFFICULTY = 1.0
SIBLING_DIFFICULTY = 2.0
SEAM_REWRITE_DIFFICULTY = 2.0
FORWARD_DIFFICULTY = 1.5
NEAR_MISS_LIMIT = 12
REGION_FLOOR_US = 5.0
HANDOFF_REASON = "handoff:HeadKernel/TuningSkillset"

# Datasheet values used only when the box has not been measured.
PEAK_TFLOPS = {
    "gfx942": {"bf16": 1307.0, "fp8": 2615.0},
    "gfx950": {"bf16": 2500.0, "fp8": 5000.0},
}
HBM_TBPS = {"gfx942": 5.3, "gfx950": 8.0}

FORWARD_KINDS = ("epilogue_forward", "fan_out")
# When two kinds produce the same member set, the more specific form is kept.
KIND_PRIORITY = ("fan_out", "epilogue_forward", "sibling", "seam_rewrite", "serial",
                 "parallel")
# Classes whose kernels get a bytes / ideal / efficiency estimate.
EFF_CLASSES = {"gemm", "elementwise", "quant", "activation", "norm", "cast", "rope"}
# Consumers that can ride in a producer's epilogue / prologue.
FORWARD_CONSUMERS = {"quant", "cast", "elementwise", "norm"}
PROLOGUE_CLASSES = {"quant", "cast", "norm"}
HARDEST_DIFFICULTY = 5.0
UNKNOWN_DIFFICULTY = 2.2
CLASS_DIFFICULTY = {
    "elementwise": 1.3, "quant": 1.3, "cast": 1.3, "rope": 1.3, "activation": 1.3,
    "norm": 1.5,
    "layout": 1.8, "kv_cache": 1.8,
    "gemm": SIBLING_DIFFICULTY,  # a small GEMM; a large one is a barrier (5)
    "topk": 2.2,
    "unknown": UNKNOWN_DIFFICULTY,
    "allreduce": HARDEST_DIFFICULTY, "attention": HARDEST_DIFFICULTY,
    "linear_attn": HARDEST_DIFFICULTY, "moe_gemm": HARDEST_DIFFICULTY,
}
BARRIER_CLASSES = ("allreduce", "attention", "linear_attn", "moe_gemm")
LIGHT_TAGS = {"norm", "add_residual", "quant", "rope", "kv_cache", "cast",
              "layout", "activation"}
HEAVY_TAGS = {"gemm", "moe", "allreduce", "attention"}
# Tags a GEMM earns by position, not by name.
_POSITION_TAGS = {"gemm_prologue", "gemm_epilogue"}
# Region dtype is the quant format; activation dtypes (bf16/fp16) would wrongly
# exclude fp8 kernels, whose inputs are bf16 too.
_QUANT_DTYPES = {"fp8", "fp8_blockscale", "fp4"}

STAGE_CLASS = {
    "norm": "norm", "rope": "rope", "kv_cache": "kv_cache", "quant": "quant",
    "activation": "activation", "topk": "topk", "gemm": "gemm",
    "attn": "attention", "attention": "attention", "linear_attn": "linear_attn",
    "communication": "allreduce", "collective": "allreduce", "moe": "moe",
    "expert_gemm": "moe_gemm", "elementwise": "elementwise",
}
# Stages that only name a family; operator / name refine them without a conflict.
_FAMILY_STAGES = {"elementwise": None, "moe": {"topk", "moe_gemm"},
                  "linear_attn": {"linear_attn", "layout", "elementwise", "norm"}}

_DATA_MOVE_OPS = {"aten::copy_", "aten::contiguous", "aten::clone", "aten::_to_copy",
                  "aten::to", "aten::fill_", "aten::zero_", "aten::zeros",
                  "aten::zeros_like", "aten::new_zeros", "aten::empty"}
# Name evidence for pure data movement is deliberately narrow: a fused
# split/reshape/cat "..._contiguous_kernel" does real layout work.
_DATA_MOVE_NAMES = ("direct_copy*", "copy_kernel", "memcpy*", "memset*",
                    "fill_buffer*", "fillbuffer*", "fill_kernel")
_GEMM_OPS = {"aten::mm", "aten::addmm", "aten::bmm", "aten::matmul", "aten::linear",
             "aten::baddbmm"}
_ELEMENTWISE_OPS = {"aten::sigmoid", "aten::mul", "aten::add", "aten::sub", "aten::div",
                    "aten::gt", "aten::lt", "aten::ge", "aten::le", "aten::eq",
                    "aten::exp", "aten::neg", "aten::where", "aten::softplus",
                    "aten::silu", "aten::tanh", "aten::relu", "aten::mul_", "aten::add_"}
_COLLECTIVE_NAMES = ("allreduce*", "all_reduce*", "cross_device_reduce*", "quickreduce",
                     "rccl*", "nccl*", "all_gather*", "allgather*", "reduce_scatter*")
_ATTENTION_NAMES = ("attention*", "fmha*", "flash_attn*", "flash_fwd*", "paged_attention*",
                    "mla_decode*", "mla_prefill*", "attn_fwd*", "unified_attention*",
                    "mha_fwd*", "mha_batch*", "pa_fwd*", "kentry_fmha*")
_LINEAR_BODY_NAMES = ("chunk_gated_delta_rule*", "chunk_fwd_kernel_o", "recompute_w_u*",
                      "fused_recurrent_gated_delta_rule*", "causal_conv1d*",
                      "chunk_local_cumsum*", "kkt_solve*", "solve_tril*",
                      "chunk_scaled_dot_kkt*", "selective_scan*", "ssd_*",
                      "mamba_chunk*")
_MOE_GEMM_NAMES = ("fmoe*", "ck_moe_stage*", "kernel_moe_gemm", "moe_gemm*",
                   "fused_moe*", "moe_stage*", "fused_experts*")
_TOPK_NAMES = ("topk*", "top_k", "moe_sorting*", "moe_sort", "append_shared_experts*",
               "moe_fused_gate", "grouped_topk")
_ELEMENTWISE_NAMES = ("elementwise*", "vectorized_elementwise*", "unrolled_elementwise*",
                      "sigmoid*", "softplus", "gating", "gdn_gating*")
_LAYOUT_NAMES = ("split", "reshape", "cat", "concat", "transpose", "permute",
                 "shuffle", "layout")
_LAYER_PREFIX = re.compile(r"^model\.layers\.\d+\.?")


# ---------------------------------------------------------------- table rows
def _name(row):
    return str(row.get("raw_name") or row.get("short_name") or "")


def _duration(row):
    return float(row.get("duration_us") or 0.0)


def _layers(row):
    try:
        return max(int(row.get("pattern_layer_count") or 1), 1)
    except (TypeError, ValueError):
        return 1


def load_rows(table):
    """Flatten pattern/phase tables. One representative layer is scaled later."""
    tables = table.get("tables") if isinstance(table, dict) else None
    rows = []
    if tables:
        for spec in tables:
            nlayers = spec.get("pattern_layer_count") or 1
            for row in spec.get("rows") or []:
                item = dict(row)
                item["phase"] = item.get("phase") or spec.get("phase") or "unresolved"
                item["pattern_id"] = item.get("pattern_id") or spec.get("pattern_id")
                item["pattern_layer_count"] = nlayers
                if not item.get("layer_total_us"):
                    item["layer_total_us"] = spec.get("layer_total_us")
                rows.append(item)
    elif isinstance(table, dict) and table.get("rows"):
        rows = [dict(row) for row in table["rows"]]
    for index, row in enumerate(rows):
        row.setdefault("row_id", "row-%d" % index)
        row.setdefault("phase", "unresolved")
        row["duration_us"] = _duration(row)
    return rows


def phase_e2e_us(rows):
    """Full-step kernel time. A representative layer is multiplied by its layer count."""
    seen = {}
    for row in rows:
        key = (row.get("phase"), row.get("pattern_id"))
        total = row.get("layer_total_us")
        if total is None:
            continue
        seen[key] = float(total) * _layers(row)
    if seen:
        out = {}
        for (phase, _), total in seen.items():
            out[phase] = out.get(phase, 0.0) + total
        return out
    out = {}
    for row in rows:
        phase = row.get("phase")
        out[phase] = out.get(phase, 0.0) + _duration(row) * _layers(row)
    return out


def phase_weights(rows, workload):
    """Forwards of each phase one request pays, or {} when the workload is unknown.

    A prefill step that carried T tokens costs ISL/T of itself per request; a
    decode step over B requests costs 1/B per request, times OSL-1 forwards.
    Without step metadata this is the legacy 1 prefill + OSL-1 decode forwards.
    """
    workload = workload or {}
    isl, osl = workload.get("isl"), workload.get("osl")
    if not isl or not osl:
        return {}
    weights = {"prefill": 1.0, "decode": float(max(int(osl) - 1, 0))}
    for row in rows:
        phase = row.get("phase")
        if phase == "prefill" and row.get("step_input_tokens"):
            weights["prefill"] = float(isl) / float(row["step_input_tokens"])
        elif phase == "decode" and row.get("step_batch_size"):
            weights["decode"] = float(max(int(osl) - 1, 0)) / float(row["step_batch_size"])
    return weights


def _effective_weights(rows, e2e_us, weights):
    """Weights used for request-level numbers: each phase step once without a workload."""
    out = {phase: 1.0 for phase in e2e_us}
    for phase, weight in (weights or {}).items():
        out[phase] = weight
    return out


# ------------------------------------------------------- hardware and c0
def hardware(gfx="gfx942", hbm_tbps=None, peak_bf16=None, peak_fp8=None):
    """HBM bandwidth and dtype peaks: CLI value, then environment, then datasheet."""
    base = PEAK_TFLOPS.get(gfx, PEAK_TFLOPS["gfx942"])

    def pick(cli, env_key, default):
        for value, source in ((cli, "cli"), (os.environ.get(env_key), "env")):
            try:
                if value not in (None, "") and float(value) > 0:
                    return float(value), source
            except (TypeError, ValueError):
                continue
        return float(default), "datasheet"

    bw, bw_src = pick(hbm_tbps, "GEAK_HBM_TBPS", HBM_TBPS.get(gfx, HBM_TBPS["gfx942"]))
    bf16, bf16_src = pick(peak_bf16, "GEAK_PEAK_TFLOPS_BF16", base["bf16"])
    fp8, fp8_src = pick(peak_fp8, "GEAK_PEAK_TFLOPS_FP8", base["fp8"])
    return {"gfx": gfx, "hbm_tbps": bw, "bw_bytes_per_us": bw * 1e6,
            "peak_tflops": {"bf16": bf16, "fp8": fp8},
            "source": {"hbm": bw_src, "bf16": bf16_src, "fp8": fp8_src}}


def phase_c0_us(rows):
    """Per-phase fixed cost of one kernel: median of the shortest 5% durations."""
    by_phase = {}
    for row in rows:
        duration = _duration(row)
        if duration > 0:
            by_phase.setdefault(row.get("phase"), []).append(duration)
    out = {}
    for phase, durations in by_phase.items():
        durations.sort()
        count = max(1, int(math.ceil(C0_FRACTION * len(durations))))
        out[phase] = float(statistics.median(durations[:count]))
    return out


# ---------------------------------------------------------- classification
def _canonical_op(row):
    parent = row.get("parent_operator")
    if isinstance(parent, dict):
        return str(parent.get("canonical_op") or "")
    return str(parent or "")


def _module(row):
    """Module path that contains the kernel, or "" when the table has none."""
    evidence = row.get("semantic_evidence")
    if isinstance(evidence, dict) and evidence.get("contained_by"):
        return str(evidence["contained_by"])
    op = _canonical_op(row)
    if op.startswith("model."):
        return op
    return str(row.get("module") or "")


def _module_rel(row):
    """Module under the layer ("mlp.gate"), "" for the layer root, None if unknown."""
    path = _module(row)
    if not path:
        return None
    return _LAYER_PREFIX.sub("", path)


def _seam_of(row):
    """Code seam a row belongs to: first module under the layer, or None."""
    rel = _module_rel(row)
    if not rel:
        return None
    return rel.split(".")[0]


def _op_class(op):
    """Class implied by the parent operator, or None when it says nothing."""
    if not op or op == "unresolved" or op.startswith("model.") or "::" not in op:
        return None
    lowered = op.lower()
    if lowered in _DATA_MOVE_OPS:
        return "data_move"
    if lowered in _GEMM_OPS:
        return "gemm"
    if lowered in _ELEMENTWISE_OPS:
        return "elementwise"
    return _name_class(catalog_lib.name_tokens(op.split("::", 1)[1]))


def _name_class(tokens):
    has = catalog_lib.has_phrase
    if has(tokens, *_DATA_MOVE_NAMES):
        return "data_move"
    if has(tokens, *_COLLECTIVE_NAMES):
        return "allreduce"
    if has(tokens, *_ATTENTION_NAMES):
        return "attention"
    if has(tokens, *_LINEAR_BODY_NAMES):
        return "linear_attn"
    if has(tokens, *_MOE_GEMM_NAMES):
        return "moe_gemm"
    tags = set(catalog_lib._op_tags(" ".join(t for t in tokens if t) or "_"))
    if "gemm" in tags:
        return "gemm"
    if has(tokens, *_TOPK_NAMES):
        return "topk"
    for cls in ("kv_cache", "norm", "rope", "activation", "quant", "cast"):
        if cls in tags:
            return cls
    if has(tokens, *_ELEMENTWISE_NAMES):
        return "elementwise"
    if has(tokens, *_LAYOUT_NAMES):
        return "layout"
    return None


def classify_row(row):
    """{cls, tags, source, conflict} for one row; cached on the row.

    Stage and parent operator first (the operator wins when both resolve and
    disagree), then whole-token name rules. Family stages (linear_attn, moe,
    elementwise) are refined by operator / name without a conflict.
    """
    cached = row.get("_cls")
    if cached is not None:
        return cached
    name = _name(row)
    tokens = catalog_lib.name_tokens(name)
    stage = str(row.get("stage") or "").strip().lower()
    stage_cls = STAGE_CLASS.get(stage)
    op = _canonical_op(row)
    op_cls = _op_class(op)
    name_cls = _name_class(tokens)
    conflict = ""
    if op_cls == "data_move" or (name_cls == "data_move" and stage_cls in (None, "elementwise")):
        cls, source = "data_move", "parent_operator" if op_cls == "data_move" else "name"
    elif stage_cls is None:
        cls = op_cls or name_cls or "unknown"
        source = "parent_operator" if op_cls else ("name" if name_cls else "none")
    elif stage_cls in _FAMILY_STAGES:
        allowed = _FAMILY_STAGES[stage_cls]
        refined = op_cls or name_cls
        if stage_cls == "linear_attn" and name_cls == "linear_attn":
            refined = "linear_attn"
        if stage_cls == "moe" and name_cls == "moe_gemm":
            refined = "moe_gemm"
        if refined is None:
            cls = {"elementwise": "elementwise", "moe": "unknown",
                   "linear_attn": "unknown"}[stage_cls]
            source = "stage"
        else:
            cls, source = refined, "stage+parent_operator" if op_cls else "stage+name"
            if allowed is not None and refined not in allowed:
                conflict = ("stage %r but %s says %s; %s wins" % (
                    stage, "parent operator %s" % op if op_cls else "name", refined,
                    "parent operator" if op_cls else "name refinement of a family stage"))
    else:
        cls, source = stage_cls, "stage"
        if op_cls and op_cls != stage_cls:
            cls, source = op_cls, "parent_operator"
            conflict = "stage %r but parent operator %s says %s; parent operator wins" % (
                stage, op, op_cls)
        elif name_cls and name_cls != stage_cls and name_cls != "data_move":
            conflict = "name suggests %s but stage says %s; stage wins" % (name_cls, stage_cls)
    tags = _class_tags(cls, name)
    result = {"cls": cls, "tags": tags, "source": source, "conflict": conflict,
              "stage": stage or None, "parent_operator": op or None}
    row["_cls"] = result
    return result


def _class_tags(cls, name):
    """Catalog op tags of a row. Heavy classes carry their class only; a light
    fused kernel also carries the light ops its name spells (add_rmsnorm_quant)."""
    if cls in ("data_move", "unknown"):
        return set()
    if cls == "gemm":
        return {"gemm"}
    if cls == "moe_gemm":
        return {"moe", "gemm"}
    if cls == "attention":
        return {"attention"}
    if cls == "linear_attn":
        return {"linear_attn"}
    if cls == "topk":
        return {"topk"}
    if cls == "elementwise":
        return {"elementwise"}
    if cls == "allreduce":
        return (catalog_lib.fused_trace_tags(name) & (LIGHT_TAGS | {"allreduce"})) | {"allreduce"}
    return (set(catalog_lib._op_tags(name)) & LIGHT_TAGS) | {cls}


def _is_data_move(row):
    return classify_row(row)["cls"] == "data_move"


def _is_unknown(row):
    return classify_row(row)["cls"] == "unknown"


def _tags(row):
    return set(classify_row(row)["tags"])


# --------------------------------------------------------------- shapes
def _dims(row):
    return (row.get("shape") or {}).get("input_dims") or []


def _types(row):
    return [str(item).lower() for item in ((row.get("shape") or {}).get("input_types") or [])]


def _elem_bytes(type_name):
    lowered = str(type_name or "").lower()
    if "float8" in lowered or "fp8" in lowered or "int8" in lowered:
        return 1
    if "int64" in lowered or "long" in lowered:
        return 8
    if "int" in lowered or "bool" in lowered:
        return 4 if "int" in lowered else 1
    if "bfloat16" in lowered or "half" in lowered or "16" in lowered:
        return 2
    if "float" in lowered or "32" in lowered:
        return 4
    return 2


def _dtype(row):
    blob = " ".join(_types(row)) + " " + _name(row).lower()
    if "fp8" in blob or "float8" in blob:
        return "fp8"
    return "bf16"


def gemm_dims(row):
    """{M, N, K, act, weight_bytes} from a GEMM row's recorded operands, or None.

    Shapes come in three layouts: A[M,K] @ B[K,N] (aten::mm), activation +
    weight [N,K] (+ output [M,N]) for aiter / triton kernels, and the legacy
    single [M,N,K] triple. A pair of identical operands is the layer-level
    fallback some tables record for a whole module, not a GEMM shape.
    """
    iface = _linear_interface(row)
    if iface:
        return iface
    dims = _dims(row)
    types = _types(row)
    for dim in dims:
        if (isinstance(dim, (list, tuple)) and len(dim) == 3 and len(dims) == 1
                and all(isinstance(v, (int, float)) and v > 1 for v in dim)):
            m, n, k = dim
            return {"M": m, "N": n, "K": k, "act": (m, k), "weight_bytes": n * k * 2}
    mats = [(index, list(dim)) for index, dim in enumerate(dims)
            if isinstance(dim, (list, tuple)) and len(dim) == 2
            and all(isinstance(v, int) and v > 0 for v in dim)]
    if len(mats) < 2:
        return None
    first, second = mats[0], mats[1]
    a, b = first[1], second[1]
    if a == b:
        return None
    op = _canonical_op(row).lower()
    if op in ("aten::mm", "aten::matmul") or (a[1] == b[0] and a[1] != b[1]):
        m, k, n = a[0], a[1], b[1]
        wtype = types[second[0]] if second[0] < len(types) else ""
        return {"M": m, "N": n, "K": k, "act": (m, k), "weight_bytes": n * k * _elem_bytes(wtype)}
    if a[1] == b[1]:
        for _, out in mats[2:]:
            if out[0] == b[0] and out[1] == a[0]:
                act, weight = second, first
                break
            if out[0] == a[0] and out[1] == b[0]:
                act, weight = first, second
                break
        else:
            return None
        m, k = act[1]
        n = weight[1][0]
        wtype = types[weight[0]] if weight[0] < len(types) else ""
        return {"M": m, "N": n, "K": k, "act": (m, k), "weight_bytes": n * k * _elem_bytes(wtype)}
    return None


def _linear_interface(row):
    """GEMM dims from a validated linear-layer interface recorded by the shape logger."""
    shape = row.get("shape") or {}
    schema = shape.get("logger_schema") or {}
    iface = schema.get("linear_interface") or {}
    if not isinstance(iface, dict) or not iface.get("validated"):
        return None
    try:
        m, n, k = (int((iface.get(axis) or {}).get("value")) for axis in ("M", "N", "K"))
    except (TypeError, ValueError):
        return None
    if min(m, n, k) <= 0:
        return None
    wtype = ""
    for tensor in schema.get("tensors") or []:
        if tensor.get("io") == "weight" and len(tensor.get("effective_shape") or tensor.get("shape") or []) == 2:
            wtype = str(tensor.get("dtype") or "")
            break
    return {"M": m, "N": n, "K": k, "act": (m, k), "weight_bytes": n * k * _elem_bytes(wtype)}


def gemm_flops(row):
    """GEMM FLOPs are 2*M*N*K. Anything else is unknown and stays a small op."""
    if classify_row(row)["cls"] != "gemm":
        return None
    dims = gemm_dims(row)
    if not dims:
        return None
    return 2.0 * dims["M"] * dims["N"] * dims["K"]


def quant_granularity(row):
    """per-group / per-token / per-tensor of a quant row: shapes first, then name."""
    dims, types = _dims(row), _types(row)
    out_k = None
    for index, dim in enumerate(dims):
        if index < len(types) and ("float8" in types[index] or "fp8" in types[index]) \
                and isinstance(dim, (list, tuple)) and dim:
            out_k = dim[-1]
            break
    if out_k:
        for index, dim in enumerate(dims):
            if index < len(types) and types[index] in ("float", "float32") \
                    and isinstance(dim, (list, tuple)):
                if not dim or (len(dim) == 1 and dim[0] == 1):
                    return "tensor"
                groups = dim[-1]
                if groups == 1:
                    # A row of <= 128 elements is one 1x128 group as well.
                    return "group" if out_k <= 128 else "token"
                if groups and out_k % groups == 0 and out_k // groups > 1:
                    return "group"
    return catalog_lib.quant_granularity(_name(row))


def classify_kernel(row, e2e_us, gfx, hw=None):
    duration = _duration(row)
    scaled = duration * _layers(row)
    time_share = scaled / e2e_us if e2e_us else 0.0
    flops = gemm_flops(row)
    compute_share = 0.0
    if flops and duration > 0:
        tflops = flops / (duration * 1e-6) / 1e12
        peaks = (hw or {}).get("peak_tflops") or PEAK_TFLOPS.get(gfx, PEAK_TFLOPS["gfx942"])
        peak = peaks[_dtype(row)]
        compute_share = tflops / peak if peak else 0.0
    large = time_share >= LARGE_TIME_SHARE and compute_share >= LARGE_COMPUTE_SHARE
    unresolved_large = (classify_row(row)["cls"] == "gemm" and not flops
                        and time_share >= LARGE_TIME_SHARE)
    return {
        "time_share": time_share,
        "compute_share": compute_share,
        "large": large or unresolved_large,
        "large_basis": ("time+compute" if large else
                        "time share, FLOPs unresolved" if unresolved_large else None),
        "flops": flops,
        "cls": classify_row(row)["cls"],
    }


def _barrier(row, classes=None):
    cls = classify_row(row)["cls"]
    if cls in BARRIER_CLASSES:
        return cls
    if cls == "gemm" and classes is not None and classes.get(row["row_id"], {}).get("large"):
        return "gemm"
    return None


def _member_difficulty(row, classes):
    cls = classify_row(row)["cls"]
    if cls == "gemm" and classes.get(row["row_id"], {}).get("large"):
        return HARDEST_DIFFICULTY
    return CLASS_DIFFICULTY.get(cls, UNKNOWN_DIFFICULTY)


# ------------------------------------------------- operands and dataflow
# Argument names that a kernel WRITES. Everything else it reads (weights and
# workspace buffers are kept apart: a weight is never produced in the layer and
# a zero-initialised workspace carries no data).
_OUT_ARG_NAMES = {"out", "output", "out_c", "residual_out", "y", "result", "scales",
                  "out_scale", "out_scales", "k_cache", "v_cache", "num_valid_ids"}
_TOPK_OUT_ARG_NAMES = {"topk_weights", "topk_indices", "topk_ids", "token_expert_indices"}
_WEIGHT_ARG_NAMES = {"weight", "bias", "mat2", "gamma", "beta"}
_WEIGHT_ARG_RE = re.compile(r"^w\d*(?:_|$)")
_WRAPPER_SHAPE_SOURCES = {"runtime_probe_wrapper", "parent_context", "module_wrapper",
                          "layer_wrapper"}


def _dtype_family(type_name):
    lowered = str(type_name or "").lower()
    if "float8" in lowered or "fp8" in lowered:
        return "fp8"
    if "bfloat16" in lowered or "bf16" in lowered:
        return "bf16"
    if "half" in lowered or "float16" in lowered or "fp16" in lowered:
        return "fp16"
    if "int8" in lowered:
        return "int8"
    if "int" in lowered or "long" in lowered:
        return "int"
    if "bool" in lowered:
        return "bool"
    if "float" in lowered or "fp32" in lowered:
        return "fp32"
    return lowered or "?"


def _arg_names(row):
    shape = row.get("shape") or {}
    schema = (shape.get("kernel_shape") or {}).get("operator_schema") or {}
    names = {}
    for index, arg in enumerate(schema.get("arguments") or []):
        if isinstance(arg, dict):
            names[int(arg.get("index", index))] = str(arg.get("name") or "")
    return names


def _operand_role(name, index, op, inplace):
    lowered = name.lower()
    if not lowered:
        return "output" if inplace and index == 0 else "input"
    if "buf" in lowered or "workspace" in lowered:
        return "scratch"
    if inplace and lowered == "self":
        return "output"
    if (lowered in _OUT_ARG_NAMES or lowered.startswith("out_") or lowered.endswith("_out")
            or lowered.startswith("sorted_")):
        return "output"
    if lowered in _TOPK_OUT_ARG_NAMES and "topk" in op:
        return "output"
    if lowered in _WEIGHT_ARG_NAMES or lowered.endswith("_weight") or _WEIGHT_ARG_RE.match(lowered):
        return "weight"
    return "input"


def _operand(role, dims, type_name):
    numel = 1
    for value in dims:
        numel *= int(value)
    return {"role": role, "shape": [int(v) for v in dims], "dtype": _dtype_family(type_name),
            "numel": numel, "bytes": numel * _elem_bytes(type_name)}


def _tensor_dims(value):
    return (isinstance(value, (list, tuple)) and len(value) > 0
            and all(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in value))


def row_operands(row):
    """Kernel-level operands [{role, shape, dtype, numel, bytes}] or [] when unknown.

    A shape recorded by a module / layer wrapper describes the wrapper's
    interface, not one kernel's operands, unless the wrapper is a single linear
    layer whose interface the logger validated. Such shapes are not used.
    """
    cached = row.get("_operands")
    if cached is not None:
        return cached
    shape = row.get("shape") or {}
    out = []
    source = str(shape.get("source") or "")
    schema = shape.get("logger_schema") or {}
    op = _canonical_op(row).lower()
    if source in _WRAPPER_SHAPE_SOURCES:
        if _linear_interface(row):
            for tensor in schema.get("tensors") or []:
                dims = tensor.get("effective_shape") or tensor.get("shape")
                role = {"input": "input", "output": "output", "weight": "weight"}.get(
                    tensor.get("io"), "input")
                if _tensor_dims(dims):
                    out.append(_operand(role, dims, tensor.get("dtype")))
        row["_operands"] = out
        return out
    dims, types = _dims(row), _types(row)
    tensors = [(i, d) for i, d in enumerate(dims) if _tensor_dims(d)
               and i < len(types) and types[i] not in ("", "scalar")]
    # A module path as the operator with every operand identical is the
    # layer-level fallback some tables record, not a kernel's operands.
    if (op.startswith("model.") and len(tensors) >= 2
            and len({tuple(d) for _, d in tensors}) == 1):
        row["_operands"] = []
        return []
    names = _arg_names(row)
    inplace = op.startswith("aten::") and op.endswith("_")
    for index, d in tensors:
        out.append(_operand(_operand_role(names.get(index, ""), index, op, inplace), d,
                            types[index]))
    if out and classify_row(row)["cls"] == "gemm":
        gdims = gemm_dims(row)
        if gdims and any(item["shape"] == [gdims["M"], gdims["N"], gdims["K"]] for item in out):
            # Legacy single [M, N, K] triple: spell out the three operands.
            m, n, k = gdims["M"], gdims["N"], gdims["K"]
            dtype = types[0] if types else "bf16"
            out = [_operand("input", [m, k], dtype), _operand("weight", [n, k], dtype),
                   _operand("output", [m, n], dtype)]
        elif gdims:
            m, n, k = gdims["M"], gdims["N"], gdims["K"]
            have_out = False
            for item in out:
                if item["role"] == "output":
                    have_out = True
                elif item["shape"] in ([n, k], [k, n]) and item["shape"] != [m, k]:
                    item["role"] = "weight"
                elif item["shape"] == [m, n] and item["shape"] != [m, k]:
                    item["role"] = "output"
                    have_out = True
                elif item["dtype"] == "fp32":
                    item["role"] = "weight" if item["numel"] != m * (k // 128 or 1) else "input"
            if not have_out:
                act = next((i for i in out if i["role"] == "input"), None)
                out.append(_operand("output", [m, n], "bfloat16" if not act else
                                    {"fp8": "bfloat16"}.get(act["dtype"], act["dtype"])))
    elif out and op.startswith("aten::") and not inplace \
            and not any(item["role"] == "output" for item in out):
        # An ATen functional op returns a fresh tensor shaped like its largest
        # input (broadcast) in the dtype of its first input.
        largest = max(out, key=lambda item: item["numel"])
        out.append(_operand("output", largest["shape"], out[0]["dtype"]))
    row["_operands"] = out
    return out


def _sig(item):
    return (item["numel"], item["dtype"])


def _layer_hidden(spec):
    """(numel, dtype, bytes) of the layer's hidden state from the table's layer_io."""
    io = (spec or {}).get("layer_io") or {}
    for key in ("input", "output"):
        tensor = io.get(key) or {}
        dims = tensor.get("effective_shape") or tensor.get("shape")
        if _tensor_dims(dims):
            return _operand("output", dims, tensor.get("dtype"))
    return None


def build_dataflow(rows, table=None):
    """Per-layer dataflow graph G: {producers, consumers, edges, stats}.

    producers[row_id] = [edge], consumers[row_id] = [edge]; an edge is
    {src, dst, sig, bytes, inferred}. Explicit ids (input_tensor_ids /
    output_tensor_ids) give exact edges; otherwise each input joins the nearest
    earlier writer with the same element count and dtype family (inferred).
    """
    hidden = {}
    specs = table.get("tables") if isinstance(table, dict) else None
    for spec in specs or []:
        item = _layer_hidden(spec)
        if item:
            hidden[(spec.get("phase"), spec.get("pattern_id"))] = item
    groups = {}
    for row in rows:
        groups.setdefault((row.get("phase"), row.get("pattern_id")), []).append(row)
    producers, consumers, edges = {}, {}, []
    writes = {}
    stats = {"explicit": 0, "inferred": 0, "assumed_collective_outputs": 0,
             "unresolved_inputs": 0, "unshaped_rows": 0}
    for key, group in groups.items():
        ordered = sorted(group, key=lambda r: (r.get("device_seq_index", r.get("pos", 0)),
                                               r.get("timestamp") or 0))
        for row in ordered:
            if not row_operands(row) and not _is_data_move(row):
                stats["unshaped_rows"] += 1
            ins, outs = _row_io(row, hidden.get(key))
            writes[row["row_id"]] = outs
            if outs and not row_operands(row):
                stats["assumed_collective_outputs"] += 1
        for index, row in enumerate(ordered):
            explicit_in = row.get("input_tensor_ids")
            if isinstance(explicit_in, list) and explicit_in:
                for tensor_id in explicit_in:
                    for other in reversed(ordered[:index]):
                        if tensor_id in (other.get("output_tensor_ids") or []):
                            _add_edge(edges, producers, consumers, other, row,
                                      ("id", tensor_id), 0, False)
                            stats["explicit"] += 1
                            break
                continue
            ins, _outs = _row_io(row, hidden.get(key))
            seen = set()
            for item in ins:
                sig = _sig(item)
                if sig in seen:
                    continue
                seen.add(sig)
                found = None
                for other in reversed(ordered[:index]):
                    if any(_sig(o) == sig for o in writes.get(other["row_id"]) or []):
                        found = other
                        break
                    if (not writes.get(other["row_id"]) and not _is_data_move(other)
                            and _module_rel(other) is not None
                            and _module_rel(other) == _module_rel(row)):
                        # An unshaped kernel of the reader's own module may be
                        # the writer; the edge cannot be inferred past it.
                        break
                if found is None:
                    stats["unresolved_inputs"] += 1
                    continue
                _add_edge(edges, producers, consumers, found, row, sig, item["bytes"], True)
                stats["inferred"] += 1
    return {"producers": producers, "consumers": consumers, "edges": edges,
            "writes": writes, "stats": stats}


def _row_io(row, hidden):
    ops = row_operands(row)
    if not ops:
        if classify_row(row)["cls"] == "allreduce" and hidden:
            # A collective without recorded operands reduces the residual
            # stream: it reads and writes the layer's hidden state.
            return [], [dict(hidden, role="output")]
        return [], []
    ins = [o for o in ops if o["role"] == "input"]
    outs = [o for o in ops if o["role"] == "output"]
    return ins, outs


def _add_edge(edges, producers, consumers, src, dst, sig, nbytes, inferred):
    for edge in producers.get(dst["row_id"], []):
        if edge["src"] == src["row_id"] and edge["sig"] == sig:
            return
    edge = {"src": src["row_id"], "dst": dst["row_id"], "sig": sig, "bytes": nbytes,
            "inferred": inferred}
    edges.append(edge)
    producers.setdefault(dst["row_id"], []).append(edge)
    consumers.setdefault(src["row_id"], []).append(edge)


# ------------------------------------------------------------ efficiency
def kernel_cost(row, hw, c0):
    """{bytes, flops, ideal_us, eff, headroom_us, inefficient} or None without shapes."""
    cls = classify_row(row)["cls"]
    if cls not in EFF_CLASSES:
        return None
    duration = _duration(row)
    if duration <= 0:
        return None
    ops = row_operands(row)
    flops = 0.0
    peak = hw["peak_tflops"]["bf16"]
    if cls == "gemm":
        dims = gemm_dims(row)
        if not dims:
            return None
        m, n, k = dims["M"], dims["N"], dims["K"]
        fp8 = _dtype(row) == "fp8" or dims["weight_bytes"] == n * k
        act_bytes = 1 if fp8 else 2
        scales = sum(o["bytes"] for o in ops if o["dtype"] == "fp32")
        nbytes = m * k * act_bytes + dims["weight_bytes"] + m * n * 2 + scales
        flops = 2.0 * m * n * k
        peak = hw["peak_tflops"]["fp8" if fp8 else "bf16"]
    else:
        if not ops:
            return None
        nbytes = sum(o["bytes"] for o in ops if o["role"] != "scratch")
    ideal = max(nbytes / hw["bw_bytes_per_us"], flops / (peak * 1e6) if flops else 0.0, c0)
    eff = min(ideal / duration, 1.0)
    return {"bytes": nbytes, "flops": flops, "ideal_us": ideal, "eff": eff,
            "headroom_us": max(0.0, duration - ideal), "inefficient": eff < INEFFICIENT_EFF}


# ------------------------------------------------------------------ windows
def _adjacent(left, right):
    if left.get("stream") != right.get("stream"):
        return False
    if (left.get("layer_id") is not None and right.get("layer_id") is not None
            and left.get("layer_id") != right.get("layer_id")):
        return False
    if left.get("timestamp") is None or right.get("timestamp") is None:
        return True
    end = float(left["timestamp"]) + _duration(left)
    return 0 <= float(right["timestamp"]) - end <= ADJACENT_GAP_US


def _ordered_groups(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row.get("phase"), row.get("pattern_id"), row.get("stream")), []).append(row)
    for key, group in groups.items():
        groups[key] = sorted(group, key=lambda row: (
            row.get("device_seq_index", row.get("pos", 0)), row.get("timestamp") or 0))
    return groups


def _runs(rows):
    """Adjacent same-stream runs, numbered, per (phase, pattern, stream)."""
    out = []
    for group in _ordered_groups(rows).values():
        current = []
        for row in group:
            if current and _adjacent(current[-1], row):
                current.append(row)
            else:
                if current:
                    out.append(current)
                current = [row]
        if current:
            out.append(current)
    return out


def _serial_chains(rows):
    """Every contiguous window of an adjacent run with 2..SEAM_SERIAL_MAX compute kernels."""
    chains = []
    for run_index, run in enumerate(_runs(rows)):
        for start in range(len(run)):
            compute = 0
            for end in range(start, len(run)):
                if not _is_data_move(run[end]):
                    compute += 1
                if compute > SEAM_SERIAL_MAX:
                    break
                if compute >= 2 and not _is_data_move(run[end]):
                    chains.append(("serial", run[start:end + 1], run_index))
    return chains


def _overlaps(left, right):
    if left.get("timestamp") is None or right.get("timestamp") is None:
        return False
    left_end = float(left["timestamp"]) + _duration(left)
    right_end = float(right["timestamp"]) + _duration(right)
    return float(left["timestamp"]) < right_end and float(right["timestamp"]) < left_end


def _parallel_chains(rows, classes):
    small = [row for row in rows if not classes[row["row_id"]]["large"]
             and not _barrier(row, classes) and not _is_data_move(row)]
    by_phase = {}
    for row in small:
        by_phase.setdefault(row.get("phase"), []).append(row)
    chains = []
    for group in by_phase.values():
        ordered = sorted(group, key=lambda row: float(row.get("timestamp") or 0))
        used = set()
        for index, row in enumerate(ordered):
            if row["row_id"] in used or row.get("timestamp") is None:
                continue
            window = [row]
            streams = {row.get("stream")}
            for other in ordered[index + 1:]:
                if other["row_id"] in used:
                    continue
                if other.get("stream") in streams:
                    continue
                if all(_overlaps(member, other) for member in window):
                    window.append(other)
                    streams.add(other.get("stream"))
            if len(window) >= 2 and len(streams) >= 2:
                chains.append(("parallel", window, None))
                used.update(member["row_id"] for member in window)
    return chains


def _sibling_chains(rows, classes):
    """Small GEMMs a few rows apart on one stream reading the same activation.

    Same activation = the same M x K operand; independent = neither reads the
    other's [M, N] output; same seam = the same module under the layer when
    the table knows it (in_proj_qkvz / in_proj_ba are both linear_attn,
    mlp.gate / mlp.shared_expert_gate are both mlp).
    """
    chains = []
    for run_index, run in enumerate(_runs(rows)):
        gemms = [(index, row) for index, row in enumerate(run)
                 if classify_row(row)["cls"] == "gemm" and not classes[row["row_id"]]["large"]]
        used = set()
        for pos, (index, row) in enumerate(gemms):
            if row["row_id"] in used:
                continue
            dims = gemm_dims(row)
            if not dims:
                continue
            group = [row]
            last = index
            for other_index, other in gemms[pos + 1:]:
                if other_index - last > SIBLING_LOOKAHEAD:
                    break
                between = run[last + 1:other_index]
                if any(_barrier(item, classes) for item in between):
                    break
                other_dims = gemm_dims(other)
                if not other_dims or other_dims["act"] != dims["act"]:
                    continue
                seams = {_seam_of(member) for member in group + [other]} - {None}
                if len(seams) > 1:
                    continue
                # An earlier member whose output has the activation's shape
                # (N == K) could be what `other` reads; without module evidence
                # placing both under one seam that is not a sibling.
                if any((gemm_dims(member) or {}).get("N") == other_dims["K"] for member in group) \
                        and not seams:
                    continue
                group.append(other)
                last = other_index
            if len(group) >= 2:
                chains.append(("sibling", group, run_index))
                used.update(member["row_id"] for member in group)
    return chains


def _code_seam(row):
    """Code seam (module forward) of a row: first module under the layer, "" for
    the layer's own forward, None when the table has no module evidence."""
    rel = _module_rel(row)
    if rel is None:
        return None
    return rel.split(".")[0] if rel else ""


def _forward_chains(rows, graph, classes):
    """epilogue_forward and fan_out windows read off the dataflow graph."""
    by_id = {row["row_id"]: row for row in rows}
    order = {}
    for group in _layer_orders(rows).values():
        for pos, row in enumerate(group):
            order[row["row_id"]] = pos
    chains, seen = [], set()
    for row in rows:
        info = classify_row(row)
        if info["cls"] not in FORWARD_CONSUMERS or _barrier(row, classes):
            continue
        for edge in graph["producers"].get(row["row_id"], []):
            producer = by_id.get(edge["src"])
            if producer is None or _is_data_move(producer) or _is_unknown(producer):
                continue
            readers = [by_id[e["dst"]] for e in graph["consumers"].get(producer["row_id"], [])
                       if e["sig"] == edge["sig"] and e["dst"] in by_id]
            prologues = [r for r in readers if classify_row(r)["cls"] in PROLOGUE_CLASSES
                         and not _barrier(r, classes)]
            raw = [r for r in readers if r not in prologues]
            if info["cls"] in PROLOGUE_CLASSES and len(readers) >= 2 and (raw or len(prologues) >= 2):
                kind, members = "fan_out", [producer] + prologues
            else:
                kind, members = "epilogue_forward", [producer, row]
            members = sorted({m["row_id"]: m for m in members}.values(),
                             key=lambda m: order.get(m["row_id"], 0))
            key = (kind, tuple(m["row_id"] for m in members))
            if key in seen or len(members) < 2:
                continue
            seen.add(key)
            chains.append((kind, members, None))
    return chains


def _layer_orders(rows):
    """Rows of each (phase, pattern) layer table in device order, all streams."""
    groups = {}
    for row in rows:
        groups.setdefault((row.get("phase"), row.get("pattern_id")), []).append(row)
    for key, group in groups.items():
        groups[key] = sorted(group, key=lambda r: (r.get("device_seq_index", r.get("pos", 0)),
                                                   r.get("timestamp") or 0))
    return groups


def _seam_segments(rows, classes):
    """Maximal adjacent runs of non-barrier kernels inside one code seam.

    Kernels of the sub-module that owns the barrier bordering a run are that
    barrier's own prologue / epilogue (e.g. the expert block's sorting and
    input quant) and are left to it. Returns [(segment, run_index, seam)].
    """
    out = []
    for run_index, run in enumerate(_runs(rows)):
        segment, seam, before = [], None, None

        def close(after):
            body = list(segment)
            while body and _is_data_move(body[-1]):
                body.pop()
            while body and _is_data_move(body[0]):
                body.pop(0)
            for edge, step in ((before, 0), (after, -1)):
                if edge is None:
                    continue
                owner = _module_rel(edge)
                while owner and body and _module_rel(body[step]) == owner:
                    body.pop(step)
                    while body and _is_data_move(body[step]):
                        body.pop(step)
            if seam is not None and sum(1 for r in body if not _is_data_move(r)) >= SEAM_REWRITE_MIN:
                out.append((body, run_index, seam))

        for row in run:
            if _is_data_move(row):
                if segment:
                    segment.append(row)
                continue
            if _barrier(row, classes):
                close(row)
                segment, seam, before = [], None, row
                continue
            row_seam = _code_seam(row)
            if row_seam is None or (segment and row_seam != seam):
                close(None)
                segment, before = [], None
            if row_seam is None:
                seam = None
                continue
            seam = row_seam
            segment.append(row)
        close(None)
    return out


def _seam_rewrite_chains(rows, classes):
    return [("seam_rewrite", segment, run_index) for segment, run_index, _seam
            in _seam_segments(rows, classes)]


# ----------------------------------------------------------------- savings
def _n_after(kind, compute, ctx, existing, run_index=None):
    """Kernels left after the fusion (section 5)."""
    if kind == "sibling":
        if existing:
            return 1
        last = compute[-1]
        group = ctx["layer_orders"].get((last.get("phase"), last.get("pattern_id")), [])
        ids = [r["row_id"] for r in group]
        if last["row_id"] in ids:
            index = ids.index(last["row_id"])
            follow = [r for r in group[index + 1:index + 1 + SIBLING_LOOKAHEAD]
                      if not _is_data_move(r)]
            if follow and classify_row(follow[0])["cls"] == "layout":
                # A split / layout kernel already reads the outputs: it takes
                # the merged output's slices, no extra split kernel.
                return 1
        return 2
    if kind == "seam_rewrite":
        return 2 if any(classify_row(m)["cls"] == "gemm" for m in compute) else 1
    if existing:
        return 1
    # Without a covering kernel, every GEMM in the window stays its own launch and
    # the remaining small ops fuse into one more kernel; a window is only a
    # one-kernel fusion when nothing in it is a GEMM. Collapsing several GEMMs
    # needs the sibling or seam_rewrite form, which price that merge themselves.
    gemms = sum(1 for m in compute if classify_row(m)["cls"] == "gemm")
    if not gemms:
        return 1
    return gemms + (1 if len(compute) > gemms else 0)


def window_savings(kind, members, ctx, existing, classes):
    """Per-step saving of one window (launch + memory + efficiency headroom)."""
    phase = members[0].get("phase")
    c0 = ctx["c0"].get(phase, LAUNCH_US)
    bw = ctx["hw"]["bw_bytes_per_us"]
    compute = [m for m in members if not _is_data_move(m)]
    ids = {m["row_id"] for m in members}
    producer_kept = None
    if kind in FORWARD_KINDS:
        producer_kept = compute[0]["row_id"]
    counted = []
    for member in compute:
        if _barrier(member, classes) and not existing:
            # Barrier time counts only when an existing kernel replaces it.
            continue
        counted.append(member)
    n_before = len(compute)
    n_after = _n_after(kind, compute, ctx, existing)
    launch_saved = max(0, n_before - n_after) * c0
    costs = ctx["costs"]
    graph = ctx["graph"]
    mem_bytes = 0
    for member in compute:
        for edge in graph["producers"].get(member["row_id"], []):
            if edge["src"] not in ids or not edge["bytes"]:
                continue
            mem_bytes += edge["bytes"]          # the consumer no longer reads it
            outside = [e for e in graph["consumers"].get(edge["src"], [])
                       if e["sig"] == edge["sig"] and e["dst"] not in ids]
            if not outside:
                mem_bytes += edge["bytes"]      # nor does the producer write it
    if kind == "sibling":
        shared = [o for o in row_operands(compute[0]) if o["role"] == "input"]
        if shared:
            mem_bytes += (len(compute) - 1) * shared[0]["bytes"]
    mem_saved = mem_bytes / bw
    rho = RHO_EXISTING if existing else RHO_SELF
    baseline = ctx["baseline_rows"]
    eff_saved = 0.0
    shaped = 0
    nbytes = 0.0
    flops_us = 0.0
    for member in counted:
        cost = costs.get(member["row_id"])
        if cost is None:
            continue
        shaped += 1
        nbytes += cost["bytes"]
        if cost["flops"]:
            peak = ctx["hw"]["peak_tflops"]["fp8" if _dtype(member) == "fp8" else "bf16"]
            flops_us += cost["flops"] / (peak * 1e6)
        if cost["inefficient"] and member["row_id"] not in baseline:
            # A kernel that already is a baseline fused kernel keeps only the
            # residual benefit of the members outside it.
            eff_saved += cost["headroom_us"] * rho
    lower_bound = shaped < len([m for m in counted if not _barrier(m, classes)]) or not counted
    counted_time = sum(_duration(m) for m in counted)
    n_after_counted = n_after - (1 if producer_kept and producer_kept not in
                                 {m["row_id"] for m in counted} else 0)
    ideal_fused = max(max(n_after_counted, 0) * c0, max(nbytes - mem_bytes, 0.0) / bw, flops_us)
    cap = max(0.0, counted_time - ideal_fused)
    raw = launch_saved + mem_saved + eff_saved
    saved = max(0.0, min(raw, cap))
    fused = counted_time - saved
    durations = [_duration(m) for m in compute]
    chain = sum(durations)
    launches = [min(c0, d) for d in durations]
    return {
        "model": "launch+memory+efficiency" if kind != "sibling" else "sibling_merge",
        "chain_us": chain,
        "counted_us": counted_time,
        "counted_row_ids": [m["row_id"] for m in counted],
        "launch_us": sum(launches),
        "launch_fraction": sum(launches) / chain if chain else 0.0,
        "memory_saved_us": mem_saved,
        "memory_fraction": mem_saved / chain if chain else 0.0,
        "tiny_us": sum(d for d in durations if d < TINY_US),
        "fused_us": fused,
        "speedup": counted_time / max(fused, 1e-3) if counted_time else 1.0,
        "gate_speedup": counted_time / max(fused, 1e-3) if counted_time else 1.0,
        "saved_us": saved,
        "launch_saved_us": launch_saved,
        "mem_saved_us": mem_saved,
        "eff_saved_us": eff_saved,
        "uncapped_saved_us": raw,
        "cap_us": cap,
        "ideal_fused_us": ideal_fused,
        "n_before": n_before,
        "n_after": n_after,
        "rho": rho,
        "c0_us": c0,
        "lower_bound": bool(lower_bound),
    }


def _optional_hit(est):
    """Descriptive: where the window's time goes (not a gate)."""
    tiny_fraction = est["tiny_us"] / est["chain_us"] if est["chain_us"] else 0.0
    hits = []
    if est["launch_fraction"] >= LAUNCH_FRACTION_MIN:
        hits.append("launch")
    if est["memory_fraction"] >= MEMORY_FRACTION_MIN:
        hits.append("memory")
    if tiny_fraction >= TINY_FRACTION_MIN:
        hits.append("tiny")
    if est.get("eff_saved_us"):
        hits.append("efficiency")
    return hits


# ---------------------------------------------------------- existing kernels
def _catalog_kernels(catalog):
    """Sanitized catalog entries (non-kernels dropped, legacy tags re-derived)."""
    if not catalog:
        return [], []
    kernels = catalog.get("kernels") if isinstance(catalog, dict) else catalog
    tagger = catalog.get("tagger") if isinstance(catalog, dict) else None
    return catalog_lib.sanitize_kernels(kernels, tagger)


def _chain_tags(compute, kind="serial"):
    """Op tags a covering kernel must carry, with GEMM prologue/epilogue by position."""
    tags = set()
    for index, member in enumerate(compute):
        member_tags = _tags(member)
        tags |= member_tags
        if kind == "serial" and "gemm" in member_tags:
            if index > 0:
                tags.add("gemm_prologue")
            if index < len(compute) - 1:
                tags.add("gemm_epilogue")
    return tags


def _sanity_reject(kernel, compute, tags):
    """Why a tag-covering kernel still does not fit the window, or ""."""
    name = kernel["name"]
    extra_heavy = (set(kernel.get("op_tags") or []) & HEAVY_TAGS) - tags
    if extra_heavy:
        return "also runs %s the window does not contain" % "/".join(sorted(extra_heavy))
    wanted = {quant_granularity(m) for m in compute if "quant" in _tags(m)} - {None}
    have = catalog_lib.quant_granularity(name)
    if have and wanted and have not in wanted:
        return "quant granularity %s, window is %s" % (have, "/".join(sorted(wanted)))
    gemms = [m for m in compute if "gemm" in _tags(m)]
    precisions = catalog_lib.gemm_precisions(name)
    if len(precisions) >= 2 and len(gemms) < 2:
        return "runs a second GEMM the window does not have"
    if len(gemms) >= 2 and "gemm" in (kernel.get("op_tags") or []) and len(precisions) < 2:
        return "one GEMM; the window has %d" % len(gemms)
    batched = catalog_lib.has_phrase(name, "batched*", "bmm")
    window_batched = any(any(isinstance(d, (list, tuple)) and len(d) == 3 for d in _dims(m))
                         for m in gemms)
    if gemms and batched != window_batched:
        return "batched=%s, window batched=%s" % (batched, window_batched)
    return ""


def _sibling_matches(compute, kernels):
    want = sorted("a8w8" if _dtype(m) == "fp8" else "a16w16" for m in compute)
    hits = []
    for kernel in kernels:
        if "gemm" not in (kernel.get("op_tags") or []):
            continue
        precisions = catalog_lib.gemm_precisions(kernel["name"])
        if sorted(precisions) != want:
            continue
        if catalog_lib.has_phrase(kernel["name"], "moe*", "batched*"):
            continue
        hits.append(kernel)
    return hits


def existing_matches(compute, kernels, kind="serial"):
    """(matches, tags, reason): catalog kernels that cover the whole window.

    Coverage needs every compute member tagged, the catalog tags to contain
    the window's tags at a compatible quant dtype, and the shape/semantics
    check to pass. aiter first, tightest cover first.
    """
    tags = _chain_tags(compute, kind)
    unknown = [_name(m) for m in compute if not _tags(m)]
    if unknown:
        return [], sorted(tags), "no cover: unclassified member(s) %s" % ", ".join(
            _short(name) for name in unknown)
    if not kernels:
        return [], sorted(tags), "no catalog"
    if kind == "sibling":
        hits = _sibling_matches(compute, kernels)
        reason = ("dual-GEMM kernel with matching operand precisions"
                  if hits else "no catalog kernel runs these GEMM precisions on one activation")
    else:
        if len(tags - _POSITION_TAGS) < 2:
            return [], sorted(tags), "fewer than two ops to fuse"
        dtypes = set()
        for member in compute:
            if "quant" in _tags(member):
                dtypes |= set(catalog_lib._dtype_tags(_name(member))) & _QUANT_DTYPES
                if _dtype(member) == "fp8":
                    dtypes.add("fp8")
        covering = catalog_lib.covers(kernels, sorted(tags), sorted(dtypes))
        hits, rejected = [], []
        for kernel in covering:
            why = _sanity_reject(kernel, compute, tags)
            if why:
                rejected.append("%s (%s)" % (kernel["name"], why))
            else:
                hits.append(kernel)
        if hits:
            reason = "op_tags cover %s at %s; shape/semantics check passed" % (
                "+".join(sorted(tags)), "/".join(sorted(dtypes)) or "any dtype")
        elif rejected:
            reason = "tags covered but rejected: " + "; ".join(rejected[:3])
        else:
            reason = "no catalog kernel carries %s" % "+".join(sorted(tags))
    wants_mrope = any(catalog_lib.has_phrase(_name(m), "mrope*") for m in compute)
    hits.sort(key=lambda k: ("aiter" not in (k.get("sources") or []),
                             catalog_lib.has_phrase(k["name"], "mrope*") != wants_mrope,
                             len(set(k["op_tags"]) - tags), k["name"]))
    return [{"name": k["name"], "op_tags": sorted(k["op_tags"]),
             "extra_tags": sorted(set(k["op_tags"]) - catalog_lib._expand_implied(tags)),
             "modules": (k.get("modules") or [])[:1]} for k in hits[:5]], sorted(tags), reason


def baseline_fused(rows, kernels, e2e_us, weights=None, denominator=0.0):
    """Trace kernels that already are a catalog fused kernel (>= 2 fused ops)."""
    found = {}
    by_name = {}
    for row in rows:
        if _is_data_move(row):
            continue
        tags = _tags(row) & (LIGHT_TAGS | {"allreduce", "topk"})
        if len(tags) < 2:
            continue
        by_name.setdefault(_name(row), []).append(row)
    if not by_name or not kernels:
        return []
    probe = [dict(kernel) for kernel in kernels]
    observed = catalog_lib.mark_observed(probe, by_name)
    weights = weights or {}
    for trace_name, names in observed.items():
        rows_here = by_name[trace_name]
        tags = _tags(rows_here[0]) & (LIGHT_TAGS | {"allreduce", "topk"})
        fits = [k for k in probe if k["name"] in names
                and len(set(k["op_tags"]) & (LIGHT_TAGS | {"allreduce", "topk"})) >= 2]
        if not fits:
            continue
        fits.sort(key=lambda k: ("aiter" not in (k.get("sources") or []),
                                 len(set(k["op_tags"]) ^ tags), k["name"]))
        best = fits[0]["name"]
        slot = found.setdefault(best, {
            "catalog_kernel": best, "also_matches": [k["name"] for k in fits[1:4]],
            "op_tags": sorted(tags), "trace_kernel": trace_name, "phases": {},
            "row_ids": [], "request_share": 0.0})
        for row in rows_here:
            slot["row_ids"].append(row["row_id"])
            phase = row.get("phase")
            share = _duration(row) * _layers(row) / (e2e_us.get(phase) or 1.0)
            slot["phases"][phase] = slot["phases"].get(phase, 0.0) + share
            if denominator:
                slot["request_share"] += (_duration(row) * _layers(row)
                                          * weights.get(phase, 1.0) / denominator)
    return sorted(found.values(), key=lambda item: -sum(item["phases"].values()))


# ------------------------------------------------------------------ records
def _record(kind, members, run_index, classes, kernels, weights, ctx):
    phase = members[0].get("phase")
    compute = [member for member in members if not _is_data_move(member)]
    matches, tags, reason = existing_matches(
        compute, kernels, "sibling" if kind == "sibling" else "serial")
    existing = bool(matches)
    est = window_savings(kind, members, ctx, existing, classes)
    nlayers = _layers(members[0])
    weight = weights.get(phase, 1.0)
    large = any(classes[member["row_id"]]["large"] for member in members)
    invalid = ""
    barriers = [index for index, member in enumerate(compute) if _barrier(member, classes)]
    seams = {_seam_of(m) for m in compute}
    seam_chain = len(seams) == 1 and None not in seams
    if kind == "serial":
        if any(0 < index < len(compute) - 1 for index in barriers):
            invalid = "barrier_inside"
        elif barriers and not existing:
            invalid = "barrier_without_existing_kernel"
        elif len(compute) > SERIAL_MAX:
            if seam_chain:
                pass
            elif not existing:
                invalid = "too_long_without_existing_kernel"
            elif len(compute) > EXISTING_SERIAL_MAX:
                invalid = "too_long_outside_one_seam"
    elif kind in FORWARD_KINDS and any(index > 0 for index in barriers):
        invalid = "barrier_consumer"
    unknown = [_name(member) for member in compute if _is_unknown(member)]
    consumers = compute[1:] if kind in FORWARD_KINDS else compute
    member_d = max([_member_difficulty(member, classes) for member in consumers] or [1.0])
    if kind in FORWARD_KINDS:
        if existing:
            # epilogue_forward with an existing kernel is that kernel; fan_out
            # still needs the producer's multi-output variant.
            difficulty = EXISTING_DIFFICULTY if kind == "epilogue_forward" else FORWARD_DIFFICULTY
        elif barriers:
            difficulty = HARDEST_DIFFICULTY
        else:
            difficulty = max(FORWARD_DIFFICULTY, member_d)
    elif existing:
        difficulty = EXISTING_DIFFICULTY
    elif kind == "sibling":
        difficulty = SIBLING_DIFFICULTY
    elif kind == "seam_rewrite":
        difficulty = max(SEAM_REWRITE_DIFFICULTY, member_d)
    else:
        difficulty = member_d
    if kind in FORWARD_KINDS:
        elim = [m["row_id"] for m in compute[1:]]
        if existing:
            elim = [compute[0]["row_id"]] + elim
    else:
        elim = [m["row_id"] for m in compute]
    step_saved = est["saved_us"] * nlayers
    seam_names = sorted(s for s in seams if s)
    code_seam = seam_names[0] if seam_chain else None
    if kind == "seam_rewrite":
        code_seam = _code_seam(compute[0]) or "(layer)"
    baseline_members = sorted(set(m["row_id"] for m in compute) & ctx["baseline_rows"])
    return {
        "kind": kind,
        "also_kinds": [],
        "phase": phase,
        "phases": [phase],
        "pattern_id": members[0].get("pattern_id"),
        "run_index": run_index,
        "row_ids": [member["row_id"] for member in members],
        "compute_row_ids": [member["row_id"] for member in compute],
        "elim_row_ids": elim,
        "kernel_names": [_name(member) for member in members],
        "classes": [classify_row(member)["cls"] for member in members],
        "data_move_row_ids": [member["row_id"] for member in members if _is_data_move(member)],
        "barrier_row_ids": [compute[index]["row_id"] for index in barriers],
        "baseline_fused_row_ids": baseline_members,
        "modules": sorted({_module_rel(m) for m in compute if _module_rel(m)}),
        "code_seam": code_seam,
        "op_tags": tags,
        "has_large_op": large,
        "required_speedup": MIN_SPEEDUP,
        "optional": _optional_hit(est),
        "invalid": invalid,
        "passed": False,
        "gate": "reject",
        "difficulty": difficulty,
        "has_existing_kernel": existing,
        "matched_existing_kernels": matches,
        "existing_match_reason": reason,
        "unknown_kernels": unknown,
        "decode_saved_us": step_saved if phase == "decode" else 0.0,
        "step_saved_us": step_saved,
        "phase_weight": weight,
        "request_saved_us": step_saved * weight,
        "request_chain_us": est["chain_us"] * nlayers * weight,
        "_scaled_us": est["chain_us"] * nlayers,
        "_sig": _signature(kind, compute),
        "seam_descriptor": _seam_descriptor(kind, compute),
        **est,
    }


def _seam_descriptor(kind, compute):
    """A measured_priors seam descriptor for this window (for prior write-back)."""
    members = []
    for member in compute:
        info = classify_row(member)
        item = {"cls": info["cls"]}
        rel = _module_rel(member)
        if rel:
            item["module"] = rel
        op = info.get("parent_operator") or ""
        if info["cls"] == "elementwise" and op.startswith("aten::"):
            item["op"] = op
        members.append(item)
    return {"kinds": [kind], "max_span": len(compute) + 2, "members": members}


def _signature(kind, compute):
    parts = []
    for member in compute:
        info = classify_row(member)
        part = (info["cls"],) + tuple(sorted(info["tags"]))
        op = info.get("parent_operator") or ""
        if info["cls"] == "elementwise" and op.startswith("aten::"):
            part += (op,)
        if info["cls"] == "gemm":
            # Weight shape tells two GEMM seams apart (router [H,E] vs in_proj).
            dims = gemm_dims(member)
            part += (("N", dims["N"], "K", dims["K"]),) if dims else (("N", None),)
        parts.append(part)
    return (kind, tuple(parts))


def _group_seams(records, denominator):
    """Same op sequence at the same occurrence = one seam, across phases and
    layer patterns. Known, different module paths veto the merge."""
    buckets = {}
    for record in records:
        if record["invalid"]:
            continue
        buckets.setdefault((record["_sig"], record["phase"], record["pattern_id"]), []).append(record)
    by_key = {}
    for (sig, _phase, _pattern), items in buckets.items():
        items.sort(key=lambda item: item["_start"])
        for occurrence, record in enumerate(items):
            record["_occurrence"] = occurrence
            by_key.setdefault((sig, occurrence), []).append(record)
    out = []
    for (sig, occurrence), items in sorted(by_key.items(), key=lambda kv: str(kv[0])):
        groups = []
        for record in sorted(items, key=lambda item: -item["request_saved_us"]):
            module = record["code_seam"]
            for group in groups:
                if any((m["phase"], m["pattern_id"]) == (record["phase"], record["pattern_id"])
                       for m in group["members"]):
                    continue
                if module and group["modules"] and module not in group["modules"]:
                    continue
                group["members"].append(record)
                if module:
                    group["modules"].add(module)
                break
            else:
                groups.append({"members": [record], "modules": {module} if module else set()})
        for group in groups:
            out.append(_group_metrics(group["members"], denominator, sig, occurrence,
                                      sorted(group["modules"])))
    return out


def _group_metrics(members, denominator, sig=None, occurrence=0, modules=()):
    counted = sum(m["counted_us"] * _member_layers(m) * m["phase_weight"] for m in members)
    fused = sum(m["fused_us"] * _member_layers(m) * m["phase_weight"] for m in members)
    chain = sum(m["request_chain_us"] for m in members)
    saved = sum(m["request_saved_us"] for m in members)
    dominant = max(members, key=lambda m: m["request_chain_us"])
    existing = all(m["has_existing_kernel"] for m in members)
    sig = sig or members[0]["_sig"]
    seam_key = "%s|%s|#%d%s" % (sig[0], "+".join(part[0] for part in sig[1]), occurrence,
                                "|" + ",".join(modules) if modules else "")
    gain = 100.0 * saved / denominator if denominator else 0.0
    return {
        "members": members,
        "seam_key": seam_key,
        "request_saved_us": saved,
        "request_chain_us": chain,
        "gate_share": chain / denominator if denominator else 0.0,
        "expected_e2e_gain_pct": gain,
        "gate_min": (EXISTING_GAIN_GATE_PCT if existing else GAIN_GATE_PCT) / 100.0,
        "speedup": counted / fused if fused > 1e-9 else (counted / 1e-3 if counted else 1.0),
        "required_speedup": MIN_SPEEDUP,
        "optional": dominant["optional"],
        "has_existing_kernel": existing,
        "lower_bound": any(m["lower_bound"] for m in members),
    }


def _gate_of(group, prior_hit=False):
    """("pass" | "pool" | "reject", reason) of a seam (section 6)."""
    if any(m["invalid"] for m in group["members"]):
        return "reject", group["members"][0]["invalid"]
    gain = group["expected_e2e_gain_pct"]
    if group["speedup"] <= MIN_SPEEDUP:
        if group["lower_bound"] and gain >= POOL_GAIN_PCT:
            # A launch-only lower bound cannot prove the speedup gate either
            # way: such a window waits in the pool instead of being dropped.
            return "pool", "lower-bound speedup %.3f <= %.1fx; expected gain >= %.3f%%" % (
                group["speedup"], MIN_SPEEDUP, gain)
        return "reject", "speedup %.3f <= %.1fx" % (group["speedup"], MIN_SPEEDUP)
    if gain >= GAIN_GATE_PCT:
        return "pass", "expected gain %.3f%% >= %.2f%%" % (gain, GAIN_GATE_PCT)
    if group["has_existing_kernel"] and gain >= EXISTING_GAIN_GATE_PCT:
        return "pass", "existing kernel and expected gain %.3f%% >= %.2f%%" % (
            gain, EXISTING_GAIN_GATE_PCT)
    if prior_hit:
        return "pass", "positive measured prior %s" % prior_hit
    if gain >= POOL_GAIN_PCT:
        return "pool", "expected gain %.3f%% in [%.1f%%, gate)" % (gain, POOL_GAIN_PCT)
    return "reject", "below_gain_gate (expected gain %.3f%% < %.1f%%)" % (gain, POOL_GAIN_PCT)


def _collapse_identical(records):
    """One window per member set: the most specific kind wins, the others are
    recorded in `also_kinds` (the same fusion found by two enumerators)."""
    best = {}
    order = []
    for record in records:
        key = (record["phase"], record["pattern_id"], frozenset(record["compute_row_ids"]))
        if key not in best:
            best[key] = record
            order.append(key)
            continue
        kept = best[key]
        rank = KIND_PRIORITY.index
        if rank(record["kind"]) < rank(kept["kind"]) and not record["invalid"]:
            record["also_kinds"] = sorted(set(kept["also_kinds"]) | {kept["kind"]})
            best[key] = record
        elif record["kind"] != kept["kind"]:
            kept["also_kinds"] = sorted(set(kept["also_kinds"]) | {record["kind"]})
    return [best[key] for key in order]


def _member_layers(record):
    return record["_scaled_us"] / record["chain_us"] if record["chain_us"] else 1.0


def _heavy_end(row_id, by_id, classes):
    row = by_id[row_id]
    return bool(_barrier(row, classes)) or classify_row(row)["cls"] in ("gemm", "moe_gemm")


def _core_window(longest, subs, classes):
    """The shortest barrier-free form of a seam: the longest window with its
    barrier / GEMM ends trimmed (its fusible core, e.g. sigmoid+mul+quant of
    sigmoid+mul+quant+o_proj). When that core did not pass the gates, the
    shortest passing barrier-free sub-window."""
    if not subs:
        return None
    by_id = longest["_rows_by_id"]
    compute = list(longest["compute_row_ids"])
    while compute and _heavy_end(compute[0], by_id, classes):
        compute.pop(0)
    while compute and _heavy_end(compute[-1], by_id, classes):
        compute.pop()
    if len(compute) >= 2 and len(compute) < len(longest["compute_row_ids"]):
        for record in subs:
            if record["compute_row_ids"] == compute:
                return record
    free = [r for r in subs if not r["barrier_row_ids"]]
    if not free:
        return None
    return min(free, key=lambda r: (len(r["compute_row_ids"]), -r["request_saved_us"], r["_start"]))


def _dedupe(passed, classes=None):
    """Per adjacency run: longest feasible window + shortest barrier-free sub-window.

    Returns (kept, appendix) where appendix entries carry the reason.
    """
    classes = classes or {}
    kept, appendix = [], []
    by_run = {}
    for record in passed:
        if record["kind"] != "serial":
            kept.append(record)
            continue
        by_run.setdefault((record["phase"], record["pattern_id"], record["run_index"]), []).append(record)
    for pool in by_run.values():
        pool = list(pool)
        while pool:
            longest = max(pool, key=lambda r: (len(r["compute_row_ids"]), r["_group"]["request_saved_us"],
                                               r["request_saved_us"], -r["_start"]))
            rows = set(longest["row_ids"])
            subs = [r for r in pool if r is not longest and set(r["row_ids"]) <= rows]
            shortest = _core_window(longest, subs, classes)
            kept.append(longest)
            if shortest is not None:
                kept.append(shortest)
            for record in subs:
                if record is shortest:
                    continue
                record["appendix_reason"] = (
                    "overlapping sub-window of the seam's longest feasible window; kept "
                    "the longest and the shortest barrier-free window instead")
                record["_superseded_by"] = longest
                appendix.append(record)
            drop = {id(longest)} | {id(r) for r in subs}
            pool = [r for r in pool if id(r) not in drop]
    return kept, appendix


# -------------------------------------------------------------------- priors
def _model_family_of(table, model=""):
    if model:
        return priors_lib.model_family(model)
    if isinstance(table, dict):
        for spec in table.get("tables") or []:
            ctx = spec.get("structural_context")
            if isinstance(ctx, str):
                text = ctx
            else:
                text = json.dumps(ctx or {})
            family = priors_lib.model_family(text)
            if family:
                return family
        for key in ("model", "model_name", "trace_path", "patterns_path"):
            family = priors_lib.model_family(table.get(key) or "")
            if family:
                return family
    return ""


def load_all_priors(measured_path=None, index_path=None):
    priors = []
    if measured_path != "":
        priors.extend(priors_lib.load_measured_priors(
            measured_path or priors_lib.DEFAULT_MEASURED_PRIORS))
    if index_path != "":
        priors.extend(priors_lib.learned_measured_priors(
            index_path or priors_lib.DEFAULT_PRIORS_INDEX))
    return priors


def _member_pattern_hit(row, pattern):
    info = classify_row(row)
    if pattern.get("cls") and info["cls"] != pattern["cls"]:
        return False
    if pattern.get("op") and (info.get("parent_operator") or "") != pattern["op"]:
        return False
    if pattern.get("name") and not catalog_lib.has_phrase(_name(row), *pattern["name"]):
        return False
    rel = _module_rel(row)
    wanted = ([pattern["module"]] if pattern.get("module") else []) + list(pattern.get("module_any") or [])
    if wanted and rel:
        parts = rel.split(".")
        hit = False
        for module in wanted:
            want = module.split(".")
            if any(parts[i:i + len(want)] == want for i in range(len(parts) - len(want) + 1)):
                hit = True
        if not hit:
            return False
    return True


def seam_occurrences(prior, rows):
    """Where a structured prior's seam sits: [(phase, pattern, fused_ids, all_ids, confirmed)]."""
    seam = prior.get("seam") or {}
    members = seam.get("members") or []
    if not members:
        return []
    span = int(seam.get("max_span") or len(members) + 2)
    out = []
    for (phase, pattern, _stream), ordered in _ordered_groups(rows).items():
        for start, row in enumerate(ordered):
            if not _member_pattern_hit(row, members[0]):
                continue
            picked = [row]
            cursor = start
            for pattern_item in members[1:]:
                found = None
                for index in range(cursor + 1, min(len(ordered), start + span)):
                    if _member_pattern_hit(ordered[index], pattern_item):
                        found = index
                        break
                if found is None:
                    picked = None
                    break
                picked.append(ordered[found])
                cursor = found
            if picked:
                fused = [r["row_id"] for r, p in zip(picked, members) if p.get("role", "fused") != "context"]
                confirmed = all(_module_rel(r) for r, p in zip(picked, members)
                                if p.get("module") or p.get("module_any"))
                out.append((phase, pattern, fused, [r["row_id"] for r in picked], confirmed))
                break
    # Module-less rows (prefill tables often lack contained_by) match any
    # module pattern, so a seam that names modules counts only where module
    # evidence confirms it. Unconfirmed look-alikes are dropped even when no
    # confirmed occurrence exists: two small GEMMs sharing an activation look
    # alike in every module, and forcing the wrong pair onto the board would
    # crowd out a real measured seam.
    if any(p.get("module") or p.get("module_any") for p in members):
        out = [item for item in out if item[4]]
    return out


def _prior_applies(prior, family, gfx):
    if not prior.get("model_family") or not family or prior["model_family"] != family:
        return False
    return not prior.get("gfx") or gfx in prior["gfx"]


def _seam_match(entry, prior, occurrences):
    seam = prior.get("seam") or {}
    kinds = seam.get("kinds")
    if kinds and not ({entry["kind"]} | set(entry.get("also_kinds") or [])) & set(kinds):
        return 0.0
    if seam.get("members"):
        best = 0.0
        for member in entry["members"]:
            rows = set(member["compute_row_ids"])
            for phase, pattern, fused, _all, _confirmed in occurrences:
                if (phase, pattern) != (member["phase"], member["pattern_id"]):
                    continue
                # Half seam coverage (did the window take the measured seam?),
                # half precision (how much of the window is that seam?).
                want = set(fused)
                both = float(len(rows & want))
                if both:
                    best = max(best, 0.5 * both / len(want) + 0.5 * both / len(rows))
        return best
    phases = seam.get("phases") or prior.get("phases")
    if phases and not set(phases) & set(entry.get("phases") or [entry["phase"]]):
        return 0.0
    want = catalog_lib._expand_implied(seam.get("tags") or [])
    have = catalog_lib._expand_implied(set(entry["op_tags"]) - _POSITION_TAGS)
    if not want or not have:
        return 0.0
    return len(want & have) / float(len(want | have))


def _polarity(prior):
    polarity = str(prior.get("polarity") or "").lower()
    if polarity in ("positive", "negative"):
        return polarity
    return "negative" if (prior.get("measured_e2e_gain_pct") or 0.0) < 0 else "positive"


def match_priors(entries, priors, rows, family, gfx):
    """Annotate entries with prior matches; return per-prior seam status.

    Positive matches feed pass 2 (prior_score); a negative match marks the
    entry negative_prior, multiplies its difficulty by NEGATIVE_PRIOR_FACTOR
    and makes it sort last.
    """
    status = []
    for prior in priors:
        applies = _prior_applies(prior, family, gfx)
        polarity = _polarity(prior)
        occurrences = seam_occurrences(prior, rows) if prior.get("seam", {}).get("members") else []
        best = None
        for entry in entries:
            match = _seam_match(entry, prior, occurrences)
            if match < PRIOR_MATCH_MIN:
                continue
            if applies:
                entry.setdefault("prior_matches", []).append({
                    "prior_id": prior["prior_id"], "title": prior["title"],
                    "origin": prior["origin"], "source": prior["source"],
                    "polarity": polarity,
                    "measured_e2e_gain_pct": prior["measured_e2e_gain_pct"],
                    "seam_match": round(match, 3)})
            rank = (match, entry.get("gate", "pass") == "pass", entry["request_saved_us"])
            if best is None or rank > best[0]:
                best = (rank, entry)
        exists = bool(occurrences) or best is not None
        status.append({
            "prior": prior, "applies": applies, "polarity": polarity,
            "occurrences": occurrences, "seam_in_trace": exists,
            "best_entry": best[1] if best else None,
            "best_match": round(best[0][0], 3) if best else 0.0})
    for entry in entries:
        matches = entry.get("prior_matches") or []
        positive = [m for m in matches if m["polarity"] == "positive"]
        negative = [m for m in matches if m["polarity"] == "negative"]
        entry["prior_score"] = max([m["measured_e2e_gain_pct"] * m["seam_match"]
                                    for m in positive] or [0.0])
        if negative:
            entry["negative_prior"] = True
            entry["prior_polarity"] = "negative"
            base = entry.get("base_difficulty", entry["difficulty"])
            entry["difficulty"] = base * NEGATIVE_PRIOR_FACTOR
            entry["score"] = entry["request_saved_us"] / entry["difficulty"]
        elif positive:
            entry["prior_polarity"] = "positive"
    return status


# -------------------------------------------------- prior hits for the gate
def _group_prior_hit(group, positive, rows_cache):
    """prior_id of a positive measured prior (gain >= noise band) the seam matches."""
    members = group["members"]
    pseudo = {"kind": members[0]["kind"],
              "also_kinds": sorted({k for m in members for k in m.get("also_kinds") or []}),
              "members": members, "phases": sorted({m["phase"] for m in members}),
              "phase": members[0]["phase"],
              "op_tags": sorted({t for m in members for t in m["op_tags"]})}
    for prior in positive:
        occurrences = rows_cache.get(prior["prior_id"], [])
        if _seam_match(pseudo, prior, occurrences) >= PRIOR_MATCH_MIN:
            return prior["prior_id"]
    return None


# --------------------------------------------------------- alt groups


def _rung(index, entry):
    return {
        "rung": index,
        "entry_id": entry.get("entry_id"),
        "kind": entry["kind"],
        "kernel_names": entry["kernel_names"],
        "row_ids": entry["row_ids"],
        "all_row_ids": entry.get("all_row_ids") or entry["row_ids"],
        "phases": entry.get("phases") or [entry.get("phase")],
        "matched_existing_kernels": entry.get("matched_existing_kernels") or [],
        "expected_e2e_gain_pct": entry.get("expected_e2e_gain_pct"),
        "difficulty": entry["difficulty"],
        "seam_key": entry.get("seam_key"),
        "gate": entry.get("gate", "pass"),
        "lower_bound": bool(entry.get("lower_bound")),
        "negative_prior": bool(entry.get("negative_prior")),
        "seam_members": (entry.get("seam_descriptor") or {}).get("members") or [],
        "matched_prior_id": _matched_prior_id(entry),
    }


def _matched_prior_id(entry):
    """The measured prior this entry IS (forced by it, or an exact seam match):
    a later measurement of the entry belongs to that prior's history."""
    if entry.get("forced_by_prior"):
        return entry["forced_by_prior"]
    exact = [m for m in entry.get("prior_matches") or [] if m.get("seam_match", 0.0) >= 0.999
             and m.get("origin") == "measured_priors"]
    return exact[0]["prior_id"] if exact else None


def _attach_ladder(group_id, ladder, also_in=None, conflicts=None):
    for rung, entry in enumerate(ladder):
        entry.setdefault("alt_group", group_id)
        if entry.get("alt_group") == group_id:
            entry["alt_rung"] = rung
    head = ladder[0]
    head["alt_group"] = group_id
    head["alt_rung"] = 0
    rungs = []
    for index, entry in enumerate(ladder[:LADDER_MAX]):
        rung = _rung(index, entry)
        others = sorted(set((also_in or {}).get(id(entry), ())) - {group_id})
        if others:
            rung["also_in_groups"] = others
        clash = sorted(set((conflicts or {}).get(id(entry), ())) - {group_id})
        if clash:
            rung["conflicts_with_groups"] = clash
        rungs.append(rung)
    head["alt_ladder"] = rungs
    head["alt_group_size"] = len(ladder)


def build_alt_groups(entries, anchors=()):
    """Alternative groups (section 7.2): entries that eliminate the same kernel
    instance compete for one Top-K slot.

    Anchors (entries a measured prior chose) come first, then passing entries by
    S/D, then pool entries. Each anchor that no earlier anchor overlaps opens a
    group (its slot). The group's ladder is the anchor followed by every other
    non-anchor entry that eliminates one of the anchor's kernels, by S/D
    (passing first); such an entry can sit on several ladders (also_in_groups),
    and a rung that also eliminates another group's anchor kernels says so
    (conflicts_with_groups). Grouping against the anchor, not transitively,
    keeps two independent fusions that a long window happens to bridge in
    separate slots.
    """
    ordered = sorted(entries, key=lambda e: (bool(e.get("negative_prior")), e.get("gate") != "pass",
                                             -e["score"], -e.get("request_saved_us", 0.0),
                                             str(_entry_key(e))))
    sequence, seen = [], set()
    for entry in list(anchors) + ordered:
        if id(entry) not in seen:
            seen.add(id(entry))
            sequence.append(entry)
    covered, heads = set(), []
    for anchor in sequence:
        if id(anchor) in covered:
            continue
        elim = _entry_rows(anchor)
        heads.append(anchor)
        covered.add(id(anchor))
        covered.update(id(e) for e in ordered if _entry_rows(e) & elim)
    head_ids = {id(h) for h in heads}
    groups = []
    for head in heads:
        elim = _entry_rows(head)
        groups.append([head] + [e for e in ordered if id(e) not in head_ids and _entry_rows(e) & elim])
    ids = ["g%02d" % index for index in range(1, len(groups) + 1)]
    also_in, conflicts = {}, {}
    for group_id, ladder in zip(ids, groups):
        for entry in ladder:
            also_in.setdefault(id(entry), []).append(group_id)
    for entry in ordered:
        rows = _entry_rows(entry)
        conflicts[id(entry)] = [gid for gid, head in zip(ids, heads) if _entry_rows(head) & rows]
    for group_id, ladder in zip(ids, groups):
        for entry in ladder[1:]:
            entry.setdefault("alt_group", group_id)
    for group_id, ladder in zip(ids, groups):
        _attach_ladder(group_id, ladder, also_in, conflicts)
    return groups


def adaptive_k(budget_s=None, item_minutes=ITEM_MINUTES, top_k=FINAL_K):
    """K = clamp(floor(budget x 0.7 / t_item), 8, 16); --top-k without a budget."""
    try:
        budget = float(budget_s) if budget_s not in (None, "") else None
    except (TypeError, ValueError):
        budget = None
    if not budget or budget <= 0:
        return int(top_k)
    item_s = max(float(item_minutes or ITEM_MINUTES), 1e-6) * 60.0
    return int(min(K_MAX, max(K_MIN, math.floor(budget * APPLYBACK_BUDGET_SHARE / item_s))))


def phase_blind(table, rows, workload=None):
    """Required phases the table lacks (section 1.3)."""
    present = {row.get("phase") for row in rows}
    coverage = (table or {}).get("phase_coverage") if isinstance(table, dict) else None
    if isinstance(coverage, dict) and coverage.get("missing_required_phases") is not None:
        return sorted(set(coverage.get("missing_required_phases") or []) - present)
    required = set()
    if isinstance(coverage, dict) and coverage.get("required_phases"):
        required = set(coverage["required_phases"])
    else:
        osl = (workload or {}).get("osl")
        if osl and int(osl) > 1:
            required.add("decode")
        if (workload or {}).get("isl"):
            required.add("prefill")
    return sorted(required - present)


# ------------------------------------------------------------------- entries
def _entry(group_members, entry_id, group):
    members = sorted(group_members, key=lambda m: -m["request_saved_us"])
    primary = members[0]
    rows = []
    elim = []
    for member in members:
        rows.extend(member["row_ids"])
        elim.extend(member["elim_row_ids"])
    tags = set()
    for member in members:
        tags |= set(member["op_tags"])
    request_saved = sum(m["request_saved_us"] for m in members)
    difficulty = max(m["difficulty"] for m in members)
    unknown = sorted({name for m in members for name in m["unknown_kernels"]})
    gain = group.get("expected_e2e_gain_pct")
    return {
        "entry_id": entry_id,
        "seam_key": group["seam_key"],
        "kind": primary["kind"],
        "also_kinds": sorted({k for m in members for k in m.get("also_kinds") or []}),
        "phase": primary["phase"],
        "phases": sorted({m["phase"] for m in members}),
        "pattern_id": primary["pattern_id"],
        "pattern_ids": sorted({str(m["pattern_id"]) for m in members}),
        "candidate_id": primary.get("candidate_id"),
        "candidate_ids": [m.get("candidate_id") for m in members],
        "row_ids": list(primary["row_ids"]),
        "partner_row_ids": [row for m in members[1:] for row in m["row_ids"]],
        "all_row_ids": rows,
        "elim_row_ids": elim,
        "merged_members": [{
            "candidate_id": m.get("candidate_id"), "phase": m["phase"],
            "pattern_id": m["pattern_id"], "row_ids": m["row_ids"],
            "step_saved_us": round(m["step_saved_us"], 3),
            "phase_weight": m["phase_weight"],
            "request_saved_us": round(m["request_saved_us"], 3),
            "launch_saved_us": round(m["launch_saved_us"], 3),
            "mem_saved_us": round(m["mem_saved_us"], 4),
            "eff_saved_us": round(m["eff_saved_us"], 3),
            "n_before": m["n_before"], "n_after": m["n_after"],
            "lower_bound": m["lower_bound"]} for m in members],
        "members": members,
        "kernel_names": primary["kernel_names"],
        "classes": primary["classes"],
        "data_move_row_ids": primary["data_move_row_ids"],
        "baseline_fused_row_ids": sorted({r for m in members for r in m["baseline_fused_row_ids"]}),
        "op_tags": sorted(tags),
        "e2e_share": group["gate_share"],
        "seam_share": group["gate_share"],
        "gate_share": group["gate_share"],
        "gate_min": group["gate_min"],
        "gate": group.get("gate", "pass"),
        "gate_reason": group.get("gate_reason", ""),
        "has_large_op": any(m["has_large_op"] for m in members),
        "required_speedup": group["required_speedup"],
        "optional": group["optional"],
        "difficulty": difficulty,
        "base_difficulty": difficulty,
        "has_existing_kernel": all(m["has_existing_kernel"] for m in members),
        "matched_existing_kernels": primary["matched_existing_kernels"],
        "existing_match_reason": primary["existing_match_reason"],
        "unknown_kernels": unknown,
        "decode_saved_us": sum(m["decode_saved_us"] for m in members),
        "step_saved_us": primary["step_saved_us"],
        "request_saved_us": request_saved,
        "expected_e2e_gain_pct": round(gain, 4) if gain is not None else None,
        "lower_bound": any(m["lower_bound"] for m in members),
        "negative_prior": False,
        "prior_polarity": None,
        "score": request_saved / difficulty if difficulty else 0.0,
        "score_basis": "request",
        "chain_us": primary["chain_us"],
        "launch_fraction": primary["launch_fraction"],
        "memory_fraction": primary["memory_fraction"],
        "launch_saved_us": primary["launch_saved_us"],
        "mem_saved_us": primary["mem_saved_us"],
        "eff_saved_us": primary["eff_saved_us"],
        "n_before": primary["n_before"],
        "n_after": primary["n_after"],
        "speedup": group["speedup"],
        "saved_us": primary["saved_us"],
        "saving_model": primary["model"],
        "code_seam": primary["code_seam"],
        "modules": sorted({mod for m in members for mod in m["modules"]}),
        "seam_descriptor": primary.get("seam_descriptor"),
    }


def _entry_rows(entry):
    """Kernel instances an entry eliminates or rewrites (its conflict set)."""
    rows = entry.get("elim_row_ids")
    if rows:
        return set(rows)
    return set(entry.get("all_row_ids") or entry.get("row_ids") or ())


def _entry_key(entry):
    return entry.get("entry_id") or entry.get("candidate_id") or tuple(entry.get("row_ids") or ())


def _resolve_conflicts(ranked):
    """Keep ranked order; an entry sharing a kernel with a kept one is dropped."""
    kept, taken = [], set()
    for item in ranked:
        rows = _entry_rows(item)
        if rows & taken:
            continue
        kept.append(item)
        taken |= rows
    return kept


def _sort_key(item):
    return (bool(item.get("negative_prior")), -item["score"], -item.get("request_saved_us", 0.0),
            str(_entry_key(item)))


def score_pass(entries, top_k=FINAL_K):
    """Pass 1: Score = request-level saved / difficulty; negative priors last."""
    return _resolve_conflicts(sorted(entries, key=_sort_key))[:top_k]


def prior_pass(entries, top_k=FINAL_K):
    """Pass 2: positive measured priors first (gain x seam match), then pass-1 order."""
    order = {id(item): index for index, item in enumerate(sorted(entries, key=_sort_key))}
    ordered = sorted(entries, key=lambda item: (bool(item.get("negative_prior")),
                                                -(item.get("prior_score") or 0.0), order[id(item)]))
    return _resolve_conflicts(ordered)[:top_k]


def merge_passes(passes, top_k=FINAL_K, forced=None, fallback=None, pool=None):
    """In both passes beats in one; then the smaller rank sum wins.

    A rank beyond K counts as missing (K+1), so the passes may be full
    rankings and the list still fills to K when entries conflict. `pool`
    adds entries no pass ranked (they sort last). `forced` entries are placed
    first and do not take one of the K slots. `fallback(item, taken)` may
    return a replacement (the entry's remaining disjoint sub-window) when an
    entry loses kernels to a better one.
    """
    missing = top_k + 1
    by_key = {}
    for pass_index, ranked in enumerate(passes):
        for rank, item in enumerate(ranked, 1):
            slot = by_key.setdefault(_entry_key(item), {"item": item, "ranks": [missing] * len(passes)})
            slot["ranks"][pass_index] = rank if rank <= top_k else missing
    for item in pool or []:
        by_key.setdefault(_entry_key(item), {"item": item, "ranks": [missing] * len(passes)})
    merged = []
    for slot in by_key.values():
        item = dict(slot["item"])
        item["pass_ranks"] = slot["ranks"]
        item["pass_hits"] = sum(1 for rank in slot["ranks"] if rank != missing)
        item["rank_sum"] = sum(slot["ranks"])
        merged.append(item)
    # A negative measured prior sorts last whatever its ranks (section 8.2).
    merged.sort(key=lambda item: (bool(item.get("negative_prior")), -item["pass_hits"],
                                  item["rank_sum"], -item.get("score", 0.0),
                                  str(_entry_key(item))))
    kept, taken = [], set()
    forced_keys = set()
    for item in forced or []:
        rows = _entry_rows(item)
        if rows & taken:
            continue
        slot = by_key.get(_entry_key(item))
        record = dict(item)
        if slot:
            record["pass_ranks"] = slot["ranks"]
            record["pass_hits"] = sum(1 for rank in slot["ranks"] if rank != missing)
            record["rank_sum"] = sum(slot["ranks"])
        else:
            record.setdefault("pass_ranks", [missing] * len(passes))
            record.setdefault("pass_hits", 0)
            record.setdefault("rank_sum", missing * len(passes))
        kept.append(record)
        forced_keys.add(_entry_key(item))
        taken |= rows
    ranked_count = 0
    for item in merged:
        if ranked_count == top_k:
            break
        if _entry_key(item) in forced_keys:
            continue
        rows = _entry_rows(item)
        if rows & taken:
            replacement = fallback(item, taken) if fallback else None
            if not replacement:
                continue
            replacement = dict(replacement)
            replacement["pass_ranks"] = item["pass_ranks"]
            replacement["pass_hits"] = item["pass_hits"]
            replacement["rank_sum"] = item["rank_sum"]
            replacement["remainder_of"] = _entry_key(item)
            item = replacement
            rows = _entry_rows(item)
        kept.append(item)
        taken |= rows
        ranked_count += 1
    return kept


# -------------------------------------------------------------- the screen
def screen_candidates(rows, gfx="gfx942", catalog=None, workload=None, table=None, hw=None,
                      priors=None, model="", noise_band_pct=NOISE_BAND_PCT):
    """Every seam window with its gate (pass / pool / reject), request-level gain first."""
    hw = hw or hardware(gfx)
    e2e_us = phase_e2e_us(rows)
    classes = {}
    for row in rows:
        classes[row["row_id"]] = classify_kernel(row, e2e_us.get(row.get("phase")) or 0.0, gfx, hw)
    kernels, dropped = _catalog_kernels(catalog)
    weights = phase_weights(rows, workload)
    eff = _effective_weights(rows, e2e_us, weights)
    denominator = sum((e2e_us.get(phase) or 0.0) * eff.get(phase, 1.0) for phase in e2e_us)
    c0 = phase_c0_us(rows)
    graph = build_dataflow(rows, table)
    costs = {}
    for row in rows:
        cost = kernel_cost(row, hw, c0.get(row.get("phase"), LAUNCH_US))
        if cost is not None:
            costs[row["row_id"]] = cost
    fused = baseline_fused(rows, kernels, e2e_us, eff, denominator)
    ctx = {"hw": hw, "c0": c0, "graph": graph, "costs": costs,
           "baseline_rows": {rid for item in fused for rid in item["row_ids"]},
           "layer_orders": _layer_orders(rows)}
    order = {}
    for index, group in enumerate(_ordered_groups(rows).values()):
        for pos, row in enumerate(group):
            order[row["row_id"]] = (index, pos)
    layer_pos = {}
    for group in ctx["layer_orders"].values():
        for pos, row in enumerate(group):
            layer_pos[row["row_id"]] = pos
    rows_by_id = {row["row_id"]: row for row in rows}
    windows = (_serial_chains(rows) + _sibling_chains(rows, classes) + _parallel_chains(rows, classes)
               + _forward_chains(rows, graph, classes) + _seam_rewrite_chains(rows, classes))
    found = []
    for kind, members, run_index in windows:
        record = _record(kind, members, run_index, classes, kernels, eff, ctx)
        record["_start"] = order.get(members[0]["row_id"], (0, 0))[1]
        span = [layer_pos.get(m["row_id"], 0) for m in members]
        record["seam_descriptor"]["max_span"] = max(span) - min(span) + 2
        found.append(record)
    found = _collapse_identical(found)
    family = _model_family_of(table, model)
    positive = [p for p in (priors or []) if _polarity(p) == "positive"
                and p.get("measured_e2e_gain_pct", 0.0) >= noise_band_pct
                and _prior_applies(p, family, gfx)]
    occurrences = {p["prior_id"]: seam_occurrences(p, rows) for p in positive
                   if (p.get("seam") or {}).get("members")}
    groups = _group_seams(found, denominator)
    for group in groups:
        hit = _group_prior_hit(group, positive, occurrences)
        gate, reason = _gate_of(group, hit)
        group["gate"], group["gate_reason"], group["prior_hit"] = gate, reason, hit
        for member in group["members"]:
            member["_group"] = group
            member["seam_key"] = group["seam_key"]
            member["gate_share"] = group["gate_share"]
            member["seam_share"] = group["gate_share"]
            member["gate_min"] = group["gate_min"]
            member["group_speedup"] = group["speedup"]
            member["seam_request_saved_us"] = group["request_saved_us"]
            member["expected_e2e_gain_pct"] = round(group["expected_e2e_gain_pct"], 4)
            member["gate"] = gate
            member["gate_reason"] = reason
            member["passed"] = gate == "pass"
    for record in found:
        record["_rows_by_id"] = rows_by_id
        if record["invalid"]:
            record["gate_share"] = record["request_chain_us"] / denominator if denominator else 0.0
            record["gate_min"] = (EXISTING_GAIN_GATE_PCT if record["has_existing_kernel"]
                                  else GAIN_GATE_PCT) / 100.0
            record["expected_e2e_gain_pct"] = round(
                100.0 * record["request_saved_us"] / denominator, 4) if denominator else 0.0
            record["gate"], record["gate_reason"] = "reject", record["invalid"]
    passed = [r for r in found if r["gate"] == "pass"]
    pooled = [r for r in found if r["gate"] == "pool"]
    kept, appendix = _dedupe(passed, classes)
    kept_pool, pool_appendix = _dedupe(pooled, classes)
    appendix += pool_appendix
    # Dedupe is per adjacency run, so a seam can keep its window in one phase /
    # pattern and lose it in another. The entry is what survived: re-gate it on
    # the kept members' own request-level gain.
    kept_ids = {id(r) for r in kept + kept_pool}
    entries = []
    for group in groups:
        if group["gate"] == "reject":
            continue
        members = [m for m in group["members"] if id(m) in kept_ids]
        if not members:
            continue
        if len(members) < len(group["members"]):
            sub = _group_metrics(members, denominator, members[0]["_sig"],
                                 members[0].get("_occurrence", 0))
            sub["seam_key"] = group["seam_key"]
            # The remnant must earn a prior hit on its own members: the seam's
            # matching window may have been the one deduped away.
            hit = _group_prior_hit(sub, positive, occurrences)
            gate, reason = _gate_of(sub, hit)
            sub["gate"], sub["gate_reason"], sub["prior_hit"] = gate, reason, hit
            if gate == "reject":
                for member in members:
                    member["appendix_reason"] = (
                        "the seam's windows in %s were deduped into longer windows; this "
                        "remnant alone is %s" % (
                            ", ".join(sorted({"%s/%s" % (m["phase"], m["pattern_id"])
                                              for m in group["members"] if id(m) not in kept_ids})),
                            reason))
                    appendix.append(member)
                    kept_ids.discard(id(member))
                continue
            for member in members:
                member["_group"] = sub
                member["gate_share"] = sub["gate_share"]
                member["seam_share"] = sub["gate_share"]
                member["seam_request_saved_us"] = sub["request_saved_us"]
                member["expected_e2e_gain_pct"] = round(sub["expected_e2e_gain_pct"], 4)
                member["gate"], member["gate_reason"] = gate, reason
                member["passed"] = gate == "pass"
            group = sub
        entries.append((group, members))
    kept = [r for r in kept + kept_pool if id(r) in kept_ids]
    initial = sorted([r for r in kept if r["gate"] == "pass"],
                     key=lambda r: (-r["_group"]["request_saved_us"], -r["request_saved_us"],
                                    r["row_ids"]))
    pool_records = sorted([r for r in kept if r["gate"] == "pool"],
                          key=lambda r: (-r["_group"]["request_saved_us"], -r["request_saved_us"],
                                         r["row_ids"]))
    for index, record in enumerate(initial, 1):
        record["candidate_id"] = "c%02d" % index
    for index, record in enumerate(pool_records, 1):
        record["candidate_id"] = "q%02d" % index
    appendix.sort(key=lambda r: (-r["request_saved_us"], r["row_ids"]))
    for index, record in enumerate(appendix, 1):
        record["candidate_id"] = "x%02d" % index
        superseded = record.pop("_superseded_by", None)
        if superseded is not None:
            record["superseded_by"] = superseded.get("candidate_id")
    entries.sort(key=lambda gm: (gm[0]["gate"] != "pass", -sum(m["request_saved_us"] for m in gm[1])))
    built = []
    for index, (group, members) in enumerate(entries, 1):
        entry = _entry(members, "s%02d" % index, group)
        for member in members:
            member["entry_id"] = entry["entry_id"]
            partners = [m for m in members if m is not member]
            if partners:
                member["partner_candidate_id"] = partners[0].get("candidate_id")
                member["partner_row_ids"] = [row for m in partners for row in m["row_ids"]]
                member["phases"] = entry["phases"]
        built.append(entry)
    rejected = {}
    for record in found:
        if record["gate"] != "reject":
            continue
        reason = record["invalid"] or record.get("gate_reason") or "rejected"
        record["reject_reason"] = reason
        bucket = reason.split(" (")[0].split(" ")[0]
        rejected[bucket] = rejected.get(bucket, 0) + 1
    near = [record for record in found if record["gate"] == "reject" and not record["invalid"]]
    near.sort(key=lambda record: -record.get("expected_e2e_gain_pct", 0.0))
    for item in fused:
        # A baseline fused kernel that still sits in a candidate leaves work
        # behind it (e.g. add_rmsnorm_quant followed by a separate group quant).
        rows_in = set(item["row_ids"])
        item["still_in_candidates"] = sorted({
            cand["candidate_id"] for cand in initial + pool_records if rows_in & set(cand["row_ids"])})
    conflicts = []
    unknown = {}
    for row in rows:
        info = classify_row(row)
        if info["conflict"]:
            conflicts.append({"row_id": row["row_id"], "phase": row.get("phase"),
                              "pattern_id": row.get("pattern_id"), "kernel": _short(_name(row)),
                              "class": info["cls"], "conflict": info["conflict"]})
        if info["cls"] == "unknown":
            unknown.setdefault(_short(_name(row)), []).append(row["row_id"])
    for conflict in conflicts:
        sys.stderr.write("[fusion_screen] classification conflict %s %s: %s\n" % (
            conflict["row_id"], conflict["kernel"], conflict["conflict"]))
    region_coverage = (fusible_region_coverage(table, rows, initial + pool_records, appendix, found)
                       if table else [])
    return {
        "e2e_us": e2e_us,
        "phase_weights": weights,
        "effective_phase_weights": eff,
        "request_denominator_us": denominator,
        "hardware": hw,
        "c0_us": c0,
        "dataflow": graph["stats"],
        "kernel_class": classes,
        "kernel_costs": costs,
        "considered": len(found),
        "rejected": rejected,
        "near_misses": near[:NEAR_MISS_LIMIT],
        "baseline_fused": fused,
        "initial_topk": initial,
        "pool_records": pool_records,
        "appendix": appendix,
        "entries": built,
        "all_records": found,
        "classification_conflicts": conflicts,
        "unclassified_kernels": unknown,
        "catalog_dropped": dropped,
        "catalog_kernel_count": len(kernels),
        "region_coverage": region_coverage,
        "seam_coverage": seam_coverage(rows, classes, initial, pool_records, appendix, found),
        "inefficient_coverage": inefficient_coverage(rows, costs, initial, pool_records,
                                                     appendix, found),
        "model_family": family,
        "_ctx": ctx,
        "_kernels": kernels,
    }


def fusible_region_coverage(table, rows, initial, appendix, found):
    """Every fusible region of the semantic report: a covering candidate or a reason."""
    try:
        from fusion_candidate_harness import _fusible_regions
    except Exception:  # noqa: BLE001 - coverage is a report, never a crash
        return []
    by_id = {row["row_id"]: row for row in rows}
    out = []
    for (phase, pattern), row_ids, stages, total_us in _fusible_regions(table, REGION_FLOOR_US):
        wanted = set(row_ids)
        record = {"phase": phase, "pattern_id": pattern, "row_ids": list(row_ids),
                  "stages": stages, "us_per_layer": total_us}
        covering = [c for c in initial if wanted <= set(c["row_ids"])
                    and (c["phase"], c["pattern_id"]) == (phase, pattern)]
        if covering:
            best = max(covering, key=lambda c: c["request_saved_us"])
            status = "covered" if best.get("gate", "pass") == "pass" else "covered_by_pool"
            record.update(status=status, covered_by=[c["candidate_id"] for c in covering],
                          entry_id=best.get("entry_id"))
            out.append(record)
            continue
        app = [c for c in appendix if wanted <= set(c["row_ids"])
               and (c["phase"], c["pattern_id"]) == (phase, pattern)]
        if app:
            record.update(status="covered_by_appendix", covered_by=[c["candidate_id"] for c in app],
                          reason="whole-region window passed but was deduped: %s (kept %s)" % (
                              app[0].get("appendix_reason"), app[0].get("superseded_by")))
            out.append(record)
            continue
        windows = [r for r in found if wanted <= set(r["row_ids"])
                   and (r["phase"], r["pattern_id"]) == (phase, pattern)]
        compute = [rid for rid in row_ids if rid in by_id and not _is_data_move(by_id[rid])]
        if len(compute) < 2:
            record.update(status="rejected",
                          reason="only %d compute kernel(s) once data movement is excluded; "
                                 "nothing to fuse" % len(compute))
        elif windows:
            tight = min(windows, key=lambda r: (len(r["row_ids"]), -r["request_saved_us"]))
            reason = tight.get("reject_reason") or tight.get("invalid") or tight.get(
                "gate_reason") or "not passed"
            record.update(status="rejected", reason=reason,
                          window_kernels=[_short(n) for n in tight["kernel_names"]])
        else:
            record.update(status="rejected",
                          reason="no window spans the region (rows not adjacent on one stream "
                                 "or more than %d compute kernels)" % SEAM_SERIAL_MAX)
        out.append(record)
    return out


def seam_coverage(rows, classes, initial, pool, appendix, found):
    """Every code seam with >= 3 non-barrier kernels in one step: a candidate or a reason."""
    seams = {}
    for row in rows:
        if _is_data_move(row) or _barrier(row, classes):
            continue
        seam = _code_seam(row)
        if seam is None:
            continue
        seams.setdefault((row.get("phase"), row.get("pattern_id"), seam), []).append(row["row_id"])
    out = []
    for (phase, pattern, seam), ids in sorted(seams.items(), key=lambda kv: str(kv[0])):
        if len(ids) < SEAM_REWRITE_MIN:
            continue
        record = {"phase": phase, "pattern_id": pattern, "code_seam": seam or "(layer)",
                  "row_ids": ids, "kernels": len(ids)}
        label = seam or "(layer)"

        def mine(items):
            return [c for c in items if (c["phase"], c["pattern_id"]) == (phase, pattern)
                    and c["kind"] == "seam_rewrite" and c.get("code_seam") == label]
        hits = mine(initial) or mine(pool)
        if hits:
            record.update(status="covered" if hits[0].get("gate") == "pass" else "covered_by_pool",
                          covered_by=[c["candidate_id"] for c in hits])
        elif mine(appendix):
            record.update(status="covered_by_appendix",
                          covered_by=[c["candidate_id"] for c in mine(appendix)])
        elif mine(found):
            window = max(mine(found), key=lambda r: r["request_saved_us"])
            record.update(status="rejected", reason="seam_rewrite window rejected: %s" % (
                window.get("reject_reason") or window.get("gate_reason")))
        else:
            touched = [c["candidate_id"] for c in initial + pool
                       if (c["phase"], c["pattern_id"]) == (phase, pattern) and set(c["row_ids"]) & set(ids)]
            record.update(status="rejected", covered_by=touched, reason=(
                "barriers split the seam's %d non-barrier kernels into adjacent runs of fewer than "
                "%d (or the runs are the bordering barrier's own prologue/epilogue)%s" % (
                    len(ids), SEAM_REWRITE_MIN,
                    "; parts are in %s" % ", ".join(touched) if touched else "")))
        out.append(record)
    return out


def inefficient_coverage(rows, costs, initial, pool, appendix, found):
    """Every inefficient kernel (eff < 0.3): in a candidate, a reason, or a hand-off."""
    out = []
    by_id = {row["row_id"]: row for row in rows}
    for row_id, cost in costs.items():
        if not cost["inefficient"]:
            continue
        row = by_id[row_id]
        record = {"row_id": row_id, "phase": row.get("phase"), "pattern_id": row.get("pattern_id"),
                  "kernel": _short(_name(row)), "class": classify_row(row)["cls"],
                  "duration_us": round(_duration(row), 3), "ideal_us": round(cost["ideal_us"], 3),
                  "eff": round(cost["eff"], 3), "headroom_us": round(cost["headroom_us"], 3)}
        hits = [c["candidate_id"] for c in initial if row_id in c["row_ids"]]
        pooled = [c["candidate_id"] for c in pool if row_id in c["row_ids"]]
        app = [c["candidate_id"] for c in appendix if row_id in c["row_ids"]]
        # A structurally invalid window (barrier inside, too long) is not a
        # fusion window: a kernel only such windows hold is an isolated one.
        windows = [r for r in found if row_id in r["row_ids"] and not r["invalid"]]
        if hits:
            record.update(status="covered", covered_by=hits)
        elif pooled:
            record.update(status="covered_by_pool", covered_by=pooled)
        elif app:
            record.update(status="covered_by_appendix", covered_by=app)
        elif windows:
            best = max(windows, key=lambda r: r["request_saved_us"])
            record.update(status="rejected", reason="every window holding it was rejected: %s" % (
                best.get("reject_reason") or best.get("gate_reason")))
        else:
            record.update(status="handoff", reason=HANDOFF_REASON)
        out.append(record)
    out.sort(key=lambda r: -r["headroom_us"])
    return out


def _forced_entries(status, rows, found, noise_band, limit, blind=()):
    """Positive measured priors that clear the noise band and whose seam is here.

    `limit` bounds how many are forced in, highest measured gain first. The
    cap is on the whole list (K+4 rows), not on the forced entries alone, so a
    short board still has room for every measured prior. A prior that does not
    name its phases cannot be confirmed on a table blind to a required phase.
    """
    forced = []
    used = set()
    by_prior = {}
    ordered = sorted((s for s in status if s["applies"] and s["seam_in_trace"]
                      and s["polarity"] == "positive"
                      and s["prior"]["measured_e2e_gain_pct"] >= noise_band),
                     key=lambda s: -s["prior"]["measured_e2e_gain_pct"])
    for item in ordered:
        phases = set(item["prior"].get("phases") or (item["prior"].get("seam") or {}).get("phases") or [])
        if blind and (not phases or phases & set(blind)):
            item["forced"] = "phase_blind:%s" % ",".join(blind)
            continue
        if len(forced) >= limit:
            item["forced"] = "cap_reached"
            continue
        entry = item.get("rep_entry") or item["best_entry"]
        if entry is None and item["occurrences"]:
            entry = _prior_seam_entry(item, rows, found)
        if entry is None:
            item["forced"] = "no_entry"
            continue
        key = _entry_key(entry)
        if key in used:
            item["forced"] = "same_entry_as_%s" % next(
                f["forced_by_prior"] for f in forced if _entry_key(f) == key)
            continue
        clash = [i for i, f in enumerate(forced) if _entry_rows(f) & _entry_rows(entry)]
        for index in clash:
            # Two measured priors may be adjacent fusions of one region: the
            # window an earlier prior adopted can run past that prior's own seam
            # into rows the next prior needs. Shrink the earlier entry back to
            # its prior's own seam rows when that frees the overlap.
            owner = by_prior[forced[index]["forced_by_prior"]]
            trimmed = _prior_seam_entry(owner, rows, found) if owner["occurrences"] else None
            if (trimmed and _entry_rows(trimmed) <= _entry_rows(forced[index])
                    and not _entry_rows(trimmed) & _entry_rows(entry)):
                trimmed = dict(trimmed)
                trimmed["forced_by_prior"] = forced[index]["forced_by_prior"]
                trimmed["trimmed_for_prior"] = item["prior"]["prior_id"]
                used.discard(_entry_key(forced[index]))
                used.add(_entry_key(trimmed))
                forced[index] = trimmed
        if _entry_rows(entry) & set().union(*[_entry_rows(f) for f in forced] or [set()]):
            item["forced"] = "overlaps_a_forced_entry"
            continue
        entry = dict(entry)
        entry["forced_by_prior"] = item["prior"]["prior_id"]
        forced.append(entry)
        by_prior[item["prior"]["prior_id"]] = item
        used.add(key)
        item["forced"] = "forced"
    return forced


def _prior_seam_entry(item, rows, found):
    """Entry built from a prior's seam rows when no screened window covers it."""
    ctx = found["_ctx"]
    by_id = {row["row_id"]: row for row in rows}
    members = []
    kinds = item["prior"]["seam"].get("kinds") or []
    kind = "sibling" if kinds == ["sibling"] else "prior_seam"
    for phase, pattern, fused, _all, _confirmed in item["occurrences"]:
        picked = [by_id[rid] for rid in fused if rid in by_id]
        if len(picked) < 2:
            continue
        record = _record(kind, picked, None, found["kernel_class"], found["_kernels"],
                         found["effective_phase_weights"], ctx)
        record["candidate_id"] = "p:%s:%s:%s" % (item["prior"]["prior_id"], phase, pattern)
        members.append(record)
    if not members:
        return None
    group = _group_metrics(members, found["request_denominator_us"])
    group["seam_key"] = "prior|%s" % item["prior"]["prior_id"]
    group["gate"], group["gate_reason"] = "pass", "positive measured prior %s" % item["prior"]["prior_id"]
    entry = _entry(members, "p-%s" % item["prior"]["prior_id"], group)
    entry["kind"] = "prior_seam"
    entry["gate_share"] = None
    entry["seam_note"] = ("no screened window covers this seam; rows located by the prior's "
                          "seam descriptor (%s)" % "; ".join(
                              "%s/%s" % (p, q) for p, q, _f, _a, _c in item["occurrences"]))
    _attach_ladder("gp-%s" % item["prior"]["prior_id"], [entry])
    return entry


def screen(rows, gfx="gfx942", catalog=None, top_k=FINAL_K, passes=SCREEN_PASSES,
           extra_passes=None, workload=None, table=None, priors=None, model="",
           noise_band_pct=NOISE_BAND_PCT, hw=None):
    priors = priors if priors is not None else []
    found = screen_candidates(rows, gfx=gfx, catalog=catalog, workload=workload, table=table,
                              hw=hw, priors=priors, model=model, noise_band_pct=noise_band_pct)
    entries = found["entries"]
    family = found["model_family"]
    blind = phase_blind(table, rows, workload) if table is not None else []
    status = match_priors(entries, priors, rows, family, gfx)
    # Measured evidence picks the form of a seam: the entry a positive prior
    # matched best heads its alternative group (priors over the noise band first).
    anchors = []
    for item in sorted(status, key=lambda s: (s["prior"]["measured_e2e_gain_pct"] < noise_band_pct,
                                              -s["prior"]["measured_e2e_gain_pct"])):
        if (item["applies"] and item["polarity"] == "positive" and item["best_entry"] is not None
                and item["prior"]["measured_e2e_gain_pct"] > 0
                and not item["best_entry"].get("negative_prior")):
            anchors.append(item["best_entry"])
    groups = build_alt_groups(entries, anchors)
    head_of = {id(entry): ladder[0] for ladder in groups for entry in ladder}
    for item in status:
        if item["best_entry"] is not None:
            item["rep_entry"] = head_of.get(id(item["best_entry"]))
    reps = [ladder[0] for ladder in groups]
    ranked_pool = [e for e in reps if e.get("gate") == "pass"]
    max_rows = top_k + FORCED_PRIOR_EXTRA

    def forced_up_to(limit):
        for item in status:
            item.pop("forced", None)
        return _forced_entries(status, rows, found, noise_band_pct, limit, blind)

    forced = forced_up_to(max_rows)
    full = max(len(ranked_pool), top_k)
    ranked = [score_pass(ranked_pool, full)]
    if passes >= 2:
        ranked.append(prior_pass(ranked_pool, full))
    for _ in range(max(0, passes - 2)):
        ranked.append(prior_pass(ranked_pool, full))
    ranked.extend(extra_passes or [])

    passed_records = found["initial_topk"] + found["appendix"]
    denominator = found["request_denominator_us"]

    def fallback(item, taken):
        """The blocked entry's remainder: passing windows inside its rows that no
        kept entry touches, re-gated on their own request-level gain."""
        rows_of = _entry_rows(item)
        options = [r for r in passed_records if r.get("gate") == "pass"
                   and set(r["elim_row_ids"]) <= rows_of and not set(r["elim_row_ids"]) & taken]
        best = None
        by_group = {}
        for record in options:
            by_group.setdefault(id(record["_group"]), []).append(record)
        for members in by_group.values():
            chosen, seen = [], set()
            for record in sorted(members, key=lambda r: -r["request_saved_us"]):
                if (record["phase"], record["pattern_id"]) in seen:
                    continue
                if any(set(record["elim_row_ids"]) & set(other["elim_row_ids"]) for other in chosen):
                    continue
                chosen.append(record)
                seen.add((record["phase"], record["pattern_id"]))
            sub = _group_metrics(chosen, denominator, chosen[0]["_sig"],
                                 chosen[0].get("_occurrence", 0))
            gate, reason = _gate_of(sub)
            if gate != "pass":
                continue
            sub["seam_key"] = chosen[0]["_group"]["seam_key"]
            sub["gate"], sub["gate_reason"] = gate, reason
            entry = _entry(chosen, "%s-r" % item.get("entry_id", "e"), sub)
            _attach_ladder("%s-r" % item.get("alt_group", "g"), [entry])
            if best is None or entry["score"] > best["score"]:
                best = entry
        return best

    final = merge_passes(ranked, top_k, forced=forced, fallback=fallback, pool=ranked_pool)
    if len(final) > max_rows:
        # Forced entries never take a K slot, but the list stops at K+4 rows:
        # keep every ranked entry and the highest-gain priors that still fit.
        ranked_rows = sum(1 for item in final if not item.get("forced_by_prior"))
        forced = forced_up_to(max(max_rows - ranked_rows, 0))
        final = merge_passes(ranked, top_k, forced=forced, fallback=fallback, pool=ranked_pool)
    chosen_groups = {item.get("alt_group") for item in final}
    for index, item in enumerate(final, 1):
        _finalize(item, "e%02d" % index, denominator)
    pool = [dict(item) for item in reps if item.get("gate") == "pool"
            and item.get("alt_group") not in chosen_groups]
    pool.sort(key=lambda item: _sort_key(item))
    for index, item in enumerate(pool, 1):
        _finalize(item, "pool%02d" % index, denominator)
    found["passes"] = [items[:top_k] for items in ranked]
    found["top_k"] = top_k
    found["k_effective"] = top_k
    found["execution_list"] = final
    found["candidate_pool"] = pool
    found["alt_groups"] = groups
    found["prior_status"] = status
    found["noise_band_pct"] = noise_band_pct
    found["phase_blind"] = blind
    return found


# ------------------------------------------------------------------ output
def _finalize(item, exec_id, denominator):
    item["exec_id"] = exec_id
    item["estimated_speedup"] = item["speedup"]
    item["compare_existing"] = item["has_existing_kernel"]
    item["self_author"] = True
    item.setdefault("forced_by_prior", None)
    item.setdefault("negative_prior", False)
    item.setdefault("prior_polarity", None)
    item.setdefault("lower_bound", False)
    if item.get("forced_by_prior"):
        item["prior_polarity"] = "positive"
    saved = item.get("request_saved_us")
    if item.get("expected_e2e_gain_pct") is None and saved is not None and denominator:
        item["expected_e2e_gain_pct"] = round(100.0 * float(saved) / denominator, 4)
    if not item.get("alt_group"):
        _attach_ladder("g-%s" % exec_id, [item])
    elif not item.get("alt_ladder"):
        item["alt_ladder"] = [_rung(0, item)]
    else:
        # Rung 0 is the entry itself (it may be a remainder or a trimmed form).
        ladder = [dict(r) for r in item["alt_ladder"]]
        if ladder[0].get("entry_id") != item.get("entry_id"):
            ladder = [_rung(0, item)] + [r for r in ladder
                                         if r.get("entry_id") != item.get("entry_id")]
        for index, rung in enumerate(ladder[:LADDER_MAX]):
            rung["rung"] = index
        ladder[0] = _rung(0, item)
        item["alt_ladder"] = ladder[:LADDER_MAX]


def _short(name):
    ident = catalog_lib.primary_identifier(name)
    return (ident or str(name))[:60]


def _ms(us):
    return "%.1f" % (us / 1000.0)


def _entry_line(item):
    existing = ", ".join(match["name"] for match in item.get("matched_existing_kernels") or [])[:80] or "no"
    prior = item.get("forced_by_prior") and ("forced: " + item["forced_by_prior"]) or ", ".join(
        "%s%s %+.2f%% x%.2f" % ("NEG " if m.get("polarity") == "negative" else "", m["prior_id"],
                                m["measured_e2e_gain_pct"], m["seam_match"])
        for m in (item.get("prior_matches") or [])[:2]) or "-"
    gain = item.get("expected_e2e_gain_pct")
    return "| %s | %s | %s | %s | %s%s | %.2f | %.1f | %s (%d) | %s | %s | %s | %s |" % (
        item["exec_id"], item["kind"], "+".join(item.get("phases") or [item["phase"]]),
        "+".join(item.get("pattern_ids") or [str(item.get("pattern_id"))]),
        "%.3f%%" % gain if gain is not None else "-", " (lb)" if item.get("lower_bound") else "",
        item.get("score", 0.0) / 1000.0, item["difficulty"], item.get("alt_group") or "-",
        len(item.get("alt_ladder") or []),
        "/".join(str(rank) for rank in item.get("pass_ranks", [])) or "-", prior, existing,
        ", ".join(_short(name) for name in item["kernel_names"]))


def _markdown(result):
    k = result.get("top_k", FINAL_K)
    hw = result.get("hardware") or {}
    lines = ["# Fusion Top-K", "",
             "Every window is scored by its request-level expected gain (launch + memory + "
             "efficiency headroom, x layers x phase weight, / request kernel time). Gate: >= %.2f%% "
             "(>= %.2f%% with an existing kernel, or a positive measured prior); %.2f%% up to the "
             "gate is the candidate pool. Alternatives that eliminate the same kernel are one alt "
             "group (one slot, a ladder). Pass 1 = S / D; pass 2 = measured priors (model family "
             "`%s`). %d passes merged into K=%d; positive priors over the %.1f%% noise band ride "
             "along without a K slot."
             % (GAIN_GATE_PCT, EXISTING_GAIN_GATE_PCT, POOL_GAIN_PCT,
                result.get("model_family") or "unknown", len(result.get("passes") or []), k,
                result.get("noise_band_pct", NOISE_BAND_PCT)), ""]
    weights = result.get("effective_phase_weights") or {}
    lines.append("Phase weights (forwards per request): " + ", ".join(
        "%s %.2f" % (phase, weight) for phase, weight in sorted(weights.items()))
        + "; request kernel time R = %.1f ms." % (result.get("request_denominator_us", 0.0) / 1000.0))
    if hw:
        lines.append("Hardware: HBM %.2f TB/s (%s), peak bf16 %.0f / fp8 %.0f TFLOPS (%s/%s); "
                     "c0 per phase: %s." % (
                         hw["hbm_tbps"], hw["source"]["hbm"], hw["peak_tflops"]["bf16"],
                         hw["peak_tflops"]["fp8"], hw["source"]["bf16"], hw["source"]["fp8"],
                         ", ".join("%s %.2f us" % kv for kv in sorted((result.get("c0_us") or {}).items()))))
    if result.get("phase_blind"):
        lines.append("")
        lines.append("**phase_blind: %s** -- the table lacks a required phase; gains are relative to "
                     "the captured phases only and this board is not evidence of \"no opportunity\" "
                     "there." % ", ".join(result["phase_blind"]))
    lines.append("")
    header = ("| exec | kind | phases | patterns | expected gain | S/D | D | alt group (rungs) "
              "| pass ranks | prior | existing kernel | kernels |")
    rule = "|---|---|---|---|---:|---:|---:|---|---|---|---|---|"
    lines.extend(["## Final", "", header, rule])
    lines.extend(_entry_line(item) for item in result["execution_list"])
    lines.append("")
    lines.append("S/D is request-level ms / difficulty; (lb) = launch-only lower bound (no shapes).")
    lines.append("")
    ladders = [item for item in result["execution_list"] if len(item.get("alt_ladder") or []) > 1]
    if ladders:
        lines.extend(["## Alternative ladders", "", "| exec | group | rung | kind | expected gain | D "
                      "| gate | kernels |", "|---|---|---:|---|---:|---:|---|---|"])
        for item in ladders:
            for rung in item["alt_ladder"]:
                lines.append("| %s | %s | %d | %s | %s | %.1f | %s | %s |" % (
                    item["exec_id"], item.get("alt_group"), rung["rung"], rung["kind"],
                    "%.3f%%" % rung["expected_e2e_gain_pct"]
                    if rung.get("expected_e2e_gain_pct") is not None else "-",
                    rung["difficulty"], rung.get("gate"),
                    ", ".join(_short(n) for n in rung["kernel_names"])))
        lines.append("")
    pool = result.get("candidate_pool") or []
    if pool:
        lines.extend(["## Candidate pool (stacking fill-ins, not Top-K)", "", header, rule])
        lines.extend(_entry_line(item) for item in pool)
        lines.append("")
    status = result.get("prior_status") or []
    if status:
        lines.append("## Measured priors")
        lines.append("")
        lines.append("| prior | polarity | effective gain | applies (family/gfx) | seam matched in trace "
                     "| best match | forced | source |")
        lines.append("|---|---|---:|---|---|---:|---|---|")
        for item in status:
            prior = item["prior"]
            best = item.get("best_entry")
            lines.append("| %s | %s | %+.2f%% | %s | %s | %s | %s | %s |" % (
                prior["prior_id"], item.get("polarity", "positive"), prior["measured_e2e_gain_pct"],
                "yes" if item["applies"] else "no",
                "yes" if item["seam_in_trace"] else "no",
                "%s %.2f" % (best["entry_id"], item["best_match"]) if best else "-",
                item.get("forced") or "-", prior["source"][:60]))
        lines.append("")
    coverage = result.get("region_coverage") or []
    if coverage:
        lines.append("## Fusible regions (01_SEMANTIC section 4)")
        lines.append("")
        lines.append("| # | phase | pattern | stages | us/layer | status | candidate / reason |")
        lines.append("|---:|---|---|---|---:|---|---|")
        for index, item in enumerate(coverage, 1):
            lines.append("| %d | %s | %s | %s | %.2f | %s | %s |" % (
                index, item["phase"], item["pattern_id"], item["stages"], item["us_per_layer"],
                item["status"], ", ".join(item.get("covered_by") or []) or item.get("reason", "")))
        lines.append("")
    seams = result.get("seam_coverage") or []
    if seams:
        lines.extend(["## Code seams with >= %d non-barrier kernels" % SEAM_REWRITE_MIN, "",
                      "| phase | pattern | seam | kernels | status | candidate / reason |",
                      "|---|---|---|---:|---|---|"])
        for item in seams:
            lines.append("| %s | %s | %s | %d | %s | %s |" % (
                item["phase"], item["pattern_id"], item["code_seam"], item["kernels"], item["status"],
                item.get("reason") or ", ".join(item.get("covered_by") or [])))
        lines.append("")
    inefficient = result.get("inefficient_coverage") or []
    if inefficient:
        lines.extend(["## Inefficient kernels (eff < %.1f)" % INEFFICIENT_EFF, "",
                      "| phase | pattern | kernel | us | ideal us | eff | status | candidate / reason |",
                      "|---|---|---|---:|---:|---:|---|---|"])
        for item in inefficient:
            lines.append("| %s | %s | %s | %.1f | %.2f | %.2f | %s | %s |" % (
                item["phase"], item["pattern_id"], item["kernel"], item["duration_us"],
                item["ideal_us"], item["eff"], item["status"],
                item.get("reason") or ", ".join(item.get("covered_by") or [])))
        lines.append("")
    if result.get("baseline_fused"):
        lines.append("## Already fused in the baseline")
        lines.append("")
        lines.append("| catalog kernel | op tags | share by phase | trace kernel | still in candidates |")
        lines.append("|---|---|---|---|---|")
        for item in result["baseline_fused"]:
            shares = ", ".join("%s %.2f%%" % (phase, 100.0 * share)
                               for phase, share in sorted(item["phases"].items()))
            lines.append("| %s | %s | %s | %s | %s |" % (
                item["catalog_kernel"], "/".join(item["op_tags"]), shares, _short(item["trace_kernel"]),
                ", ".join(item.get("still_in_candidates") or []) or "no (nothing left to fuse)"))
        lines.append("")
    if result.get("appendix"):
        lines.append("## Appendix: overlapping sub-windows (deduped)")
        lines.append("")
        lines.append("| id | phase | pattern | superseded by | request saved ms | kernels |")
        lines.append("|---|---|---|---|---:|---|")
        for item in result["appendix"][:40]:
            lines.append("| %s | %s | %s | %s | %s | %s |" % (
                item["candidate_id"], item["phase"], item["pattern_id"], item.get("superseded_by"),
                _ms(item["request_saved_us"]), ", ".join(_short(n) for n in item["kernel_names"])))
        if len(result["appendix"]) > 40:
            lines.append("")
            lines.append("(%d more in fusion_candidates.json)" % (len(result["appendix"]) - 40))
        lines.append("")
    if result.get("near_misses"):
        lines.append("## Rejected windows with the highest expected gain")
        lines.append("")
        lines.append("| phase | expected gain | reason | kernels |")
        lines.append("|---|---:|---|---|")
        for item in result["near_misses"]:
            lines.append("| %s | %.3f%% | %s | %s |" % (
                "+".join(item.get("phases") or [item["phase"]]),
                item.get("expected_e2e_gain_pct") or 0.0, item.get("reject_reason") or "-",
                ", ".join(_short(name) for name in item["kernel_names"])))
        lines.append("")
    unknown = result.get("unclassified_kernels") or {}
    if unknown:
        lines.append("## Unclassified kernels (difficulty %.1f, never covered by an existing kernel)"
                     % UNKNOWN_DIFFICULTY)
        lines.append("")
        lines.extend("- `%s` (%d rows)" % (name, len(ids)) for name, ids in sorted(unknown.items()))
        lines.append("")
    if result.get("classification_conflicts"):
        lines.append("## Classification conflicts")
        lines.append("")
        lines.extend("- %s `%s` -> %s: %s" % (c["row_id"], c["kernel"], c["class"], c["conflict"])
                     for c in result["classification_conflicts"])
        lines.append("")
    if result.get("catalog_dropped"):
        lines.append("Catalog hygiene: %d entries dropped (not kernels / no op tag), %d kept."
                     % (len(result["catalog_dropped"]), result.get("catalog_kernel_count", 0)))
        lines.append("")
    if result.get("rejected"):
        lines.append("Rejected windows: " + ", ".join(
            "%s %d" % (reason, count) for reason, count in sorted(result["rejected"].items())))
        lines.append("")
    return "\n".join(lines) + "\n"


def _public(item):
    keep = (
        "candidate_id", "entry_id", "exec_id", "candidate_ids", "kind", "also_kinds", "phase",
        "phases", "pattern_id", "pattern_ids", "seam_key", "code_seam", "modules",
        "row_ids", "partner_row_ids", "partner_candidate_id", "all_row_ids", "elim_row_ids",
        "merged_members", "data_move_row_ids", "baseline_fused_row_ids", "kernel_names", "classes",
        "op_tags", "e2e_share", "seam_share", "gate_share", "gate_min", "gate", "gate_reason",
        "has_large_op", "required_speedup", "optional", "difficulty", "base_difficulty",
        "has_existing_kernel", "matched_existing_kernels", "existing_match_reason",
        "unknown_kernels", "decode_saved_us", "step_saved_us", "phase_weight", "request_saved_us",
        "seam_request_saved_us", "expected_e2e_gain_pct", "lower_bound", "launch_saved_us",
        "mem_saved_us", "eff_saved_us", "n_before", "n_after", "rho", "c0_us", "counted_row_ids",
        "score", "score_basis", "chain_us", "launch_fraction", "memory_fraction", "speedup",
        "group_speedup", "estimated_speedup", "saved_us", "saving_model", "model", "pass_ranks",
        "pass_hits", "rank_sum", "compare_existing", "self_author", "prior_matches",
        "prior_score", "prior_polarity", "negative_prior", "forced_by_prior", "remainder_of",
        "trimmed_for_prior", "alt_group", "alt_rung", "alt_ladder", "alt_group_size",
        "seam_descriptor", "superseded_by", "appendix_reason", "reject_reason", "invalid",
        "seam_note",
    )
    out = {key: item[key] for key in keep if key in item}
    if "model" in out and "saving_model" not in out:
        out["saving_model"] = out.pop("model")
    else:
        out.pop("model", None)
    for key in ("expected_e2e_gain_pct", "lower_bound", "negative_prior", "prior_polarity"):
        out.setdefault(key, None if key in ("expected_e2e_gain_pct", "prior_polarity") else False)
    if "kind" in out:
        out["seam_members"] = (item.get("seam_descriptor") or {}).get("members") or []
        out["matched_prior_id"] = _matched_prior_id(item)
    return out


def _int_or_none(value):
    try:
        return int(value) if str(value).strip() else None
    except (TypeError, ValueError):
        return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--semantic-table", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--catalog", default="")
    parser.add_argument("--gfx", default="gfx942")
    parser.add_argument("--top-k", type=int, default=FINAL_K,
                        help="K when --fusion-budget-s is not given (default %d)" % FINAL_K)
    parser.add_argument("--fusion-budget-s", default="",
                        help="KernelFusion time budget in seconds; K = clamp(floor(budget x %.1f / "
                             "item), %d, %d)" % (APPLYBACK_BUDGET_SHARE, K_MIN, K_MAX))
    parser.add_argument("--item-minutes", type=float, default=ITEM_MINUTES,
                        help="expected minutes per apply-back item (unit test + A/B)")
    parser.add_argument("--passes", type=int, default=SCREEN_PASSES)
    parser.add_argument("--isl", default="")
    parser.add_argument("--osl", default="")
    parser.add_argument("--conc", default="")
    parser.add_argument("--md", default="")
    parser.add_argument("--model", default="",
                        help="model name for prior matching; default: inferred from the table")
    parser.add_argument("--measured-priors", default=priors_lib.DEFAULT_MEASURED_PRIORS,
                        help="knowledge/fusion/measured_priors.json ('' disables)")
    parser.add_argument("--priors-index", default=priors_lib.DEFAULT_PRIORS_INDEX,
                        help="knowledge/learned/INDEX.md ('' disables)")
    parser.add_argument("--noise-band-pct", type=float, default=NOISE_BAND_PCT,
                        help="a positive measured prior at or above this e2e gain is forced in")
    parser.add_argument("--hbm-tbps", default="", help="measured HBM bandwidth (TB/s)")
    parser.add_argument("--peak-tflops-bf16", default="", help="measured bf16 peak (TFLOPS)")
    parser.add_argument("--peak-tflops-fp8", default="", help="measured fp8 peak (TFLOPS)")
    args = parser.parse_args(argv)
    with open(args.semantic_table) as handle:
        table = json.load(handle)
    catalog = None
    if args.catalog and os.path.isfile(args.catalog):
        with open(args.catalog) as handle:
            catalog = json.load(handle)
    workload = {"isl": _int_or_none(args.isl), "osl": _int_or_none(args.osl),
                "conc": _int_or_none(args.conc)}
    k_effective = adaptive_k(args.fusion_budget_s, args.item_minutes, args.top_k)
    hw = hardware(args.gfx, args.hbm_tbps, args.peak_tflops_bf16, args.peak_tflops_fp8)
    priors = load_all_priors(args.measured_priors, args.priors_index)
    result = screen(load_rows(table), gfx=args.gfx, catalog=catalog, top_k=k_effective,
                    passes=args.passes, workload=workload, table=table, priors=priors,
                    model=args.model, noise_band_pct=args.noise_band_pct, hw=hw)
    os.makedirs(args.out_dir, exist_ok=True)
    candidates_path = os.path.join(args.out_dir, "fusion_candidates.json")
    topk_path = os.path.join(args.out_dir, "fusion_topk.json")
    md_path = args.md or os.path.join(os.path.dirname(args.out_dir.rstrip("/")), "03_FUSION_TOPK.md")
    coverage = result["region_coverage"]
    coverage_check = _coverage_check(table, candidates_payload_regions=coverage, result=result)
    pool = [_public(item) for item in result["candidate_pool"]]
    by_id = {row["row_id"]: row for row in load_rows(table)}
    efficiency = []
    for row_id, cost in sorted(result["kernel_costs"].items(),
                               key=lambda kv: -kv[1]["headroom_us"]):
        row = by_id.get(row_id) or {}
        efficiency.append({"row_id": row_id, "phase": row.get("phase"),
                           "pattern_id": row.get("pattern_id"), "kernel": _short(_name(row)),
                           "duration_us": round(_duration(row), 3),
                           "bytes": int(cost["bytes"]), "flops": cost["flops"],
                           "ideal_us": round(cost["ideal_us"], 3), "eff": round(cost["eff"], 3),
                           "headroom_us": round(cost["headroom_us"], 3),
                           "inefficient": cost["inefficient"]})
    common = {
        "k_effective": result["k_effective"],
        "noise_band_pct": args.noise_band_pct,
        "phase_blind": result["phase_blind"],
        "hardware": result["hardware"],
        "c0_us": result["c0_us"],
        "gates": {"gain_pct": GAIN_GATE_PCT, "existing_gain_pct": EXISTING_GAIN_GATE_PCT,
                  "pool_gain_pct": POOL_GAIN_PCT, "min_speedup": MIN_SPEEDUP},
    }
    candidates = dict(common, **{
        "schema_version": 4,
        "phase": "screen",
        "considered": result["considered"],
        "passed": len(result["initial_topk"]),
        "workload": workload,
        "phase_weights": result["phase_weights"],
        "effective_phase_weights": result["effective_phase_weights"],
        "request_denominator_us": result["request_denominator_us"],
        "model_family": result["model_family"],
        "rejected": result["rejected"],
        "baseline_fused": result["baseline_fused"],
        "near_misses": [_public(item) for item in result["near_misses"]],
        "candidates": [_public(item) for item in result["initial_topk"]],
        "pool_candidates": [_public(item) for item in result["pool_records"]],
        "candidate_pool": pool,
        "seam_entries": [_public(item) for item in result["entries"]],
        "alt_groups": [{"alt_group": ladder[0].get("alt_group"),
                        "entries": [e.get("entry_id") for e in ladder],
                        "gate": ladder[0].get("gate")} for ladder in result["alt_groups"]],
        "appendix": [_public(item) for item in result["appendix"]],
        "region_coverage": coverage,
        "seam_coverage": result["seam_coverage"],
        "inefficient_coverage": result["inefficient_coverage"],
        "region_coverage_check": coverage_check,
        "dataflow": result["dataflow"],
        "kernel_efficiency": efficiency,
        "classification_conflicts": result["classification_conflicts"],
        "unclassified_kernels": result["unclassified_kernels"],
        "catalog_hygiene": {"kept": result["catalog_kernel_count"],
                            "dropped": result["catalog_dropped"]},
        "priors": [{
            "prior_id": s["prior"]["prior_id"], "title": s["prior"]["title"],
            "origin": s["prior"]["origin"], "source": s["prior"]["source"],
            "polarity": s.get("polarity"),
            "measured_e2e_gain_pct": s["prior"]["measured_e2e_gain_pct"],
            "applies": s["applies"], "seam_in_trace": s["seam_in_trace"],
            "best_entry": s["best_entry"]["entry_id"] if s.get("best_entry") else None,
            "best_match": s["best_match"], "forced": s.get("forced")}
            for s in result["prior_status"]],
    })
    topk = dict(common, **{
        "schema_version": 4,
        "phase": "final_topk",
        "top_k": result["k_effective"],
        "passes": [[item.get("entry_id") for item in ranked] for ranked in result["passes"]],
        # Every final entry is tier B for apply-back: by then the unit-side gate has either
        # matched an existing kernel or authored one, so apply-back only wires it in.
        "execution_list": [dict(_public(item), tier="B") for item in result["execution_list"]],
        "candidate_pool": pool,
    })
    with open(candidates_path, "w") as handle:
        json.dump(candidates, handle, indent=2)
    with open(topk_path, "w") as handle:
        json.dump(topk, handle, indent=2)
    parent = os.path.dirname(md_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(md_path, "w") as handle:
        handle.write(_markdown(result))
    summary = {
        "status": "pass" if topk["execution_list"] else "no_candidate",
        "fusion_candidates_json": os.path.abspath(candidates_path),
        "fusion_topk_json": os.path.abspath(topk_path),
        "fusion_topk_md": os.path.abspath(md_path),
        "execution_list": topk["execution_list"],
        "candidate_count": candidates["passed"],
        "candidate_pool_count": len(pool),
        "k_effective": result["k_effective"],
        "phase_blind": result["phase_blind"],
        "region_coverage_ok": coverage_check.get("ok", True),
    }
    print(json.dumps(summary))
    return 0


def _coverage_check(table, candidates_payload_regions, result=None):
    try:
        from fusion_candidate_harness import check_screen_region_coverage, check_screen_item_coverage
    except Exception:  # noqa: BLE001
        return {"ok": True, "errors": [], "note": "fusion_candidate_harness unavailable"}
    errors = check_screen_region_coverage(table, {"region_coverage": candidates_payload_regions},
                                          REGION_FLOOR_US)
    if result is not None:
        errors += check_screen_item_coverage({"seam_coverage": result.get("seam_coverage"),
                                              "inefficient_coverage": result.get("inefficient_coverage")})
    return {"ok": not errors, "errors": errors}


if __name__ == "__main__":
    sys.exit(main())
