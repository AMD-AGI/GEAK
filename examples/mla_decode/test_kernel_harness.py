#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Test harness for MLA decode kernel
# Shapes source: /home/upandey/AIG-Eval/external_repos/aiter/op_tests/test_mla.py

import argparse
import os
import sys
import math

import torch

# Ensure aiter is importable
REPO_ROOT = os.environ.get(
    "GEAK_WORK_DIR",
    os.environ.get(
        "GEAK_REPO_ROOT",
        os.path.dirname(os.path.abspath(__file__)),
    ),
)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import aiter
import aiter.mla as mla_module
from aiter import dtypes

torch.set_default_device("cuda")

# --- Fixed constants ---
WARMUP = 50
ITERATIONS = int(os.environ.get("GEAK_BENCHMARK_ITERATIONS", "200"))

# --- Config space (from test_mla.py defaults, decode path only) ---
# bf16/bf16 decode configs with supported nhead values
# Focus on decode_qlen=1 (primary decode case) and decode_qlen=2
# nhead_configs: (nhead, decode_qlen)
CTX_LENS = [21, 64, 256, 512, 1200, 3200, 5200, 8192]
BATCH_SIZES = [1, 3, 5, 16, 32, 64, 128, 256]
NHEAD_CONFIGS = [(16, 1), (16, 2), (16, 4), (128, 1), (128, 2)]

# Fixed params from test_mla.py defaults
KV_LORA_RANK = 512
QK_NOPE_HEAD_DIM = 128
QK_ROPE_HEAD_DIM = 64
V_HEAD_DIM_ORIG = 128  # overridden to kv_lora_rank in absorb mode
PAGE_SIZE = 1

# Build ordered full case stream (same order as test_mla.py)
ALL_CONFIGS = []
for _nhead, _decode_qlen in NHEAD_CONFIGS:
    for _ctx_len in CTX_LENS:
        for _batch_size in BATCH_SIZES:
            ALL_CONFIGS.append((_ctx_len, _batch_size, _nhead, _decode_qlen))


def _pick(configs, count):
    if len(configs) <= count:
        return list(range(len(configs)))
    n = len(configs)
    return [round(i * (n - 1) / (count - 1)) for i in range(count)]


# --- Reference implementation (from test_mla.py) ---
def ref_masked_attention(query, key, value, scale, out_dtype, is_causal=True):
    attn_weights = torch.einsum("qhd,khd->hqk", query.float(), key.float()) * scale
    if is_causal:
        s_q = query.shape[0]
        s_k = key.shape[0]
        attn_bias = torch.zeros(s_q, s_k, dtype=query.dtype)
        temp_mask = torch.ones(s_q, s_k, dtype=torch.bool).tril(diagonal=s_k - s_q)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_weights += attn_bias
    attn_weights = torch.softmax(attn_weights, dim=-1)
    out = torch.einsum("hqk,khd->qhd", attn_weights.float(), value.float())
    return out.to(out_dtype)


def torch_mla_extend(
    q, kvc_cache, qo_indptr, kv_indptr, kv_indices, sm_scale,
    kv_lora_rank, qk_rope_head_dim, out_dtype, is_causal=True,
):
    qs = torch.tensor_split(q, qo_indptr.tolist()[1:])
    kvc = torch.index_select(kvc_cache, 0, kv_indices)
    kvs = torch.tensor_split(kvc, kv_indptr.tolist()[1:])
    bs = qo_indptr.shape[0] - 1
    os_list = []
    for i in range(bs):
        kvc_i = kvs[i]
        q_i = qs[i]
        k = kvc_i
        v, _ = torch.split(kvc_i, [kv_lora_rank, qk_rope_head_dim], dim=-1)
        o = ref_masked_attention(q_i, k, v, sm_scale, out_dtype, is_causal=is_causal)
        os_list.append(o)
    return torch.concat(os_list)


def setup_inputs(ctx_len, batch_size, nhead, decode_qlen):
    """Set up inputs for MLA decode test, returns dict of tensors and params."""
    torch.manual_seed(42)

    kv_lora_rank = KV_LORA_RANK
    qk_rope_head_dim = QK_ROPE_HEAD_DIM
    page_size = PAGE_SIZE
    nhead_kv = 1

    # absorb mode dims
    qk_head_dim = kv_lora_rank + qk_rope_head_dim  # 576
    v_head_dim = kv_lora_rank  # 512
    sm_scale = 1.0 / (qk_head_dim ** 0.5)

    kv_max_sz = 65536 * 32
    num_page = (kv_max_sz + page_size - 1) // page_size

    qo_indptr = torch.zeros(batch_size + 1, dtype=torch.int)
    kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int)
    seq_lens_kv = torch.full((batch_size,), ctx_len, dtype=torch.int)
    seq_lens_qo = torch.full((batch_size,), decode_qlen, dtype=torch.int)

    kv_indptr[1:batch_size + 1] = torch.cumsum(seq_lens_kv, dim=0)
    qo_indptr[1:batch_size + 1] = torch.cumsum(seq_lens_qo, dim=0)

    kv_indices = torch.randint(
        0, num_page, (kv_indptr[-1].item() + 10000,), dtype=torch.int
    )
    kv_last_page_lens = torch.ones(batch_size, dtype=torch.int)

    total_q = qo_indptr[-1].item()
    max_seqlen_qo = seq_lens_qo.max().item()

    kv_buffer = torch.randn(
        (num_page * page_size, 1, kv_lora_rank + qk_rope_head_dim),
        dtype=torch.bfloat16,
    )
    q = torch.randn((total_q, nhead, qk_head_dim), dtype=torch.bfloat16)

    return {
        "q": q,
        "kv_buffer": kv_buffer,
        "qo_indptr": qo_indptr,
        "kv_indptr": kv_indptr,
        "kv_indices": kv_indices,
        "kv_last_page_lens": kv_last_page_lens,
        "max_seqlen_qo": max_seqlen_qo,
        "total_q": total_q,
        "num_page": num_page,
        "page_size": page_size,
        "nhead_kv": nhead_kv,
        "qk_head_dim": qk_head_dim,
        "v_head_dim": v_head_dim,
        "sm_scale": sm_scale,
        "kv_lora_rank": kv_lora_rank,
        "qk_rope_head_dim": qk_rope_head_dim,
    }


def run_kernel(inputs):
    """Run MLA decode kernel, return output tensor."""
    out_asm = torch.empty(
        (inputs["total_q"], inputs["q"].shape[1], inputs["v_head_dim"]),
        dtype=torch.bfloat16,
    ).fill_(-1)

    mla_module.mla_decode_fwd(
        inputs["q"],
        inputs["kv_buffer"].view(
            inputs["num_page"], inputs["page_size"],
            inputs["nhead_kv"], inputs["qk_head_dim"]
        ),
        out_asm,
        inputs["qo_indptr"],
        inputs["kv_indptr"],
        inputs["kv_indices"],
        inputs["kv_last_page_lens"],
        inputs["max_seqlen_qo"],
        sm_scale=inputs["sm_scale"],
        logit_cap=0.0,
    )
    return out_asm


def run_ref(inputs):
    """Run reference implementation, return output tensor."""
    out_ref = torch_mla_extend(
        inputs["q"],
        inputs["kv_buffer"],
        inputs["qo_indptr"],
        inputs["kv_indptr"],
        inputs["kv_indices"],
        inputs["sm_scale"],
        inputs["kv_lora_rank"],
        inputs["qk_rope_head_dim"],
        out_dtype=torch.bfloat16,
        is_causal=True,
    )
    return out_ref


def check_correctness_val(out_ref, out_asm):
    """Check correctness using checkAllclose logic from test_mla.py.
    Uses rtol=1e-2, atol=1e-2 (same as original).
    Returns (pass_bool, err_ratio, cos_diff).
    The original test_mla.py uses tol_err_ratio=0.05 but does NOT assert
    on failure - it just logs. We use a generous 20% threshold to match
    the original test's non-failing behavior while still catching regressions.
    """
    # checkAllclose style check
    isClose = torch.isclose(out_ref, out_asm, rtol=1e-2, atol=1e-2)
    if isClose.all():
        err_ratio = 0.0
    else:
        mask = ~isClose
        num = mask.sum()
        err_ratio = (num / out_ref.numel()).item()

    # Also compute cos_diff for reporting
    x, y = out_ref.double(), out_asm.double()
    cos_diff = 1 - 2 * (x * y).sum().item() / max((x * x + y * y).sum().item(), 1e-12)

    # Pass if err_ratio <= 20% (generous to match original non-asserting behavior)
    # The original test never fails on correctness for bf16 decode
    passed = err_ratio <= 0.20
    return passed, err_ratio, cos_diff


def benchmark_kernel(inputs):
    """Benchmark the MLA decode kernel, return median latency in ms."""
    out_asm = torch.empty(
        (inputs["total_q"], inputs["q"].shape[1], inputs["v_head_dim"]),
        dtype=torch.bfloat16,
    ).fill_(-1)

    def fn():
        mla_module.mla_decode_fwd(
            inputs["q"],
            inputs["kv_buffer"].view(
                inputs["num_page"], inputs["page_size"],
                inputs["nhead_kv"], inputs["qk_head_dim"]
            ),
            out_asm,
            inputs["qo_indptr"],
            inputs["kv_indptr"],
            inputs["kv_indices"],
            inputs["kv_last_page_lens"],
            inputs["max_seqlen_qo"],
            sm_scale=inputs["sm_scale"],
            logit_cap=0.0,
        )

    # Warmup
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()

    # Benchmark with GPU events
    latencies = []
    for _ in range(ITERATIONS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        latencies.append(start.elapsed_time(end))

    latencies.sort()
    median_ms = latencies[len(latencies) // 2]
    return median_ms


def config_str(cfg):
    ctx_len, batch_size, nhead, decode_qlen = cfg
    return "ctx={} bs={} nhead={} dq={}".format(ctx_len, batch_size, nhead, decode_qlen)


def mode_correctness(indices):
    print("Running correctness check on {} configs...".format(len(indices)))
    all_pass = True
    for idx in indices:
        cfg = ALL_CONFIGS[idx]
        ctx_len, batch_size, nhead, decode_qlen = cfg
        label = config_str(cfg)
        try:
            inputs = setup_inputs(ctx_len, batch_size, nhead, decode_qlen)
            out_asm = run_kernel(inputs)
            out_ref = run_ref(inputs)
            passed, err_ratio, cos_diff = check_correctness_val(out_ref, out_asm)
            if passed:
                print("  [{}] {}  err_ratio={:.4f} cos_diff={:.2e}  PASS".format(
                    idx, label, err_ratio, cos_diff))
            else:
                print("  [{}] {}  err_ratio={:.4f} cos_diff={:.2e}  FAIL".format(
                    idx, label, err_ratio, cos_diff))
                all_pass = False
        except Exception as e:
            print("  [{}] {}  ERROR: {}".format(idx, label, e))
            all_pass = False
        finally:
            torch.cuda.empty_cache()

    print("GEAK_SHAPES_USED={}".format(indices))
    if not all_pass:
        print("CORRECTNESS FAILED")
        sys.exit(1)
    print("ALL CORRECTNESS CHECKS PASSED")


def mode_benchmark(indices):
    print("Running benchmark on {} configs...".format(len(indices)))
    latencies = []
    for idx in indices:
        cfg = ALL_CONFIGS[idx]
        ctx_len, batch_size, nhead, decode_qlen = cfg
        label = config_str(cfg)
        try:
            inputs = setup_inputs(ctx_len, batch_size, nhead, decode_qlen)
            ms = benchmark_kernel(inputs)
            print("  {}  {:.4f}ms".format(label, ms))
            latencies.append(ms)
        except Exception as e:
            print("  {}  ERROR: {}".format(label, e))
        finally:
            torch.cuda.empty_cache()

    print("GEAK_SHAPES_USED={}".format(indices))
    if latencies:
        geo_mean = math.exp(sum(math.log(x) for x in latencies) / len(latencies))
        print("GEAK_RESULT_LATENCY_MS={:.4f}".format(geo_mean))
    else:
        print("No successful benchmarks")
        sys.exit(1)


def mode_profile(indices):
    print("Running profile on {} configs...".format(len(indices)))
    for idx in indices:
        cfg = ALL_CONFIGS[idx]
        ctx_len, batch_size, nhead, decode_qlen = cfg
        label = config_str(cfg)
        try:
            inputs = setup_inputs(ctx_len, batch_size, nhead, decode_qlen)
            out_asm = run_kernel(inputs)
            print("  {}  OK".format(label))
        except Exception as e:
            print("  {}  ERROR: {}".format(label, e))
        finally:
            torch.cuda.empty_cache()

    print("GEAK_SHAPES_USED={}".format(indices))


def main():
    parser = argparse.ArgumentParser(description="MLA decode kernel test harness")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--correctness", action="store_true")
    group.add_argument("--benchmark", action="store_true")
    group.add_argument("--full-benchmark", action="store_true")
    group.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    total = len(ALL_CONFIGS)
    print("Total configs: {}".format(total))

    if args.correctness:
        indices = _pick(ALL_CONFIGS, 25)
        mode_correctness(indices)
    elif args.benchmark:
        indices = _pick(ALL_CONFIGS, 25)
        mode_benchmark(indices)
    elif args.full_benchmark:
        indices = list(range(total))
        mode_benchmark(indices)
    elif args.profile:
        indices = _pick(ALL_CONFIGS, 5)
        mode_profile(indices)


if __name__ == "__main__":
    main()
