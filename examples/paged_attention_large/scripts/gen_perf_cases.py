#!/usr/bin/env python3
"""Inject memory-bound performance cases into ``test_cases.json``.

The captured cases record real tensor *shapes* but no *values*; the runtime then
synthesizes structured index tensors (block_table / seq_lens / query_start_loc)
and clamps seq_lens to ~32 for crash-safety. At that length the whole KV cache
stays resident in MI300X's last-level cache, so paged_attention profiles as
latency-bound and a roofline reads <2% HBM — unrepresentative of real decode.

This script appends a ``perf_only`` case carrying *real, self-consistent* values
for the structured index tensors (each sequence owns a disjoint run of KV blocks,
seq_lens = L), sized so the KV working set far exceeds the cache and the kernel
must stream from HBM. The big float tensors stay shape-only — their values don't
affect bandwidth/timing, only the index structure does.

  ``seq_lens``        : every sequence has context length L
  ``block_table``     : seq i owns blocks [i*nblk, (i+1)*nblk)  -> disjoint HBM
  ``query_start_loc`` : unit-step prefix sum [0,1,2,…] (1 query token/seq = decode)

Replay is handled by ``_runtime._make_tensor`` honoring the ``"data"`` field.

  python3 scripts/gen_perf_cases.py            # default: L=4096, S=2048, GQA 16:1

Idempotent: existing ``perf_only`` cases are removed before re-appending.
"""
import argparse
import copy
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
TEST_CASES = os.path.join(os.path.dirname(HERE), "test_cases.json")

# Kernel layout constants (match the captured cases + the compiled template:
# paged_attention_ll4mi_QKV_mfma16_kernel<…, D, PART, …, X, …>).
D = 128            # head size
X = 8              # key_cache packing factor (D // X = 16)
BLK = 16           # block_size (KV cache page size)
PART = 256         # PARTITION_SIZE -> exp_sums/max_logits/tmp_out 3rd dim = ceil(L/PART)


def _t(dtype, shape, data=None):
    sig = {"kind": "tensor", "dtype": dtype, "shape": list(shape), "device": "cuda:0"}
    if data is not None:
        sig["data"] = data
    return sig


def build_case(S: int, L: int, H: int, KVH: int, template: dict) -> dict:
    """Construct one perf_only case from a captured case template.

    Scalar / opaque args (kv_cache_dtype, mfma_type, alibi_slopes, …) are copied
    verbatim from the template so their exact recorded encoding is preserved;
    only tensor shapes, the coupled scalars, and the structured ``data`` change.
    """
    nblk = math.ceil(L / BLK)          # blocks per sequence
    num_blocks = S * nblk              # disjoint -> forces real HBM traffic
    P = math.ceil(L / PART)            # partitions per sequence

    block_table = [[i * nblk + j for j in range(nblk)] for i in range(S)]
    seq_lens = [L] * S
    query_start_loc = list(range(S + 1))   # 1 query token per sequence (decode)

    tc = copy.deepcopy(template)
    a = tc["args_sig"]
    a[0] = _t("bfloat16", [S, H, D])                       # out
    a[1] = _t("float32", [S, H, P])                        # exp_sums
    a[2] = _t("float32", [S, H, P])                        # max_logits
    a[3] = _t("bfloat16", [S, H, P, D])                    # tmp_out
    a[4] = _t("bfloat16", [S, H, D])                       # query
    a[5] = _t("bfloat16", [num_blocks, KVH, D // X, BLK, X])  # key_cache
    a[6] = _t("bfloat16", [num_blocks, KVH, D, BLK])       # value_cache
    a[7]["value"] = KVH                                    # num_kv_heads
    a[8]["value"] = 1.0 / math.sqrt(D)                     # scale
    a[9] = _t("int32", [S, nblk], data=block_table)        # block_tables
    a[10] = _t("int32", [S], data=seq_lens)                # seq_lens
    a[11] = _t("int32", [S + 1], data=query_start_loc)     # query_start_loc
    a[12]["value"] = BLK                                   # block_size
    a[13]["value"] = L                                     # max_seq_len
    a[16] = _t("float32", [], data=1.0)                    # k_scale
    a[17] = _t("float32", [], data=1.0)                    # v_scale
    # a[14] alibi_slopes(None), a[15] kv_cache_dtype, a[18] fp8_out_scale(None),
    # a[19] mfma_type are left exactly as captured.

    kv_gb = 2 * num_blocks * KVH * (D // X) * BLK * X * 2 / 1e9  # K+V, bf16
    tc["test_case_id"] = f"perf_L{L}_S{S}_h{H}kv{KVH}"
    tc["perf_only"] = True
    tc["count"] = 1
    tc["params_repr"] = {
        "S_seqs": S, "ctx_len": L, "out_len": 1, "heads": H, "kv_heads": KVH,
        "head_size": D, "block_size": BLK, "partition": PART,
        "kv_alloc_gb": round(kv_gb, 1), "note": "decode; disjoint KV blocks -> HBM-streaming",
    }
    return tc


# Default memory-bound sweep tested by DEFAULT (no env var needed). This is the
# roofline_probe.py (S, L) sweep plus the original single (S=2048, L=4096) case,
# all GQA 16:1 (H=16, KVH=1). Every entry carries baked, self-consistent index
# ``data`` (disjoint KV blocks) so the kernel actually streams KV from HBM.
DEFAULT_CONFIGS = [
    # (S sequences, L context length)
    (1024, 1024), (1024, 4096),
    (4096, 2048), (8192, 2048),
    (2048, 8192), (8192, 8192),
    (16384, 4096),
    (2048, 4096),               # the original gen_perf_cases default
]
DEFAULT_H = 16
DEFAULT_KVH = 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-S", type=int, default=None, help="single override: concurrent sequences (batch)")
    ap.add_argument("-L", type=int, default=None, help="single override: context / KV length per sequence")
    ap.add_argument("-H", type=int, default=DEFAULT_H, help="query heads")
    ap.add_argument("--kvh", type=int, default=DEFAULT_KVH, help="kv heads (GQA group = H/kvh)")
    args = ap.parse_args()

    # -S/-L given -> single case; otherwise emit the full default sweep.
    if args.S is not None or args.L is not None:
        configs = [(args.S or 2048, args.L or 4096)]
    else:
        seen, configs = set(), []          # de-dup on (S, L), preserve order
        for S, L in DEFAULT_CONFIGS:
            if (S, L) not in seen:
                seen.add((S, L)); configs.append((S, L))

    with open(TEST_CASES) as f:
        cases = json.load(f)
    captured = [c for c in cases if not c.get("perf_only")]

    # Pick a captured template whose GQA layout matches the request so scalar /
    # opaque arg encodings carry over cleanly.
    def heads_of(c):
        q = c["args_sig"][4]
        kv = c["args_sig"][7]
        return (q.get("shape", [0, 0])[1], kv.get("value"))
    tmpl = next((c for c in captured if heads_of(c) == (args.H, args.kvh)), captured[0])

    new_cases = [build_case(S, L, args.H, args.kvh, tmpl) for S, L in configs]
    out = captured + new_cases
    with open(TEST_CASES, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[gen_perf_cases] wrote {len(captured)} captured + {len(new_cases)} perf_only:")
    for c in new_cases:
        print(f"    {c['test_case_id']:28s} KV~{c['params_repr']['kv_alloc_gb']}GB")
    print(f"  -> {TEST_CASES}")


if __name__ == "__main__":
    main()
