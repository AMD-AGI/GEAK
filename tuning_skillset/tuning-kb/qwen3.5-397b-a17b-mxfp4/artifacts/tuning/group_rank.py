#!/usr/bin/env python3
"""Group the kernel ranking into functional buckets."""
import json, sys, re
j = json.load(open(sys.argv[1]))
G = [
 ("MoE gemm1 (mxfp4)",      r"mfma_moe1"),
 ("MoE gemm2 (mxfp4)",      r"mfma_moe2"),
 ("MoE quant+sort",         r"fused_mx_quant_moe_sort|moe_sorting_entry"),
 ("MoE routing softmax",    r"topkGatingSoftmax"),
 ("Full attn (paged)",      r"paged_attention"),
 ("GDN linear attn core",   r"fused_recurrent_gated_delta_rule"),
 ("GDN aux (conv/norm/split/gate)", r"causal_conv1d|layer_norm_fwd|qkvzba_split|fused_gate_sigmoid|fused_sigmoid_mul"),
 ("TP all-reduce",          r"cross_device_reduce|allgather_vec"),
 ("Dense GEMM (proj)",      r"^Cijk_|hgemm_bf16"),
 ("RMSNorm",                r"rmsnorm"),
 ("Activation (shared exp)",r"act_and_mul"),
 ("RoPE",                   r"mrope"),
 ("KV cache store",         r"store_kvcache"),
]
tot = j["total_us"]; acc = {}; seen = set()
for name, pat in G:
    s = c = 0
    for k in j["kernels"]:
        if re.search(pat, k["name"]) and k["name"] not in seen:
            s += k["us"]; c += k["calls"]; seen.add(k["name"])
    acc[name] = (s, c)
other = sum(k["us"] for k in j["kernels"] if k["name"] not in seen)
print("%-34s %10s %8s %8s" % ("bucket", "us", "% dev", "calls"))
for n, (s, c) in sorted(acc.items(), key=lambda kv: -kv[1][0]):
    print("%-34s %10.0f %7.2f%% %8d" % (n, s, 100*s/tot, c))
print("%-34s %10.0f %7.2f%%" % ("other", other, 100*other/tot))
print("-"*62)
print("%-34s %10.0f" % ("TOTAL device us (4 ranks x 8 steps)", tot))
moe = sum(acc[n][0] for n in acc if n.startswith("MoE"))
att = acc["Full attn (paged)"][0] + acc["GDN linear attn core"][0] + acc["GDN aux (conv/norm/split/gate)"][0]
print("%-34s %7.2f%%" % ("MoE total", 100*moe/tot))
print("%-34s %7.2f%%" % ("attention total (both kinds)", 100*att/tot))
