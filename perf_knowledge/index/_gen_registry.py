#!/usr/bin/env python3
"""Regenerate sota_registry.yaml + sota_matrix.md from the per-card frontmatter.

Single source of truth = operators/<op>/backends/<backend>.md frontmatter
(kind: sota_card, operator, backend, status, gens, dtypes, regimes, sources).
Run from perf_knowledge/: python index/_gen_registry.py
"""
import os, re, glob, sys

KK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CARDS = sorted(glob.glob(os.path.join(KK, "operators", "*", "backends", "*.md")))

BADGE = {"sota":"🟢","competitive":"🟡","experimental":"🧪","legacy":"🟤","na":"⚪"}

# operator -> family (display grouping for the matrix); order preserved
FAMILY = [
 ("GEMM", ["dense_gemm","batched_gemm","grouped_gemm_moe","splitk_streamk_gemm","scaled_quant_gemm","gemm_epilogue_fused","skinny_gemv_decode"]),
 ("Attention", ["attention_prefill_fmha","attention_decode_paged","mla_attention","gqa_mqa_attention","sliding_window_attention","chunked_prefill","sparse_attention_nsa","linear_attention_gated_delta","context_parallel_attention","speculative_decode_verify"]),
 ("Norm / Act / Pos", ["rmsnorm","layernorm","softmax","act_and_mul_silu_gelu","fused_add_rmsnorm","fused_norm_quant","rope","mrope","alibi"]),
 ("MoE", ["moe_routing_topk","moe_dispatch_combine","fused_moe_grouped_gemm","shared_expert_fusion"]),
 ("Collectives", ["allreduce","allgather","reduce_scatter","fused_allreduce_rmsnorm"]),
 ("Quantization", ["quant_dequant_fp8","quant_int8","quant_fp4_mxfp","kv_cache_quant"]),
 ("Embedding / Sampling", ["embedding","lm_head_logits","sampling_topk_topp"]),
 ("Elementwise / Reduction", ["elementwise","reduction","cumsum_scan","argmax_topk","cast_fill_copy"]),
 ("Convolution", ["causal_conv1d","depthwise_conv","conv2d"]),
 ("Data movement", ["transpose","gather_scatter","all_to_all_dispatch_combine","paged_kv_copy","layout_shuffle"]),
]

def parse_fm(path):
    txt = open(path, encoding="utf-8").read()
    m = re.match(r"^---\n(.*?)\n---", txt, re.S)
    if not m: return None
    fm = {}
    body = m.group(1)
    # simple key: value and key: [list] parser
    cur = None
    for line in body.splitlines():
        lm = re.match(r"^([a-zA-Z_]+):\s*(.*)$", line)
        if lm:
            k, v = lm.group(1), lm.group(2).strip()
            if v.startswith("[") and v.endswith("]"):
                fm[k] = [x.strip() for x in v[1:-1].split(",") if x.strip()]
            elif v == "":
                fm[k] = []; cur = k
            else:
                fm[k] = v; cur = None
        else:
            lm2 = re.match(r"^\s*-\s*(.*)$", line)
            if lm2 and cur:
                fm.setdefault(cur, [])
                if isinstance(fm[cur], list): fm[cur].append(lm2.group(1).strip())
    return fm

recs = {}
for c in CARDS:
    fm = parse_fm(c)
    if not fm:
        print("WARN no frontmatter:", c, file=sys.stderr); continue
    op = fm.get("operator") or os.path.basename(os.path.dirname(os.path.dirname(c)))
    bk = fm.get("backend") or os.path.splitext(os.path.basename(c))[0]
    rel = os.path.relpath(c, os.path.join(KK, "index"))
    recs.setdefault(op, {})[bk] = {
        "status": fm.get("status","competitive"),
        "gens": fm.get("gens",[]), "dtypes": fm.get("dtypes",[]),
        "regimes": fm.get("regimes",[]), "card": rel,
        "sources": fm.get("sources",[]),
    }

# ---------- sota_registry.yaml ----------
def yl(xs): return "[" + ", ".join(xs) + "]"
out = []
out.append("# sota_registry.yaml — HUMAN/AUDIT view (status badges for browsing). AUTO-GENERATED.")
out.append("# NOT consumed by the workflows. The `status` field is TIME-SENSITIVE, dated evidence — it goes")
out.append("# stale as ROCm/aiter evolve. Do NOT use it to pick a backend. Workflows consume")
out.append("# capability_index.yaml (no status/perf) to ENUMERATE candidates, then DECIDE by measurement.")
out.append("# Edit card frontmatter and re-run index/_gen_registry.py; do not hand-edit entries.")
out.append("")
out.append("schema:")
out.append("  version: 2")
out.append("  entry: {operator, backend, status: sota|competitive|experimental|legacy|na, gens, dtypes, regimes, card, sources}")
out.append("")
out.append("entries:")
nentry = 0
for op in [o for _,ops in FAMILY for o in ops] + [o for o in recs if o not in {o for _,ops in FAMILY for o in ops}]:
    if op not in recs: continue
    for bk in sorted(recs[op]):
        r = recs[op][bk]; nentry += 1
        out.append(f"  - operator: {op}")
        out.append(f"    backend: {bk}")
        out.append(f"    status: {r['status']}")
        out.append(f"    gens: {yl(r['gens'])}")
        out.append(f"    dtypes: {yl(r['dtypes'])}")
        out.append(f"    regimes: {yl(r['regimes'])}")
        out.append(f"    card: {r['card']}")
        if r["sources"]:
            out.append("    sources:")
            for s in r["sources"][:4]:
                out.append(f"      - {s}")
open(os.path.join(KK,"index","sota_registry.yaml"),"w",encoding="utf-8").write("\n".join(out)+"\n")

# ---------- capability_index.yaml (what the WORKFLOWS consume) ----------
# Reference-only CAPABILITY index: which backends have a documented impl for an op and what they
# support. NO status, NO perf, NO ranking — those are time-sensitive and decisions belong to the
# workflow + measurement, not the knowledge base. Use this to ENUMERATE candidates; never to pick a
# winner. The agent always measures; this only widens/locates the candidate set.
cap = []
cap.append("# capability_index.yaml — reference-only candidate/capability index (AUTO-GENERATED).")
cap.append("# Purpose: let a workflow ENUMERATE which backends have a documented implementation for an")
cap.append("#   operator, and what gens/dtypes/regimes each supports, with a pointer to the card + sources.")
cap.append("# NOT a ranking. There is deliberately NO status/perf here — 'best' is decided by the workflow")
cap.append("#   via on-box measurement, never by this file. Incomplete/■stale entries can only widen the")
cap.append("#   candidate set or be ignored; they can never prune the agent's own candidates or pick a winner.")
cap.append("# Query: filter by (operator, gen, dtype, regime) -> candidate backends -> read card -> MEASURE.")
cap.append("")
cap.append("schema: {operator, backend, gens, dtypes, regimes, card, sources}")
cap.append("candidates:")
ncap = 0
for op in [o for _,ops in FAMILY for o in ops] + [o for o in recs if o not in {o for _,ops in FAMILY for o in ops}]:
    if op not in recs: continue
    for bk in sorted(recs[op]):
        r = recs[op][bk]; ncap += 1
        cap.append(f"  - operator: {op}")
        cap.append(f"    backend: {bk}")
        cap.append(f"    gens: {yl(r['gens'])}")
        cap.append(f"    dtypes: {yl(r['dtypes'])}")
        cap.append(f"    regimes: {yl(r['regimes'])}")
        cap.append(f"    card: {r['card']}")
        if r["sources"]:
            cap.append("    sources:")
            for s in r["sources"][:4]:
                cap.append(f"      - {s}")
open(os.path.join(KK,"index","capability_index.yaml"),"w",encoding="utf-8").write("\n".join(cap)+"\n")

# ---------- sota_matrix.md ----------
m = []
m.append("# SOTA matrix — operator × backend")
m.append("")
m.append("AUTO-GENERATED from per-card frontmatter (`index/_gen_registry.py`). Each cell links to the SOTA card.")
m.append("Legend: 🟢 sota · 🟡 competitive · 🧪 experimental · 🟤 legacy · ⚪ na · `·` no card.")
m.append("")
m.append(f"Coverage: **{len(recs)} operators**, **{nentry} backend cards**.")
m.append("")
for fam, ops in FAMILY:
    # column order: core authoring langs, then other authoring langs (gluon/hipkittens/…), then libs
    core = ["triton","flydsl","hip","ck","asm","tilelang"]
    authoring_extra = ["gluon","hipkittens","rocwmma","mojo","cutlass_port"]
    present = set()
    for op in ops:
        present |= set(recs.get(op,{}).keys())
    extra = [a for a in authoring_extra if a in present]
    libs = sorted(b for b in present if b not in core and b not in authoring_extra)
    cols = core + extra + libs
    m.append(f"## {fam}")
    m.append("| operator | " + " | ".join(cols) + " |")
    m.append("|" + "---|"*(len(cols)+1))
    for op in ops:
        if op not in recs:
            m.append(f"| {op} | " + " | ".join(["·"]*len(cols)) + " |"); continue
        cells = [f"[{op}](../operators/{op}/overview.md)"]
        for bk in cols:
            if bk in recs[op]:
                r = recs[op][bk]
                badge = BADGE.get(r["status"],"🟡")
                cells.append(f"[{badge}](../operators/{op}/backends/{bk}.md)")
            else:
                cells.append("·")
        m.append("| " + " | ".join(cells) + " |")
    m.append("")
open(os.path.join(KK,"index","sota_matrix.md"),"w",encoding="utf-8").write("\n".join(m)+"\n")

print(f"OK: {len(recs)} operators, {nentry} cards -> sota_registry.yaml + sota_matrix.md")
