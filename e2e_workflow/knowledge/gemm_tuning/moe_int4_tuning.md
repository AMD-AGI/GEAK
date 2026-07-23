# int4 fused-MoE Triton config tuning — the memory-free vLLM MoE lever (MI300X / gfx942)

> **Verified 2026-06-10/11 on moonshotai-Kimi-K2.6** (DeepSeek-V3-class int4 MoE, compressed-tensors
> w4a16, vLLM v1, TP=8, gfx942, ISL/OSL=8192/1024 conc=64). Tuning the missing int4_w4a16 fused-MoE
> Triton config and deploying it via `VLLM_TUNED_CONFIG_FOLDER` gave **+16.4% e2e** (514.55 → 598.98
> tok/s, parity-preserved, GSM8K 0.965 → 0.9733), with **zero extra HBM**. This is a head-kernel Tier-B
> tune (an env winner), orthogonal to config-tuner flags, and it COMPOUNDS with `--max-num-batched-tokens`.

## Why this file exists (the lesson)
vLLM ships per-shape fused-MoE Triton configs (`E=…,N=…,device_name=…,dtype=int4_w4a16.json`) generated
by an offline sweep and loaded at startup (no runtime autotune). For a model whose `(E,N,int4_w4a16)`
shape has **no shipped config**, the int4 expert grouped-GEMM falls back to a slow default — vLLM logs
**"Using default MoE config"**. On Kimi-K2.6 that expert GEMM was **~57% of GPU time**, so the missing
config was the single biggest e2e lever.

Two ways to get this wrong (both observed):
- ❌ **Reach for an fp8 / quant rewrite of the same op first.** It caches a SECOND (fp8) weight copy →
  at memory parity it **OOMs at KV-cache init** (op-level ~1.5x but e2e-undeployable; the e2e Integrator
  correctly rejects it with `mem_footprint_starves_kv`). For an already-int4 model, the Triton config
  tune below gets a comparable kernel speedup at **no memory cost** — do it FIRST.
- ❌ **Use the sglang knob (`SGLANG_MOE_CONFIG_DIR`) on a vLLM server.** Wrong stack. vLLM's env is
  `VLLM_TUNED_CONFIG_FOLDER` (a directory holding the per-shape JSONs).

## How the live lookup works (so you know what to tune)
vLLM resolves the file via `get_config_file_name(E, N, "int4_w4a16", None)` →
`E=<E>,N=<N>,device_name=<dev>,dtype=int4_w4a16.json`, searched under `VLLM_TUNED_CONFIG_FOLDER`. The
JSON maps each **M bucket** → a Triton tile config `{BLOCK_SIZE_M/N/K, GROUP_SIZE_M, num_warps,
num_stages, ...}`; at runtime vLLM nearest-M-bucket-looks-up per forward. So you tune **one config per
M bucket**, and the prefill (large-M) buckets matter most under a large `--max-num-batched-tokens`.

### The shape is per-rank — derive it from the model config (+TP), never hardcode
- `E`   = `text_config.n_routed_experts` (or `num_experts` / `num_local_experts`)
- `N`   = `text_config.moe_intermediate_size // TP`  (per-tensor-parallel-rank intermediate = lookup N)
- `K`   = `text_config.hidden_size`
- `topk`= `text_config.num_experts_per_tok`
- `group_size`, `num_bits` = `quantization_config.config_groups.*.weights.{group_size,num_bits}`
- only applies when `quant_method == compressed-tensors`, `num_bits == 4`, `input_activations == null`
  (weight-only w4a16). A w4a8 / fp8-activation MoE uses a different kernel — do NOT use this recipe.

(Kimi-K2.6 @ TP=8 resolves to E=384, N=256, K=7168, group_size=32, topk=8.)

## The recipe (sweep → deploy → gate), no package edits
1. **Sweep** each M bucket against the REAL vLLM int4_w4a16 path (`fused_experts` +
   `fused_moe.override_config`) with synthetic correctly-shaped weights: time the default vs each
   candidate tile (hipEvent median), keep the fastest that passes **parity rel<1e-2** vs the default
   output. Run each bucket in its OWN subprocess (the Triton in-memory JIT cache contaminates shared-mem
   state when sweeping >300 configs in one process). Buckets that find no winner keep the default config
   (so vLLM still has an entry).
2. **Write** the tuned JSON into a fresh dir → that dir is your `VLLM_TUNED_CONFIG_FOLDER`.
3. **Return it as a head-kernel env winner** (op_benchmarker): `winner_kind=env`,
   `apply_env="VLLM_TUNED_CONFIG_FOLDER=<dir>"`, `tuning_artifact=<dir>`. Pair it with
   `--max-num-batched-tokens` (large prefill batch so the tuned large-M buckets execute).
4. **Gate via the e2e Integrator** (the existing general path — do NOT hand-roll an A/B): the Integrator
   builds candidate env = current + `apply_env`, runs the tight 2-launch A/B on `bench_e2e.sh`, proves
   engagement (the "Using default MoE config" warning DISAPPEARS), checks greedy parity, and applies the
   Amdahl + `mem_footprint_starves_kv` gates. Tile-only config changes don't alter math (parity is
   per-bucket rel<1e-2 + greedy e2e), and they add no memory, so the memory gate is a no-op here.

## At runtime: write the driver into the eval dir, NOT into `scripts/`
The shared `scripts/` dir holds only generic infra (`bench_e2e.sh`, `op_bench.py`, …). This tuner is
experiment-specific (shape from the model), so the Op Benchmarker **writes the driver into
`$EVAL_DIR/config/` at runtime** and runs it (mirrors the aiter-GEMM recipe). A complete, generic,
env-driven driver to adapt — it derives the shape from the model config, sweeps per bucket in isolated
subprocesses, and writes the lookup JSON + a report:

```python
#!/usr/bin/env python3
# $EVAL_DIR/config/tune_moe_int4.py — int4_w4a16 fused-MoE Triton config tuner.
# Generic: shape from --model-config (+--tp) OR explicit --e/--n/--k. Writes the
# E=...,int4_w4a16.json that vLLM loads from VLLM_TUNED_CONFIG_FOLDER.  No package edits.
import argparse, json, os, statistics, subprocess, sys, time

M_BUCKETS = [1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
WARMUP, ITERS, REL_TOL, BIT8_PACK, DT = 5, 20, 1e-2, 2, "int4_w4a16"

def derive_shape(cfg_path, tp):
    c = json.load(open(cfg_path)); tc = c.get("text_config", c)
    g = lambda d, *k: next((d[x] for x in k if d.get(x) is not None), None)
    e = g(tc, "n_routed_experts", "num_experts", "num_local_experts")
    mi = g(tc, "moe_intermediate_size", "intermediate_size"); k = g(tc, "hidden_size")
    topk = g(tc, "num_experts_per_tok", "moe_topk") or 8
    gs, qc = 32, (tc.get("quantization_config") or c.get("quantization_config") or {})
    for _n, grp in (qc.get("config_groups") or {}).items():
        w = (grp or {}).get("weights") or {}
        if w.get("group_size"): gs = int(w["group_size"]); break
    return {"E": int(e), "N": int(mi) // int(tp), "K": int(k), "group_size": int(gs), "topk": int(topk)}

def candidates(M):
    bms = ([16,32] if M<=16 else [16,32,64] if M<=64 else [32,64,128] if M<=256 else [64,128] if M<=1024 else [64,128,256])
    bns = [32,64,128] if M<=64 else [64,128,256]
    gms = [1] if M<128 else [1,8] if M<1024 else [1,8,16]
    out, seen = [], set()
    for bm in bms:
        for bn in bns:
            for bk in [32,64,128]:
                for gm in gms:
                    for w in [4,8]:
                        if M>=512 and bm==16: continue
                        if bm*bn>=128*256 and w==4: continue
                        if bm<=16 and bn<=32 and w==8: continue
                        key=(bm,bn,bk,gm,w)
                        if key in seen: continue
                        seen.add(key)
                        out.append({"BLOCK_SIZE_M":bm,"BLOCK_SIZE_N":bn,"BLOCK_SIZE_K":bk,
                                    "GROUP_SIZE_M":gm,"SPLIT_K":1,"num_warps":w,"num_stages":2})
    return out

def bench(run):
    import torch
    for _ in range(WARMUP): run()
    torch.cuda.synchronize(); ts=[]
    for _ in range(ITERS):
        s,e=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        s.record(); run(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    return statistics.median(ts)

def tune_bucket(M, sh, out_dir):
    import torch
    from vllm.model_executor.layers.fused_moe import fused_experts, override_config
    from vllm.model_executor.layers.fused_moe.config import int4_w4a16_moe_quant_config
    E,N,K,GS,TK = sh["E"],sh["N"],sh["K"],sh["group_size"],sh["topk"]; bs=[0,GS]; dt,dev=torch.bfloat16,"cuda"
    gen=torch.Generator(device=dev).manual_seed(0)
    w1=torch.randint(0,256,(E,2*N,K//BIT8_PACK),dtype=torch.uint8,device=dev,generator=gen)
    w2=torch.randint(0,256,(E,K,N//BIT8_PACK),dtype=torch.uint8,device=dev,generator=gen)
    w1s=(torch.randn(E,2*N,K//GS,dtype=dt,device=dev,generator=gen)*0.01).to(dt)
    w2s=(torch.randn(E,K,N//GS,dtype=dt,device=dev,generator=gen)*0.01).to(dt)
    qc=int4_w4a16_moe_quant_config(w1_scale=w1s,w2_scale=w2s,w1_zp=None,w2_zp=None,block_shape=bs)
    g2=torch.Generator(device=dev).manual_seed(1234+M)
    hid=torch.randn(M,K,dtype=dt,device=dev,generator=g2)
    tid=torch.randint(0,E,(M,TK),dtype=torch.int32,device=dev,generator=g2)
    tw=torch.rand(M,TK,dtype=torch.float32,device=dev,generator=g2); tw=tw/tw.sum(1,keepdim=True)
    def run(cfg):
        if cfg is None: return fused_experts(hid,w1,w2,tw,tid,inplace=False,global_num_experts=E,quant_config=qc)
        with override_config(cfg): return fused_experts(hid,w1,w2,tw,tid,inplace=False,global_num_experts=E,quant_config=qc)
    ref=run(None); torch.cuda.synchronize(); dms=bench(lambda: run(None))
    best=None
    for cfg in candidates(M):
        try:
            out=run(cfg); torch.cuda.synchronize()
            if not torch.isfinite(out).all(): continue
            rd=(out.float()-ref.float()).abs().max().item()/(ref.float().abs().max().item() or 1.0)
            if rd>REL_TOL: continue
            ms=bench(lambda: run(cfg))
            if best is None or ms<best["ms"]: best={"cfg":cfg,"ms":ms}
        except Exception:
            torch.cuda.empty_cache(); continue
    rec={"M":M,"default_ms":dms}
    if best: rec.update(best_ms=best["ms"],best_cfg=best["cfg"],speedup=dms/best["ms"])
    os.makedirs(os.path.join(out_dir,"bucket_results"),exist_ok=True)
    json.dump(rec,open(os.path.join(out_dir,"bucket_results",f"M_{M}.json"),"w"),indent=2)

def main():
    if len(sys.argv)>=3 and sys.argv[1]=="--bucket":
        tune_bucket(int(sys.argv[2]), json.loads(os.environ["SHAPE"]), os.environ["OUTDIR"]); return
    ap=argparse.ArgumentParser(); ap.add_argument("--out-dir",required=True)
    ap.add_argument("--model-config",default=""); ap.add_argument("--tp",type=int,default=1)
    ap.add_argument("--e",type=int); ap.add_argument("--n",type=int); ap.add_argument("--k",type=int)
    ap.add_argument("--group-size",type=int,default=0); ap.add_argument("--topk",type=int,default=0)
    a=ap.parse_args()
    sh = ({"E":a.e,"N":a.n,"K":a.k,"group_size":a.group_size or 32,"topk":a.topk or 8}
          if a.e and a.n and a.k else derive_shape(a.model_config, a.tp))
    out=os.path.abspath(a.out_dir); os.makedirs(out,exist_ok=True)
    from vllm.model_executor.layers.fused_moe.fused_moe import get_config_file_name
    from vllm.platforms import current_platform
    fname=get_config_file_name(sh["E"],sh["N"],DT,None)
    print(f"[tune] dev={current_platform.get_device_name()} shape={sh} -> {fname}")
    res={}
    for M in M_BUCKETS:
        rp=os.path.join(out,"bucket_results",f"M_{M}.json")
        if os.path.exists(rp): os.remove(rp)
        env=dict(os.environ, SHAPE=json.dumps(sh), OUTDIR=out,
                 HIP_VISIBLE_DEVICES=os.environ.get("HIP_VISIBLE_DEVICES","0"))
        subprocess.run([sys.executable,"-u",os.path.abspath(__file__),"--bucket",str(M)],env=env,cwd=out)
        res[M]=json.load(open(rp)) if os.path.exists(rp) else {"M":M}
    tuned={str(M): (r["best_cfg"] if r.get("best_ms",9e9)<=r.get("default_ms",9e9) and "best_cfg" in r
                    else r.get("best_cfg")) for M,r in res.items() if r.get("best_cfg")}
    json.dump(tuned, open(os.path.join(out,fname),"w"), indent=4)
    print(f"[tune] wrote {os.path.join(out,fname)} ({len(tuned)} buckets) -> VLLM_TUNED_CONFIG_FOLDER={out}")

if __name__=="__main__": main()
```

Invoke it (single GPU; no server up), then deploy + gate via the Integrator:
```bash
# 1) tune (derives E/N/K/gs/topk from the model config + TP)
HIP_VISIBLE_DEVICES=0 python3 "$EVAL_DIR/config/tune_moe_int4.py" \
  --out-dir "$EVAL_DIR/config/moe_tuned" --model-config "$MODEL_PATH/config.json" --tp "$TP"
# 2) hand to op_benchmarker return:  winner_kind=env,
#    apply_env="VLLM_TUNED_CONFIG_FOLDER=$EVAL_DIR/config/moe_tuned",  also recommend
#    EXTRA_SERVER_ARGS+=" --max-num-batched-tokens <~2*ISL, clamp 8192..32768>"
# 3) the e2e Integrator A/B-gates it (engagement = "Using default MoE config" warning gone).
```

## Gate / verify (uses the existing general path)
- **Engagement proof:** the server log line `Using default MoE config` for this shape must DISAPPEAR
  once `VLLM_TUNED_CONFIG_FOLDER` is set (the equivalent of aiter's `is tuned on cu_num` check).
- **Parity:** per-bucket rel<1e-2 in the sweep (tile-only config; no algorithm change) + the Integrator's
  greedy temp=0 e2e probe.
- **Amdahl + memory:** the Integrator's standard gates. This change adds **no HBM**, so it passes the
  `mem_footprint_starves_kv` gate trivially (unlike an fp8 rewrite of the same op).
- **Tight A/B:** ref (current accepted) vs cand (+ `VLLM_TUNED_CONFIG_FOLDER` + mnbt), same session,
  `delta% > NOISE_BAND` AND `cand_min > ref_max`.

## Gotchas
- **`BLOCK_SIZE_M=256` is legal** for the largest buckets and is where the big prefill wins live
  (matches AMD's MI350X reference trend); vLLM accepts any `BLOCK_SIZE_M` from the JSON.
- **mnbt matters:** without a large `--max-num-batched-tokens` the prefill M stays small and the tuned
  large-M buckets never execute — the win shrinks. Derive it from the workload (≈2·ISL, clamped), don't
  ship a magic constant.
- **Isolate each bucket in a subprocess** — sweeping >300 configs in one process corrupts the Triton
  shared-memory JIT state and produces spurious LDS-overflow failures.
- **Report the A/B ratio, not absolute tok/s** — it survives box/convention drift (fixed vs variable
  `random_range_ratio` shifts both legs together).

## Confirmed result
Kimi-K2.6 / vLLM / MI300X, TP=8, ISL/OSL=8192/1024 conc=64: tuned int4 fused-MoE config (M=8192 1.59x,
4096/16384 1.53x kernel) + `--max-num-batched-tokens 16384` → **+16.4% e2e** (514.55 → 598.98 tok/s),
parity preserved, **zero extra HBM**. Full write-up:
[`../../../perf_knowledge/case_studies/by_model/kimi_k2.6_int4_moe_mi300x.md`](../../../perf_knowledge/case_studies/by_model/kimi_k2.6_int4_moe_mi300x.md).
Operator background: [`../../../perf_knowledge/operators/fused_moe_grouped_gemm/tuning.md`](../../../perf_knowledge/operators/fused_moe_grouped_gemm/tuning.md).
