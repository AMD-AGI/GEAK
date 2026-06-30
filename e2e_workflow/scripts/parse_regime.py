#!/usr/bin/env python3
"""Parse the ONLINE serving REGIME from the server launch flags (+ model config).

The #1 cause of "isolated win, e2e loss" is a microbench that runs in a regime the live server never
uses: testing an UNQUANTIZED gemm when the server runs `--quantization fp8` (so the real seam is the
fp8 path and the unquantized one only serves lm_head), verifying attention under bf16 KV when the
server runs `--kv-cache-dtype fp8` (bf16 stride over fp8 bytes -> GPU fault), or comparing a Triton
norm against eager when the server fuses it via torch.compile (strawman baseline).

None of that is visible in a shape. It lives in the LAUNCH FLAGS and the model's own quant config. This
parser turns those into a `regime` descriptor that the extractor writes into meta.json, so every
downstream step (oracle capture, baseline choice, shape/dtype, weight attribution) matches online.

Output (json):
{
  "quant": {"method": "fp8|fp8_blockscale|awq|gptq|compressed-tensors|none",
            "weight_dtype": "fp8_e4m3fnuz|int4|...", "act_dtype": "fp8|bf16|...",
            "block_size": [..]|null, "source": "flag|model_config|none"},
  "kv_cache_dtype": "fp8|bf16|auto",
  "compile": "torch_compile|eager",      # the baseline-relevant fusion state
  "cuda_graph": true|false,
  "attention_backend": "<str>|''",
  "notes": "..."
}

Stdlib only.
"""
import argparse, json, os, re, sys


def _tokenize(server_args):
    """Split a launch flag string into a {flag: value} map. Handles `--k v`, `--k=v`, and bare flags."""
    toks = (server_args or "").split()
    out = {}
    i = 0
    while i < len(toks):
        t = toks[i]
        if t.startswith("--"):
            key = t[2:]
            if "=" in key:
                k, v = key.split("=", 1)
                out[k] = v
            elif i + 1 < len(toks) and not toks[i + 1].startswith("--"):
                out[key] = toks[i + 1]
                i += 1
            else:
                out[key] = True   # bare boolean flag
        i += 1
    return out


def _load_model_quant(model_config_path):
    """Read the model's own quantization_config from config.json (a pre-quantized checkpoint)."""
    if not model_config_path or not os.path.isfile(model_config_path):
        return None
    try:
        with open(model_config_path) as fh:
            cfg = json.load(fh)
    except Exception:
        return None
    qc = cfg.get("quantization_config") or cfg.get("compression_config")
    if not qc:
        return None
    method = (qc.get("quant_method") or qc.get("format") or qc.get("method") or "").lower()
    # fp8 block-scale (e.g. DeepSeek/Qwen fp8) exposes weight_block_size
    block = qc.get("weight_block_size") or qc.get("block_size")
    fmt = (qc.get("fmt") or qc.get("activation_scheme") or "").lower()
    wdt = "fp8_e4m3" if "fp8" in method or "fp8" in fmt else (
        "int4" if ("4" in method or "awq" in method or "gptq" in method) else method or "")
    return {"method": method or "fp8", "weight_dtype": wdt, "block_size": block, "fmt": fmt}


def parse_regime(server_args, model_config_path=""):
    flags = _tokenize(server_args)
    notes = []

    # ---- quantization: flag wins, else the model's own config ----
    q_flag = flags.get("quantization")
    model_q = _load_model_quant(model_config_path)
    quant = {"method": "none", "weight_dtype": "", "act_dtype": "", "block_size": None, "source": "none"}
    if isinstance(q_flag, str) and q_flag:
        ql = q_flag.lower()
        quant = {
            "method": ql,
            "weight_dtype": "fp8_e4m3" if "fp8" in ql else ("int4" if ("4" in ql or "awq" in ql or "gptq" in ql) else ql),
            "act_dtype": "fp8" if "fp8" in ql else "bf16",
            "block_size": (model_q or {}).get("block_size"),
            "source": "flag",
        }
        if model_q and "fp8" in (model_q.get("weight_dtype") or "") and "fp8" not in ql:
            notes.append(f"flag quantization='{q_flag}' but model config says fp8 — verify which wins online.")
    elif model_q:
        quant = {
            "method": ("fp8_blockscale" if model_q.get("block_size") and "fp8" in (model_q.get("weight_dtype") or "")
                       else model_q.get("method", "")),
            "weight_dtype": model_q.get("weight_dtype", ""),
            "act_dtype": "fp8" if "fp8" in (model_q.get("weight_dtype") or "") else "bf16",
            "block_size": model_q.get("block_size"),
            "source": "model_config",
        }

    # ---- KV cache dtype ----
    kv = flags.get("kv-cache-dtype") or flags.get("kv_cache_dtype") or "auto"
    if isinstance(kv, str):
        kv = kv.lower()
    if kv in ("auto", True, None):
        kv = "auto"
        notes.append("kv-cache-dtype=auto -> follows model compute dtype (usually bf16); confirm if fp8 desired.")

    # ---- compile / fusion state (the baseline-relevant axis) ----
    compile_on = bool(flags.get("enable-torch-compile") or flags.get("enable_torch_compile")
                      or flags.get("torch-compile"))
    compile_state = "torch_compile" if compile_on else "eager"

    # ---- cuda graph (decode is graph-captured unless disabled) ----
    cuda_graph = not bool(flags.get("disable-cuda-graph") or flags.get("disable_cuda_graph"))

    attn = flags.get("attention-backend") or flags.get("attention_backend") or ""
    if attn is True:
        attn = ""

    return {
        "quant": quant,
        "kv_cache_dtype": kv,
        "compile": compile_state,
        "cuda_graph": cuda_graph,
        "attention_backend": attn,
        "notes": " ".join(notes),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server-args", default="",
                    help="the server launch flag string (e.g. EXTRA_SERVER_ARGS / the recipe flags)")
    ap.add_argument("--model-config", default="",
                    help="path to the model's config.json (for a pre-quantized checkpoint)")
    ap.add_argument("--out", default="", help="write regime json here (also printed to stdout)")
    args = ap.parse_args()
    regime = parse_regime(args.server_args, args.model_config)
    js = json.dumps(regime, indent=2)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(js)
        sys.stderr.write(f"wrote {args.out}\n")
    print(js)


if __name__ == "__main__":
    main()
