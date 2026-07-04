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

Flags can come from a flag STRING (`--server-args`, e.g. EXTRA_SERVER_ARGS) and/or the actual server
LAUNCH SCRIPT (`--server-script`, e.g. the recipe `launch_baseline.sh`). The script often carries flags
the live EXTRA_SERVER_ARGS does not — notably the chunked-prefill budget — so pass it when available. On
overlap, `--server-args` wins (it is the live/override config).

Output (json):
{
  "quant": {"method": "fp8|fp8_blockscale|awq|gptq|compressed-tensors|none",
            "weight_dtype": "fp8_e4m3fnuz|int4|...", "act_dtype": "fp8|bf16|...",
            "block_size": [..]|null, "source": "flag|model_config|none"},
  "kv_cache_dtype": "fp8|bf16|auto",
  "compile": "torch_compile|eager",      # the baseline-relevant fusion state
  "enforce_eager": true|false,           # --enforce-eager / --disable-cuda-graph: eager (strawman) baseline
  "cuda_graph": true|false,
  "attention_backend": "<str>|''",
  "prefill_chunk": <int>|null,           # chunked-prefill token budget (chunked-prefill-size /
                                         # max-num-batched-tokens); null = one prefill pass over the prompt.
                                         # attribute_weights.py uses it to size the serving-lifecycle
                                         # prefill pass count (ceil(isl/prefill_chunk)).
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


def _read_script_flags(script_path):
    """Extract `--flag value` / `--flag=value` pairs from a server LAUNCH SCRIPT. Shell line
    continuations (`\\`+newline) are joined first so a flag and its value on separate lines still pair
    up; `_tokenize` then ignores everything that isn't a `--` flag (echo/if/env lines pass through
    harmlessly). Returns {} if the path is missing/unreadable."""
    if not script_path or not os.path.isfile(script_path):
        return {}
    try:
        with open(script_path) as fh:
            text = fh.read()
    except Exception:
        return {}
    return _tokenize(text.replace("\\\n", " "))


def _prefill_chunk(flags):
    """The chunked-prefill token budget from the launch flags: sglang `--chunked-prefill-size`, vllm
    `--max-num-batched-tokens`. A value <= 0 (sglang's -1 = disabled) or absent -> None (the caller then
    treats prefill as a single pass over the whole prompt)."""
    for k in ("chunked-prefill-size", "chunked_prefill_size",
              "max-num-batched-tokens", "max_num_batched_tokens"):
        v = flags.get(k)
        if isinstance(v, bool):
            continue
        if isinstance(v, int):
            return v if v > 0 else None
        if isinstance(v, str) and v.lstrip("-").isdigit():
            iv = int(v)
            return iv if iv > 0 else None
    return None


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


def parse_regime(server_args, model_config_path="", server_script=""):
    # Launch-script flags fill the base; the live --server-args string overrides on overlap.
    flags = {**_read_script_flags(server_script), **_tokenize(server_args)}
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

    # ---- enforce-eager: the STRAWMAN-baseline flag. vllm's --enforce-eager disables BOTH cuda graphs
    # and compilation; sglang's equivalent is --disable-cuda-graph. When set, the online baseline runs
    # every forward pass eagerly, so launch/dispatch overhead is unmasked and an e2e A/B over-credits
    # launch-overhead kernels (isolated win, e2e loss). Recorded so the unittest-writing module times its
    # baseline under deployment_graph_mode (harness_lib) and the live-seam guard can flag the strawman. ----
    enforce_eager = bool(flags.get("enforce-eager") or flags.get("enforce_eager")
                         or flags.get("disable-cuda-graph") or flags.get("disable_cuda_graph"))
    if enforce_eager:
        notes.append("enforce-eager/disable-cuda-graph set: the online baseline runs eagerly (no "
                     "graph replay); an e2e A/B on it is a strawman for launch-overhead kernels.")

    # ---- cuda graph (decode is graph-captured unless disabled) ----
    cuda_graph = not bool(flags.get("disable-cuda-graph") or flags.get("disable_cuda_graph")
                          or flags.get("enforce-eager") or flags.get("enforce_eager"))

    attn = flags.get("attention-backend") or flags.get("attention_backend") or ""
    if attn is True:
        attn = ""

    # ---- chunked-prefill budget (sizes the serving prefill pass count in attribute_weights.py) ----
    prefill_chunk = _prefill_chunk(flags)

    return {
        "quant": quant,
        "kv_cache_dtype": kv,
        "compile": compile_state,
        "enforce_eager": enforce_eager,
        "cuda_graph": cuda_graph,
        "attention_backend": attn,
        "prefill_chunk": prefill_chunk,
        "notes": " ".join(notes),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server-args", default="",
                    help="the server launch flag string (e.g. EXTRA_SERVER_ARGS / the recipe flags)")
    ap.add_argument("--model-config", default="",
                    help="path to the model's config.json (for a pre-quantized checkpoint)")
    ap.add_argument("--server-script", default="",
                    help="path to the server launch script (e.g. launch_baseline.sh); "
                         "carries flags EXTRA_SERVER_ARGS may omit, notably the chunked-prefill budget")
    ap.add_argument("--out", default="", help="write regime json here (also printed to stdout)")
    args = ap.parse_args()
    regime = parse_regime(args.server_args, args.model_config, args.server_script)
    js = json.dumps(regime, indent=2)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(js)
        sys.stderr.write(f"wrote {args.out}\n")
    print(js)


if __name__ == "__main__":
    main()
