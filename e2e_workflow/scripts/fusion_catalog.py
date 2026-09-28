#!/usr/bin/env python3
"""Deterministic fusion-kernel catalog builder (P2.0).

Enumerates the fusion kernels ACTUALLY present in the installed backends
(aiter / sglang / vllm / flashinfer / …, auto-detected) so downstream harnesses
can *falsify* an analyst's "no existing kernel -> author-track" claim instead of
trusting it. This is the authority for "what exists"; the LLM only decides
"what to fuse".

Why source-grep and not just dir(): aiter's `@compile_ops`-registered kernels
are not module attributes until first call, so a pure-import catalog under-reports
exactly the way the analyst did. We union:
  1. python importable callables (dir over aiter.ops.*)   -> source="py_import"
  2. python `def NAME(` across each provider tree          -> source=<provider>
  3. C++/HIP kernel symbols in csrc                        -> source="csrc"
  4. (optional) kernels observed in a clean trace          -> source="trace"

Output: available_fusion_kernels.json — a list of
  {name, op_tags[], dtype_tags[], modules[], sources[], observed_in_trace}
plus the tag vocab and the providers scanned (so the catalog declares its scope).

Run INSIDE the runtime container (needs the providers importable):
  python3 fusion_catalog.py --out <dir>/available_fusion_kernels.json \
      [--trace <rank0.trace.json.gz>] [--extra-provider name=/path]
"""
import argparse
import gzip
import json
import os
import re

# --- semantic tag vocabulary (name-substring -> op tag) ----------------------
# One shared vocabulary with the candidate harness's region op-tags, so
# "region op-set subset of kernel op_tags" is a real containment test.
_OP_RULES = [
    ("norm", ("rmsnorm", "rms_norm", "layernorm", "layer_norm", "groupnorm",
              "group_norm", "l2norm", "qk_norm", "_norm_", "_norm", "norm_")),
    ("add_residual", ("with_add", "add_rmsnorm", "add_rms", "_residual",
                      "fused_add")),
    ("quant", ("quant", "_fp8", "fp8_", "_fp4", "fp4_", "a8w8", "a4w4",
               "blockscale", "block_quant", "dynamicquant", "e4m3", "e5m2",
               "mxfp")),
    ("rope", ("rope", "rotary", "pos_encoding", "mrope", "_rope_")),
    ("kv_cache", ("kv_cache", "kvcache", "_cache_", "cache_quant",
                  "cache_block", "reshape_and_cache", "concat_and_cast",
                  "kv_buffer", "set_mla_kv", "store_kv")),
    ("cast", ("_cast", "cast_", "concat_and_cast", "convert")),
    ("layout", ("shuffle", "layout", "trans_ragged", "permute", "transpose")),
    ("allreduce", ("all_reduce", "allreduce", "fused_ar", "custom_fused_ar")),
    ("moe", ("moe", "fmoe", "expert", "ck_moe", "g1u1")),
    ("topk", ("topk", "grouped_topk", "fused_gate", "moe_sort", "gating")),
    ("activation", ("silu", "gelu", "act_mul", "activation", "swiglu",
                    "geglu")),
    ("gemm", ("gemm", "_bmm", "bmm_", "batched_gemm", "matmul")),
    ("gemm_prologue", ("prequant", "prologue", "_a_per_token", "a_per_group")),
    ("gemm_epilogue", ("epilogue", "_bias", "gemm_a8w8")),
]
_DTYPE_RULES = [
    ("fp8_blockscale", ("blockscale", "block_quant", "per_group", "per_1x128",
                        "group_quant", "a8w8")),
    ("fp8", ("fp8", "_f8", "f8_", "e4m3", "e5m2")),
    ("fp4", ("fp4", "_f4", "f4_", "mxfp4", "a4w4", "e2m1")),
    ("bf16", ("bf16", "bfloat16")),
    ("fp16", ("fp16", "float16", "_half")),
]
# names that are NOT kernels: decorators, generators, config/tuner helpers
_NOISE_EXACT = {"compile_ops", "torch_compile_guard", "per_block_quant_wrapper"}
_NOISE_PREFIX = ("gen_", "get_", "compute_", "cmdgenfunc", "_", "test_")
_NOISE_SUBSTR = ("_fake_tensor", "_tune_", "_config", "_torch", "fake_tensors")
# a name must look like a fusion kernel: carry >=1 of these tokens
_FUSED_HINT = re.compile(
    r"(fused|_quant|rmsnorm|rms_norm|layernorm|rope|allreduce|all_reduce|"
    r"moe|fmoe|topk|batched_gemm|gemm_a8w8|gemm_a4w4|act_mul|silu|gelu|"
    r"concat_and_cast|kv_cache|cache_quant|blockscale|group_quant|g1u1|"
    r"pertoken|per_token|per_group|dynamicquant|shuffle|"
    r"kv_buffer|set_mla|store_kv|reshape_and_cache)", re.I)
_DEF_RE = re.compile(r"^\s*def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", re.M)
# C++/HIP entry symbols: torch bindings (m.def / TORCH_LIBRARY) or __global__
_CPP_RE = re.compile(
    r'(?:m\.def\("|def\("|TORCH_LIBRARY[^{]*\{[^}]*?"|__global__[^;{]*?\b)'
    r"([A-Za-z_][A-Za-z0-9_]*)")


def _op_tags(name):
    low = name.lower()
    return sorted({tag for tag, toks in _OP_RULES
                  if any(t in low for t in toks)})


# A kernel's NAME is a lossy projection of what it does: fused_allreduce_rmsnorm
# takes a `residual` arg (it does the add) but the name never says "add". So we
# add two more, non-name sources of truth.
#
# (1) SIGNATURE tags — param names reveal ops the name omits. Only py_import
#     kernels carry a signature; it is real evidence, not a guess.
_SIG_RULES = [
    ("add_residual", ("residual", "res_in", "residual_in", "res_out")),
    ("rope", ("cos", "sin", "freqs", "positions", "rotary", "rope")),
    ("kv_cache", ("kv_buffer", "kv_cache", "kvcache", "slot_mapping", "loc",
                  "cache_k", "cache_v", "k_cache", "v_cache")),
    ("quant", ("scale", "fp8", "qscale", "x_scale", "y_scale")),
]


def _sig_tags(sig):
    low = (sig or "").lower()
    return {tag for tag, toks in _SIG_RULES if any(t in low for t in toks)}


# (2) IMPLIED-op closure — some ops are BUNDLED modifiers, not independent ops.
# add_residual travels with norm: the fused norm-kernel families always ship an
# add-variant (add_rmsnorm alongside rmsnorm). So a kernel that does `norm` is
# credited with being able to do `add_residual` too, and a region that needs
# `add_residual+norm` matches it. This is a TARGETED relaxation of the strict
# `region subset of kernel` rule for a genuinely-universal pairing — NOT a
# global loosening (rope/quant/kv_cache/gemm stay strict, where a false match
# would be costly). Extend only when a pairing is truly universal in this build.
_IMPLIED_OPS = {"norm": ("add_residual",)}


def _expand_implied(tags):
    out = set(tags)
    for tag in list(out):
        out.update(_IMPLIED_OPS.get(tag, ()))
    return out


def _dtype_tags(name):
    low = name.lower()
    tags = {tag for tag, toks in _DTYPE_RULES
            if any(t in low for t in toks)}
    # blockscale/fp4-variant are SUBTYPES of the base precision: a region that
    # only knows it is "fp8" must still match an "fp8_blockscale" kernel (and
    # must NOT match an fp4 one). Make the subtype imply its base.
    if "fp8_blockscale" in tags:
        tags.add("fp8")
    return sorted(tags)


def _is_noise(name):
    low = name.lower()
    if name in _NOISE_EXACT or low.startswith(_NOISE_PREFIX):
        return True
    return any(s in low for s in _NOISE_SUBSTR)


def _keep(name):
    return (not _is_noise(name)) and bool(_FUSED_HINT.search(name))


def enumerate_py_import(catalog):
    """Importable callables under aiter.ops.* (misses lazy @compile_ops)."""
    import importlib
    import inspect
    import pkgutil
    import aiter.ops as ops
    for mod_info in pkgutil.iter_modules(ops.__path__):
        modname = "aiter.ops." + mod_info.name
        try:
            mod = importlib.import_module(modname)
        except Exception:
            continue
        for attr in dir(mod):
            if not _keep(attr):
                continue
            obj = getattr(mod, attr, None)
            if not callable(obj):
                continue
            if not str(getattr(obj, "__module__", "")).startswith("aiter"):
                continue
            try:
                sig = str(inspect.signature(obj))
            except (ValueError, TypeError):
                sig = ""
            _add(catalog, attr, "py_import", modname, sig)


def _scan_file(catalog, path, rel, py_source):
    if path.endswith(".py"):
        try:
            text = open(path, encoding="utf-8", errors="ignore").read()
        except OSError:
            return
        for name in _DEF_RE.findall(text):
            if _keep(name):
                _add(catalog, name, py_source, rel, "")
    elif path.endswith((".cu", ".cpp", ".hip", ".cuh")):
        try:
            text = open(path, encoding="utf-8", errors="ignore").read()
        except OSError:
            return
        for name in _CPP_RE.findall(text):
            if _keep(name):
                _add(catalog, name, "csrc", rel, "")


def enumerate_source(catalog, root, py_source="py_def"):
    """`def NAME(` (+ csrc symbols) in a python/native tree. Catches lazy
    @compile_ops (aiter) + provider triton/fused seams. `root` may be a directory
    OR a single file; `py_source` labels the provider (aiter/sglang/vllm/...)."""
    anchor = os.path.dirname(root.rstrip("/"))
    if os.path.isfile(root):
        _scan_file(catalog, root, os.path.relpath(root, anchor), py_source)
        return
    for base, _dirs, files in os.walk(root):
        for fn in files:
            path = os.path.join(base, fn)
            _scan_file(catalog, path, os.path.relpath(path, anchor), py_source)


def enumerate_trace(catalog, trace_path):
    """Kernel names actually seen in a clean trace = ground-truth present."""
    opener = gzip.open if trace_path.endswith(".gz") else open
    try:
        with opener(trace_path, "rt") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return
    events = data.get("traceEvents", data) if isinstance(data, dict) else data
    for ev in events or []:
        raw = ev.get("name", "")
        # demangle the leading identifier out of a C++ symbol
        m = re.search(r"([A-Za-z_][A-Za-z0-9_]{3,})", raw)
        if not m:
            continue
        name = m.group(1)
        if _keep(name):
            _add(catalog, name, "trace", "", "", observed=True)


def _add(catalog, name, source, module, sig, observed=False):
    entry = catalog.setdefault(name, {
        "name": name,
        "op_tags": set(_op_tags(name)),
        "dtype_tags": _dtype_tags(name),
        "modules": set(),
        "signatures": set(),
        "sources": set(),
        "observed_in_trace": False,
    })
    if module:
        entry["modules"].add(module)
    if sig:
        entry["signatures"].add(sig)
        # signature params reveal ops the name omits (residual -> add_residual…)
        entry["op_tags"].update(_sig_tags(sig))
    entry["sources"].add(source)
    if observed:
        entry["observed_in_trace"] = True


# Providers to scan. A catalog limited to one library (the DSR1 2026-08-28 bug:
# aiter-only) is blind to every fused op another installed backend ships — vllm
# has fused_add_rms_norm / fused_moe / silu_mul_fp8_quant, sglang has the MLA
# set_mla_kv_buffer_triton_fp8_quant kv-write+quant, etc. Each provider is
# AUTO-DETECTED (import, else a fallback source root — editable installs often
# have __file__=None); a new backend is picked up by adding one row here, or by
# --extra-provider name=/path at the CLI. `subdirs` scopes the walk to the fused
# hotspots so we don't crawl an entire multi-GB package (torch is skipped for
# that reason — its fusions are inductor-generated, not named kernels).
PROVIDERS = (
    {"name": "aiter", "import": "aiter",
     "roots": ["/sgl-workspace/aiter/aiter"], "subdirs": [""]},
    {"name": "sglang", "import": "sglang",
     "roots": ["/sgl-workspace/sglang/python/sglang"], "subdirs": ["srt"]},
    {"name": "vllm", "import": "vllm", "roots": [],
     "subdirs": ["model_executor", "_custom_ops.py", "attention"]},
    {"name": "flashinfer", "import": "flashinfer", "roots": [], "subdirs": [""]},
)


def _provider_root(provider):
    """Resolve a provider's package dir: import first, else a fallback path.

    Editable installs (sglang here) can import with __file__=None or fail; the
    fallback roots make discovery robust to that."""
    try:
        mod = __import__(provider["import"])
        path = getattr(mod, "__file__", None)
        if path:
            return os.path.dirname(path)
    except Exception:
        pass
    for root in provider.get("roots", []):
        if os.path.exists(root):
            return root
    return None


def build(out_path, trace_path="", extra_providers=None):
    catalog = {}
    scanned = []
    providers = list(PROVIDERS) + list(extra_providers or [])
    for provider in providers:
        root = _provider_root(provider)
        if not root:
            continue
        # aiter also gets a python-import pass (real signatures for ops.*).
        if provider["name"] == "aiter":
            try:
                enumerate_py_import(catalog)
            except Exception:
                pass
        for subdir in provider.get("subdirs", [""]):
            target = os.path.join(root, subdir) if subdir else root
            if os.path.exists(target):
                enumerate_source(catalog, target, py_source=provider["name"])
        scanned.append(provider["name"])
    print("[fusion_catalog] scanned providers: %s" % ", ".join(scanned or ["<none>"]))
    if trace_path:
        enumerate_trace(catalog, trace_path)

    entries = []
    for entry in catalog.values():
        # keep only entries that carry at least one op tag (a fusion needs one)
        if not entry["op_tags"]:
            continue
        entries.append({
            "name": entry["name"],
            "op_tags": sorted(entry["op_tags"]),
            "dtype_tags": entry["dtype_tags"],
            "modules": sorted(entry["modules"]),
            "signatures": sorted(entry["signatures"])[:2],
            "sources": sorted(entry["sources"]),
            "observed_in_trace": entry["observed_in_trace"],
        })
    entries.sort(key=lambda e: e["name"])
    result = {
        "schema_version": 2,
        "artifact": "available_fusion_kernels",
        # The catalog DECLARES its own scope: whoever reads it knows which
        # backends were scanned and — by omission — which are blind spots
        # (a provider not listed here was not installed / not scanned).
        "providers_scanned": scanned,
        "op_tag_vocab": [t for t, _ in _OP_RULES],
        "dtype_tag_vocab": [t for t, _ in _DTYPE_RULES],
        "kernel_count": len(entries),
        "kernels": entries,
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2)
    return result


# --- query helper the harnesses import ---------------------------------------
def load_index(catalog_path):
    """Return (kernels_by_name, list). Query with covers()."""
    data = json.load(open(catalog_path))
    kernels = data.get("kernels", [])
    return {k["name"]: k for k in kernels}, kernels


def covers(kernels, region_op_tags, region_dtype_tags=None):
    """Kernels whose op_tags superset region_op_tags and dtype-compatible.

    dtype compatibility: if the region declares a dtype family, at least one of
    the kernel's dtype_tags must match (or the kernel declares none = generic).
    Returns the matching kernel entries, widest-coverage first.
    """
    want = set(region_op_tags or [])
    if not want:
        return []
    want_dt = set(region_dtype_tags or [])
    hits = []
    for k in kernels:
        if not want.issubset(_expand_implied(k["op_tags"])):
            continue
        kdt = set(k["dtype_tags"])
        if want_dt and kdt and not (want_dt & kdt):
            continue  # region is fp8 but kernel is fp4-only -> not a match
        hits.append(k)
    hits.sort(key=lambda k: len(k["op_tags"]), reverse=True)
    return hits


# --- fusion strategy priors (knowledge/fusion) -------------------------------
def load_strategies(path):
    """Load knowledge/fusion/fusion_strategies.json -> list of strategies."""
    return json.load(open(path)).get("strategies", [])


def match_strategies(strategies, region_op_tags, region_dtype_tags=None):
    """Known strategies whose op_set covers the region — the same containment
    test as covers(), but against the provider-agnostic PRIOR (used to fill a
    gap the installed-kernel scan could not). Widest op_set first."""
    want = set(region_op_tags or [])
    if not want:
        return []
    want_dt = set(region_dtype_tags or [])
    hits = []
    for strat in strategies:
        if not want.issubset(_expand_implied(strat.get("op_set", []))):
            continue
        sdt = set(strat.get("dtype", []))
        if want_dt and sdt and not (want_dt & sdt):
            continue
        hits.append(strat)
    hits.sort(key=lambda s: len(s.get("op_set", [])), reverse=True)
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--trace", default="")
    ap.add_argument(
        "--extra-provider", action="append", default=[], metavar="NAME=PATH",
        help="scan an additional backend not in the built-in registry, e.g. "
             "--extra-provider mylib=/path/to/mylib. Repeatable.")
    args = ap.parse_args()
    extra = []
    for spec in args.extra_provider:
        if "=" in spec:
            name, path = spec.split("=", 1)
            extra.append({"name": name.strip(), "import": name.strip(),
                          "roots": [path.strip()], "subdirs": [""]})
    res = build(args.out, args.trace, extra_providers=extra)
    print(json.dumps({"kernel_count": res["kernel_count"],
                      "providers": res["providers_scanned"], "out": args.out}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
