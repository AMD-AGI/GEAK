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

Only real kernel entry points are kept: predicates (is_/should_/can_/has_/
check_), *_fake / *_cpu / *_npu fallbacks and config/tuner helpers are dropped
(is_kernel_name). Op tags come from whole name TOKENS (plus signature params),
never from bare substrings -- "gemma" is not "gemm", "gating" is not "topk".

observed_in_trace is set by matching every catalog kernel against the kernel
names of a clean trace (--trace) or of the semantic table (--semantic-table):
by token stem (add_rmsnorm_quant in add_rmsnorm_quant_kernel<...>) or, for a
multi-op trace kernel, by its op set (allreduce_fusion_kernel_1stage = AR + add
+ RMSNorm marks fused_allreduce_rmsnorm). A catalog entry seen there is
"already fused in the baseline".

Output: available_fusion_kernels.json — a list of
  {name, op_tags[], dtype_tags[], modules[], sources[], observed_in_trace}
plus the tag vocab, the tagger version and the providers scanned (so the
catalog declares its scope). Readers re-tag catalogs from an older tagger
(sanitize_kernels / load_index).

Run INSIDE the runtime container (needs the providers importable):
  python3 fusion_catalog.py --out <dir>/available_fusion_kernels.json \
      [--trace <rank0.trace.json.gz>] [--semantic-table <table.json>]
      [--extra-provider name=/path]
"""
import argparse
import gzip
import json
import os
import re

# --- semantic tag vocabulary (name TOKENS -> op tag) --------------------------
# One shared vocabulary with the candidate harness's region op-tags and the
# fusion screen's chain tags, so "region op-set subset of kernel op_tags" is a
# real containment test.
#
# Matching is on whole name TOKENS, never on bare substrings. Substrings were
# the root of the Qwen3.5-397B (2026-10-11) mis-screen: "gemma" contained
# "gemm" (qk_norm became a GEMM barrier), "gating" tagged fused_gdn_gating /
# fused_sigmoid_gating_delta_rule_update / ff_a16w16_fused_gated as topk, and
# "wvsplit" never matched wv_splitk. A phrase is a "_"-joined token sequence
# that must appear at token boundaries; a trailing "*" lets the LAST token be a
# prefix ("rmsnorm*" also matches rmsnorm2d). Names are tokenized after
# demangling Itanium symbols and splitting camelCase (topkGatingSoftmax ->
# topk, gating, softmax).
TAGGER_VERSION = "token-v1"
_OP_RULES = [
    ("norm", ("rmsnorm*", "rms_norm", "layernorm*", "layer_norm", "groupnorm",
              "group_norm", "l2norm", "norm", "gemma_norm", "ar_rms", "rms")),
    ("add_residual", ("with_add", "add_rmsnorm*", "add_rms", "add_layernorm*",
                      "fused_add*", "residual*", "res_in")),
    ("quant", ("~quant", "~lowprec", "quantize*", "fp8*", "fp4*", "a8w8", "a4w4",
               "afp4wfp4", "blockscale", "e4m3*", "e5m2*", "mxfp*")),
    ("rope", ("rope*", "rotary*", "mrope*", "pos_encoding")),
    ("kv_cache", ("kv_cache", "kvcache", "store_kvcache", "store_kv",
                  "store_cache", "cache_quant", "reshape_and_cache*",
                  "concat_and_cache*", "kv_buffer", "set_mla_kv*", "kv_write",
                  "~cache")),
    ("cast", ("cast", "convert", "concat_and_cast")),
    ("layout", ("shuffle", "layout", "trans_ragged", "permute", "transpose",
                "split", "reshape", "cat", "concat")),
    ("allreduce", ("allreduce*", "all_reduce*", "fused_ar", "custom_fused_ar*",
                   "cross_device_reduce*", "quick_reduce", "quickreduce",
                   "collective")),
    ("moe", ("moe*", "fmoe*", "expert", "experts", "g1u1", "ck_moe*")),
    ("topk", ("topk*", "top_k", "grouped_topk", "moe_fused_gate",
              "moe_sorting*", "moe_sort", "append_shared_experts*")),
    ("activation", ("silu*", "gelu*", "act_mul*", "act_and_mul*", "activation",
                    "swiglu", "geglu")),
    ("gemm", ("~gemm", "bmm", "matmul", "batched_gemm", "cijk", "wv_splitk*",
              "wvsplitk*", "hipblaslt*", "skinny_gemm")),
    ("gemm_prologue", ("prequant", "prologue", "a_per_token", "a_per_group")),
    ("gemm_epilogue", ("epilogue", "bias")),
]
_DTYPE_RULES = [
    ("fp8_blockscale", ("blockscale", "block_quant", "per_group", "per_1x128",
                        "group_quant", "a8w8")),
    ("fp8", ("fp8*", "f8", "e4m3*", "e5m2*", "float8*")),
    ("fp4", ("fp4", "f4", "mxfp4", "a4w4", "afp4wfp4", "e2m1")),
    ("bf16", ("bf16", "bfloat16", "a16w16")),
    ("fp16", ("fp16", "float16", "half")),
]
# Quant granularity is part of a kernel's contract: a per-token kernel is not a
# drop-in for a per-group-128 seam even though both carry the "quant" tag.
_GRANULARITY_RULES = [
    ("group", ("per_group", "group_quant", "blockscale", "block_quant",
               "per_1x128", "1x128", "group")),
    ("token", ("per_token", "pertoken", "token", "dynamicquant",
               "smoothquant")),
    ("tensor", ("per_tensor", "tensorwise", "static_quant", "static")),
]
# GEMM operand-precision tokens; a name with two of them runs two GEMMs
# (fused_gemm_a8w8_blockscale_a16w16 = one fp8 + one bf16 GEMM, shared input).
_GEMM_PRECISION = re.compile(r"^a(?:\d+|fp\d+)w(?:\d+|fp\d+)$")
# names that are NOT kernels: decorators, generators, config/tuner helpers
_NOISE_EXACT = {"compile_ops", "torch_compile_guard", "per_block_quant_wrapper"}
_NOISE_PREFIX = ("gen_", "get_", "compute_", "cmdgenfunc", "_", "test_")
_NOISE_SUBSTR = ("_fake_tensor", "_tune_", "_config", "_torch", "fake_tensors")
# Predicates, dispatch switches and per-device fallbacks. They mention a fused
# op in their name but never launch it here, so as catalog entries they are
# false "existing kernel" answers (is_activation_quantization_format "covered"
# SiLU+quant on Qwen3; can_use_custom_all_reduce_with_nvlink "covered" AR).
_NOT_KERNEL_PREFIX = ("is_", "should_", "can_", "has_", "check_", "use_",
                      "enable_", "disable_", "init_", "maybe_", "need_",
                      "supports_", "set_custom_", "set_mscclpp_")
_NOT_KERNEL_SUFFIX = ("_fake", "_cpu", "_npu", "_meta", "_ref", "_reference",
                      "_tune", "_config", "_configs")
_NOT_KERNEL_TOKENS = {"fake", "cpu", "npu", "xpu", "mps", "musa", "config",
                      "configs", "tuner", "ref", "reference", "benchmark"}
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
_CAMEL_LOWER_UPPER = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_CAMEL_ACRONYM = re.compile(r"(?<=[A-Z])(?=[A-Z][a-z])")
# A "cache" token is a KV-cache write only next to attention operands; alone it
# is too generic (rope_cache, lru caches).
_KV_CONTEXT = {"kv", "qk", "k", "v", "mla", "paged", "attn", "kvcache"}


def _demangle_identifiers(name):
    """Identifiers of an Itanium-mangled symbol (_ZN5aiter24add_rmsnorm...).

    Not a demangler: it only reads the <length><identifier> runs, which is
    all the tagger needs (aiter, add_rmsnorm_quant_kernel, silu_kernel ...).
    """
    out = []
    index = 0
    while index < len(name):
        if name[index].isdigit():
            end = index
            while end < len(name) and name[end].isdigit():
                end += 1
            length = int(name[index:end])
            ident = name[end:end + length]
            if length and len(ident) == length and re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", ident):
                out.append(ident)
                index = end + length
                continue
            index = end
            continue
        index += 1
    return out


def name_tokens(name):
    """Lower-case word tokens of a kernel / function name.

    Function arguments are dropped (they are types, not semantics); template
    arguments are kept because they often carry the op
    (elementwise_kernel<direct_copy_kernel_cuda>, act_and_mul_kernel<silu>).
    """
    text = str(name or "")
    if text.startswith("_Z"):
        parts = _demangle_identifiers(text) or [text]
    else:
        text = text.replace("(anonymous namespace)::", "")
        if text.startswith("void "):
            text = text[5:]
        text = text.split("(", 1)[0]
        parts = [text]
    tokens = []
    raw = []
    for part in parts:
        split = _CAMEL_ACRONYM.sub("_", _CAMEL_LOWER_UPPER.sub("_", part))
        tokens.extend(tok for tok in re.split(r"[^a-z0-9]+", split.lower()) if tok)
        raw.extend(tok for tok in re.split(r"[^a-z0-9]+", part.lower()) if tok)
    # camelCase splitting breaks acronyms (RoPE -> ro, pe), so the unsplit
    # tokens ride along after an empty separator that no phrase can span.
    extra = [tok for tok in raw if tok not in tokens]
    return tokens + ([""] + extra if extra else [])


def _phrase_hit(tokens, phrase):
    """`phrase` ("a_b", "a_b*", "~sub") at token boundaries in `tokens`."""
    if phrase.startswith("~"):
        word = phrase[1:]
        if word == "quant":
            return any(tok.endswith("quant") or tok.startswith("quant")
                       for tok in tokens)
        if word == "gemm":
            return any(re.match(r"^gemm\d*$", tok) for tok in tokens)
        if word == "lowprec":
            return any(_GEMM_PRECISION.match(tok) and not re.match(r"^a16w16$", tok)
                       for tok in tokens)
        if word == "cache":
            return "cache" in tokens and bool(_KV_CONTEXT & set(tokens))
        return word in tokens
    prefix = phrase.endswith("*")
    words = (phrase[:-1] if prefix else phrase).split("_")
    span = len(words)
    for start in range(len(tokens) - span + 1):
        window = tokens[start:start + span]
        if window[:-1] != words[:-1]:
            continue
        last = window[-1]
        if last == words[-1] or (prefix and last.startswith(words[-1])):
            return True
    return False


def has_phrase(name_or_tokens, *phrases):
    tokens = (name_or_tokens if isinstance(name_or_tokens, list)
              else name_tokens(name_or_tokens))
    return any(_phrase_hit(tokens, phrase) for phrase in phrases)


def _op_tags(name):
    tokens = name_tokens(name)
    return sorted({tag for tag, phrases in _OP_RULES
                   if any(_phrase_hit(tokens, p) for p in phrases)})


def quant_granularity(name):
    """per-group / per-token / per-tensor from the name, or None if it does not say."""
    tokens = name_tokens(name)
    for granularity, phrases in _GRANULARITY_RULES:
        if any(_phrase_hit(tokens, p) for p in phrases):
            return granularity
    return None


def gemm_precisions(name):
    """GEMM operand-precision tokens (a8w8, a16w16, afp4wfp4) in name order."""
    return [tok for tok in name_tokens(name) if _GEMM_PRECISION.match(tok)]


def is_kernel_name(name):
    """False for predicates, fakes, per-device fallbacks and config helpers."""
    low = str(name or "").lower()
    if not low:
        return False
    if low.startswith(_NOT_KERNEL_PREFIX) or low.endswith(_NOT_KERNEL_SUFFIX):
        return False
    return not (_NOT_KERNEL_TOKENS & set(name_tokens(name)))


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
    tokens = name_tokens(name)
    tags = {tag for tag, phrases in _DTYPE_RULES
            if any(_phrase_hit(tokens, p) for p in phrases)}
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
    if any(s in low for s in _NOISE_SUBSTR):
        return True
    return not is_kernel_name(name)


def _keep(name):
    return (not _is_noise(name)) and bool(_FUSED_HINT.search(name))


def sanitize_kernels(kernels, tagger=None, require_tags=True):
    """Catalog entries the screen may match against, plus what was dropped.

    Drops non-kernel entries (predicates, fakes, per-device fallbacks, config
    helpers). A catalog written by an older tagger (no `tagger` field, or a
    different version) carries substring-derived op_tags, so those are
    re-derived from the name tokens plus the stored signatures -- the only
    other evidence an old catalog kept. Entries from the current tagger are
    trusted as written.
    """
    kept, dropped = [], []
    for kernel in kernels or []:
        entry = {"name": kernel} if isinstance(kernel, str) else dict(kernel)
        name = str(entry.get("name") or "")
        if not name:
            continue
        if not is_kernel_name(name):
            dropped.append({"name": name, "reason": "not a kernel entry point"})
            continue
        if tagger != TAGGER_VERSION or not entry.get("op_tags"):
            tags = set(_op_tags(name))
            for sig in entry.get("signatures") or []:
                tags |= _sig_tags(sig)
            stale = sorted(set(entry.get("op_tags") or []) - tags)
            if stale:
                entry["retag_dropped"] = stale
            entry["op_tags"] = sorted(tags)
            entry["dtype_tags"] = _dtype_tags(name)
        if entry.get("dtype_tags") is None:
            entry["dtype_tags"] = _dtype_tags(name)
        entry.setdefault("sources", [])
        entry.setdefault("observed_in_trace", False)
        if require_tags and not entry["op_tags"]:
            dropped.append({"name": name, "reason": "no op tag after token re-tagging"})
            continue
        kept.append(entry)
    return kept, dropped


def _stem(tokens):
    """Name tokens without the generic kernel / launcher / provider words."""
    generic = {"kernel", "kernels", "fwd", "impl", "void", "aiter", "sglang",
               "vllm", "triton", "hip", "ck", "cuda", "launch", "op", "ops",
               "native", "at", "sgl", "torch", "run", "forward"}
    return [tok for tok in tokens if tok and tok not in generic and not tok.isdigit()]


_NAMESPACES = {"aiter", "ck_tile", "ck", "vllm", "moe", "at", "native", "sglang",
               "sgl_kernel", "quickreduce", "std", "void", "detail", "internal"}


def primary_identifier(trace_name):
    """The kernel's own identifier, without namespaces and template/function args.

    Matching a catalog name against the WHOLE mangled symbol let template
    arguments answer for the kernel (an FMHA prefill kernel "matched"
    reshape_and_cache_* through its KV-cache layout enums).
    """
    text = str(trace_name or "")
    if text.startswith("_Z"):
        for ident in _demangle_identifiers(text):
            if ident.lower() not in _NAMESPACES:
                return ident
        return text
    text = text.replace("(anonymous namespace)::", "")
    if text.startswith("void "):
        text = text[5:]
    text = text.split("(", 1)[0].split("<", 1)[0]
    return text.split("::")[-1]


def trace_kernel_matches(kernel, trace_name):
    """Length of the catalog kernel's token stem inside the trace kernel's
    primary identifier (0 = no match).

    add_rmsnorm_quant matches _ZN5aiter24add_rmsnorm_quant_kernel... with 3;
    add_rmsnorm also matches with 2 and loses to the longer stem in
    mark_observed. Multi-op trace kernels whose name does not spell their ops
    are matched by op set instead (fused_trace_tags).
    """
    want = _stem(name_tokens(kernel["name"]))
    have = _stem(name_tokens(primary_identifier(trace_name)))
    if len(want) < 2 or len(want) > len(have):
        return 0
    for i in range(len(have) - len(want) + 1):
        if have[i:i + len(want)] == want:
            return len(want)
    return 0


# Fused trace kernels whose name does not spell every op they run. The aiter
# allreduce_fusion kernel is the custom all-reduce with the residual add and
# RMSNorm folded in (--enable-aiter-allreduce-fusion); naming it AR alone made
# the baseline look unfused.
_FUSED_TRACE_SEMANTICS = (
    (("allreduce_fusion*",), ("allreduce", "add_residual", "norm")),
    (("all_reduce_fusion*",), ("allreduce", "add_residual", "norm")),
    (("fused_allreduce_rmsnorm*",), ("allreduce", "add_residual", "norm")),
)


def fused_trace_tags(name):
    """Op tags a trace kernel actually runs (name tags + known fused semantics)."""
    tags = set(_op_tags(name))
    tokens = name_tokens(name)
    for phrases, extra in _FUSED_TRACE_SEMANTICS:
        if any(_phrase_hit(tokens, p) for p in phrases):
            tags |= set(extra)
    return tags


def mark_observed(kernels, trace_names):
    """Set observed_in_trace / observed_as on catalog entries seen in the trace.

    Returns {trace_name: [catalog names]} for the report. A multi-op trace
    kernel with no name match is matched semantically to the catalog kernels
    that run exactly its op set (so allreduce_fusion_kernel_1stage marks
    fused_allreduce_rmsnorm as already in the baseline).
    """
    found = {}
    names = sorted({str(n) for n in trace_names or [] if n})
    for trace_name in names:
        scored = [(trace_kernel_matches(k, trace_name), k) for k in kernels]
        best = max([score for score, _ in scored] or [0])
        hits = [k for score, k in scored if score and score == best]
        tags = fused_trace_tags(trace_name) - {"gemm_prologue", "gemm_epilogue"}
        if not hits and tags != set(_op_tags(trace_name)) - {"gemm_prologue", "gemm_epilogue"}:
            # Known fused semantics the name does not spell: match by op set.
            hits = [k for k in kernels
                    if _expand_implied(k.get("op_tags") or []) >= tags
                    and set(k.get("op_tags") or []) <= _expand_implied(tags)]
        for kernel in hits:
            kernel["observed_in_trace"] = True
            seen = kernel.setdefault("observed_as", [])
            if trace_name not in seen:
                seen.append(trace_name)
        if hits:
            found[trace_name] = sorted(k["name"] for k in hits)
    return found


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


def trace_kernel_names(trace_path):
    """GPU kernel names in a clean trace (ground truth for observed_in_trace)."""
    opener = gzip.open if trace_path.endswith(".gz") else open
    try:
        with opener(trace_path, "rt") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return set()
    events = data.get("traceEvents", data) if isinstance(data, dict) else data
    names = set()
    for ev in events or []:
        if not isinstance(ev, dict):
            continue
        cat = str(ev.get("cat") or "").lower()
        if cat and cat not in ("kernel", "gpu_op", "cuda_runtime_kernel"):
            continue
        if ev.get("name"):
            names.add(str(ev["name"]))
    return names


def table_kernel_names(table_path):
    """Kernel names in a semantic pattern-layer table (raw_name per row)."""
    try:
        with open(table_path) as fh:
            table = json.load(fh)
    except (OSError, ValueError):
        return set()
    names = set()
    for spec in table.get("tables") or []:
        for row in spec.get("rows") or []:
            name = row.get("raw_name") or row.get("short_name")
            if name:
                names.add(str(name))
    return names


def enumerate_trace(catalog, trace_path):
    """Back-compat shim: mark existing entries seen in the trace.

    Trace names used to be ADDED as catalog entries; their mangled leading
    identifier (_ZN5aiter24add_rmsnorm...) was never a usable kernel name and
    nothing else was marked observed. Now the catalog entries are matched
    against the trace names instead (see mark_observed).
    """
    names = trace_kernel_names(trace_path)
    entries = [dict(entry, op_tags=sorted(entry["op_tags"])) for entry in catalog.values()]
    mark_observed(entries, names)
    for entry in entries:
        if entry.get("observed_in_trace"):
            catalog[entry["name"]]["observed_in_trace"] = True
            catalog[entry["name"]]["observed_as"] = entry.get("observed_as", [])
    return names


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
    # vllm: _aiter_ops.py wraps the aiter fused ops vllm routes to on ROCm,
    # third_party/flash_linear_attention holds the GDN chunk kernels, kernels/ the
    # standalone triton fusions, compilation/passes the fused ops its graph passes
    # substitute (e.g. rms+quant) -- all were invisible to candidates' existing_apis.
    {"name": "vllm", "import": "vllm", "roots": [],
     "subdirs": ["model_executor", "_custom_ops.py", "attention", "_aiter_ops.py",
                 "kernels", "third_party/flash_linear_attention",
                 "compilation/passes", "v1/attention"]},
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


def build(out_path, trace_path="", extra_providers=None, semantic_table=""):
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
    trace_names = set()
    if trace_path:
        trace_names |= trace_kernel_names(trace_path)
    if semantic_table:
        trace_names |= table_kernel_names(semantic_table)

    entries = []
    for entry in catalog.values():
        # keep only entries that carry at least one op tag (a fusion needs one)
        if not entry["op_tags"] or not is_kernel_name(entry["name"]):
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
    observed = mark_observed(entries, trace_names) if trace_names else {}
    result = {
        "schema_version": 3,
        "artifact": "available_fusion_kernels",
        # Tags were derived by this token tagger; older catalogs (no field)
        # are re-tagged on load by sanitize_kernels().
        "tagger": TAGGER_VERSION,
        "observed_trace_kernels": observed,
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
    """Return (kernels_by_name, list). Query with covers().

    Non-kernel entries are dropped and a catalog from an older tagger is
    re-tagged (sanitize_kernels), so a stale catalog file cannot answer
    "existing kernel" with a predicate or a substring-derived tag.
    """
    data = json.load(open(catalog_path))
    kernels, _dropped = sanitize_kernels(
        data.get("kernels", []) if isinstance(data, dict) else data,
        data.get("tagger") if isinstance(data, dict) else None, require_tags=False)
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
        "--semantic-table", default="",
        help="pattern_layer_kernel_table.json; its kernel names mark catalog "
             "entries observed_in_trace (baseline already runs them)")
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
    res = build(args.out, args.trace, extra_providers=extra,
                semantic_table=args.semantic_table)
    print(json.dumps({"kernel_count": res["kernel_count"],
                      "providers": res["providers_scanned"], "out": args.out}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
