
#!/usr/bin/env python3
"""EvoK library lookup for the GEAK e2e workflow — DETERMINISTIC, no agent judgement.

Answers exactly one question for one extracted op task:
  "Does the EvoK library already hold an implementation of this op for this arch,
   and if so, WHERE is it, HOW is it called, and is it safe to rebind into the live server?"

Emits ONE json object on stdout (and optionally to --out). Never raises on a miss:
a miss is `{"hit": false, "reason": "..."}` with exit 0, so the caller can treat
"EvoK has nothing" as a normal, boring outcome.

GEAK is READ-ONLY w.r.t. EvoK: this script imports evok and reads the kernels tree.
It writes nothing under --evok-root.

Usage:
  python3 evok_lookup.py --evok-root /wekafs/EvoK --task <op_task_dir> --gfx gfx942
                         [--regime prefill|decode] [--shape-set Qwen3-14B-FP8] [--out f.json]
"""
import argparse, importlib, inspect, json, os, re, sys, traceback


# ---------------------------------------------------------------- vocabulary alignment
# GEAK's extracted-task vocabulary (op_kind / kernel_class / quant_scheme / short_name)
# -> EvoK's `spec/meta.json:resolve_key`. This table is the ONLY place the two repos'
# names are joined; everything downstream is data read out of the leaf itself.
#
# Rules are evaluated top-to-bottom; the first whose predicate matches wins.
# Keep this table EXPLICIT — a wrong guess here silently benches the wrong kernel.
#
# The full set of keys evok currently answers to (registry().resolve_keys(), 54 of them,
# across 65 leaves / 12 op_classes -- re-read 2026-08-11; was 41/51 when this table was
# first written, so DO NOT trust a stale copy of this list, re-run:
#     python3 -c "import evok; print(len(evok.registry().resolve_keys()))"
#   activation.silu_mul  activation.silu_mul_per_block_quant
#   activation.silu_mul_per_token_group_quant_colmajor  activation.swiglu_oai
#   attention.context_prefill  attention.index_block_score_decode  attention.mla
#   attention.mla_decode  attention.mla_prefill  attention.paged_decode
#   attention.paged_decode_2d  attention.sparse_decode  attention.split_kv_reduce
#   attention.unified  attention.unified_diffkv  comm.tp_collective
#   elementwise.chunk_copy  elementwise.copy  elementwise.mrope_gather_zerofill
#   elementwise.slice_copy  gemm.dense  gemm.skinny_decode  gemm.split_k_reduce
#   kv.mla_concat  kv.reshape_cache  kv.reshape_cache_diffkv  kv.reshape_cache_flash
#   linear_attn.chunk_output  linear_attn.chunk_scan  linear_attn.conv1d_update
#   linear_attn.recurrent  misc.quant  misc.rope  moe.align_block_size  moe.bitmatrix
#   moe.bitmatrix_meta  moe.combine  moe.combine_masked  moe.gating  moe.gating_softmax
#   moe.grouped_gating  moe.grouped_matmul  moe.permute  moe.ragged_metadata  moe.routing
#   norm.fused_add_rmsnorm  norm.gated_rmsnorm  norm.gated_rmsnorm_per_token_group_quant
#   norm.gemma_fused_add_rmsnorm  norm.gemma_fused_add_rmsnorm_single_site
#   norm.qk_norm_rope  norm.qk_rmsnorm_sumsq  norm.rmsnorm_per_block_quant
#   reduction.reduce
#
# NOTE the trap this list exists to prevent: `activation.silu_mul_per_block_quant` and
# `activation.silu_mul_per_token_group_quant_colmajor` are BOTH present and BOTH plausible
# for a "silu + quant" task, but only the second one is the key of the Qwen3-14B-FP8 leaf
# (kernels/activation/silu_mul_block_scaled/bf16). Picking the first benches a different
# kernel and reports a perfectly well-formed wrong number.
def _has(s, *needles):
    s = str(s or "").lower()
    return any(n in s for n in needles)


RULES = [
    # (predicate(meta) -> bool,                                        resolve_key)
    (lambda m: m.get("op_kind") == "gemm" and _has(m.get("kernel_class"), "moe", "grouped", "experts"),
     "moe.grouped_matmul"),
    (lambda m: m.get("op_kind") == "gemm" and _has(m.get("quant_scheme"), "blockscale", "block_scaled"),
     "gemm.dense"),
    (lambda m: m.get("op_kind") == "gemm",
     "gemm.dense"),
    (lambda m: m.get("op_kind") == "norm" and _has(m.get("short_name"), "fused_add", "add_rms"),
     "norm.fused_add_rmsnorm"),
    (lambda m: m.get("op_kind") == "norm" and _has(m.get("short_name"), "qk_norm", "rope"),
     "norm.qk_norm_rope"),
    (lambda m: m.get("op_kind") == "norm" and _has(m.get("short_name"), "quant"),
     "norm.rmsnorm_per_block_quant"),
    (lambda m: m.get("op_kind") in ("quant", "misc") and _has(m.get("short_name"), "quant"),
     "misc.quant"),
    # colmajor FIRST: the Qwen3-14B-FP8 leaf kernels/activation/silu_mul_block_scaled/bf16
    # declares resolve_key=activation.silu_mul_per_token_group_quant_colmajor. The older
    # ..._per_block_quant key also exists (a different leaf) -- matching it here would bench
    # the wrong kernel silently. Keep the colmajor rule above it.
    (lambda m: _has(m.get("short_name"), "silu") and _has(m.get("short_name"), "quant")
               and _has(m.get("short_name"), "colmajor", "per_token_group", "token_group"),
     "activation.silu_mul_per_token_group_quant_colmajor"),
    (lambda m: _has(m.get("short_name"), "silu") and _has(m.get("short_name"), "quant"),
     "activation.silu_mul_per_token_group_quant_colmajor"),
    (lambda m: _has(m.get("short_name"), "swiglu") and _has(m.get("short_name"), "oai"),
     "activation.swiglu_oai"),
    (lambda m: _has(m.get("short_name"), "silu", "swiglu"),
     "activation.silu_mul"),
    (lambda m: m.get("op_kind") == "norm" and _has(m.get("short_name"), "gemma"),
     "norm.gemma_fused_add_rmsnorm"),
    (lambda m: m.get("op_kind") == "norm" and _has(m.get("short_name"), "gated")
               and _has(m.get("short_name"), "quant"),
     "norm.gated_rmsnorm_per_token_group_quant"),
    (lambda m: m.get("op_kind") == "norm" and _has(m.get("short_name"), "sumsq"),
     "norm.qk_rmsnorm_sumsq"),
    (lambda m: _has(m.get("short_name"), "mrope") and _has(m.get("short_name"), "gather"),
     "elementwise.mrope_gather_zerofill"),
    (lambda m: _has(m.get("short_name"), "rope"),
     "misc.rope"),
    (lambda m: _has(m.get("short_name"), "reshape_cache", "kv_cache"),
     "kv.reshape_cache"),
    (lambda m: m.get("op_kind") == "attn" and _has(m.get("short_name"), "prefill", "context"),
     "attention.context_prefill"),
    (lambda m: m.get("op_kind") == "attn" and _has(m.get("short_name"), "mla"),
     "attention.mla_decode"),
    (lambda m: m.get("op_kind") == "attn" and _has(m.get("short_name"), "sparse", "index_block"),
     "attention.sparse_decode"),
    (lambda m: m.get("op_kind") == "attn",
     "attention.paged_decode"),
    (lambda m: m.get("op_kind") in ("linear_attn", "ssm", "mamba")
               and _has(m.get("short_name"), "conv1d"),
     "linear_attn.conv1d_update"),
    (lambda m: m.get("op_kind") in ("linear_attn", "ssm", "mamba"),
     "linear_attn.recurrent"),
]


# ---------------------------------------------------------------- deployability policy
# resolve_key -> (deployable, reason). Absent = deployable pending the generic checks.
# A leaf can be MEASURABLE (it competes fine in the isolated bake-off) yet NOT
# REBINDABLE into the live server. Saying so here, up front, stops the integrator from
# burning a full e2e A/B on a seam that cannot engage.
NOT_DEPLOYABLE = {
    # vLLM V1's torch.compile backbone DECOMPOSES the native fused_add_rms_norm and Inductor
    # FUSES it into triton_red_fused_fused_add_rms_norm_0 -> there is no installed-package
    # symbol to rebind. The only seam is the IR op registration
    # (vllm.ir.ops.layernorm.fused_add_rms_norm.register_impl(...)), and registering a
    # non-native impl makes it an OPAQUE call: Inductor stops fusing, BOTH _0 (1.78% GPU) and
    # _2 (1.76%) change shape at once, and the candidate must honour the full (out, newres)
    # contract. EvoK's triton_2 entry `fused_add_rms_norm_0(x, residual, weight, eps)` returns
    # ONLY `out` (kernel.py:192 `return out.reshape(x.shape)`) -> contract violation.
    # => measurable, NOT drop-in. Use it as an author-lane SEED instead.
    "norm.fused_add_rmsnorm": (False, "ir_contract_mismatch: entry returns `out` only, the vLLM IR "
                                      "seam requires (out, newres); rebinding also de-fuses Inductor "
                                      "and moves the whole norm cluster -> seed an author lane instead"),
}


# ---------------------------------------------------------------- signature classification
# How the resolved callable wants to be invoked. This travels to op_bench.py as
# `signature_form` so the bake-off adapter is pure ARGUMENT REORDERING — never a
# reshape/transpose (any data movement would contaminate the isolated timing).
#
#   vllm6   fn(A, B, As, Bs, block_size: list[int], output_dtype) -> C     [EvoK gemm fp8]
#   sglang5 fn(x, w, x_scale, w_scale, dtype=out) -> C                     [aiter / GEAK bake-off]
#   dense3  fn(x, w, bias) -> C                                            [EvoK gemm bf16]
#
# The scale LAYOUTS already agree byte-for-byte, which is why reordering suffices:
#   op_bench._synth_blockscale_case ->  x_scale [M, ceil(K/128)] NON-transposed
#                                       w_scale [ceil(N/128), ceil(K/128)]
#   EvoK meta.math_contract         ->  As[M,ceil(K/128)] fp32 NON-transposed
#                                       Bs[ceil(N/128),ceil(K/128)] fp32
_FORM_BY_PARAMS = [
    (("a", "b", "as_", "bs"), "vllm6"),
    (("a", "b", "as", "bs"), "vllm6"),
    (("x", "w", "x_scale", "w_scale"), "sglang5"),
    (("x", "w", "bias"), "dense3"),
    (("a", "b", "bias"), "dense3"),
]


def classify_signature(fn):
    """(form, note). Name-based, with the param list echoed so a miss is diagnosable
    rather than a silent wrong call."""
    try:
        params = [p.name.lower().rstrip("_") for p in inspect.signature(fn).parameters.values()]
    except (TypeError, ValueError):
        return "", "signature not introspectable (compiled/triton object?)"
    head = tuple(p.rstrip("_") for p in params[:4])
    for want, form in _FORM_BY_PARAMS:
        w = tuple(x.rstrip("_") for x in want)
        if head[:len(w)] == w:
            return form, "params=%s" % (params,)
    # positional fallback: 6 params ending in a dtype-ish name == the vLLM form
    if len(params) >= 6 and _has(params[-1], "dtype"):
        return "vllm6", "params=%s (matched by arity+dtype tail)" % (params,)
    return "", "unrecognised signature params=%s" % (params,)


# ---------------------------------------------------------------- main
def pick_resolve_key(meta):
    for pred, key in RULES:
        try:
            if pred(meta):
                return key
        except Exception:
            continue
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evok-root", required=True)
    ap.add_argument("--task", required=True, help="GEAK op task dir (with meta.json)")
    ap.add_argument("--gfx", default=os.environ.get("EVOK_GFX", ""))
    ap.add_argument("--regime", default="")
    ap.add_argument("--shape-set", default="", dest="shape_set")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    def emit(d):
        d.setdefault("hit", False)
        s = json.dumps(d, indent=2, default=str)
        if a.out:
            with open(a.out, "w") as fh:
                fh.write(s)
        print(s)
        sys.exit(0)              # a miss is NOT an error; never break the bake-off

    try:
        with open(os.path.join(a.task, "meta.json")) as fh:
            meta = json.load(fh)
    except Exception as e:
        emit({"reason": "task meta.json unreadable: %r" % (e,)})

    key = pick_resolve_key(meta)
    if not key:
        emit({"reason": "no_resolve_key_mapping",
              "op_kind": meta.get("op_kind"), "short_name": meta.get("short_name"),
              "kernel_class": meta.get("kernel_class")})

    # regime: honour the task's dominant bucket unless overridden. decode_m_buckets that
    # contain the dominant M => decode, else prefill. Matches how EvoK's leaves are measured.
    regime = a.regime
    if not regime:
        buckets = [int(b) for b in (meta.get("m_buckets") or []) if str(b).isdigit()]
        dom = max(buckets) if buckets else 0
        dec = [int(b) for b in (meta.get("decode_m_buckets") or []) if str(b).isdigit()]
        regime = "decode" if dom and dom in dec else "prefill"

    dtype = str(meta.get("dtype", "")).lower()
    # normalise GEAK's dtype spelling into EvoK's leaf dtype segment
    if "fp8" in dtype or "e4m3" in dtype:
        dtype = "fp8_e4m3"
    elif "bf16" in dtype or "bfloat16" in dtype:
        dtype = "bf16"

    lib = os.path.join(a.evok_root, "library")
    if lib not in sys.path:
        sys.path.insert(0, lib)
    if a.gfx:
        os.environ["EVOK_GFX"] = a.gfx
    try:
        import evok
    except Exception as e:
        emit({"reason": "evok not importable from %s: %r" % (lib, e),
              "trace": traceback.format_exc()[-600:]})

    gfx = a.gfx or evok.current_gfx()
    reg = evok.registry()
    variants = reg.variants(key, gfx, dtype, regime) or reg.variants(key, gfx, dtype, None)
    if not variants:
        emit({"reason": "no_leaf_for_key", "resolve_key": key, "dtype": dtype, "gfx": gfx,
              "known_keys": reg.resolve_keys()[:80]})

    why = []
    try:
        # strict=False deliberately. dispatch.resolve(strict=True) RAISES when neither
        # shape_set nor shape is given, which is the right default for a serving path but the
        # wrong one for a lookup: `--shape-set` is optional here and a task with no matching
        # shape set must degrade to "hit, unranked", not to a traceback. When the caller DOES
        # pass --shape-set we still get shape-set-aware ranking; strict only governs the
        # missing-argument case.
        fn = evok.resolve(key, regime=regime, gfx=gfx, dtype=dtype, strict=False,
                          shape_set=(a.shape_set or None), _explain=why)
    except Exception as e:
        emit({"reason": "resolve raised: %r" % (e,), "resolve_key": key,
              "trace": traceback.format_exc()[-600:]})
    if fn is None:
        emit({"reason": "resolve returned None", "resolve_key": key, "why": why})

    # Which leaf + which impl actually answered. Derive from the loaded module name rather
    # than re-implementing _pick_backend's precedence (best_pointer > verified candidate >
    # baseline) — reimplementing it would drift from dispatch.py the moment it changes.
    modname = getattr(fn, "__module__", "") or ""
    leaf, backend = None, ""
    for m in variants:
        pfx = "_evokpkg_" + "".join(c if c.isalnum() else "_" for c in m["kernel_id"]) + "_"
        if modname.startswith(pfx):
            leaf = m
            backend = modname[len(pfx):].split(".", 1)[0]
            break
    if leaf is None:
        leaf, backend = variants[0], (variants[0].get("baseline_backend") or "")

    impl_dir = os.path.join(leaf["_dir"], "impls", backend) if backend else ""
    load = (leaf.get("impl_load") or {}).get(backend) or {}
    sel = (leaf.get("impl_selectors") or {}).get(backend) or {}
    role = (leaf.get("impl_roles") or {}).get(backend, "")
    verified = bool((leaf.get("impl_verified") or {}).get(backend, False))
    archs = sel.get("archs") or ["*"]
    arch_ok = ("*" in archs) or (gfx in archs)

    form, sig_note = classify_signature(fn)

    # WHAT ANOTHER PROCESS CAN ACTUALLY IMPORT.
    #
    # `fn.__module__` is `_evokpkg_<kernel_id>_<backend>...`, a SYNTHETIC package that
    # dispatch._exec_entry/_exec_package materialises in `sys.modules` at resolve time. It exists only
    # inside a process that has already called `evok.resolve()`. Handing that name to
    # `op_bench.py --extra-callables`, which does a plain `importlib.import_module`, fails with
    # "callable not importable" -- which is exactly what it did the first time this was run end to end,
    # and it fails as a MISSING CANDIDATE rather than as an error, so the bake-off just quietly went
    # back to being a two-horse race.
    #
    # The importable form is the entry file loaded off `impls/<backend>/` with that dir on sys.path.
    # It is not always available: `_exec_entry` execs inside a package precisely so that RELATIVE
    # imports resolve too, and a `kind:"package"` impl has no single entry module. Where the entry
    # needs the package, say so and emit no bench callable rather than one that raises at import.
    entry_file = load.get("entry", "") or ""
    bench_module = os.path.splitext(os.path.basename(entry_file))[0] if entry_file else ""
    direct_note = ""
    if not bench_module:
        direct_note = "impl declares no entry file"
    elif (load.get("kind") or "") == "package":
        direct_note = "kind:package -- the impl is a package, importable only through evok.resolve()"
    else:
        try:
            with open(os.path.join(impl_dir, entry_file)) as _fh:
                _src = _fh.read()
            if re.search(r"^\s*from\s+\.", _src, re.M) or re.search(r"^\s*import\s+\.", _src, re.M):
                direct_note = ("entry uses relative imports; it resolves only inside the synthetic "
                               "package evok.resolve() builds")
        except OSError as _e:
            direct_note = "entry file unreadable: %r" % (_e,)
    if direct_note:
        bench_module = ""
    bench_attr = "%s:%s" % (bench_module, getattr(fn, "__name__", "")) if bench_module else ""

    # Oracle provenance, read straight off the leaf's spec. The library moved captured blobs
    # OUT of leaves into a content-addressed store (`<root>/oracles/<sha[:2]>/<sha>.pt`) and
    # replaced most captures with references COMPUTED in-test from `evok.refs`. Both facts
    # matter downstream: hop-3 (ingest_run.py re-verification) can run for an in_test leaf on
    # any machine, but for a capture-only leaf it needs the blob to be physically present.
    try:
        with open(os.path.join(leaf.get("_spec_dir") or leaf["_dir"], "meta.json")) as fh:
            _smeta = json.load(fh)
    except Exception:
        _smeta = {}
    _sha = leaf.get("reference_io_sha256") or _smeta.get("reference_io_sha256") or ""
    _oracle_blob = False
    if _sha:
        try:
            from evok import oracle_store as _os_mod
            _root = _os_mod.root_for_leaf(leaf["_dir"]) or a.evok_root
            _oracle_blob = bool(_os_mod.resolve(_root, _sha))
        except Exception:
            _oracle_blob = False

    # measured speedup + staleness straight out of the leaf's recorded results.
    #
    # SIGNATURE, exactly: results.load(leaf_dir: str, gfx: str, shape_set: str|None).
    # An earlier draft of this file called it as load(leaf, shape_set, gfx) -- a DICT in the
    # path slot and the last two arguments swapped -- under a bare `except: pass`, so
    # `measured_speedup` came back null on every single hit and nothing said why. Do not
    # swallow this: a lookup that cannot read the leaf's own scoreboard is a broken lookup,
    # and it must show up in the JSON, not in the silence.
    measured, stale, results_note = None, [], ""
    measured_correct = False
    if not a.shape_set:
        # `shape_set` is not optional to `results.load` despite the `or None` reading like it is: it
        # is joined into the record's PATH, so None reaches `shape_set + ".json"` and raises TypeError.
        # A caller that named no shape set has not asked a question about recorded numbers, so say that
        # instead of reporting a crash as if the leaf's scoreboard were unreadable.
        results_note = "no --shape-set given; recorded numbers not consulted"
    else:
        try:
            rec = evok.results.load(leaf["_dir"], gfx, a.shape_set) or {}
            stale = list(rec.get("stale") or [])
            for e in rec.get("entries", []) or []:
                if e.get("backend") == backend:
                    measured = e.get("weighted_speedup")
                    measured_correct = bool(e.get("correct"))
            if measured is None and rec:
                results_note = "no entry for backend %r in results/%s/%s.json" % (
                    backend, gfx, a.shape_set)
        except Exception as e:
            results_note = "results.load failed: %r" % (e,)
    for w in why:                            # resolve()'s own staleness note is authoritative
        if w.get("stale"):
            stale = w["stale"]

    # GROUND-TRUTH ARCH OVERRIDE (mirror of EvoK dispatch._dispatchable):
    # selector.archs is a *hint* about where an impl is expected to run, and it goes stale the
    # moment an impl is proven on a chip its author never listed. The leaf's own scoreboard is
    # the authority: a verified candidate that has a `correct` measured result for THIS backend
    # on the live gfx has demonstrably run correctly here, so a stale arch pin must not veto it.
    # This keeps deployability driven by evidence, not by an out-of-date label -- and, per the
    # no-refusal-path rule, it removes a gate that would otherwise strand a working kernel.
    arch_proven = bool(measured is not None and measured_correct and verified)
    arch_ok = arch_ok or arch_proven

    deployable, reason = NOT_DEPLOYABLE.get(key, (True, ""))
    if deployable and not arch_ok:
        deployable, reason = False, "arch_not_eligible: impl selector archs=%s, live gfx=%s" % (archs, gfx)
    if deployable and role == "candidate" and not verified:
        deployable, reason = False, "not_verified: role=candidate but impl.json verified=false"
    if deployable and not form:
        deployable, reason = False, "signature_unknown: " + sig_note
    if deployable and not (leaf.get("seam") or "").strip():
        deployable, reason = False, "no_rebind_seam: leaf declares no live seam"

    emit({
        "hit": True,
        "resolve_key": key,
        "leaf": os.path.relpath(leaf["_dir"], a.evok_root),
        "kernel_id": leaf.get("kernel_id"),
        "dtype": leaf.get("dtype"),
        "gfx": gfx,
        "regime": regime,
        "shape_set": a.shape_set,
        "backend": backend,
        "role": role,
        "verified": verified,
        # ADVISORY ONLY -- never gate on it. Since the oracle migration (captured blobs moved
        # to the content-addressed store, references computed in-test by evok.refs),
        # manifest._resolve_status still decides `onboarded` by stat()ing an in-leaf
        # reference_io.pt that no longer exists anywhere in the tree -> ALL 65 leaves now read
        # "static_only". A gate on this field rejects 100% of the library. See D4 (§8.10).
        "status": leaf.get("status"),
        "has_oracle": bool(leaf.get("has_oracle")),
        # The three fields that actually say whether a re-verification is POSSIBLE here, which
        # is what `status` used to be a proxy for. oracle_reference=="in_test" means the leaf
        # computes its own reference and needs no blob (gemm/block_scaled is one of these);
        # a non-empty sha with oracle_blob==False means this machine simply lacks the blob.
        "oracle_reference": _smeta.get("oracle_reference", ""),
        "reference_io_sha256": leaf.get("reference_io_sha256", ""),
        "oracle_blob": bool(_oracle_blob),
        "measured_speedup": measured,
        "results_note": results_note,
        "stale": stale,                                # non-empty => EvoK's own number is NOT scored
        "impl_dir": impl_dir,
        "entry": load.get("entry", ""),
        "callable": load.get("callable", ""),
        # THE ONE the bake-off may pass to `--extra-callables`: importable by a plain process with
        # `evok_pythonpath` on PYTHONPATH. Empty when the impl can only be reached through
        # evok.resolve() -- in that case the op is still DEPLOYABLE (the overlay shim resolves lazily,
        # it does not import this name), it just cannot be entered as an isolated bake-off row.
        "module_attr": bench_attr,
        "module_attr_note": direct_note,
        # The in-process synthetic package name, for provenance and for reading stack traces. NOT
        # importable from anywhere else; see the comment where bench_attr is built.
        "package_module_attr": "%s:%s" % (modname, getattr(fn, "__name__", "")),
        "seam": leaf.get("seam", ""),
        "seam_attr": leaf.get("seam_attr", ""),
        "math_contract": leaf.get("math_contract", ""),
        "signature_form": form,
        "signature_note": sig_note,
        "arch_ok": arch_ok,
        "arch_selector": archs,
        "deployable": deployable,
        "reason": reason,
        "why": why,
        # How EvoK decided which arch it is running on. dispatch.GFX_DETECTION names the keys
        # ({"gfx","source","tried"}) that carry the FACT of detection rather than a warning
        # string, so this survives rewording of the warning. If `source` is not "env" and not
        # a real device query, every arch_ok below was computed against a GUESS -- surface it.
        "gfx_detection": [
            {k: w[k] for k in evok.dispatch.GFX_DETECTION if k in w}
            for w in why if any(k in w for k in evok.dispatch.GFX_DETECTION)
        ],
        # Backends whose arch pin could not be checked at all (dispatch._arch_pinned). An
        # unpinnable backend is not the same as an eligible one; it just means nobody knows.
        "arch_unpinned": sorted(getattr(evok.dispatch, "_arch_pinned", lambda *_: [])(leaf, gfx) or []),
        # PYTHONPATH a consumer needs, in order: the impl dir (so `module_attr` imports, and so the
        # entry's absolute sibling imports -- `_v_vllm_shim`, `_prefill_configs` -- resolve from the
        # same path entry the way dispatch's synthetic package makes them), then evok's library (so
        # the overlay shim can call evok.resolve()). The impl dir is omitted when there is no
        # importable entry, rather than pointing PYTHONPATH at a dir nothing will import from.
        "evok_pythonpath": (impl_dir + os.pathsep + lib) if (bench_module and impl_dir) else lib,
    })


if __name__ == "__main__":
    main()
