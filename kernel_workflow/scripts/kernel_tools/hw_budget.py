#!/usr/bin/env python3
"""Print the hardware budget for a kernel BEFORE you touch it. No GPU, no compile.

Assembles the SKU peaks (sku.json), the arch microarch facts (hw_constants.json) and, when a
workload shape is given, the analytic FLOP/byte model (workload_models.json) into the six resources
that bound every CDNA kernel, plus the MFMA-only floor and your multiple over it.

This answers the Gluon pack's references/method/budget.md: "what is my budget, and where is the floor" -- the number that
sizes the prize before the first round. It does NOT profile; it is the analytic upper bound the
measured profile is read against.

TWO WAYS TO GIVE IT A BYTE MODEL, and the second is the general one:

  --workload NAME --shapes ...   a named archetype's formula (gemm, attention, moe, ...). Fast when
                                 the kernel really is that archetype.
  --tensors "name:dir:dtype:dims[:xN]"
                                 a MANIFEST of what the kernel actually moves. Use this whenever the
                                 kernel does not match an archetype -- which is most real kernels.
                                 An archetype formula cannot express per-tensor dtypes, arguments
                                 that are never dereferenced, a workspace much larger than the part
                                 touched, or an operand re-read once per tile; a manifest is just
                                 bookkeeping for all of them. It reports a BRACKET (unique footprint
                                 to issued traffic) instead of one confident number.

BEFORE EITHER, THREE PRECONDITIONS decide whether a roofline is the right model at all. Each one is
a situation where the two-term model is not merely imprecise but inapplicable, and the tool refuses
rather than printing a gap that would send the next round at the wrong resource:

  --grid N          fewer workgroups than CUs -> part of the machine is idle by construction, so
                    neither the compute peak nor the memory ceiling is reachable. (At ~1 workgroup
                    per CU it warns instead: the machine is full but has too few requests in flight
                    to stream, so the applicable ceiling is the one probed at THAT program count.)
  --footprint-mb X  a working set inside the memory-side LLC is served from cache in the steady state
                    of a repeated-iteration benchmark -> the MEMORY floor is void (the compute floor
                    survives, and is then the binding one).
  --dispatches N    N > 1 means the timed region is several kernels, so a per-kernel floor cannot be
                    compared against the region's total time.

Usage (gfx950 first; the SKU key is a row of perf_knowledge/hardware/data/sku.json, and --sku is
required -- the tool never assumes a part):
  hw_budget.py --sku MI355X
  hw_budget.py --sku MI355X --workload attn_bwd --shapes b=1,h=32,d=128,s=8192 --dtype bf16
  hw_budget.py --sku MI355X --workload gemm --shapes M=4096,N=4096,K=8192 --dtype fp4 [--json]
  hw_budget.py --sku MI325X --workload gemm --shapes M=4096,N=4096,K=8192 --dtype bf16   # gfx942
  hw_budget.py --sku MI355X --tensors "A:r:fp8:MxK, B:r:fp8:KxN:x<M/BLOCK_M>, C:w:bf16:MxN" \
               --flops <2MNK> --grid <wgs> --footprint-mb <unique> --measured-ms X

Shapes are free-form k=v (case-insensitive); the model's own variable names are accepted plus the
usual aliases (h->heads, s->seqlen, d->head_dim). --measured-ms X prints your multiple over the
floor and over the roofline.

BOTH CEILINGS ARE CALIBRATABLE, and both default to the optimistic datasheet reading:

  --measured-hbm-tb-s LO,HI   the in-shape memory ceiling (mem_bw_probe.py)
  --measured-tflops LO,HI     the in-shape compute ceiling (a dense back-to-back loop on whichever
                              engine the FLOPs issue on)

Neither datasheet peak is reachable at any real access shape, and the error has a DIRECTION: a peak
that is too high puts the floor too low, so every multiple over it overstates the prize. The caveats
are therefore scoped to the axis the kernel is BOUND ON -- a compute-bound kernel is told to probe
the compute ceiling, not the memory one. Off-axis warnings are how on-axis ones stop being read.

A FLOP COUNT IS NOT A COMPUTE FLOOR. `flops / MFMA_peak` is a floor only if the FLOPs go through
MFMA, so the engine is declared, not assumed: a named --workload carries `flops_engine` in
workload_models.json, and on the --tensors path you say `--flops-engine mfma|valu`. Pricing VALU work
(reduction / elementwise / norm / scan / gather are all flops = c*n, no matrix multiply anywhere) at
the matrix rate understates the compute floor by one to two ORDERS, after which it never binds and
"memory" comes out as a DEFAULT rather than a finding -- wrong in exactly the direction that opens a
byte-removal round on a kernel whose arithmetic is the constraint.

Which leaves three states for the compute term, and they carry different instructions:
  0 FLOPs               -> the term CANNOT bind. A conclusion.
  a rate for this dtype -> a real floor.
  no rate (dtype absent from the SKU table, or FLOPs on an unpriced engine)
                        -> the floor is UNKNOWN, so a memory verdict is provisional, not measured.
                           Never substituted with another dtype's rate: that error runs the
                           dangerous way, lifting the compute floor until the binding flips.
"""
import argparse
import ast
import json
import math
import os
import sys
import textwrap

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import _hwdata  # noqa: E402  (sibling locator: perf_knowledge/hardware/data + the Gluon pack)


def _find(rel):
    """Path of a shared hardware data file (perf_knowledge/hardware/data via _hwdata), or None."""
    p = _hwdata.find(rel)
    return str(p) if p else None


def _load(rel):
    p = _find(rel)
    if not p:
        sys.exit(f"[hw_budget] cannot find {rel} (looked in $GEAK_HW_DATA_DIR and "
                 f"perf_knowledge/hardware/data via {HERE}/_hwdata.py)")
    return json.load(open(p))


def _pack_root():
    """The Gluon pack root (the reading-route pages are pack-relative), or None."""
    p = _hwdata.pack_dir()
    return str(p) if p else None


def _denominator_basis(sku, bw_src, ceiling_source_ref=None):
    """datasheet | empirical@<tool>-<version> | in-shape probe -- what the HBM ceiling rests on.

    A typed-in ceiling with no probe artifact is empirical but of unknown origin, so it is labelled
    empirical@hand-entered rather than promoted to an in-shape probe. A table row whose own peak is
    MEASURED (sku.json peak_hbm_basis) is empirical at the tool its measured_ceilings entry names.
    """
    if bw_src == "calibrated":
        return "in-shape probe" if ceiling_source_ref else "empirical@hand-entered"
    if (sku or {}).get("peak_hbm_basis") == "measured":
        m = next((c for c in sku.get("measured_ceilings") or [] if c.get("metric") == "hbm_tb_s"), {})
        return f"empirical@{m.get('tool', 'unrecorded')}-{m.get('tool_version', 'unrecorded')}"
    return "datasheet"


def _normalize_workload(name, refused, aliases, models_doc):
    """Resolve the name as typed first, then under case/separator folding.

    Exact match wins so that nothing already in the vocabulary changes meaning. The folded pass
    exists because the names that most need to be caught are the ones a caller types by hand from
    a paper -- `MQA`, `GQA`, `KV-Cache` -- and an exact-match table silently misses all three.
    """
    known = set(refused) | set(aliases) | set(models_doc.get("models") or {}) | set(
        ((models_doc.get("reading_route") or {}).get("route_only") or {}))
    if name in known:
        return name
    folded = {}
    for k in known:
        folded.setdefault(k.lower().replace("_", "-"), k)
    return folded.get(str(name).strip().lower().replace("_", "-"), name)


def _reading_route(models_doc, key):
    """The reading this archetype routes to, as a citation the caller can open.

    Rides this tool on purpose. A document path listed on a stage card reached 0 of 19 measured
    kernel/phase runs; the roofline reached 17, because a tool computed it and printed it. The
    archetype is already a required argument here, so routing the reading off it costs the caller
    nothing new -- and unlike a list, it arrives at the moment the archetype is decided.

    Returns None when the pack ships no such page: a route nobody can open reads as covered, which
    is worse than an honest silence (the tool is shared by packs that do not carry a workload index).
    """
    spec = models_doc.get("reading_route") or {}
    page = spec.get("page")
    if not page or not key:
        return None
    pack_root = _pack_root()
    if not pack_root or not os.path.exists(os.path.join(pack_root, page)):
        return None
    full = os.path.join(pack_root, page)
    try:
        rel = os.path.relpath(full, str(_hwdata.REPO_ROOT))
        full = rel if not rel.startswith("..") else full
    except ValueError:
        pass
    # `page`/`cite` stay pack-relative (the anchor contract); `path` is where it sits from the repo
    # root, because this tool is no longer inside the pack and a bare pack path would not open.
    route = {"page": page, "section": key, "cite": f"{page} ## {key}", "path": full}
    note = ((spec.get("route_only") or {}).get(key))
    if note:
        route["route_only"] = note
    return route


# ---- safe arithmetic evaluator for the workload_models expressions -------------------
_ALLOWED = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.IfExp, ast.Compare, ast.Call,
            ast.Name, ast.Load, ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.Div,
            ast.FloorDiv, ast.Mod, ast.Pow, ast.USub, ast.UAdd,
            ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq, ast.BoolOp, ast.And, ast.Or)
_FUNCS = {"ceil": math.ceil, "floor": math.floor, "min": min, "max": max, "abs": abs}


def _safe_eval(expr, env):
    """Evaluate a workload-model arithmetic expression against env. Rejects anything but
    arithmetic/compare/ternary/whitelisted-calls so a model file cannot execute code."""
    tree = ast.parse(expr, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED):
            raise ValueError(f"disallowed syntax {type(node).__name__!r} in model expr: {expr!r}")
        if isinstance(node, ast.Call) and (not isinstance(node.func, ast.Name)
                                           or node.func.id not in _FUNCS):
            raise ValueError(f"disallowed call in model expr: {expr!r}")
    return eval(compile(tree, "<model>", "eval"), {"__builtins__": {}}, {**_FUNCS, **env})  # noqa: S307


_SHAPE_ALIAS = {"h": "heads", "s": "seqlen", "d": "head_dim", "bh": "BH"}

# `b` is RESERVED for dtype bytes: every model's `bytes_min` in workload_models.json is written
# `b*(...)`. But `b` is ALSO the documented attention batch spelling (`--shapes b=1,h=32,d=128,s=4096`
# appears in this file's own usage, in the Gluon skill's skill.md and in the selftest), so a user `b=` used to land in
# `env["b"]` and shadow the dtype. That made every attention/attn_bwd byte model a QUADRATIC in batch
# and silenced `--dtype` completely -- 0.5x at b=1, 2x at b=4, 4x at b=8, with fp8/bf16/fp32 all
# returning the same byte count. A user `b` is therefore routed to `batch` and never into `env["b"]`.
_DTYPE_BYTES_VAR = "b"


def _build_env(shapes, dtype_bytes, models_doc, scale_overhead=0.0):
    """Map the user's k=v shapes onto the variable names the model expressions use."""
    env = {_DTYPE_BYTES_VAR: dtype_bytes, "c": 1, "causal": shapes.get("causal", 0),
           "scale_overhead": scale_overhead}
    for k, v in shapes.items():
        if k.lower() == _DTYPE_BYTES_VAR:
            env["batch"] = v          # batch, not dtype bytes -- see _DTYPE_BYTES_VAR above
            continue
        env[k] = v
        env[k.upper()] = v
        if k.lower() in _SHAPE_ALIAS:
            env[_SHAPE_ALIAS[k.lower()]] = v
    # attention family uses BH = batch*heads, S, D
    if "BH" not in env:
        b = shapes.get("b", shapes.get("batch", 1))
        h = shapes.get("h", shapes.get("heads"))
        if h is not None:
            env["BH"] = b * h
    for name in ("S", "D", "M", "N", "K"):
        if name not in env and name.lower() in shapes:
            env[name] = shapes[name.lower()]
    env.setdefault("BLOCK_M", 128)
    # streaming ops (reduction/elementwise/scan/gather) count bytes as n_in + n_out. Derive them
    # from `n` when not given explicitly: a reduction collapses to a scalar-ish output (n_out≈1),
    # elementwise/scan preserve size (n_out≈n). The user can override any of them directly.
    if "n" in env:
        env.setdefault("n_in", env["n"])
        env.setdefault("n_out", env["n"])          # safe upper bound on bytes; override for a true reduction
        env.setdefault("n_gathered", env["n"])
    # MoE: the routed-expert count is the whole byte model (weights dominate). Charging all E
    # over-counts by E/E_touched, which at decode M puts the HBM floor BELOW the measured time.
    # min(E, M*k) is the analytic cap on distinct routed experts; collisions push the truth lower,
    # so pass a measured E_touched when you have one.
    if "E" in env:
        cap = env["E"]
        if "M" in env and "k" in env:
            cap = min(cap, env["M"] * env["k"])
        env.setdefault("E_touched", cap)
    # Which MoE GEMM is being budgeted. The default is 1 (w1, gate+up), NOT "both": a budget is
    # computed per KERNEL, and the two GEMMs are two kernels with two different measured times, so
    # charging both stages' weights against one of them over-counts by 1.5x. `stage` also scales the
    # FLOPs term, so the two stay describable as the same kernel -- an AI that mixes stage-1 FLOPs
    # with both stages' bytes is not a number about anything.
    env.setdefault("stage", 1)                     # 1 = w1 (gate+up); 2 = w2 (down); 0 = one fused kernel doing both
    return env


# Payload bytes per element. A dtype this table does not know falls back to 2 -- which is a 4x
# byte error on a 4-bit weight GEMM, so the fallback WARNS instead of passing silently.
_DTYPE_BYTES = {
    "fp4": 0.5, "mxfp4": 0.5, "nvfp4": 0.5, "e2m1": 0.5, "int4": 0.5, "uint4": 0.5,
    "fp6": 0.75, "mxfp6": 0.75, "e2m3": 0.75, "e3m2": 0.75,
    "fp8": 1, "mxfp8": 1, "e4m3": 1, "e5m2": 1, "int8": 1, "uint8": 1,
    "bf16": 2, "fp16": 2, "f16": 2, "int16": 2,
    "fp32": 4, "f32": 4, "tf32": 4, "int32": 4,
    "fp64": 8, "f64": 8,
}

# Micro-scaling: one scale byte per group of G elements, charged as a FRACTION of payload bytes
# (1 / (G * bytes_per_element)). OCP MX is G=32 with an E8M0 byte -> 6.25% at fp4, 3.125% at fp8;
# NVFP4 is G=16 with an E4M3 byte -> 12.5%. int4 group-128 is the common weight-quant convention.
# A raw weight-byte count misses this entirely; override with `scale_overhead=` in --shapes.
_MICRO_SCALE_GROUP = {"fp4": 32, "mxfp4": 32, "e2m1": 32, "mxfp6": 32, "mxfp8": 32,
                      "nvfp4": 16, "int4": 128, "uint4": 128}


def _dtype_bytes(dt, warn=False):
    b = _DTYPE_BYTES.get(dt.lower())
    if b is None:
        if warn:
            print(f"[hw_budget] WARNING: unknown dtype {dt!r} -> assuming 2 B/element. Every byte "
                  f"count and the HBM floor below are wrong by 2/actual. Known: "
                  f"{', '.join(sorted(_DTYPE_BYTES))}", file=sys.stderr)
        return 2
    return b


def _scale_overhead(dt, dtype_bytes):
    """Micro-scaling scale bytes as a fraction of payload bytes (0 for a non-scaled dtype)."""
    g = _MICRO_SCALE_GROUP.get(dt.lower())
    return (1.0 / (g * dtype_bytes)) if g and dtype_bytes else 0.0


_DIRS = {"r": 1, "read": 1, "w": 1, "write": 1, "rw": 2, "readwrite": 2, "atomic": 2, "rmw": 2}


def _parse_tensors(spec, warn=False):
    """Byte model as a MANIFEST of the tensors a kernel actually touches, not a formula keyed on a
    workload name.

    `--workload NAME --shapes ...` can only express kernels that match a named archetype, and real
    kernels routinely do not. Each of these breaks a name-keyed formula, and all of them are just
    bookkeeping in a manifest:

      * MIXED DTYPES. One `b` cannot describe fp8 operands with a bf16 output and fp32 scales, and
        the output is frequently the largest term -- so a single-dtype count is wrong by the ratio.
      * POINTERS THAT ARE NEVER DEREFERENCED. Kernels take arguments they do not read on the taken
        path. Summing the argument list charges them; a manifest just omits them.
      * ALLOCATION IS NOT TRAFFIC. A workspace or KV cache sized for the general case, then used for
        one sequence, moves a tiny fraction of what it allocates. You declare TOUCHED elements.
      * RE-TRAVERSAL. A tile loop re-reads an operand once per tile of the other operand, so issued
        traffic is a multiple of the footprint. That multiple is `xN` here, per tensor, because it
        differs per tensor (the re-read operand and the streamed one are not the same).
      * REDUNDANT GRID. When a grid dimension is dead on the taken path, every program is duplicated
        along it; `grid_redundancy` multiplies issued traffic without changing the footprint.

    Grammar, comma-separated:  name:dir:dtype:dims[:xN]
      dir    r | w | rw (rw/atomic/rmw = read+write = 2x, which is what a read-modify-write costs)
      dtype  any key in the dtype table, or `bN` for an explicit N bytes/element
      dims   AxBxC..., or a plain element count
      xN     times this tensor is traversed (default 1); affects ISSUED bytes only

    Returns unique-footprint bytes (perfect reuse, a LOWER bound) and issued bytes (the declared
    traversals, an UPPER bound). Those two ends are the honest byte interval: the truth is where the
    cache hierarchy puts it, and no formula knows that -- only a measurement does."""
    tensors, uniq, issued = [], 0.0, 0.0
    for part in [p.strip() for p in spec.split(",") if p.strip()]:
        f = part.split(":")
        if len(f) < 4:
            raise ValueError(f"tensor {part!r}: want name:dir:dtype:dims[:xN]")
        name, d, dt, dims = f[0], f[1].strip().lower(), f[2].strip(), f[3].strip()
        if d not in _DIRS:
            raise ValueError(f"tensor {name!r}: dir {d!r} not in {sorted(_DIRS)}")
        trav = 1.0
        if len(f) > 4 and f[4].strip():
            t = f[4].strip().lower().lstrip("x")
            trav = float(t)
            if trav < 1:
                raise ValueError(f"tensor {name!r}: traversals {trav} < 1 -- a tensor that is "
                                 f"touched cannot be traversed less than once; to charge only part "
                                 f"of it, reduce the element count instead")
        elems = 1.0
        for tok in dims.replace("*", "x").split("x"):
            tok = tok.strip()
            if tok:
                elems *= float(tok)
        eb = float(dt[1:]) if (dt.startswith("b") and dt[1:].replace(".", "", 1).isdigit()) \
            else _dtype_bytes(dt, warn=warn)
        # a read-modify-write moves the line twice; scale bytes ride along with the payload
        base = elems * eb * _DIRS[d] * (1.0 + _scale_overhead(dt, eb))
        tensors.append({"name": name, "dir": d, "dtype": dt, "bytes_per_elem": eb,
                        "elements": elems, "traversals": trav,
                        "unique_bytes": round(base), "issued_bytes": round(base * trav)})
        uniq += base
        issued += base * trav
    if not tensors:
        raise ValueError("--tensors is empty")
    return {"tensors": tensors, "bytes_unique": uniq, "bytes_issued": issued}


def _bound_direction(model, key, shapes, bytes_measured=False, routes_agree=True):
    """The model's directional caveat, minus the parts whose cause the caller already removed.

    `bound_direction` says which way a model's byte count is wrong, which is the single most
    important thing about it -- and also a fixed string, so it keeps warning about causes the
    caller has already removed. A warning that fires when its cause is absent gets skimmed past,
    and then it is not read on the run where it matters. Two removals:

      - measured bytes: the byte model is no longer in the path at all, so none of its caveat is;
      - a measured `E_touched`: the MoE weight term is real, but the activation term still
        under-counts cross-block re-reads, so the sentence is narrowed rather than dropped.

    Measured-but-disagreeing routes are NOT the same as measured: the model is still out of the
    path, but the numerator is a range rather than a fact, so the caveat moves from the model to
    the instrument instead of disappearing."""
    txt = model.get("bound_direction")
    if bytes_measured and not routes_agree:
        return ("not in the path, but the measurement is: bytes are MEASURED and the counter routes "
                "DISAGREE, so the numerator is a range. Both terms still need work.")
    if bytes_measured:
        return ("not in the path: bytes are MEASURED, so the model's direction does not bound "
                "this number. The ceiling is now the only calibratable term left.")
    if key == "moe" and txt and (shapes or {}).get("E_touched") is not None:
        return ("a LOWER bound at a measured E_touched: the weight term is now real, but the "
                "activation term still under-counts cross-block re-reads. Confirm with "
                "--measured-dram-mb before sizing a prize.")
    return txt


def budget(sku_name, workload=None, shapes=None, dtype="bf16", measured_ms=None,
           measured_hbm_tb_s=None, measured_bytes_mb=None, warn=False, ceiling_source_ref=None,
           tensors=None, flops=None, grid=None, footprint_mb=None, dispatches=None,
           grid_redundancy=1.0, measured_tflops=None, flops_engine=None):
    skus = _load("sku.json")["skus"]
    if sku_name not in skus:
        sys.exit(f"[hw_budget] unknown SKU {sku_name!r}; known: {', '.join(skus)}")
    sku = skus[sku_name]
    arch = sku["arch"]
    if sku.get("geak_support") == "unsupported" and warn:
        print(f"[hw_budget] NOTE: {sku_name} ({arch}) is kept in sku.json for reference only -- GEAK "
              f"does not support or validate this part.", file=sys.stderr)
    hwc = _load("hw_constants.json")["arch"].get(arch, {})
    db = _dtype_bytes(dtype, warn=warn)
    # NO SILENT SUBSTITUTION. This used to fall back to the bf16 rate for any dtype the table does
    # not list, which is most of the interesting ones: CDNA4 fp4/fp6 are native and several times
    # the bf16 rate, Instinct rows carry no int8/int4, RDNA3 rows carry no fp8 -- and a misspelled
    # dtype landed on bf16 too. The substitution is silent and its error runs the DANGEROUS way: a
    # peak that is too low lifts the compute floor, which can flip the binding to compute and send
    # the round at an engine that was never the constraint. An absent entry is not a number.
    peak_tflops = sku["peak_tflops"].get(dtype.lower())
    peak_dtypes = sorted(sku["peak_tflops"])
    # THE RIDGE IS PER-DTYPE, for exactly the reason the paragraph above gives. It is
    # peak_tflops[dtype]/peak_hbm, so it moves with the numerator: on MI325X it is 218 ops/byte at
    # fp16, 436 at fp8 and 27 at fp32. sku.json used to ship it as a stored field taken at fp16 and
    # every read substituted that one dtype's crossover for the caller's -- the SAME silent
    # substitution refused two lines up, and it put every fp32 kernel whose intensity fell between
    # 27 and 218 on the wrong side. Derived here, never stored, and absent for a dtype with no rate.
    # (Both terms are datasheet peaks, so the crossover is a property of the SKU and does not move
    # with each probe. It orients the reader; the FLOORS below are what a round gets sized on.)
    ridge = (peak_tflops / sku["peak_hbm_tb_s"]) if peak_tflops else 0
    # A LOWER BOUND ON THE RIDGE, for the one reader that can use a bound instead of a value: the
    # engine gate below, which only ever asks "is AI far BELOW the crossover". A lower bound settles
    # that question and nothing else, so it is kept apart from `ridge` -- what gets reported stays
    # None when it is unknown, and never quietly becomes another dtype's number.
    #
    # The bound rests on a monotonicity the matrix pipeline actually has: narrowing the dtype never
    # makes it slower. So for a dtype with no rate of its own, peak[dt] >= peak[d] for every listed
    # d that is at least as WIDE, and the largest such peak is a floor under the true ridge. fp4 on
    # a table listing fp8 lands on the fp8 ridge (625 ops/B on MI355X, above the fp16 one) -- which
    # is the direction the old fp16-for-everything substitution happened to get right. It got the
    # other direction wrong: a dtype WIDER than everything listed has no bound here and the gate
    # simply does not fire, instead of firing on a crossover an order of magnitude too high.
    ridge_floor = ridge
    if not ridge:
        want_b = _dtype_bytes(dtype)
        wider = [tf for d, tf in sku["peak_tflops"].items() if _dtype_bytes(d) >= want_b and tf]
        ridge_floor = (max(wider) / sku["peak_hbm_tb_s"]) if wider else 0

    # The register and LDS notes are ARCH-SPECIFIC and cannot be written once: CDNA shares one
    # 512-entry ArchVGPR+AGPR file per SIMD, RDNA has a 1536-entry file with a 256/wave cap and
    # no AGPRs at all, and RDNA's LDS is a per-WGP pool with a separate per-workgroup limit.
    family = (hwc.get("family") or "").upper()
    is_rdna = family.startswith("RDNA")
    # SAME RULE AS THE DTYPE BLOCK ABOVE, applied to the hardware geometry. `hwc` is
    # `hw_constants["arch"].get(arch, {})`, so an arch missing from the table makes EVERY lookup
    # below fall to its default -- and 512 is CDNA's register file. Handed to an RDNA part it is
    # 3x low (1536) and on CDNA5 2x low (1024), and a too-small register file overstates VGPR
    # pressure, which caps the search from above: the axis is narrowed, not errored.
    # The defaults are therefore gone. An absent entry is not a number; it is None, and the
    # consumers (tuning_laws.derive -> `unknown`) already have a path for that.
    vgpr_file = sku.get("vgpr_per_simd") or hwc.get("vgpr_file_per_simd")
    vgpr_wave = sku.get("vgpr_per_wave") or hwc.get("vgpr_per_wave") or vgpr_file
    waves_cap = hwc.get("max_waves_per_simd")
    # THREE DISTINCT LDS QUANTITIES, and collapsing them into one "LDS size" is the bug this
    # block exists to prevent. CDNA has a single per-CU pool (`lds_per_cu_kib`). RDNA has a
    # per-WGP pool (`lds_per_wgp_kib`; a WGP is 2 CUs) AND a separate per-workgroup allocation
    # limit (`lds_per_wg_kib`, 64 KiB). Only the POOL prices occupancy; the per-workgroup limit
    # is a ceiling on one allocation and says nothing about how many workgroups fit.
    #   sku.json used to carry `lds_per_cu_kib: 64.0` on every RDNA row. That is pool/2: the
    # right arithmetic on the WRONG SHARING STRUCTURE -- two workgroups landing on one WGP
    # contend for ONE 128 KiB pool, not for two independent 64s -- so it is not a smaller true
    # quantity, it is a quantity that does not exist at that granularity. And 64 is exactly
    # gfx942's genuine per-CU figure, so an RDNA reading was numerically indistinguishable from
    # a CDNA3 one: a value that is correct somewhere else never looks wrong on inspection.
    # Those rows are null now. Two rules follow, and both are enforced below:
    #   (1) never emit a bare KiB number -- the SCOPE travels with it, or the reader cannot tell
    #       which of the three quantities they are holding;
    #   (2) where the pool is not recorded, say occupancy is NOT PRICED. Do not halve the
    #       per-workgroup limit to manufacture a per-CU figure.
    lds_kib_per_cu = sku.get("lds_per_cu_kib") or hwc.get("lds_per_cu_kib")
    lds_bytes_per_cu = int(lds_kib_per_cu * 1024) if lds_kib_per_cu else None
    lds_kib_per_wgp = hwc.get("lds_per_wgp_kib")
    lds_kib_per_wg = hwc.get("lds_per_wg_kib")   # NO default: absent means absent, not 64
    if is_rdna:
        pool_kib, pool_scope = lds_kib_per_wgp, ("WGP" if lds_kib_per_wgp else None)
    else:
        pool_kib, pool_scope = lds_kib_per_cu, ("CU" if lds_kib_per_cu else None)
    lds_priced = pool_kib is not None
    reg_note = (f"file/SIMD; <={vgpr_wave} VGPR addressable per wave; no AGPR file"
                if is_rdna else "ArchVGPR+AGPR COMBINED per SIMD")
    if waves_cap:
        reg_note += f"; cap {waves_cap} waves/SIMD"
    reg_note += " -- waves/SIMD: amd_occupancy.py (or LLVM's `; Occupancy:`)"
    if not is_rdna:
        lds_basis = (f"{pool_kib} KiB per CU (hw_constants lds_per_cu_kib) -- on CDNA the LDS "
                     f"pool is per-CU, so this figure IS the occupancy pool"
                     if lds_priced else
                     f"OCCUPANCY NOT PRICED: hw_constants.json records no lds_per_cu_kib for "
                     f"{arch}")
    elif lds_priced:
        lds_basis = (f"{pool_kib} KiB per WGP (hw_constants lds_per_wgp_kib), shared by the 2 "
                     f"CUs of a WGP -- this, not any per-CU figure, is the occupancy pool"
                     + (f"; one workgroup may allocate at most {lds_kib_per_wg} KiB "
                        f"(lds_per_wg_kib), an allocation ceiling that does NOT price how many "
                        f"workgroups fit" if lds_kib_per_wg else ""))
    else:
        lds_basis = (f"OCCUPANCY NOT PRICED on {arch}: RDNA's occupancy pool is per-WGP and "
                     f"hw_constants.json records no lds_per_wgp_kib for this arch"
                     + (f". Only lds_per_wg_kib={lds_kib_per_wg} KiB is known, and that is the "
                        f"per-workgroup ALLOCATION LIMIT, not a pool -- halving it to a per-CU "
                        f"figure would manufacture a quantity that does not exist at that "
                        f"granularity" if lds_kib_per_wg else ""))
    lds_note = "second, independent occupancy limiter; " + lds_basis

    out = {
        "sku": sku_name, "arch": arch, "dtype": dtype,
        "resources": {
            "mfma_issue": {"peak_tflops": peak_tflops, "note": "per-dtype; VALU co-executes"},
            "registers": {"vgpr_per_simd": vgpr_file, "vgpr_per_wave": vgpr_wave,
                          "max_waves_per_simd": waves_cap, "note": reg_note},
            # The SCOPE ships with the number. `occupancy_pool_kib` alone is unreadable --
            # 128 could be a per-WGP pool and 64 could be a per-CU pool, a per-WGP half, or a
            # per-workgroup cap. Read the pair, or read `basis`, which spells it out. The three
            # raw fields are kept separate and are None where the arch does not have that
            # quantity; `occupancy_priced` is False when the pool is unknown, and in that case
            # there is no LDS limiter here to compare against the register one.
            "lds_capacity": {"occupancy_pool_kib": pool_kib,
                             "occupancy_pool_scope": pool_scope,
                             "occupancy_priced": lds_priced,
                             "kib_per_cu": lds_kib_per_cu,
                             "kib_per_wgp": lds_kib_per_wgp,
                             "kib_per_wg": lds_kib_per_wg,
                             "basis": lds_basis,
                             "note": lds_note},
            "lds_bandwidth": {"bytes_per_clk": hwc.get("lds_bytes_per_clk"),
                              "banks": hwc.get("lds_banks"),
                              "ds_read_b128_cf_cyc": hwc.get("ds_read_b128_conflict_free_cyc")},
            "hbm": {"peak_tb_s": sku.get("peak_hbm_tb_s"),
                    "ridge_ops_per_byte": round(ridge, 1) if ridge else None,
                    "ridge_dtype": dtype if ridge else None},
            # Two capacities, and they are not interchangeable: the L2 is per-XCD and is the gateway
            # onto the fabric, while the memory-side LLC sits BEHIND the fabric in front of DRAM. The
            # second one is the bar a footprint has to clear before any measurement is about HBM, and
            # it is an order of magnitude larger -- so using the L2 as that bar admits cache readings.
            "l2_fabric": {"l2_mb": sku.get("l2_mb"), "mall_mb": sku.get("mall_mb"),
                          "note": "every TCC/EA counter is measured at the L2/fabric boundary, IN "
                                  "FRONT of the memory-side LLC -- so it counts fabric traffic and "
                                  "cannot see whether a byte came from that cache or from HBM"},
        },
        "cus": sku.get("cus"),
        # Canonical flat pruning inputs — the ONE shape plain_autotune.feasible()/model_rank() read.
        # (Before this, feasible() looked for budget["ideal"]["lds_bytes_per_cu"] which hw_budget
        # never emitted, so a --budget file pruned nothing.) All bytes; dtype_bytes for LDS math.
        "ideal": {
            # `or 64` used to sit at the end of this chain. It was not an unknown-arch edge
            # case: every RDNA row in hw_constants.json carries `lds_per_cu_kib: null` ON
            # PURPOSE, because RDNA's shared memory is a per-WGP pool with a separate
            # per-workgroup cap and there IS no per-CU figure -- so the default converted a
            # deliberate "this quantity does not exist in this form here" into 64 KiB, for
            # every RDNA part, silently. On gfx950 the same default is 2.5x low (160 KiB) and
            # on gfx1250 5x (320 KiB), and it is too LOW, so it prunes reachable tiles.
            # A wrong constant that is the right answer on some other arch never looks wrong.
            # Now: absent -> None. `tuning_laws.lds_budget` already returns None for that and
            # reports `lds_bytes_per_cu` as unknown, which is the honest outcome.
            "lds_bytes_per_cu": lds_bytes_per_cu,
            "vgpr_per_simd": vgpr_file,
            "vgpr_per_wave": vgpr_wave,
            # 64 is CDNA's wave. Defaulting to it on an unlisted arch silently halves every
            # per-lane figure on a wave32 part, so this refuses too.
            "wave_size": sku.get("wave_size") or hwc.get("wave_size"),
            "arch": arch,
            "arch_family": hwc.get("family", "CDNA"),
            "tcp_inflight_cap_kib": hwc.get("tcp_inflight_cap_kib"),
        },
        "dtype_bytes": _dtype_bytes(dtype),
    }

    manifest = _parse_tensors(tensors, warn=warn) if tensors else None
    if workload or manifest:
        model, key, env = {}, None, {}
        model_lo = model_hi = None
        reading_route = None
        if workload:
            models_doc = _load("workload_models.json")
            aliases = models_doc.get("aliases", {})
            # REFUSE BEFORE RESOLVING. A handful of names are words that two archetypes answer to,
            # and the two disagree by orders of magnitude rather than by a little (`decode` is a
            # gather on the attention side and a Small-M GEMM on the other; `mqa`/`gqa` need two
            # head counts where this page has one `BH`). Aliasing them would pick a side silently;
            # the generic unknown-name exit below would read as "the vocabulary is incomplete" and
            # send the caller to the nearest modelled name -- which is that same wrong side. So
            # these get their own exit that names the ambiguity and the disambiguation.
            refused = models_doc.get("refused") or {}
            # Match on the name as typed, then on a normalized form. MQA and GQA are acronyms and
            # arrive in caps more often than not; `kv_cache` and `kv-cache` are the same word. A
            # refusal that fires only on one casing is a refusal that does not fire.
            lookup = _normalize_workload(workload, refused, aliases, models_doc)
            if lookup in refused:
                page = ((models_doc.get("reading_route") or {}).get("page")) or ""
                sys.exit(
                    f"[hw_budget] {workload!r} is REFUSED, not unknown -- this page will not pick "
                    f"a side for you.\n  {refused[lookup]}\n"
                    f"  HOW TO PROCEED, either way:\n"
                    f"    - to take one side deliberately, pass THAT archetype's own name; its "
                    f"reading route prints with the budget.\n"
                    f"    - to declare the traffic instead, re-run with NO --workload and pass "
                    f"--tensors (and --flops). The refusal is on the name, not on the measurement, "
                    f"so keeping --workload here will only refuse again."
                    + (f"\n  The reading this name would have routed to is still worth opening: "
                       f"{page}." if page else ""))
            key = aliases.get(lookup, lookup)
            route_only = ((models_doc.get("reading_route") or {}).get("route_only") or {})
            model = models_doc["models"].get(key)
            if not model and key in route_only:
                # A ROUTE-ONLY archetype is a deliberate model gap, not a typo. It still routes the
                # reading, and it still refuses to invent a ceiling: the caller must declare what the
                # kernel moves. Erroring here instead would push the caller to the nearest MODELLED
                # name, which is worse than no model -- it hands back a confident wrong byte count.
                model = {"route_only": route_only[key]}
            elif not model:
                sys.exit(f"[hw_budget] unknown workload {workload!r}; modelled: "
                         f"{', '.join(models_doc['models'])}; route-only (no byte/FLOP model, "
                         f"declare --tensors/--flops): {', '.join(route_only) or 'none'} (+aliases)"
                         + (f". Deliberately REFUSED as ambiguous (ask for the reason by name): "
                            f"{', '.join(refused)}" if refused else ""))
            reading_route = _reading_route(models_doc, key)
            if model.get("route_only") and not (manifest or flops is not None):
                # Refuse rather than default. This archetype has no byte/FLOP formula BY DESIGN, so
                # the useful failure names the missing declaration and hands over the reading -- the
                # unhelpful one would be to price it as the nearest modelled name.
                cite = f"\n  read first: {reading_route['cite']}" if reading_route else ""
                sys.exit(f"[hw_budget] {key!r} is a ROUTE-ONLY archetype: no byte/FLOP model.\n"
                         f"  {model['route_only']}\n"
                         f"  declare what the kernel MOVES: --tensors \"name:dir:dtype:dims[:xN]\" "
                         f"[--flops N --flops-engine mfma|valu]{cite}")
            env = _build_env(shapes or {}, db, models_doc, _scale_overhead(dtype, db))
            if model.get("flops") or model.get("bytes_min") or model.get("bytes_hbm"):
                try:
                    if flops is None:
                        flops = _safe_eval(model["flops"], env)
                    # The model already carries BOTH ends: bytes_min is the all-cached reading and
                    # bytes_hbm adds the re-reads. Using only one of them throws away the model's own
                    # statement of its uncertainty (they are the same expression where there is none).
                    ends = [_safe_eval(model[f], env)
                            for f in ("bytes_min", "bytes_hbm") if model.get(f)]
                    model_lo, model_hi = min(ends), max(ends)
                except Exception as e:  # noqa: BLE001
                    sys.exit(f"[hw_budget] model eval failed ({e}). Provide the shape vars this "
                             f"model needs: {models_doc.get('_variables', '')}")
        if manifest:
            # the manifest REPLACES a name-keyed formula: perfect reuse at the bottom, the declared
            # traversals (times any dead-grid duplication) at the top
            model_lo = manifest["bytes_unique"]
            model_hi = manifest["bytes_issued"] * (grid_redundancy or 1.0)
            key = key or "tensor-manifest"
            model = dict(model or {})
            model.setdefault("bound_direction",
                             "BRACKETING: the low end assumes every declared tensor is read exactly "
                             "once (perfect reuse), the high end assumes the declared traversals all "
                             "reach the fabric (no reuse). The truth is wherever the cache hierarchy "
                             "puts it, which no manifest knows -- measure it.")
        # An explicit --flops-engine wins; otherwise the archetype declares it (workload_models.json
        # keys it to the formula that produces the FLOPs, which is where the fact belongs). The
        # manifest path has no archetype to ask, so an unlabelled --flops N is "unknown" rather than
        # assumed-MFMA: assuming it is how a VALU kernel acquires a matrix-rate floor.
        flops_engine = (flops_engine or model.get("flops_engine")
                        or ("unknown" if flops else "none (no FLOPs)"))
        if flops is None:
            sys.exit("[hw_budget] no FLOP count: --tensors gives bytes only. Pass --flops N (a "
                     "gather / scatter / quantize / index kernel may legitimately have ~0 useful "
                     "FLOPs, and 0 is a real answer -- it says the compute floor cannot bind), or "
                     "add --workload to take the FLOPs from a named model.")
        model_bytes = model_hi

        # --- CALIBRATION (roofline-models.md ## Calibration) -----------------------------------
        # Both terms of a roofline are calibratable, and both default to the OPTIMISTIC datasheet
        # reading: the analytic byte model for the numerator, the datasheet HBM peak for the
        # denominator. The datasheet peak is unreachable at ANY real access shape, so a gap
        # computed against it is part measurement, part fiction -- every number below therefore
        # carries whether it came from the datasheet/model or from a measurement.
        # The numerator gets the same treatment as the denominator: it is an INTERVAL, because the
        # independent counter routes (TCC_MISS*128B, EA RDREQ/WRREQ*64B, FETCH_SIZE+WRITE_SIZE)
        # agree to within a fraction of a percent on a pure stream and can diverge several-fold on
        # a real kernel -- partial-line writes, write-allocate, atomics and split-K partials are
        # visible to some routes and invisible to others. Which route is "right" is a property of
        # THIS kernel's access shape, so the tool must not pick one: pass every route you measured
        # and let the width of the answer show how much of the prize is actually known.
        if measured_bytes_mb is None:
            b_lo, b_hi = model_lo, model_hi
            bytes_src = "model" if b_lo == b_hi else "model (bracket)"
        else:
            vals = measured_bytes_mb if isinstance(measured_bytes_mb, (list, tuple)) \
                else [measured_bytes_mb]
            b_lo, b_hi = min(vals) * 1e6, max(vals) * 1e6
            bytes_src = "measured" if b_lo == b_hi else "measured (routes disagree)"
        hbm_bytes = b_lo
        peak_bw = sku.get("peak_hbm_tb_s")
        if measured_hbm_tb_s:
            bw_lo, bw_hi = min(measured_hbm_tb_s), max(measured_hbm_tb_s)
            bw_src = "calibrated"
        else:
            bw_lo = bw_hi = peak_bw
            bw_src = "datasheet"
        # PHYSICS TIGHTENS THE BRACKET. A byte-count end that implies a rate above the fastest rate
        # this box can reach is not merely pessimistic, it is impossible: that traffic never got to
        # the fabric. Clamping turns a useless bracket end into information -- "at most this much of
        # the declared re-traversal actually misses". Clamp against the best available ceiling (a
        # probed one is far tighter than the datasheet peak), and only ever narrow.
        bytes_clamped = None
        cap_rate = bw_hi if bw_src == "calibrated" else peak_bw
        if measured_ms and cap_rate and b_hi > cap_rate * 1e12 * measured_ms * 1e-3 > b_lo:
            cap = cap_rate * 1e12 * measured_ms * 1e-3
            bytes_clamped = (f"the high end of the byte bracket ({b_hi / 1e6:.0f} MB) would need "
                             f"{b_hi / (measured_ms * 1e-3) / 1e12:.2f} TB/s, above the fastest rate "
                             f"this box reaches ({cap_rate} TB/s, {bw_src}) -- impossible, so it is "
                             f"clamped to {cap / 1e6:.0f} MB. Read that as a measurement: at most "
                             f"that much of the declared re-traversal actually reaches the fabric, "
                             f"and the rest is absorbed by cache.")
            b_hi = cap

        ai = flops / hbm_bytes if hbm_bytes else float("inf")
        ai_lo = flops / b_hi if b_hi else float("inf")

        # --- PRECONDITIONS FOR A ROOFLINE AT ALL -------------------------------------------------
        # Both roofline terms assume the whole machine is streaming. Two common situations break that
        # assumption outright, and when either holds the floors below are not floors -- they are
        # unreachable numbers whose distance from the measurement invites a round on the wrong axis.
        fill = None
        if grid:
            cus = sku.get("cus") or 0
            waves = math.ceil(grid / cus) if cus else None
            fill = {"workgroups": grid, "cus": cus,
                    "workgroups_per_cu": round(grid / cus, 2) if cus else None,
                    # with fewer workgroups than CUs, the rest of the machine is idle by construction
                    "machine_fill_pct": round(100.0 * min(grid, cus) / cus, 1) if cus else None,
                    # and past that, a grid that is not a multiple of the CU count pays a tail: the
                    # last wave runs with only (grid mod cus) CUs busy
                    "tail_waste_pct": (round(100.0 * (waves * cus - grid) / (waves * cus), 1)
                                       if cus and waves else None)}
            if cus and grid < cus:
                fill["_refused"] = (
                    f"{grid} workgroups on {cus} CUs: {100.0 * (1 - grid / cus):.0f}% of the machine "
                    f"is idle by construction. NEITHER roofline term applies -- the compute peak and "
                    f"the memory ceiling both assume every CU is streaming, and a kernel that cannot "
                    f"fill the machine is bounded by issue/latency and by how few CUs it engages. "
                    f"The tell is a measured time that barely moves when the byte count changes by "
                    f"an order of magnitude. Fix the decomposition (more parallelism: split-K, a "
                    f"finer tile, a persistent grid) before reading any gap below.")
            elif cus and grid < 2 * cus:
                # Filling every CU once is not the same as saturating memory. Bandwidth is
                # requests-in-flight x bytes-per-request / latency, so a CU holding a single
                # workgroup has too few outstanding requests to reach a streaming rate no matter how
                # perfect its access pattern is. This is measurable rather than arguable: the
                # in-shape ceiling RISES with program count over a fixed footprint, which is what
                # mem_bw_probe.py --nprog-sweep exists to show.
                fill["_warning"] = (
                    f"{grid} workgroups on {cus} CUs is at most {grid / cus:.2f} per CU. Every CU has "
                    f"work, but a CU with one workgroup keeps too few requests in flight to reach a "
                    f"streaming rate -- bandwidth is requests-in-flight x bytes-per-request / "
                    f"latency, and this grid caps the first factor. The ceiling that applies here is "
                    f"the one probed AT THIS program count, which is well below the high-parallelism "
                    f"ceiling: mem_bw_probe.py --nprog-sweep {grid},{2 * grid},{8 * grid}. Quoting a "
                    f"high-parallelism ceiling against a low-parallelism kernel invents a gap that "
                    f"more parallelism, not a better access pattern, is what closes.")
        resident = None
        # A MISSING CAPACITY IS NOT A SMALLER CAPACITY. This used to read
        # `mall_mb or l2_mb`, which is what sku.json's own _mall_doc tells it not to do: null
        # means the memory-side LLC was never established for this part, and standing the L2 in
        # for it understates the bar by more than an order (RX7900XTX: 6 MB L2, 96 MB Infinity
        # Cache). Understating the bar clears footprints that never left cache, and the verdict
        # that follows -- a DRAM floor and a %-of-HBM-ceiling -- is then a category error on a
        # kernel that never touched DRAM. So the bar is used only in the direction it can carry:
        #   fits inside the L2            -> resident for certain, whatever the LLC turns out to
        #                                    be, since the LLC only ever ADDS capacity behind it
        #   larger than the L2, LLC null  -> UNKNOWN. Not "clears it".
        llc_mb, llc_which = sku.get("mall_mb"), "memory-side LLC"
        if footprint_mb and not llc_mb and sku.get("l2_mb"):
            l2 = sku["l2_mb"]
            resident = {"footprint_mb": footprint_mb, "llc_mb": None, "llc": "unknown",
                        "l2_mb": l2, "footprint_over_llc": None,
                        # what the refusal below, if it fires, actually compared against
                        "bar_mb": l2, "bar": "L2 (memory-side LLC unknown, and it only adds to it)"}
            if footprint_mb < l2:
                resident["_refused"] = (
                    f"footprint {footprint_mb:g} MiB fits inside the {l2} MiB L2 alone, so it is "
                    f"cache-resident whatever this part's memory-side LLC turns out to be -- that "
                    f"cache sits BEHIND the L2 and only adds capacity. A benchmark averaging "
                    f"repeated iterations over the same buffers measures the steady state, and in "
                    f"the steady state this working set never reaches DRAM: the HBM floor is not a "
                    f"floor and a %-of-HBM-ceiling is a category error.")
            else:
                resident["_warning"] = (
                    f"this SKU's memory-side LLC capacity is not established (sku.json mall_mb is "
                    f"null), so there is NO residency bar to compare {footprint_mb:g} MiB against. "
                    f"The {l2} MiB L2 is not that bar -- on the parts where the LLC IS known it is "
                    f"8-40x the L2 -- so 'clears the L2' says nothing about whether this working "
                    f"set reaches DRAM. Read the memory verdict below as conditional on it doing "
                    f"so. To settle it: establish the part's Infinity Cache size and fill in "
                    f"mall_mb, or measure a cold first iteration against the averaged one.")
        elif footprint_mb and llc_mb:
            resident = {"footprint_mb": footprint_mb, "llc_mb": llc_mb, "llc": llc_which,
                        "footprint_over_llc": round(footprint_mb / llc_mb, 2),
                        "bar_mb": llc_mb, "bar": llc_which}
            if llc_mb <= footprint_mb < 4 * llc_mb:
                # Clearing the cache by a hair is not clearing it. A footprint a small multiple of
                # the LLC still has a large fraction of its re-reads served from cache, so the
                # achieved rate is a blend of cache and DRAM and the HBM floor is optimistic by an
                # unknown amount. Same margin the ceiling probe demands of itself.
                resident["_warning"] = (
                    f"footprint {footprint_mb:g} MiB is only {footprint_mb / llc_mb:.2f}x the "
                    f"{llc_mb} MiB {llc_which}, so a large share of the re-reads is still served "
                    f"from cache and the achieved rate is a cache/DRAM blend. The HBM floor is "
                    f"optimistic here by an amount nobody has measured. Treat the memory verdict as "
                    f"provisional until the footprint clears ~4x the LLC (the same margin the "
                    f"ceiling probe requires of itself) or you measure the split.")
            if footprint_mb < llc_mb:
                resident["_refused"] = (
                    f"footprint {footprint_mb:g} MiB fits inside the {llc_mb} MiB {llc_which}. A "
                    f"single cold pass still reaches DRAM, but a benchmark that averages repeated "
                    f"iterations over the SAME buffers measures the steady state, and in the steady "
                    f"state this working set is served from cache -- so the HBM floor is not a floor "
                    f"and a %-of-HBM-ceiling is a category error. Either report against the cache's "
                    f"bandwidth, or raise the footprint past the LLC, or time a cold first "
                    f"iteration. Note that the byte counters cannot settle this for you: they sit at "
                    f"the L2/fabric boundary, in front of this cache.")
        # analytic floors: compute-bound floor vs memory-bound floor. The HBM floor is a RANGE
        # whenever the ceiling is (a probed ceiling drifts 3-4% between sessions and rises with
        # parallelism), and a gap narrower than that range is not a reason to open a round.
        # BOTH ceilings are datasheet numbers by default, so both get the same treatment. The MFMA
        # peak is no more reachable than the HBM peak: it assumes back-to-back MFMA issue with the
        # operands already in register, which no kernel sustains once it also has to load, convert
        # and address. Leaving the compute term uncalibrated while refusing the memory term put the
        # refusal on whichever axis was NOT binding -- a compute-bound kernel got a loud memory
        # refusal and then an unlabelled datasheet "prize" on the axis the round would open on.
        #
        # A FLOP COUNT DOES NOT IMPLY A COMPUTE CEILING; the engine it issues on does. `flops /
        # MFMA_peak` is a floor only for FLOPs that go through MFMA. The `valu` archetypes (flops =
        # c*n) contain no matrix multiply, so charging them at the matrix rate produces a floor one
        # to two ORDERS too low, which then never binds -- and the tool reports "memory" by default.
        # That default is the dangerous part: it is not a measurement, and for a VALU-heavy kernel
        # with modest bytes it is wrong in the direction that opens a byte-removal round which can
        # collect nothing.
        compute_unknown = None
        if measured_tflops:
            # a measured rate is the ceiling for whatever engine it was measured on, so it settles
            # the engine question too -- this is the escape hatch for every branch below
            tf_lo, tf_hi = min(measured_tflops), max(measured_tflops)
            tf_src = "calibrated"
        elif not flops:
            # 0 FLOPs takes 0 time on ANY engine, so no ceiling is needed. This stays a real floor
            # (and a real conclusion: the compute term cannot bind) rather than becoming an unknown.
            tf_lo = tf_hi = None
            tf_src = "not-needed (no FLOPs)"
        elif peak_tflops is None:
            tf_lo = tf_hi = None
            tf_src = "unavailable"
            compute_unknown = (
                f"no {dtype} entry in this SKU's peak_tflops table (it has: "
                f"{', '.join(peak_dtypes)}), so there is NO compute ceiling to divide by and no "
                f"compute floor. This is reported rather than substituted: the old behaviour "
                f"silently used the bf16 rate, which for a native low-precision dtype is several "
                f"times too low and lifts the compute floor enough to flip the binding. Measure the "
                f"rate at this dtype and pass --measured-tflops lo,hi.")
        elif flops_engine != "mfma":
            tf_lo = tf_hi = None
            tf_src = f"wrong-engine ({flops_engine})"
            compute_unknown = (
                f"these FLOPs issue on the {flops_engine.upper()}, not on MFMA, so the MFMA peak is "
                f"not their ceiling and dividing by it would understate the compute floor by one to "
                f"two orders. No VALU/transcendental peak is carried in the tables, so there is no "
                f"compute floor here until you supply one: time the op at this rate and pass "
                f"--measured-tflops lo,hi. Note the hardware facts that make this its own axis -- "
                f"MFMA and VALU CO-EXECUTE (hw_constants mfma_valu_coexec) and transcendentals run "
                f"at a fraction of the VALU rate (exp_rate_vs_valu), so a softmax or a norm is "
                f"priced on neither the matrix peak nor the plain VALU peak.")
        else:
            tf_lo = tf_hi = peak_tflops
            tf_src = "datasheet"
        # fastest rate -> lowest floor, and the reverse; identical ends when uncalibrated
        if not flops:
            mfma_floor_lo = mfma_floor_hi = 0.0
        elif tf_hi:
            mfma_floor_lo = 1e3 * flops / (tf_hi * 1e12)
            mfma_floor_hi = 1e3 * flops / (tf_lo * 1e12)
        else:
            mfma_floor_lo = mfma_floor_hi = None
        mfma_floor_ms = mfma_floor_lo
        # fewest bytes over the fastest ceiling -> lowest floor, and the reverse for the highest
        hbm_floor_lo = 1e3 * b_lo / (bw_hi * 1e12) if bw_hi else None
        hbm_floor_hi = 1e3 * b_hi / (bw_lo * 1e12) if bw_lo else None
        # An UNKNOWN compute floor is not a zero compute floor, and collapsing the two is how the
        # tool used to hand back an unearned verdict: with no compute floor to compare, memory won by
        # default and printed as plain "memory". "Cannot bind" (0 FLOPs) is a conclusion; "unknown"
        # (no ceiling for this dtype, or FLOPs on another engine) is the absence of one, and the
        # reader has to be able to tell which they were given.
        if mfma_floor_hi is None and hbm_floor_hi is None:
            binding, floor_lo, floor_hi = "UNKNOWN (neither floor computable)", None, None
        elif mfma_floor_hi is None:
            # Whether a missing compute floor MATTERS depends on intensity: a kernel far below the
            # ridge is memory-bound on intensity grounds alone, and escalating there would be an
            # off-axis warning -- the kind that trains readers to skip the on-axis ones.
            #
            # But the ridge is an MFMA ridge (peak_tflops / peak_hbm), so it can only settle the
            # question for MFMA FLOPs. Two directions, and only one of them is safe:
            #   missing DTYPE, still MFMA -> the gate runs on `ridge_floor`, a provable lower bound
            #     on this dtype's crossover (see its derivation above), because "AI is far below a
            #     floor under the ridge" implies "far below the ridge". This used to run on the
            #     fp16 ridge standing in for the caller's, which was right for narrower dtypes and
            #     wrong for wider ones -- fp32's peak is 1/8 of fp16's, so its ridge is 1/8 too and
            #     "far below the fp16 ridge" says nothing at all there.
            #   FLOPs on the VALU -> the VALU peak is a small fraction of the matrix peak, so the
            #     real ridge is roughly an order LOWER and a kernel comfortably below the MFMA ridge
            #     can sit on the compute side of the VALU one. Using the MFMA ridge here would
            #     manufacture exactly the false "memory" confirmation this whole branch exists to
            #     prevent -- the same category error, one level up. Still gated on flops_engine.
            if ridge_floor and ai <= 0.5 * ridge_floor and flops_engine == "mfma":
                binding = ("memory (confirmed by intensity: AI is far below the ridge, so the "
                           "un-priced compute term could not bind here anyway)")
            else:
                binding = "memory (compute floor UNKNOWN -- memory is NOT confirmed as the bound)"
            floor_lo, floor_hi = hbm_floor_lo, hbm_floor_hi
        elif hbm_floor_hi is None:
            binding, floor_lo, floor_hi = "compute", mfma_floor_lo, mfma_floor_hi
        elif (mfma_floor_lo or 0) > hbm_floor_hi:
            binding, floor_lo, floor_hi = "compute", mfma_floor_lo, mfma_floor_hi
        elif (mfma_floor_hi or 0) < hbm_floor_lo:
            binding, floor_lo, floor_hi = "memory", hbm_floor_lo, hbm_floor_hi
        else:
            # the MFMA floor sits INSIDE the HBM floor range: the two floors are not separable at
            # this ceiling uncertainty, so do not pick one. Read the bound from the profiler.
            binding = "ambiguous (MFMA floor inside the HBM floor range)"
            floor_lo = min(mfma_floor_lo, hbm_floor_lo)
            floor_hi = max(mfma_floor_hi, hbm_floor_hi)
        # A precondition failure outranks the floor arithmetic -- but the two failures void DIFFERENT
        # floors, and collapsing them throws away a usable answer. An unfilled machine engages
        # neither resource, so both floors go. A cache-resident working set only voids the MEMORY
        # floor: the arithmetic still has to happen, so the compute floor stands and is then the
        # binding one. Refusing it too would discard the only floor that still applies.
        preconditions = [p["_refused"] for p in (fill, resident) if p and p.get("_refused")]
        if fill and fill.get("_refused"):
            binding = "REFUSED (machine not filled -- neither resource is engaged)"
        elif resident and resident.get("_refused"):
            if mfma_floor_ms:
                binding = "compute (memory floor VOID: working set is cache-resident)"
                floor_lo, floor_hi = mfma_floor_lo, mfma_floor_hi
            elif mfma_floor_hi is None:
                # residency voided the memory term and the compute term was never computable, so
                # nothing is left -- but for a DIFFERENT reason than the 0-FLOP case below, and the
                # instruction that follows differs with it: there is a number to go get here.
                binding = "UNKNOWN (memory floor VOID, compute floor never computed)"
                floor_lo = floor_hi = None
                preconditions.append(
                    "no floor to report: the memory term is void (cache-resident working set) and "
                    "the compute term was never computable -- see the compute-floor note. Unlike a "
                    "0-FLOP kernel, this one HAS arithmetic that could bind; supply its rate "
                    "(--measured-tflops) and the compute floor comes back as the only one that "
                    "applies. Do not read this as 'no roofline answer exists'.")
            else:
                # Both terms are gone: no FLOPs means the compute term cannot bind, and residency
                # voided the memory term. There is NO analytic floor here, and that is the answer --
                # a 0 ms floor would report an infinite multiple, and calling it "unmeasured" would
                # imply a number exists to be found.
                binding = "NONE (no analytic floor exists for this kernel)"
                floor_lo = floor_hi = None
                preconditions.append(
                    "no analytic floor at all: the compute term cannot bind (no useful FLOPs) and "
                    "the memory term is void (cache-resident working set). Whatever this kernel "
                    "spends its time on is not on either roofline axis -- it is launch overhead, "
                    "issue/latency, or serialization. Measure it there; there is no roofline answer "
                    "to find, so do not compute a gap and do not open a byte-removal round.")

        w = out["workload"] = {
            "model": key, "multi_reduction": model.get("multi_reduction", False),
            # The archetype routes the reading, and it lands in the artifact so "was it read" is a
            # question the record can answer rather than a thing to hope about.
            "reading_route": reading_route,
            "flops": flops,
            "hbm_bytes_min": hbm_bytes, "hbm_bytes_model": model_bytes,
            "hbm_bytes_range": [b_lo, b_hi], "bytes_source": bytes_src,
            "bytes_clamped_by_physics": bytes_clamped,
            "hbm_ceiling_tb_s": [bw_lo, bw_hi], "hbm_ceiling_source": bw_src,
            # THE ROOFLINE BASIS PAIR (GEAK-wide rule, perf_knowledge/profiling/roofline_on_mi.md):
            # every %-of-roofline names what its numerator and denominator rest on. A datasheet
            # denominator may RANK only; a measured numerator over a probed/calibrated denominator
            # is the only combination that may gate or close.
            "numerator_basis": "counters" if measured_bytes_mb is not None else "model",
            "denominator_basis": _denominator_basis(sku, bw_src, ceiling_source_ref),
            "may_gate": (measured_bytes_mb is not None
                         and _denominator_basis(sku, bw_src, ceiling_source_ref) != "datasheet"),
            "arithmetic_intensity": round(ai, 1),
            "ridge_ops_per_byte": round(ridge, 1) if ridge else None,
            # the crossover belongs to a dtype, so the label that names it has to say which one --
            # a bare "ridge 218" reads as a property of the SKU, which is how one dtype's ridge got
            # quoted for all of them in the first place
            "ridge_dtype": dtype if ridge else None,
            # a byte interval that straddles the ridge cannot name a regime -- the SAME kernel is
            # compute-bound on one counter route and memory-bound on another, and picking either
            # one silently is how a round gets opened on the wrong axis
            "regime": (
                # NO RIDGE, NO REGIME. With no rate for this dtype there is no crossover to compare
                # against, and `ai > 0` would have called every such kernel compute-bound -- a
                # verdict manufactured out of a missing table entry.
                "unknown (no compute rate for this dtype -- no crossover to compare against)"
                if not ridge else
                "ambiguous (byte routes straddle the ridge)"
                if ai_lo <= ridge <= ai else
                ("compute" if ai > ridge else "memory") + (
                    " (near ridge)" if 0.5 * ridge < ai < 2 * ridge else "")),
            # a 0-FLOP kernel has a 0 ms compute floor, and that is a real, informative answer: the
            # compute term CANNOT bind, so printing None ("nobody measured it") would be a lie
            "mfma_only_floor_ms": round(mfma_floor_ms, 4) if mfma_floor_ms is not None else None,
            "mfma_floor_ms": ([round(mfma_floor_lo, 4), round(mfma_floor_hi, 4)]
                              if mfma_floor_hi is not None else None),
            "compute_ceiling_tflops": [tf_lo, tf_hi], "compute_ceiling_source": tf_src,
            # which engine the FLOPs issue on, and (when it could not be priced) why. Both are
            # first-class output: the engine is what makes a FLOP count into a floor at all.
            "flops_engine": flops_engine,
            "compute_floor_unknown": compute_unknown,
            "sku_peak_dtypes": peak_dtypes,
            "hbm_floor_ms": [round(hbm_floor_lo, 4), round(hbm_floor_hi, 4)] if hbm_floor_hi else None,
            "binding_floor": binding,
            "binding_floor_ms": [round(floor_lo, 4), round(floor_hi, 4)] if floor_hi else None,
            # The MoE caveat is about the DEFAULTED expert count, so it must not be repeated at a
            # caller who measured and passed one -- a warning that fires when its cause is absent
            # trains the reader to skip it, and then it is not there when it matters.
            "bound_direction": _bound_direction(model, key, shapes,
                                                bytes_measured=bytes_src.startswith("measured"),
                                                routes_agree=bytes_src == "measured"),
            # "calibrated" is scoped to the axis this kernel is BOUND ON, because that is the axis
            # every decision comes off. A memory-bound kernel does not need the MFMA peak probed,
            # and a compute-bound one does not need its counter routes settled -- demanding both
            # would mark every honest answer uncalibrated, which trains the reader to ignore it.
            "binding_floor_source": (tf_src if binding.startswith("compute") else
                                     bw_src if binding.startswith("memory") else
                                     "calibrated" if tf_src == bw_src == "calibrated" else "datasheet"),
            # An unevaluated term is not a calibrated model: when memory is the bound only because
            # the compute term was never priced, the ceilings being measured does not make the
            # VERDICT measured, so this stays False until the missing term is supplied or intensity
            # settles it independently.
            "calibrated": ("NOT confirmed" not in binding and
                           (tf_src == "calibrated" if binding.startswith("compute") else
                            bw_src == "calibrated" and bytes_src == "measured"
                            if binding.startswith("memory") else
                            tf_src == bw_src == "calibrated" and bytes_src == "measured")),
            # WHERE the calibrated ceiling came from. A probed ceiling and a remembered one are the
            # same float by the time they reach the denominator, and both print as [calibrated] --
            # so `calibrated` above answers "was a measurement supplied" and cannot answer "by
            # what". Passing --measured-from records the probe artifact, and passing a ceiling
            # without one records `hand_entered`, which is not refused (a number recalled from a
            # previous session is often right) but is no longer indistinguishable from a probe run
            # against this shape on this box.
            "ceiling_provenance": {
                "entry": ("probe_artifact" if ceiling_source_ref else
                          "hand_entered" if (measured_hbm_tb_s or measured_tflops) else
                          "datasheet"),
                "refs": list(ceiling_source_ref or []),
            },
            # ...and when it is NOT calibrated, WHICH term is still uncalibrated decides what to go
            # measure next. Collapsing "you never probed the ceiling" and "your counter routes
            # disagree" into one warning sends the reader to the wrong instrument.
            "uncalibrated_terms": [t for t, bad in (
                ("ceiling (datasheet HBM peak, never reachable) -> mem_bw_probe.py in-shape",
                 bw_src != "calibrated" and not binding.startswith("compute")),
                ("ceiling (datasheet MFMA peak: assumes back-to-back MFMA issue with operands "
                 "already in register, which no real kernel sustains) -> time a dense back-to-back "
                 "MFMA loop at this dtype and pass --measured-tflops",
                 tf_src == "datasheet" and not binding.startswith("memory")),
                # this one fires even on a memory-BINDING kernel, because the binding is only
                # provisional while a term is missing -- that is the whole point of the state
                (f"compute floor NOT COMPUTED ({tf_src}) -> the memory verdict is provisional until "
                 f"this term exists; see compute_floor_unknown for the instrument",
                 compute_unknown is not None and "NOT confirmed" in binding),
                ("bytes (analytic model) -> parse_pmc.py memory block",
                 bytes_src.startswith("model") and not binding.startswith("compute")),
                ("bytes (counter routes disagree; which one holds is a property of THIS access "
                 "shape) -> settle it with a known-bytes probe at this shape, or carry the width",
                 bytes_src.startswith("measured (") and not binding.startswith("compute")),
            ) if bad],
            "machine_fill": fill,
            "cache_residency": resident,
            "roofline_preconditions_refused": preconditions or None,
            # AN UNCHECKED PRECONDITION IS NOT A PASSED ONE. This is guardrail R9, which
            # `close_audit.py` already applies to a check whose input is missing, and it was absent
            # exactly here -- where the preconditions are OPTIONAL ARGUMENTS. With none of the three
            # supplied, `roofline_preconditions_refused` is None, which reads identically to "all
            # three were checked and none fired", and the gap below is then printed with the same
            # confidence either way.
            #
            # Measured, on a run that had every number to hand: `structure_census.py fill` was given
            # grid_wgs=128, cus=304, footprint_mb=160, llc_mb=256, dispatches=1 and computed
            # `device_fill: 0.421` with the verdict "fewer workgroups than CUs -- neither roofline
            # term binds" and `roofline_applies: False`. The SAME THREE NUMBERS were never passed
            # here, so all three budget artifacts on that tree carry machine_fill=None,
            # cache_residency=None, dispatches_in_timed_region=None, an unrefused datasheet ceiling
            # and `calibrated: False`. Two tools, one set of inputs, and only one of them got them.
            # `--from-census` closes the supply route; this field makes the omission legible when
            # nobody uses it.
            "roofline_preconditions_unchecked": [
                p for p, supplied in (
                    ("device fill (--grid WGS): with fewer workgroups than CUs part of the machine "
                     "is idle by construction and NEITHER roofline term applies", grid is not None),
                    ("cache residency (--footprint-mb MiB): a working set inside the memory-side "
                     "LLC voids the MEMORY floor in a repeated-iteration benchmark's steady state",
                     footprint_mb is not None),
                    ("dispatch count (--dispatches N): a per-kernel floor cannot be compared "
                     "against the total time of a region containing several kernels",
                     dispatches is not None),
                ) if not supplied] or None,
            "tensor_manifest": manifest["tensors"] if manifest else None,
            # A measured time that covers N dispatches cannot be read against a floor computed for
            # one kernel. This is not a rounding issue: MoE and split-K "kernels" routinely time a
            # region containing a zero-fill, one or two GEMMs, a requant and a reduction, and the
            # per-kernel floor is then compared against the sum of all of them.
            "dispatches_in_timed_region": dispatches,
        }
        if dispatches and dispatches > 1:
            w["dispatch_warning"] = (
                f"the timed region contains {dispatches} dispatches but this budget describes ONE "
                f"kernel. Every floor and gap below is per-kernel, so comparing it against the "
                f"region's total time inflates the apparent gap by whatever the other "
                f"{dispatches - 1} dispatches cost. Either budget each dispatch against its own "
                f"time (rocprofv3 reports per-dispatch durations), or declare every dispatch's "
                f"traffic in --tensors and compare against the region total -- not a mix of the two.")
        if measured_ms is not None and floor_hi:
            w["measured_ms"] = measured_ms
            w["multiple_over_floor"] = [round(measured_ms / floor_hi, 2),
                                        round(measured_ms / floor_lo, 2)]
            # An uncalibrated ceiling puts the floor BELOW the real one, so the multiple over it is
            # an upper bound on the prize, not the prize. The direction is what makes this worth
            # saying: the error only ever flatters the opportunity, so a round sized on this number
            # is sized on the most optimistic reading available, and "2.4x on the table" can be
            # 1.8x once the ceiling is real. Reported (it is a genuine bound) but never as fact.
            if w["binding_floor_source"] == "datasheet" and not binding.startswith(("REFUSED", "NONE")):
                w["prize_is_upper_bound"] = (
                    f"{w['multiple_over_floor'][0]}x is an UPPER BOUND on the prize, not the prize: "
                    f"the {binding.split()[0]} floor rests on a datasheet peak, which sits below the "
                    f"real floor, so the multiple over it can only overstate what is reachable. "
                    f"Calibrate that ceiling before sizing a round on this number.")
            if hbm_floor_hi:
                w["achieved_hbm_tb_s"] = [round(b_lo / (measured_ms * 1e-3) / 1e12, 2),
                                          round(b_hi / (measured_ms * 1e-3) / 1e12, 2)]
                # %-of-ceiling and the gap are the two numbers a round is opened on, and BOTH are
                # just the ceiling read two ways. Against the datasheet peak they do not merely
                # skew: a kernel already at its reachable ceiling reads as having tens of percent
                # of headroom, which is a prize that does not exist. So they are REFUSED, not
                # printed with a caveat -- the datasheet peak survives only as the loosest possible
                # floor, which is all it is good for.
                if preconditions:
                    w["pct_of_hbm_ceiling"] = w["hbm_gap_ms"] = None
                    w["ceiling_refused"] = (
                        "no %-of-ceiling and no gap: this kernel does not meet the preconditions for "
                        "a roofline in the first place (see above). A fraction-of-ceiling computed "
                        "anyway would be measuring the wrong resource.")
                elif bw_src == "datasheet":
                    w["pct_of_hbm_ceiling"] = w["hbm_gap_ms"] = None
                    w["ceiling_refused"] = (
                        "no %-of-ceiling and no gap: the only ceiling available is this SKU's "
                        "datasheet peak, which no real access shape reaches. Probe the in-shape "
                        "ceiling (mem_bw_probe.py at this run length / stride / read:write mix / "
                        "program count) and pass --measured-hbm-tb-s lo,hi. The datasheet floor "
                        "above is a plausibility bound only -- a measurement under it is "
                        "impossible, a measurement over it means nothing.")
                else:
                    # A fraction of a ceiling cannot exceed 1 and a gap cannot be negative. When the
                    # arithmetic says otherwise, the byte bracket's high end already saturates the
                    # ceiling -- which is not an error to hide but the strongest finding available:
                    # this kernel MAY ALREADY BE AT THE WALL, and the whole apparent gap may be zero.
                    p_hi = 100 * hbm_floor_hi / measured_ms
                    w["pct_of_hbm_ceiling"] = [round(100 * hbm_floor_lo / measured_ms, 1),
                                               round(min(p_hi, 100.0), 1)]
                    w["hbm_gap_ms"] = [round(max(measured_ms - hbm_floor_hi, 0.0), 4),
                                       round(measured_ms - hbm_floor_lo, 4)]
                    if p_hi >= 100.0:
                        w["at_ceiling_possible"] = (
                            "the high end of the byte bracket saturates the ceiling: if that much "
                            "traffic is real, this kernel is ALREADY AT the memory wall and the gap "
                            "is zero. That is the single most decision-relevant possibility on the "
                            "table, and only measured bytes can rule it in or out -- do not open a "
                            "round against the low end of the bracket while this is live.")
            # A measured time BELOW the most optimistic floor is not a record, it is the model
            # announcing that it does not apply to this shape. Reporting it as "0.17x over the
            # floor" hands the caller a prize that cannot exist.
            if measured_ms < floor_lo and resident and resident.get("_refused"):
                # cause already established: a cache-resident working set does not have to obey a
                # DRAM floor, so this is the residency check being CONFIRMED, not a new mystery
                w["floor_breached"] = True
                w["_error"] = (
                    f"measured {measured_ms} ms is below the memory floor {round(floor_lo, 4)} ms, "
                    f"which is exactly what the cache-residency refusal above predicts: the working "
                    f"set fits in the {resident['bar_mb']} MiB {resident['bar']}, so the traffic "
                    f"did not have to come from DRAM and a DRAM floor does not bind it. This is "
                    f"CONFIRMATION, not a separate error -- fix the measurement (cold iteration or a "
                    f"footprint past the LLC) rather than hunting the byte model.")
            elif measured_ms < floor_lo:
                w["floor_breached"] = True
                w["_error"] = (
                    f"measured {measured_ms} ms is BELOW the {binding} floor {round(floor_lo, 4)} ms "
                    f"-- physically impossible, so an input is wrong. Usual causes, in order: "
                    f"(1) the byte model over-counts for this shape (MoE at decode M charging all E "
                    f"experts; a per-stage kernel charged both stages) -> pass --measured-dram-mb; "
                    f"(2) --dtype is wider than the real one (a 4-bit weight read as 2 B is 4x); "
                    f"(3) the stream is cache-resident, so the DRAM bytes are not what the model "
                    f"assumes -> measure them (parse_pmc.py memory block). Do NOT read the "
                    f"multiple-over-floor as a prize until this closes."
                    # (3) is not a hypothesis here, it is the leading one -- this SKU has no
                    # established LLC size, so nothing above could have ruled residency out.
                    + (f" NOTE: cause (3) is the FIRST thing to check on this part -- its "
                       f"memory-side LLC capacity is unknown (mall_mb null), so the residency "
                       f"check above could not run and a resident working set is exactly what a "
                       f"sub-floor time looks like."
                       if resident and resident.get("llc_mb") is None else ""))
    return out


def _nm(v, unit=""):
    """A missing microarch constant prints as `needs-measure`, never as `None <unit>` -- a
    bare None reads as 'this resource is zero/absent' instead of 'nobody measured it yet'."""
    return "needs-measure" if v is None else f"{v}{unit}"


def _rng(v, unit=""):
    """A [lo, hi] pair prints as `lo-hi unit`, and collapses to one number only when lo == hi.
    A ceiling (and therefore every floor/gap derived from it) is an interval with an error bar."""
    if v is None:
        return "needs-measure"
    if not isinstance(v, (list, tuple)):
        return f"{v}{unit}"
    lo, hi = v
    return f"{lo}{unit}" if lo == hi else f"{lo}-{hi}{unit}"


def _print_human(b):
    print(f"\n=== hardware budget: {b['sku']} ({b['arch']}, {b['dtype']}) ===")
    r = b["resources"]
    print(f"  MFMA issue     peak {r['mfma_issue']['peak_tflops']} TFLOPS ({b['dtype']})  "
          f"-- {r['mfma_issue']['note']}")
    print(f"  Registers      {_nm(r['registers']['vgpr_per_simd'])}/SIMD  -- {r['registers']['note']}")
    # The unit is NOT a constant here. Printing "KiB/CU" against an RDNA part labels a per-WGP
    # pool (or nothing at all) with a granularity it does not have, which is how "64 KiB/CU"
    # ended up on RDNA in the first place. Take the scope from the data, and when the pool is
    # unknown print NOT PRICED rather than an "n/a" that still carries a unit.
    _lc = r["lds_capacity"]
    if _lc["occupancy_priced"]:
        print(f"  LDS capacity   {_lc['occupancy_pool_kib']} KiB/{_lc['occupancy_pool_scope']}"
              f"  -- {_lc['note']}")
    else:
        print(f"  LDS capacity   NOT PRICED  -- {_lc['note']}")
    lb = r["lds_bandwidth"]
    print(f"  LDS bandwidth  {_nm(lb['bytes_per_clk'], ' B/clk')}, {_nm(lb['banks'], ' banks')}, "
          f"ds_read_b128 {_nm(lb['ds_read_b128_cf_cyc'], ' cyc')} conflict-free")
    print(f"  HBM/MALL       {r['hbm']['peak_tb_s']} TB/s, ridge "
          + (f"{r['hbm']['ridge_ops_per_byte']} ops/byte AT {r['hbm']['ridge_dtype'].upper()}"
             " (per-dtype: it scales with that dtype's peak)"
             if r["hbm"]["ridge_ops_per_byte"] else "n/a (no compute rate for this dtype)"))
    print(f"  L2/fabric      {r['l2_fabric']['l2_mb']} MB L2 + "
          f"{_nm(r['l2_fabric']['mall_mb'], ' MB memory-side LLC')}")
    for line in textwrap.wrap(r["l2_fabric"]["note"], 88):
        print(f"                 {line}")
    print(f"  CUs            {b['cus']}")
    w = b.get("workload")
    if w:
        print(f"\n  workload: {w['model']}"
              f"{'  [multi-reduction]' if w['multi_reduction'] else ''}")
        # Printed HERE, next to the archetype that selects it, because that is the moment the choice
        # is made. Naming the same pages on a stage card instead reached 0 of 19 measured runs.
        rr = w.get("reading_route")
        if rr:
            print(f"    READ FIRST -> {rr['cite']}  ({rr.get('path', rr['page'])})")
            print("      the archetype decides the tile, the layer order and the version gates; "
                  "the bound decides the lever")
        if w.get("tensor_manifest"):
            print("    tensor manifest (bytes a kernel MOVES, not tensors it is handed):")
            for t in w["tensor_manifest"]:
                trav = f" x{t['traversals']:g}" if t["traversals"] != 1 else ""
                print(f"      {t['name']:<14s} {t['dir']:<2s} {t['dtype']:<6s} "
                      f"{t['elements']:.4g} elem{trav}  -> {t['unique_bytes']/1e6:9.2f} MB unique, "
                      f"{t['issued_bytes']/1e6:9.2f} MB issued")
        print(f"    FLOPs {w['flops']:.3e}   HBM bytes {_rng([f'{x:.3e}' for x in w['hbm_bytes_range']])} "
              f"[{w['bytes_source']}]   AI {w['arithmetic_intensity']} ops/byte "
              + (f"(ridge {w['ridge_ops_per_byte']} at {w['ridge_dtype']})"
                 if w["ridge_ops_per_byte"] else "(no ridge: this dtype has no compute rate)"))
        print(f"    regime: {w['regime']}")
        if w.get("bytes_clamped_by_physics"):
            for i, line in enumerate(textwrap.wrap(w["bytes_clamped_by_physics"], 88)):
                print(f"      {'! ' if i == 0 else '  '}{line}")
        if w.get("dispatch_warning"):
            print("    *** MULTI-DISPATCH TIMED REGION ***")
            for line in textwrap.wrap(w["dispatch_warning"], 92):
                print(f"      {line}")
        for line in textwrap.wrap(f"bytes model is {w['bound_direction']}", 92):
            print(f"      {line}")
        print(f"    HBM ceiling {_rng(w['hbm_ceiling_tb_s'], ' TB/s')} [{w['hbm_ceiling_source']}]"
              + ("   <-- datasheet peak is unreachable at ANY real access shape; probe the in-shape "
                 "ceiling (mem_bw_probe.py) and pass --measured-hbm-tb-s"
                 if w["hbm_ceiling_source"] == "datasheet" else ""))
        f_ = w.get("machine_fill")
        if f_:
            print(f"    machine fill: {f_['workgroups']} workgroups on {f_['cus']} CUs = "
                  f"{f_['workgroups_per_cu']}/CU, {f_['machine_fill_pct']}% engaged, "
                  f"tail waste {f_['tail_waste_pct']}%")
            if f_.get("_warning"):
                for i, line in enumerate(textwrap.wrap(f_["_warning"], 88)):
                    print(f"      {'! ' if i == 0 else '  '}{line}")
        c_ = w.get("cache_residency")
        if c_:
            print(f"    footprint {c_['footprint_mb']:g} MiB = {c_['footprint_over_llc']}x the "
                  f"{c_['llc_mb']} MiB {c_['llc']}"
                  if c_["llc_mb"] else
                  f"    footprint {c_['footprint_mb']:g} MiB, residency bar UNKNOWN "
                  f"(memory-side LLC not established; L2 alone is {c_['l2_mb']} MiB)")
            if c_.get("_warning"):
                for i, line in enumerate(textwrap.wrap(c_["_warning"], 88)):
                    print(f"      {'! ' if i == 0 else '  '}{line}")
        at = ("n/a -- no floor exists" if w["binding_floor"].startswith("NONE")
              else _rng(w["binding_floor_ms"], " ms"))
        ceil_txt = ("n/a" if w["compute_ceiling_tflops"][1] is None
                    else _rng(w["compute_ceiling_tflops"], " TFLOPS"))
        print(f"    compute ceiling {ceil_txt} [{w['compute_ceiling_source']}]  "
              f"(FLOPs on: {w['flops_engine']})"
              + ("   <-- datasheet peak assumes back-to-back MFMA issue with operands already in "
                 "register; time a dense MFMA loop at this dtype and pass --measured-tflops"
                 if w["compute_ceiling_source"] == "datasheet" else ""))
        print(f"    roofline basis: numerator={w['numerator_basis']}, "
              f"denominator={w['denominator_basis']} -> "
              + ("may gate/close" if w["may_gate"] else
                 "RANK ONLY (a gate or close needs a counters numerator over a probed denominator)"))
        if w.get("compute_floor_unknown"):
            print("      *** NO COMPUTE FLOOR ***")
            for line in textwrap.wrap(w["compute_floor_unknown"], 86):
                print(f"        {line}")
        prov = w.get("ceiling_provenance") or {}
        if prov.get("entry") == "probe_artifact":
            print(f"      calibrated from: {', '.join(prov['refs'])}")
        elif prov.get("entry") == "hand_entered":
            for i, line in enumerate(textwrap.wrap(
                    "ceiling was typed in, not read from a probe artifact -- it prints as "
                    "[calibrated] either way, so every percent-of-ceiling below inherits whatever "
                    "shape and session that number came from. Pass --measured-from <probe.json> to "
                    "make the provenance travel with it.", 86)):
                print(f"      {'! ' if i == 0 else '  '}{line}")
        print(f"    compute floor {_rng(w['mfma_floor_ms'], ' ms')} | HBM floor "
              f"{_rng(w['hbm_floor_ms'], ' ms')}  -> binding: {w['binding_floor']} @ {at}")
        if w.get("roofline_preconditions_refused"):
            voided = ("the MEMORY floor is void; the compute floor still applies"
                      if w["binding_floor"].startswith("compute (memory floor VOID")
                      else "the floors above do not apply")
            print(f"\n    *** ROOFLINE PRECONDITIONS NOT MET -- {voided} ***")
            for p in w["roofline_preconditions_refused"]:
                for i, line in enumerate(textwrap.wrap(p, 88)):
                    print(f"      {'- ' if i == 0 else '  '}{line}")
        if w.get("roofline_preconditions_unchecked"):
            # Printed even when nothing was refused, because "nothing fired" and "nothing ran" are
            # the two readings this line exists to separate, and only one of them licenses the gap.
            print(f"\n    *** {len(w['roofline_preconditions_unchecked'])} ROOFLINE PRECONDITION(S) "
                  f"NOT CHECKED -- not the same as met ***")
            for p in w["roofline_preconditions_unchecked"]:
                for i, line in enumerate(textwrap.wrap(p, 88)):
                    print(f"      {'? ' if i == 0 else '  '}{line}")
            print("      pass them, or pass --from-census <structure_census.json> and let the three")
            print("      numbers come from the file that already holds them.")
        if w.get("floor_breached"):
            print("\n    *** FLOOR BREACHED ***")
            for line in textwrap.wrap(w["_error"], 92):
                print(f"      {line}")
        elif "multiple_over_floor" in w:
            bf = w["binding_floor"]
            short = bf if bf in ("compute", "memory") else bf.split(" (")[0]
            # "this is the prize" is a claim about a REACHABLE floor. When the preconditions failed
            # the floor is not reachable, so the multiple is arithmetic about nothing.
            tag = "   (this is the prize)"
            if w.get("roofline_preconditions_refused"):
                tag = ("   (a REAL prize, but against the compute floor -- the memory floor is void)"
                       if bf.startswith("compute (memory floor VOID")
                       else "   (NOT a prize -- the floor is not reachable; see the refusal above)")
            elif w.get("prize_is_upper_bound"):
                tag = "   (an UPPER BOUND on the prize -- uncalibrated ceiling)"
            print(f"    measured {w['measured_ms']} ms  ->  {_rng(w['multiple_over_floor'], 'x')} over "
                  f"the {short} floor{tag}")
            if w.get("prize_is_upper_bound"):
                for i, line in enumerate(textwrap.wrap(w["prize_is_upper_bound"], 88)):
                    print(f"      {'! ' if i == 0 else '  '}{line}")
            # A multiple whose LOW end is under 1x means the top of the byte bracket puts the floor
            # above the measured time -- so that much traffic demonstrably did not happen. The
            # bracket is still too wide to size a prize; it needs measured bytes, not a wider model.
            if w["multiple_over_floor"][0] < 1:
                for i, line in enumerate(textwrap.wrap(
                        f"the low end of that multiple is under 1x, which is not a record: it means "
                        f"the TOP of the byte bracket implies a floor above the measured time, so "
                        f"that much traffic provably did not reach the fabric. The bracket is too "
                        f"wide to size anything -- measure the bytes (parse_pmc.py) instead of "
                        f"widening the model.", 88)):
                    print(f"      {'! ' if i == 0 else '  '}{line}")
            if w.get("ceiling_refused"):
                print(f"    achieved {_rng(w['achieved_hbm_tb_s'], ' TB/s')} [bytes: {w['bytes_source']}]")
                print("    *** %-OF-CEILING AND GAP REFUSED ***")
                for line in textwrap.wrap(w["ceiling_refused"], 92):
                    print(f"      {line}")
            elif "achieved_hbm_tb_s" in w:
                print(f"    achieved {_rng(w['achieved_hbm_tb_s'], ' TB/s')} = "
                      f"{_rng(w['pct_of_hbm_ceiling'], '%')} of the HBM ceiling; gap to it "
                      f"{_rng(w['hbm_gap_ms'], ' ms')}")
                if w.get("at_ceiling_possible"):
                    for i, line in enumerate(textwrap.wrap(w["at_ceiling_possible"], 88)):
                        print(f"      {'! ' if i == 0 else '  '}{line}")
                print("      that gap is a RATE gap. Only a floor probe (Gluon pack references/method/profile.md) says "
                      "whether any of it is\n      removable BYTES -- if the non-stream floor "
                      "exceeds the memory ideal, memory is not binding\n      and none of it is.")
        elif w["binding_floor"].startswith("NONE"):
            pass                       # no floor exists, so there is no multiple to print
        else:
            print(f"    (pass --measured-ms X to see your multiple over the floor -- the prize size)")
        if not w.get("calibrated"):
            print("    NOT CALIBRATED -- do not size a prize on this gap until these close:")
            for t in w["uncalibrated_terms"]:
                for i, line in enumerate(textwrap.wrap(t, 86)):
                    print(f"      {'- ' if i == 0 else '  '}{line}")
            print("      and confirm memory is binding AT ALL with a floor probe "
                  "(Gluon pack references/method/profile.md).")
    else:
        print("\n  (pass --workload + --shapes to get the FLOP/byte model + the MFMA-only floor)")
    print()


def _parse_shapes(s):
    out = {}
    if not s:
        return out
    for tok in s.replace(" ", "").split(","):
        if not tok:
            continue
        k, _, v = tok.partition("=")
        if v.lower() in ("true", "false"):
            out[k] = 1 if v.lower() == "true" else 0
        else:
            try:
                out[k] = int(v)
            except ValueError:
                try:
                    out[k] = float(v)
                except ValueError:
                    out[k] = v
    return out


_FIX = "MI355X~fp16-bf16-fp8-fp32-only"   # selftest-only fixture row, see _selftest


def _selftest():
    global _load
    _saved = _load
    try:
        return _selftest_body()
    finally:
        _load = _saved


def _selftest_body():
    global _load
    # ---- THE SKU TABLE ITSELF, BEFORE ANY ARITHMETIC ON IT -----------------------------------
    # Every floor, ceiling and bound class below is arithmetic on these cells, so a cell that was
    # mis-transcribed argues for the wrong round just as convincingly as a right one -- and until
    # this block existed, nothing checked them at all. Two errors had been sitting in the gfx950
    # rows: fp8 quoted 8-10% high, and MI355X's fp32 was a verbatim copy of MI350X's despite a
    # different fp16 peak, which is 2x off the published vector rate.
    _skus = _load("sku.json")["skus"]
    for name, s in _skus.items():
        for field in ("arch", "cus", "peak_hbm_tb_s"):
            assert s.get(field), f"{name}.{field} is missing -- every floor below divides by it"
        assert s["peak_tflops"].get("fp16"), f"{name} carries no fp16 rate"
        for dt, tf in s["peak_tflops"].items():
            assert tf and tf > 0, f"{name}.peak_tflops.{dt} = {tf!r}: an absent rate is not a zero"
        assert s.get("basis") in ("datasheet", "derived", "orc-derived"), (
            f"{name}.basis = {s.get('basis')!r}")
        assert s.get("peak_hbm_basis") in ("datasheet", "measured", "orc-derived"), (
            f"{name}.peak_hbm_basis = {s.get('peak_hbm_basis')!r}")
        if s["peak_hbm_basis"] == "measured":
            # a measured ceiling stored as THE ceiling must say what the pin rate is and which
            # tool measured it -- otherwise it reads as a datasheet number one probe later
            assert s.get("datasheet_hbm_tb_s"), f"{name}: measured HBM row without datasheet_hbm_tb_s"
            assert any(m.get("metric") == "hbm_tb_s" and m.get("tool")
                       for m in s.get("measured_ceilings") or []), (
                f"{name}: measured HBM row names no tool in measured_ceilings")
        # bf16 and fp16 run at the same matrix-core rate on every part tabulated here
        if "bf16" in s["peak_tflops"]:
            assert s["peak_tflops"]["bf16"] == s["peak_tflops"]["fp16"], f"{name}: bf16 != fp16"
        # A DERIVED QUANTITY MAY NOT BE STORED. ridge = peak_tflops[dtype]/peak_hbm, so a stored
        # ridge is a cache of one division taken at ONE dtype -- and it was read back for every
        # dtype, which is a 16x error between fp8 and fp32 on the same SKU. Computed at call time
        # now (see `ridge` in budget()); this assert stops it being re-added as data.
        assert "ridge_ops_per_byte" not in s, (
            f"{name} stores a derived ridge. It is peak_tflops[dtype]/peak_hbm_tb_s and is only "
            f"valid for the dtype it was taken at; compute it, do not cache it.")
    # Within one arch every SKU is the same CU design -- only the CU count and the clock differ,
    # and both cancel in a rate RATIO. So a dtype ratio that varies across an arch means a cell was
    # mis-transcribed, whichever cell it is. (Tolerance covers datasheet figures published rounded
    # to 3-4 significant digits; it does NOT cover a cell that is uniformly wrong across the whole
    # arch, which only a cross-check against the published spec catches.)
    _by_arch = {}
    for name, s in _skus.items():
        _by_arch.setdefault(s["arch"], []).append((name, s))
    for arch, members in _by_arch.items():
        for dt in sorted({d for _, s in members for d in s["peak_tflops"]} - {"fp16"}):
            ratios = [(n, s["peak_tflops"][dt] / s["peak_tflops"]["fp16"])
                      for n, s in members if dt in s["peak_tflops"]]
            if len(ratios) < 2:
                continue
            lo, hi = min(r for _, r in ratios), max(r for _, r in ratios)
            assert (hi - lo) / hi < 0.02, (
                f"{arch}: {dt}:fp16 ratio varies across SKUs of one arch -- {ratios}. Same CU "
                f"design, so this ratio is fixed; one of these cells is mis-transcribed.")

    # A FIXTURE ROW for the missing-dtype paths. The real gfx950 rows now list fp4 / int8 / fp64,
    # so the refusal, ridge-lower-bound and wider-dtype branches below are exercised on a copy of
    # MI355X restricted to the dtypes it used to list (fp16/bf16/fp8/fp32). It exists only inside
    # this selftest (the module-level _load is restored on exit) and is never written anywhere.
    _real_load = _load

    def _fixture_load(rel):
        d = _real_load(rel)
        if rel == "sku.json":
            row = json.loads(json.dumps(d["skus"]["MI355X"]))
            row["peak_tflops"] = {k: v for k, v in row["peak_tflops"].items()
                                  if k in ("fp16", "bf16", "fp8", "fp32")}
            row["roofline_default_for_arch"] = False
            d["skus"][_FIX] = row
        return d
    _load = _fixture_load

    _defaults = {}
    for name, s in _skus.items():
        if s.get("roofline_default_for_arch"):
            assert s.get("geak_support") == "supported", f"{name}: an unsupported row is an arch default"
            assert s["arch"] not in _defaults, f"{s['arch']}: two arch defaults ({_defaults.get(s['arch'])}, {name})"
            _defaults[s["arch"]] = name
    assert _defaults.get("gfx950") == "MI355X" and _defaults.get("gfx942") == "MI300X", _defaults
    assert "gfx1201" not in _defaults, "R9700 peaks are product-scoped; gfx1201 must have no arch default"

    b = budget("MI300X", "attn_bwd", {"b": 1, "h": 16, "d": 128, "s": 4096}, "bf16", measured_ms=5.0)  # synthetic
    w = b["workload"]
    assert w["model"] == "attention_bwd" and w["multi_reduction"], w
    assert w["flops"] > 0 and w["hbm_bytes_min"] > 0 and w["binding_floor_ms"][0] > 0, w
    assert w["multiple_over_floor"][0] > 1, w

    # The shape variable `b` means BATCH; `b` inside a model expression means DTYPE BYTES. That
    # collision once made every attention byte model a quadratic in batch and silenced --dtype, and
    # `hbm_bytes_min > 0` above walked straight past it. Two assertions pin the two halves:
    # (1) bytes scale with the dtype at a FIXED shape, and (2) `b=` and `batch=` are the same shape.
    _shape = {"b": 2, "h": 32, "d": 128, "s": 4096}
    _by_dtype = {dt: budget("MI350X", "attention", dict(_shape), dt)["workload"]["hbm_bytes_min"]
                 for dt in ("fp8", "bf16", "fp32")}
    assert _by_dtype["bf16"] == 2 * _by_dtype["fp8"] and _by_dtype["fp32"] == 4 * _by_dtype["fp8"], (
        f"attention bytes must scale with dtype width, got {_by_dtype} -- a shape variable is "
        f"shadowing the dtype-bytes variable `{_DTYPE_BYTES_VAR}` in _build_env")
    _spelled_batch = budget("MI350X", "attention", {"batch": 2, "h": 32, "d": 128, "s": 4096},
                            "bf16")["workload"]["hbm_bytes_min"]
    assert _spelled_batch == _by_dtype["bf16"], (
        f"`b=2` and `batch=2` must be the same shape, got {_by_dtype['bf16']} vs {_spelled_batch}")
    # and the absolute value: 2 tensors (K,V) x BH x S x D x dtype_bytes, linear in batch.
    assert _by_dtype["bf16"] == 2 * (2 * (2 * 32) * 4096 * 128), _by_dtype
    g = budget("MI300X", "gemm", {"M": 4096, "N": 4096, "K": 8192}, "bf16")
    assert g["workload"]["flops"] == 2 * 4096 * 4096 * 8192, g["workload"]["flops"]

    # dtype table: a 4-bit weight is 0.5 B/element, not the 2 B fallback (a 4x byte error).
    assert _dtype_bytes("fp4") == 0.5 and _dtype_bytes("mxfp4") == 0.5, "fp4 must be 0.5 B"
    assert _dtype_bytes("fp8") == 1 and _dtype_bytes("bf16") == 2
    assert _dtype_bytes("no_such_dtype") == 2                      # documented fallback, warns
    # mx block-32 scale bytes: 1/(32*b) of payload -> 6.25% at fp4, 3.125% at fp8, 0 for bf16.
    assert abs(_scale_overhead("fp4", 0.5) - 0.0625) < 1e-9
    assert abs(_scale_overhead("mxfp8", 1) - 0.03125) < 1e-9
    assert _scale_overhead("bf16", 2) == 0.0

    # MoE at decode M: the byte model must charge the ROUTED experts, not all E. Reference point is
    # a measured fused-MoE gemm1 (mxfp4 w1, E=896, top_k=16, I=384, H=3584): 246.35 MB of DRAM at
    # 53.52 us. Charging all E gave a 0.308 ms "floor" -- 5.8x SLOWER than the measured kernel.
    moe_shapes = {"M": 32, "k": 16, "I": 384, "H": 3584, "E": 896, "E_touched": 165, "stage": 1}
    m = budget(_FIX, "moe", moe_shapes, "fp4", measured_ms=0.05352)["workload"]
    err = abs(m["hbm_bytes_min"] - 246.35e6) / 246.35e6
    assert err < 0.10, f"moe byte model {m['hbm_bytes_min']:.3e} is {err:.0%} off the measured 246.35 MB"
    assert m["multiple_over_floor"][0] > 1 and not m.get("floor_breached"), m
    # the MoE caveat names the DEFAULTED expert count, so it must change once one is measured --
    # otherwise it warns about a decision the caller did not make, and gets skimmed past when it
    # DOES apply
    assert "default E_touched" not in (m["bound_direction"] or ""), m["bound_direction"]
    assert "measured E_touched" in m["bound_direction"], m["bound_direction"]
    # ... while the half of it that still stands (activation re-reads) is kept, not dropped
    assert "under-counts" in m["bound_direction"], m["bound_direction"]

    # The DEFAULT stage is the one the flops term describes (w1), so a per-kernel budget is right
    # without an extra flag -- the reference command in the review passed no `stage` at all.
    dflt = budget(_FIX, "moe", {k: v for k, v in moe_shapes.items() if k != "stage"},
                  "fp4", measured_ms=0.05352)["workload"]
    assert dflt["hbm_bytes_min"] == m["hbm_bytes_min"] and dflt["flops"] == m["flops"], dflt
    # ... and `stage` scales BOTH terms together, so every stage describes ONE real kernel. w2 is
    # half of w1 (I*H vs 2*I*H per expert); a fused both-stages kernel is 1.5x w1 on each term.
    s2 = budget(_FIX, "moe", dict(moe_shapes, stage=2), "fp4")["workload"]
    s0 = budget(_FIX, "moe", dict(moe_shapes, stage=0), "fp4")["workload"]
    assert abs(s2["flops"] / m["flops"] - 0.5) < 1e-9, (s2["flops"], m["flops"])
    assert abs(s0["flops"] / m["flops"] - 1.5) < 1e-9
    for key in ("hbm_bytes_min", "flops"):
        assert s2[key] < m[key] < s0[key], (key, s2[key], m[key], s0[key])
    # the AI is then a property of one kernel, not a blend of two
    ai0, ai1 = s0["arithmetic_intensity"], m["arithmetic_intensity"]
    assert abs(ai0 - ai1) / ai1 < 0.02, (ai0, ai1)
    # ... and the same shape with the default E_touched (min(E, M*k) = 512) over-counts enough to
    # breach the floor, which must FAIL LOUDLY rather than print a sub-1x "prize".
    over = budget(_FIX, "moe", {k: v for k, v in moe_shapes.items() if k != "E_touched"},
                  "fp4", measured_ms=0.05352)["workload"]
    assert over.get("floor_breached") and "_error" in over, over
    assert "default E_touched" in over["bound_direction"], over["bound_direction"]
    # a wrong dtype (2 B instead of 0.5 B) is the other way to breach it
    wide = budget("MI355X", "moe", moe_shapes, "bf16", measured_ms=0.05352)["workload"]
    assert wide.get("floor_breached"), wide

    # calibration: the ceiling replaces the datasheet peak, carries its error bar into every
    # derived number, and the provenance is never lost.
    cal = budget(_FIX, "moe", moe_shapes, "fp4", measured_ms=0.05352,
                 measured_hbm_tb_s=[5.31, 5.72], measured_bytes_mb=246.35)["workload"]
    assert cal["hbm_ceiling_source"] == "calibrated" and cal["bytes_source"] == "measured", cal
    # with measured bytes the byte model is out of the path entirely, so its caveat must retire
    assert "not in the path" in cal["bound_direction"], cal["bound_direction"]
    assert cal["calibrated"] and cal["hbm_floor_ms"][0] < cal["hbm_floor_ms"][1], cal
    lo, hi = cal["pct_of_hbm_ceiling"]
    assert 75 < lo < hi < 95, cal          # ~80-87% of the in-shape ceiling, not 58% of datasheet
    # ...and with NO probed ceiling the two ceiling-relative numbers are refused outright. A
    # datasheet-peak %-of-ceiling would read a kernel that is already AT its reachable ceiling as
    # having tens of percent of headroom, so printing it with a caveat still hands over a prize
    # that does not exist. The floor survives (it is a real impossibility bound); the gap does not.
    dat = budget(_FIX, "moe", moe_shapes, "fp4", measured_ms=0.05352,
                 measured_bytes_mb=246.35)["workload"]
    assert dat["pct_of_hbm_ceiling"] is None and dat["hbm_gap_ms"] is None, dat
    assert "mem_bw_probe" in dat["ceiling_refused"], dat["ceiling_refused"]
    assert dat["hbm_floor_ms"] and dat["achieved_hbm_tb_s"], "the floor and achieved rate survive"
    assert not dat["calibrated"], dat
    # the refusal is about the CEILING only: a probed ceiling with modelled bytes still reports,
    # because then the denominator is real and only the numerator carries a caveat.
    probed_only = budget(_FIX, "moe", moe_shapes, "fp4", measured_ms=0.05352,
                         measured_hbm_tb_s=[5.31, 5.72])["workload"]
    assert probed_only["pct_of_hbm_ceiling"] and "ceiling_refused" not in probed_only, probed_only

    # THE COMPUTE CEILING GETS THE SAME TREATMENT, and the case that made it necessary is a
    # compute-bound kernel: the memory refusal fires loudly on the axis that is NOT binding, while
    # the multiple over the compute floor -- the number the round actually gets sized on -- rested
    # on an unlabelled datasheet MFMA peak. So: the caveat must land on the BINDING axis, and the
    # datasheet reading must be framed as an upper bound rather than as the prize.
    big = {"M": 8192, "N": 8192, "K": 8192}
    dsc = budget("MI325X", "gemm", big, "bf16", measured_ms=2.0)["workload"]
    assert dsc["binding_floor"] == "compute" and dsc["compute_ceiling_source"] == "datasheet", dsc
    assert dsc["binding_floor_source"] == "datasheet" and not dsc["calibrated"], dsc
    assert "UPPER BOUND" in dsc["prize_is_upper_bound"], dsc
    # the instrument named must be the compute one; sending a compute-bound kernel to mem_bw_probe
    # is what the un-scoped version did
    assert any("--measured-tflops" in t for t in dsc["uncalibrated_terms"]), dsc["uncalibrated_terms"]
    assert not any("mem_bw_probe" in t for t in dsc["uncalibrated_terms"]), dsc["uncalibrated_terms"]
    # probing it moves the prize DOWN (a datasheet floor is too low, so it flatters the opportunity)
    calc = budget("MI325X", "gemm", big, "bf16", measured_ms=2.0,
                  measured_tflops=[980, 1120])["workload"]
    assert calc["compute_ceiling_source"] == "calibrated" and calc["calibrated"], calc
    assert "prize_is_upper_bound" not in calc, calc
    assert calc["mfma_floor_ms"][0] < calc["mfma_floor_ms"][1], calc
    assert calc["multiple_over_floor"][1] < dsc["multiple_over_floor"][0], (
        "a probed compute ceiling must shrink the apparent prize, never grow it")
    # a MEMORY-bound kernel must NOT be told to go probe the MFMA peak: scoping cuts both ways, and
    # a warning that fires off-axis is how the on-axis one stops being read.
    assert not any("--measured-tflops" in t for t in dat["uncalibrated_terms"]), dat["uncalibrated_terms"]
    assert dat["binding_floor_source"] == "datasheet", dat

    # A COMPUTE CEILING IS PER-DTYPE AND MUST NOT BE SUBSTITUTED. The lookup used to fall back to
    # the bf16 rate for any dtype the table omits -- which is most of the interesting ones (CDNA4
    # fp4/fp6, int8/int4 on Instinct rows, fp8 on RDNA3) and every typo. Silent, and wrong in the
    # DANGEROUS direction: too low a peak lifts the compute floor, which can flip the binding onto
    # an engine that was never the constraint.
    tbl = _load("sku.json")["skus"][_FIX]["peak_tflops"]
    assert "fp4" not in tbl, f"test premise: {_FIX} carries no fp4 rate"
    # ...while the REAL MI355X row now carries its datasheet fp4 / int8 / fp64 rates, and those
    # must be used as datasheet ceilings, not reported unavailable
    for listed in ("fp4", "int8", "fp64"):
        assert budget("MI355X", "gemm", big, listed, measured_ms=1.0)["workload"][
            "compute_ceiling_source"] == "datasheet", listed
    for absent in ("fp4", "int8", "definitely_not_a_dtype"):
        u = budget(_FIX, "gemm", big, absent, measured_ms=1.0)["workload"]
        assert u["compute_ceiling_source"] == "unavailable", (absent, u["compute_ceiling_source"])
        assert u["mfma_floor_ms"] is None, (absent, u["mfma_floor_ms"])
        # the substituted rate was bf16's, so the giveaway is a floor equal to the bf16 one
        assert u["mfma_only_floor_ms"] != dsc["mfma_only_floor_ms"], (absent, "bf16 rate substituted")
        assert absent in u["compute_floor_unknown"] and "--measured-tflops" in u["compute_floor_unknown"], u
    # and the dtypes it DOES carry are unaffected
    assert budget("MI355X", "gemm", big, "fp8", measured_ms=1.0)["workload"][
        "compute_ceiling_source"] == "datasheet"

    # A FLOP COUNT IS NOT A COMPUTE FLOOR; the engine it issues on is what makes it one. These
    # archetypes are flops = c*n -- no matrix multiply anywhere -- so the matrix rate priced them one
    # to two orders too low, the compute floor never bound, and the tool reported "memory" by
    # DEFAULT rather than by measurement.
    for wl, shp in (("elementwise", {"n": 1e8, "c": 200}), ("norm", {"M": 8192, "N": 8192, "c": 8}),
                    ("reduction", {"n": 1e9, "c": 4}), ("scan", {"n": 1e9, "c": 4})):
        v = budget("MI355X", wl, shp, "bf16", measured_ms=0.3)["workload"]
        assert v["flops_engine"] == "valu", (wl, v["flops_engine"])
        assert v["mfma_floor_ms"] is None, (wl, "VALU FLOPs priced at the matrix rate")
        assert "VALU" in v["compute_floor_unknown"], (wl, v["compute_floor_unknown"])
    assert budget("MI355X", "gemm", big, "bf16")["workload"]["flops_engine"] == "mfma"

    # THE CASE THAT MAKES IT MATTER: VALU-heavy, bytes modest. The old tool said "memory binds, you
    # are 6x over the floor" -- a byte-removal round that could collect nothing, because pricing the
    # arithmetic at a VALU rate puts the compute floor ABOVE the memory floor.
    hot = {"n": 1e8, "c": 200}
    prov = budget("MI355X", "elementwise", hot, "bf16", measured_ms=0.3)["workload"]
    assert "NOT confirmed" in prov["binding_floor"], prov["binding_floor"]
    assert not prov["calibrated"], "a verdict resting on an unevaluated term is not calibrated"
    assert any("NOT COMPUTED" in t for t in prov["uncalibrated_terms"]), prov["uncalibrated_terms"]
    priced = budget("MI355X", "elementwise", hot, "bf16", measured_ms=0.3,
                    measured_tflops=[157])["workload"]
    assert priced["mfma_floor_ms"][0] > priced["hbm_floor_ms"][1], priced
    assert priced["binding_floor"] == "compute", priced["binding_floor"]

    # ...and the intensity shortcut that suppresses this escalation must be gated on the ENGINE. The
    # ridge is peak_MFMA/peak_HBM, so it settles the question only for MFMA FLOPs: the VALU peak is a
    # fraction of the matrix peak, so the real VALU ridge is roughly an order LOWER and a kernel
    # comfortably below the MFMA ridge (AI 50 vs ridge 312) can sit on the compute side of it.
    assert prov["arithmetic_intensity"] <= 0.5 * prov["ridge_ops_per_byte"], (
        "test premise: this kernel IS far below the MFMA ridge, so only the engine gate saves it")
    # the missing-DTYPE case is the direction a LOWER BOUND on the ridge can settle, so it stays
    # quiet: fp4 is narrower than every dtype MI355X lists, and the matrix pipeline does not slow
    # down as the dtype narrows, so its true crossover is at or above the fp8 one.
    assert "confirmed by intensity" in cal["binding_floor"], cal["binding_floor"]
    assert cal["compute_ceiling_source"] == "unavailable" and cal["calibrated"], cal
    # ...and the reported ridge must NOT have been filled in from that bound. A bound is enough to
    # answer "far below?"; it is not a crossover, and printing it as one is how a substituted
    # number becomes a quoted fact.
    assert cal["ridge_ops_per_byte"] is None and cal["regime"].startswith("unknown"), cal
    # the OTHER direction gets no bound and the gate must stay shut: fp64 is wider than everything
    # MI355X lists, so nothing there floors its crossover.
    wide_dt = budget(_FIX, "moe", moe_shapes, "fp64", measured_ms=0.05352,
                     measured_hbm_tb_s=[5.31, 5.72], measured_bytes_mb=246.35)["workload"]
    assert "NOT confirmed" in wide_dt["binding_floor"], wide_dt["binding_floor"]

    # THE RIDGE IS A PER-DTYPE CROSSOVER, and the whole point of computing it instead of storing it
    # is that the SAME SKU and the SAME shape must land on different sides for different dtypes.
    # sku.json shipped one fp16-derived ridge that every dtype read back, so this never held.
    _r = {dt: budget("MI325X", "gemm", {"M": 4096, "N": 4096, "K": 4096}, dt)["workload"]
          for dt in ("fp16", "fp8", "fp32")}
    assert abs(_r["fp8"]["ridge_ops_per_byte"] / _r["fp16"]["ridge_ops_per_byte"] - 2.0) < 0.01, _r
    assert abs(_r["fp32"]["ridge_ops_per_byte"] / _r["fp16"]["ridge_ops_per_byte"] - 0.125) < 0.01, _r
    for dt, v in _r.items():
        assert v["ridge_dtype"] == dt, (dt, v["ridge_dtype"])
    # ...and the band between the fp32 and fp16 crossovers is not hypothetical: it is where every
    # fp32 kernel of moderate intensity lives, and the stored fp16 ridge called all of them memory.
    lo_r, hi_r = _r["fp32"]["ridge_ops_per_byte"], _r["fp16"]["ridge_ops_per_byte"]
    assert lo_r < 30 < 200 < hi_r, (lo_r, hi_r)
    # a square fp32 gemm has AI = M/6, so M=768 lands at 128 ops/byte -- between the two crossovers
    band = budget("MI325X", tensors="A:r:fp32:768x768, B:r:fp32:768x768, C:w:fp32:768x768",
                  flops=2 * 768 ** 3, dtype="fp32")["workload"]
    assert lo_r < band["arithmetic_intensity"] < hi_r, band["arithmetic_intensity"]
    assert band["regime"].startswith("compute"), (
        f"AI {band['arithmetic_intensity']} is above the fp32 crossover ({lo_r:.0f}) and this "
        f"kernel IS compute-bound; the stored fp16 ridge ({hi_r:.0f}) reported it as memory")

    # BOTH floors gone, but for which reason? Cache residency voids the memory floor and an unpriced
    # compute term leaves nothing to fall back on -- yet this is NOT the 0-FLOP case, and saying "no
    # analytic floor exists" there would send the reader off the roofline entirely when in fact there
    # is a number to go get. The two states carry opposite instructions, so they must not share one.
    gone = budget(_FIX, "gemm", {"M": 2048, "N": 2048, "K": 2048}, "fp4",
                  footprint_mb=32, grid=4 * 256, measured_ms=0.5)["workload"]
    assert gone["binding_floor"].startswith("UNKNOWN"), gone["binding_floor"]
    assert gone["binding_floor_ms"] is None, gone
    why = " ".join(gone["roofline_preconditions_refused"])
    assert "never computable" in why and "--measured-tflops" in why, why
    assert "no useful FLOPs" not in why, "this kernel HAS arithmetic; only its rate is missing"

    # DIVERGING COUNTER ROUTES: the numerator is an interval for the same reason the denominator is.
    # Feeding several routes must widen every derived number instead of silently adopting one -- the
    # width IS the finding, because a 4x spread in bytes is a 4x spread in the size of the prize.
    div = budget(_FIX, "moe", moe_shapes, "fp4", measured_ms=0.05352,
                 measured_hbm_tb_s=[5.31, 5.72], measured_bytes_mb=[120.0, 240.0])["workload"]
    assert div["bytes_source"] == "measured (routes disagree)", div["bytes_source"]
    assert div["hbm_bytes_range"] == [120e6, 240e6], div["hbm_bytes_range"]
    # the model's caveat retires, but it must NOT be replaced by "the ceiling is the only term left"
    assert "the measurement is" in div["bound_direction"], div["bound_direction"]
    assert "only calibratable term" not in div["bound_direction"], div["bound_direction"]
    glo, ghi = div["hbm_gap_ms"]
    plo, phi = div["pct_of_hbm_ceiling"]
    assert glo < ghi and plo < phi, div         # both must widen, not collapse to a point
    # a 2x byte spread must not be reported as a tighter answer than a single route
    single = budget(_FIX, "moe", moe_shapes, "fp4", measured_ms=0.05352,
                    measured_hbm_tb_s=[5.31, 5.72], measured_bytes_mb=[240.0])["workload"]
    assert (phi - plo) > (single["pct_of_hbm_ceiling"][1] - single["pct_of_hbm_ceiling"][0]), \
        "diverging routes must produce a WIDER %-of-ceiling than any single route"
    assert single["bytes_source"] == "measured", single["bytes_source"]
    # ...and when the byte interval straddles the ridge, no regime may be named at all
    strad = budget("MI355X", "gemm", {"M": 4096, "N": 4096, "K": 4096}, "bf16",
                   measured_bytes_mb=[40.0, 40000.0])["workload"]
    assert strad["regime"].startswith("ambiguous"), strad["regime"]

    # ---- the byte model as a TENSOR MANIFEST -------------------------------------------------
    # A formula keyed on a workload NAME cannot express a real kernel. Each assertion below is one
    # way that fails, and the manifest has to express it as ordinary bookkeeping.
    # (a) per-tensor dtypes. A fp8-operand / wide-output kernel is not describable by one `b`, and
    #     the output is often the LARGEST term -- so the single-dtype count is wrong by that ratio.
    mix = budget("MI325X", tensors="A:r:fp8:64x64, C:w:fp32:64x64", flops=0)["workload"]
    a_bytes, c_bytes = 64 * 64 * 1, 64 * 64 * 4
    assert mix["hbm_bytes_range"] == [a_bytes + c_bytes] * 2, mix["hbm_bytes_range"]
    one_dtype = budget("MI325X", tensors="A:r:fp8:64x64, C:w:fp8:64x64", flops=0)["workload"]
    assert one_dtype["hbm_bytes_range"][0] < mix["hbm_bytes_range"][0], \
        "charging the output at the operand dtype must under-count"
    # (b) re-traversal is PER TENSOR: a tiled loop re-reads one operand per tile of the other, so a
    #     single global multiplier cannot express it. Only the issued end moves; the footprint is
    #     the same bytes either way, which is exactly why the two ends bracket the truth.
    trav = budget("MI325X", tensors="A:r:fp8:64x64, B:r:fp8:64x64:x8", flops=0)["workload"]
    assert trav["hbm_bytes_range"][0] == 2 * a_bytes, trav
    assert trav["hbm_bytes_range"][1] == a_bytes + 8 * a_bytes, trav
    assert trav["bytes_source"] == "model (bracket)", trav["bytes_source"]
    # (c) a dead grid dimension duplicates every program: issued traffic scales, footprint does not
    red = budget("MI325X", tensors="A:r:fp8:64x64", flops=0, grid_redundancy=4)["workload"]
    assert red["hbm_bytes_range"] == [a_bytes, 4 * a_bytes], red["hbm_bytes_range"]
    # (d) a rw/atomic tensor moves the line twice -- the round trip a split-K reduction really pays
    assert budget("MI325X", tensors="P:rw:fp32:1000", flops=0)["workload"]["hbm_bytes_range"][0] \
        == 2 * 1000 * 4
    # (e) bytes without FLOPs is legal (gather/scatter/quantize), and a 0 compute floor is a real
    #     answer: it says the compute term cannot bind. It must not read as "not measured".
    zf = budget("MI325X", tensors="idx:r:int32:1000", flops=0)["workload"]
    assert zf["mfma_only_floor_ms"] == 0.0 and zf["binding_floor"] == "memory", zf
    # ...but omitting the FLOPs entirely is NOT the same as declaring 0, and must not silently pass
    try:
        budget("MI325X", tensors="A:r:fp8:64")
        raise AssertionError("a manifest with no --flops must not be accepted")
    except SystemExit:
        pass
    for bad in ("A:r:fp8", "A:zzz:fp8:64", "A:r:fp8:64:x0.5", ""):
        try:
            _parse_tensors(bad)
            raise AssertionError(f"_parse_tensors accepted {bad!r}")
        except ValueError:
            pass

    # ---- ROOFLINE PRECONDITIONS: is a roofline the right model at all? -----------------------
    cus = _load("sku.json")["skus"]["MI325X"]["cus"]
    # (f) fewer workgroups than CUs: neither term applies, so BOTH floors go and the multiple over
    #     them is not a prize. Refusing here is the point -- the gap is real but it is not bytes.
    thin = budget("MI325X", tensors="A:r:fp8:1e6", flops=0, grid=cus // 4,
                  measured_ms=1.0, measured_hbm_tb_s=[3.5, 3.8])["workload"]
    assert thin["binding_floor"].startswith("REFUSED"), thin["binding_floor"]
    assert thin["pct_of_hbm_ceiling"] is None and thin["hbm_gap_ms"] is None, thin
    assert thin["machine_fill"]["machine_fill_pct"] == 25.0, thin["machine_fill"]
    # (g) exactly one workgroup per CU fills the machine but cannot saturate memory (too few
    #     requests in flight), so it WARNS rather than refuses -- the ceiling to compare against is
    #     the one probed at this program count, not the high-parallelism one.
    thin1 = budget("MI325X", tensors="A:r:fp8:1e6", flops=0, grid=cus)["workload"]
    assert not thin1["roofline_preconditions_refused"], thin1
    assert "requests in flight" in thin1["machine_fill"]["_warning"], thin1["machine_fill"]
    assert not budget("MI325X", tensors="A:r:fp8:1e6", flops=0,
                      grid=4 * cus)["workload"]["machine_fill"].get("_warning"), "a filled machine is quiet"
    # (h) a working set inside the memory-side LLC voids the MEMORY floor only. The compute floor
    #     survives and becomes binding -- refusing it too would throw away the usable answer.
    llc = _load("sku.json")["skus"]["MI325X"]["mall_mb"]
    res = budget("MI325X", "gemm", {"M": 2048, "N": 2048, "K": 2048}, "bf16",
                 footprint_mb=llc / 4, grid=4 * cus, measured_ms=1.0,
                 measured_hbm_tb_s=[3.5, 3.8])["workload"]
    assert res["binding_floor"].startswith("compute (memory floor VOID"), res["binding_floor"]
    assert res["binding_floor_ms"][0] == res["mfma_only_floor_ms"], res
    assert res["pct_of_hbm_ceiling"] is None, "a %-of-HBM-ceiling is a category error in cache"
    # ...and clearing the LLC restores the ordinary two-floor comparison
    ok = budget("MI325X", "gemm", {"M": 2048, "N": 2048, "K": 2048}, "bf16",
                footprint_mb=4 * llc, grid=4 * cus, measured_ms=1.0,
                measured_hbm_tb_s=[3.5, 3.8])["workload"]
    assert not ok["roofline_preconditions_refused"] and ok["pct_of_hbm_ceiling"], ok
    # ...and that run supplied grid + footprint, so only the dispatch precondition is unchecked.
    assert [p.split(" (")[0] for p in ok["roofline_preconditions_unchecked"]] == ["dispatch count"], \
        ok["roofline_preconditions_unchecked"]

    # (h2) AN UNCHECKED PRECONDITION IS NOT A PASSED ONE (R9). Supplying none of the three leaves
    #      `refused` empty, which is what a clean roofline looks like -- so the two readings have to
    #      be separable in the artifact. Measured: on one run all three numbers existed in a
    #      structure_census.json and none reached here.
    blind = budget("MI325X", "gemm", {"M": 2048, "N": 2048, "K": 2048}, "bf16",
                   measured_ms=1.0)["workload"]
    assert blind["roofline_preconditions_refused"] is None, "nothing fired..."
    assert len(blind["roofline_preconditions_unchecked"]) == 3, "...because nothing ran"
    assert [p.split(" (")[0] for p in blind["roofline_preconditions_unchecked"]] == [
        "device fill", "cache residency", "dispatch count"], blind["roofline_preconditions_unchecked"]
    # All three supplied -> the field is absent, and only then does silence mean "checked".
    seen = budget("MI325X", "gemm", {"M": 2048, "N": 2048, "K": 2048}, "bf16",
                  footprint_mb=4 * llc, grid=4 * cus, dispatches=1, measured_ms=1.0)["workload"]
    assert seen["roofline_preconditions_unchecked"] is None, seen["roofline_preconditions_unchecked"]
    # (j) physics narrows the bracket, and the reported fractions stay physical. A byte-count end
    #     implying a rate above the reachable ceiling is impossible, so it is clamped -- and if it
    #     still saturates the ceiling, "already at the wall" must be stated as a live possibility
    #     rather than emitted as >100%-of-ceiling with a negative gap.
    wide = budget("MI325X", tensors="A:r:bf16:1e8, B:r:bf16:1e8:x1000", flops=0,
                  grid=4 * cus, measured_ms=1.0, measured_hbm_tb_s=[3.5, 3.8])["workload"]
    assert wide["bytes_clamped_by_physics"], "an impossible bracket end must be clamped"
    assert wide["hbm_bytes_range"][1] <= 3.8e12 * 1e-3 * 1.0001, wide["hbm_bytes_range"]
    assert wide["pct_of_hbm_ceiling"][1] <= 100.0, wide["pct_of_hbm_ceiling"]
    assert wide["hbm_gap_ms"][0] >= 0.0, wide["hbm_gap_ms"]
    assert "ALREADY AT the memory wall" in wide["at_ceiling_possible"], wide
    # a bracket that stays reachable is left alone, and claims nothing about being at the wall
    tight = budget("MI325X", tensors="A:r:bf16:1e6", flops=0, grid=4 * cus,
                   measured_ms=1.0, measured_hbm_tb_s=[3.5, 3.8])["workload"]
    assert not tight["bytes_clamped_by_physics"] and "at_ceiling_possible" not in tight, tight
    # (k) no FLOPs AND a cache-resident set = no analytic floor at all, and that is the answer
    none = budget("MI325X", tensors="idx:r:int32:1000", flops=0, grid=4 * cus,
                  footprint_mb=1.0, measured_ms=0.01)["workload"]
    assert none["binding_floor"].startswith("NONE"), none["binding_floor"]
    assert none["binding_floor_ms"] is None and "multiple_over_floor" not in none, none
    assert any("no analytic floor at all" in p for p in none["roofline_preconditions_refused"]), none
    # (l) clearing the LLC by a hair is not clearing it: warn in the marginal band, quietly pass above
    marg = budget("MI325X", "gemm", {"M": 4096, "N": 4096, "K": 4096}, "bf16",
                  footprint_mb=2 * llc, grid=4 * cus)["workload"]
    assert not marg["roofline_preconditions_refused"], "2x the LLC must not be refused"
    assert "cache/DRAM blend" in marg["cache_residency"]["_warning"], marg["cache_residency"]
    assert "_warning" not in budget("MI325X", "gemm", {"M": 4096, "N": 4096, "K": 4096}, "bf16",
                                    footprint_mb=8 * llc, grid=4 * cus)["workload"]["cache_residency"]
    # (l2) AN UNESTABLISHED LLC IS NOT A SMALL ONE. Eight of the thirteen SKUs carry mall_mb null,
    #      and this used to substitute the L2 -- 6 MB where RX7900XTX actually has 96 MB of
    #      Infinity Cache. The substitution reads as a PASS for every footprint above 4x the L2,
    #      which is the dangerous direction: it certifies as DRAM traffic a stream that may never
    #      have left cache. The bar may only be used where it is provable.
    unk_l2 = _load("sku.json")["skus"]["RX7900XTX"]["l2_mb"]
    assert _load("sku.json")["skus"]["RX7900XTX"]["mall_mb"] is None, "test premise"
    #   above the L2 -> unknown, and it must NOT read as a clean pass
    amb = budget("RX7900XTX", "gemm", {"M": 4096, "N": 4096, "K": 4096}, "bf16",
                 footprint_mb=20 * unk_l2, grid=4 * 96)["workload"]
    assert amb["cache_residency"]["llc_mb"] is None, amb["cache_residency"]
    assert "not established" in amb["cache_residency"]["_warning"], amb["cache_residency"]
    assert not amb["roofline_preconditions_refused"], "unknown is not a refusal"
    #   inside the L2 -> resident for certain: whatever the LLC is, it sits BEHIND the L2
    ins = budget("RX7900XTX", "gemm", {"M": 4096, "N": 4096, "K": 4096}, "bf16",
                 footprint_mb=unk_l2 / 2, grid=4 * 96, measured_ms=1.0)["workload"]
    assert ins["binding_floor"].startswith("compute (memory floor VOID"), ins["binding_floor"]
    assert "whatever this part's memory-side LLC" in ins["cache_residency"]["_refused"], ins
    #   and a sub-floor time on such a part must name the unknown bar as the leading suspect
    brk = budget("RX7900XTX", tensors="A:r:bf16:1e9", flops=0, grid=4 * 96,
                 footprint_mb=20 * unk_l2, measured_ms=0.001)["workload"]
    assert brk["floor_breached"] and "mall_mb null" in brk["_error"], brk.get("_error")
    # (i) a multi-dispatch timed region cannot be read against a per-kernel floor
    md = budget("MI325X", tensors="A:r:fp8:1e6", flops=0, dispatches=5)["workload"]
    assert "5 dispatches" in md["dispatch_warning"], md
    assert "dispatch_warning" not in budget("MI325X", tensors="A:r:fp8:1e6", flops=0,
                                            dispatches=1)["workload"]
    # a single-value ceiling still works and collapses the range
    one = budget(_FIX, "moe", moe_shapes, "fp4", measured_hbm_tb_s=[5.31])["workload"]
    assert one["hbm_floor_ms"][0] == one["hbm_floor_ms"][1], one
    assert _rng([1.0, 1.0], "x") == "1.0x" and _rng([1.0, 2.0], "x") == "1.0-2.0x"

    # safe-eval must reject code
    try:
        _safe_eval("__import__('os').system('echo x')", {})
        raise AssertionError("safe_eval did not reject import")
    except ValueError:
        pass
    # every model expression must still EVALUATE (a prose fragment inside an expression string is
    # a hard eval failure, not a wrong number -- it shipped once in a per-DSL copy of this file)
    doc = _load("workload_models.json")
    probe_env = _build_env({"M": 64, "N": 64, "K": 64, "b": 2, "h": 8, "d": 128, "s": 1024,
                            "n": 1 << 20, "E": 8, "k": 2, "I": 128, "H": 512, "t_exp": 8},
                           2, doc, 0.0)
    for name, mdl in doc["models"].items():
        for field in ("flops", "bytes_min", "bytes_hbm"):
            expr = mdl.get(field)
            if not expr:
                continue
            try:
                _safe_eval(expr, probe_env)
            except Exception as e:  # noqa: BLE001
                raise AssertionError(f"model {name}.{field} does not evaluate: {expr!r} ({e})")

    # ceiling provenance: a typed ceiling and a probed one both print [calibrated], so the state
    # that distinguishes them has to survive into the JSON a later reader picks up.
    sku0 = sorted(_load("sku.json")["skus"])[0]
    base = dict(workload="gemm", shapes={"M": 4096, "N": 4096, "K": 4096}, measured_ms=1.0)
    b_ds = budget(sku0, **base)["workload"]["ceiling_provenance"]
    assert b_ds == {"entry": "datasheet", "refs": []}, b_ds
    _w = budget("MI355X", **base)["workload"]
    assert (_w["numerator_basis"], _w["denominator_basis"], _w["may_gate"]) == (
        "model", "datasheet", False), _w
    _w = budget("MI355X", measured_hbm_tb_s=[5.3, 5.6], ceiling_source_ref=["probe.json"],
                measured_bytes_mb=[100.0], **base)["workload"]
    assert (_w["numerator_basis"], _w["denominator_basis"], _w["may_gate"]) == (
        "counters", "in-shape probe", True), _w
    _w = budget("AI_MAX_395", **base)["workload"]
    assert _w["denominator_basis"].startswith("empirical@rocm_bandwidth_test"), _w["denominator_basis"]
    b_hand = budget(sku0, measured_hbm_tb_s=[4.0, 4.2], **base)["workload"]["ceiling_provenance"]
    assert b_hand == {"entry": "hand_entered", "refs": []}, b_hand
    b_ref = budget(sku0, measured_hbm_tb_s=[4.0, 4.2],
                   ceiling_source_ref=["probe.json"], **base)["workload"]["ceiling_provenance"]
    assert b_ref == {"entry": "probe_artifact", "refs": ["probe.json"]}, b_ref
    # ...and the calibrated FLAG is unchanged by provenance: it answers "was a measurement
    # supplied", which is a different question and must not start depending on this one.
    assert (budget(sku0, measured_hbm_tb_s=[4.0], measured_bytes_mb=[100.0], **base)["workload"]
            ["calibrated"]
            is budget(sku0, measured_hbm_tb_s=[4.0], measured_bytes_mb=[100.0],
                      ceiling_source_ref=["p.json"], **base)["workload"]["calibrated"])

    # ---- THE VOCABULARY, AND THE PROSE PAGE IT IS SUPPOSED TO MATCH -------------------------
    # Three invariants that nothing checked until now. The third is the one that rots: the
    # archetype keys here and the `## <key>` sections of the reading-route page are two hand-kept
    # lists with no mechanical tie, so a key added on one side reaches a reader who then finds no
    # section, or a section is written that `--workload` will not accept.
    _keys = set(doc["models"]) | set(doc["reading_route"]["route_only"])
    for _a, _t in doc.get("aliases", {}).items():
        assert _t in _keys, f"alias {_a!r} points at {_t!r}, which is neither a model nor route-only"
        assert _a not in _keys, f"alias {_a!r} shadows a real archetype key"
    for _r in doc.get("refused", {}):
        # A refused name that is ALSO resolvable is the worst of both: the refusal never fires.
        assert _r not in doc.get("aliases", {}), f"{_r!r} is both refused and aliased"
        assert _r not in _keys, f"{_r!r} is both refused and a real archetype key"
    _page = doc["reading_route"]["page"]
    # The page is PACK-relative and the pack is found through _hwdata.pack_dir() (this tool lives
    # in kernel_workflow/scripts/kernel_tools, not inside the pack). Where the pack is reachable the
    # page MUST exist and the checks below MUST run: a skip there would be green for a check that
    # never ran, on the one invariant with no other guard. Only a tool copied out with no pack
    # reachable at all skips, and it says so.
    _root = _pack_root()
    _hit = os.path.join(_root, _page) if _root else None
    # A reachable pack that carries no workload index at all (its layout predates
    # references/workloads/) is the no-pack case: nothing to route to, so say so rather than fail.
    # A pack that HAS that directory but lost the page still fails the assertion below.
    _no_index = bool(_root) and not os.path.isdir(os.path.dirname(_hit))
    if _root and not _no_index:
        assert os.path.isfile(_hit), (
            f"workload_models.json reading_route.page {_page!r} does not resolve under the pack "
            f"root {_root} -- fix the page path (it is pack-relative) or restore the page")
        # Proof for the caller that this block RAN (a CI line can grep for it).
        _PAGE_CHECK_RAN = True
    elif _no_index:
        print(f"[hw_budget] note: the Gluon pack at {_root} has no {os.path.dirname(_page)}/ "
              f"(no workload index in this pack) -- reading-route page checks SKIPPED")
        _hit = None
        _PAGE_CHECK_RAN = False
    else:
        print(f"[hw_budget] note: Gluon pack not reachable ($GEAK_GLUON_PACK_DIR unset and no "
              f"repo checkout around {HERE}) -- reading-route page checks SKIPPED")
        _PAGE_CHECK_RAN = False
    if _hit:
        import re as _re
        _secs = set(_re.findall(r"^## (\S+)\s*$", open(_hit).read(), _re.M))
        assert _secs == _keys, (
            f"{_page} and this table disagree: sections with no archetype {sorted(_secs - _keys)}, "
            f"archetypes with no section {sorted(_keys - _secs)}")

        # The two shared prechecks on that page are the only place a reader is told to settle
        # roofline applicability, the per-dtype ridge, the counter convention and the oracle type
        # BEFORE a bound claim. They are prose, so nothing else in this repo notices if an edit
        # removes them -- and prose with no guard is exactly the class of content that went missing
        # last time. `_secs` above deliberately matches single-word `## <archetype>` headings only,
        # so these multi-word ones do not disturb it.
        # Match the WHOLE heading line, not a substring: `## Before any bound claim` is a prefix of
        # `## Before any bound claim RENAMED`, so a substring test would call a rename a pass.
        _heads = set(_re.findall(r"^#{2,3} (.+?)\s*$", open(_hit).read(), _re.M))
        _required = ("Before any bound claim", "Before any correctness-gated lever",
                     "When no row fits")
        _absent = [h for h in _required if h not in _heads]
        assert not _absent, (
            f"{_page} lost its shared precheck section(s): {_absent}. These gate the bound claim "
            f"itself; per-row text cannot replace them. Restore the heading(s) or, if the content "
            f"genuinely moved, update this assertion to name where it went.")

    if _PAGE_CHECK_RAN:
        print(f"[hw_budget] reading-route page checked: {_hit} ({len(_keys)} archetype sections + "
              f"{len(_required)} shared prechecks)")
    print("[hw_budget] SELFTEST PASS")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sku")
    ap.add_argument("--workload", help="gemm | attention | attn_bwd | reduction | norm | moe | ... (+aliases)")
    ap.add_argument("--shapes", help="k=v,k=v (e.g. b=1,h=32,d=128,s=8192 or M=4096,N=4096,K=8192)")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--measured-ms", type=float, help="your kernel's ms -> multiple over the floor")
    # --- the general byte model: declare what the kernel MOVES ---------------------------------
    ap.add_argument("--tensors", metavar="SPEC",
                    help="byte model as a manifest, REPLACING the named-workload formula: "
                         "`name:dir:dtype:dims[:xN]` comma-separated (dir=r|w|rw, dims=AxBxC or a "
                         "count, xN=traversals). Use this whenever the kernel does not match an "
                         "archetype -- mixed dtypes per tensor, arguments that are never "
                         "dereferenced, a workspace far larger than the part touched, an operand "
                         "re-read once per tile. Emits the honest bracket: unique footprint at the "
                         "low end, declared traversals at the high end. Needs --flops.")
    ap.add_argument("--flops", type=float,
                    help="useful FLOPs, for --tensors without a named workload. 0 is a real answer "
                         "for a gather/scatter/quantize kernel and means the compute floor cannot "
                         "bind.")
    ap.add_argument("--grid-redundancy", type=float, default=1.0, metavar="N",
                    help="issued traffic multiplier from duplicated programs -- a grid dimension "
                         "that is dead on the taken path runs the same work N times. Multiplies the "
                         "issued (high) end only; the footprint is unchanged.")
    # --- preconditions: is a roofline even the right model for this kernel? --------------------
    ap.add_argument("--grid", type=int, metavar="WGS",
                    help="number of workgroups launched. Fewer workgroups than CUs means the machine "
                         "is idle by construction and NEITHER roofline term applies.")
    ap.add_argument("--footprint-mb", type=float, metavar="MiB",
                    help="unique MiB the kernel touches (binary, to match how cache capacity is "
                         "quoted). Compared against the memory-side LLC: a working set that fits "
                         "inside it is served from cache in the steady state of a "
                         "repeated-iteration benchmark, so no HBM floor binds it. On a SKU whose "
                         "LLC capacity is not established there is no bar to compare against, and "
                         "the L2 is NOT a substitute for it.")
    ap.add_argument("--dispatches", type=int, metavar="N",
                    help="kernels inside the TIMED region. >1 means this per-kernel budget cannot be "
                         "compared against the region's total time.")
    # --- calibration: replace the datasheet/model terms with measurements ---------------------
    ap.add_argument("--measured-hbm-tb-s", metavar="TB_S|LO,HI",
                    help="in-shape HBM ceiling from a probe (mem_bw_probe.py), REPLACING the "
                         "datasheet peak. Pass `lo,hi` to carry the probe's error bar -- a probed "
                         "ceiling drifts 3-4%% between sessions and rises with parallelism, so the "
                         "honest form is a range and every floor/gap below is emitted as one.")
    ap.add_argument("--flops-engine", choices=("mfma", "valu"),
                    help="which engine the --flops issue on. Only needed with --tensors: a named "
                         "--workload declares it (workload_models.json). A FLOP count alone is not "
                         "a compute floor -- `flops / MFMA_peak` is one only if the FLOPs go through "
                         "MFMA, and pricing VALU work at the matrix rate understates the floor by "
                         "one to two orders, after which it never binds and the tool reports "
                         "'memory' by default. Unset with --flops > 0 means UNKNOWN, and the compute "
                         "floor is withheld rather than guessed.")
    ap.add_argument("--measured-tflops", metavar="TFLOPS|LO,HI",
                    help="in-shape MFMA ceiling, REPLACING the datasheet compute peak. The compute "
                         "term needs this for the same reason the memory term needs a probe: the "
                         "datasheet rate assumes back-to-back MFMA issue with operands already in "
                         "register, and a kernel that also loads, converts and addresses does not "
                         "sustain it. Time a dense back-to-back MFMA loop at THIS dtype (or take "
                         "the best rate a tuned dense GEMM reaches here) and pass `lo,hi`. Without "
                         "it a compute-bound kernel's multiple over the floor is only an upper "
                         "bound on its prize.")
    ap.add_argument("--measured-dram-mb", metavar="MB[,MB...]",
                    help="measured DRAM traffic in MB, REPLACING the analytic byte model. Pass "
                         "EVERY counter route you measured (parse_pmc.py memory block reports "
                         "TCC_MISS*128B, EA RDREQ/WRREQ*64B and FETCH_SIZE+WRITE_SIZE) rather than "
                         "picking one: the routes agree to a fraction of a percent on a pure "
                         "stream and can diverge several-fold on a real kernel, and that width is "
                         "how much of the prize is genuinely unknown. Which route is right is a "
                         "property of this kernel's access shape, not a global constant.")
    ap.add_argument("--measured-from", metavar="PATH[,PATH...]",
                    help="the probe artifact(s) the --measured-* ceilings were read out of "
                         "(mem_bw_probe.py / an MFMA-loop timing JSON). A remembered number and a "
                         "probed one are the same float here and both print as [calibrated], so "
                         "without this the label answers 'a measurement was supplied' and not 'by "
                         "what'. Each path must EXIST -- a provenance ref that points nowhere is "
                         "worse than none. Omitting it is allowed and records `hand_entered`.")
    ap.add_argument("--from-census", metavar="STRUCTURE_CENSUS_JSON", dest="from_census",
                    help="adopt --grid / --footprint-mb / --dispatches from a structure_census.json "
                         "`facts` block, which is where those three numbers are already recorded. "
                         "One flag instead of three looked-up values, because the three-value form "
                         "is measurably not passed: on one run the census had all of them and "
                         "computed roofline_applies=False, and every budget artifact on the same "
                         "tree was produced without any of them. An explicit flag still wins; a "
                         "flag that DISAGREES with the census is refused")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
        return
    if not a.sku:
        ap.error("--sku is required (or --selftest)")
    if a.from_census:
        try:
            facts = (json.load(open(a.from_census)).get("facts") or {})
        except (OSError, json.JSONDecodeError) as e:
            ap.error(f"--from-census {a.from_census!r}: {e}")
        if not facts:
            ap.error(f"--from-census {a.from_census!r} has an empty `facts` block -- run "
                     f"`structure_census.py fill --grid-wgs .. --cus ..` first. An empty census "
                     f"cannot supply a precondition, and adopting nothing silently would leave the "
                     f"roofline unchecked while looking like it had been sourced")
        for flag, census_key, cast in (("grid", "grid_wgs", int),
                                       ("footprint_mb", "footprint_mb", float),
                                       ("dispatches", "dispatches", int)):
            have = getattr(a, flag)
            want = facts.get(census_key)
            if want is None:
                continue
            want = cast(want)
            if have is None:
                setattr(a, flag, want)
            elif cast(have) != want:
                # Two numbers for one quantity is the defect, not the disagreement's resolution.
                ap.error(f"--{flag.replace('_', '-')}={have} disagrees with the census's "
                         f"{census_key}={want} ({a.from_census}). One of them is stale; fix the "
                         f"source rather than letting this tool pick")
    def _nums(raw, flag, cap):
        if not raw:
            return None
        try:
            v = [float(x) for x in raw.replace(" ", "").split(",") if x]
        except ValueError:
            v = []
        if not 1 <= len(v) <= cap:
            ap.error(f"{flag} wants {'a number or `lo,hi`' if cap == 2 else 'one or more numbers'}")
        return v

    bw = _nums(a.measured_hbm_tb_s, "--measured-hbm-tb-s", 2)
    mb = _nums(a.measured_dram_mb, "--measured-dram-mb", 8)
    tf = _nums(a.measured_tflops, "--measured-tflops", 2)
    refs = [p for p in (a.measured_from or "").split(",") if p.strip()]
    for p in refs:
        if not os.path.exists(p):
            ap.error(f"--measured-from {p!r} does not exist -- a provenance ref that points nowhere "
                     f"credits a hand-typed ceiling as a probed one, which is the exact confusion "
                     f"this flag exists to remove")
    if refs and not (bw or tf):
        ap.error("--measured-from names a probe artifact but no --measured-hbm-tb-s / "
                 "--measured-tflops was passed -- there is no measured ceiling for it to back")
    try:
        b = budget(a.sku, a.workload, _parse_shapes(a.shapes), a.dtype, a.measured_ms,
                   measured_hbm_tb_s=bw, measured_bytes_mb=mb, warn=True,
                   ceiling_source_ref=refs,
                   tensors=a.tensors, flops=a.flops, grid=a.grid,
                   footprint_mb=a.footprint_mb, dispatches=a.dispatches,
                   grid_redundancy=a.grid_redundancy, measured_tflops=tf,
                   flops_engine=a.flops_engine)
    except ValueError as e:
        ap.error(str(e))
    if a.json:
        print(json.dumps(b, indent=2))
    else:
        _print_human(b)
    # A breached floor exits non-zero so a driver script cannot carry the impossible prize forward
    # (the JSON is still printed, so a --json consumer keeps a readable, self-describing record).
    if (b.get("workload") or {}).get("floor_breached"):
        sys.exit(2)


if __name__ == "__main__":
    main()
