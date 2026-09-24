#!/usr/bin/env python3
"""Op-level RELATIVE numeric gate for the integrate seam (`accuracy_gate=op_tolerance`).

Why this exists
---------------
The integrate gate's default bar is e2e byte-exact greedy parity vs the true baseline. For a
QUANTIZED kernel that bar is already waived (`accuracy_gate=gsm8k`). It is ALSO the wrong bar for a
dtype-preserving KERNEL/BLAS SWAP: a vendor BLAS picks its solution per shape, so swapping tables
changes reduction ORDER, which flips borderline greedy argmaxes exactly the way rounding does.
Measured on gfx1151/vLLM/Qwen3-4B: a +14.38% TunableOp table diverged on 3/12 greedy prompts, every
divergence coherent and correct, against a baseline that was byte-exact against itself 12/12.

The tuning lane already gates that class of change, and it gates it on a RELATIVE OP-LEVEL error.
This script brings the integrate seam onto that same standard. The rules are not invented here; they
come from perf_knowledge/expert_skills/tuning/tuning-core/correctness_gates.md:

  * Relative, never absolute. bf16 `max_abs` grows ~8x over a 64x K sweep while the relative error
    stays flat — an absolute threshold is a function of K, not of correctness.
  * The reference must be HIGHER PRECISION than either leg, and INDEPENDENT of it. Here: the
    BASELINE seam (`--ref-target`, defaulting to the name the oracle was captured under) re-run on
    CPU in float32. CPU fp32 is chosen deliberately — it cannot be perturbed by the very knob under
    test (a tuned BLAS table, a GPU env var), so both legs are scored against the same fixed truth.
    "Same fixed truth" is VERIFIED, not assumed: each leg fingerprints its own fp32 reference and
    the compare step refuses to rule if the two disagree. Without that check a candidate that
    changes the function ITSELF would also move its own reference and score ~0 error — measured: a
    deliberate out*1.05 candidate scored 7.4e-3 and would have passed.
  * "Establish the baseline error first. That number is your floor; a candidate is suspect when it is
    materially worse, not when it is merely nonzero." Hence two legs and `--floor-mult`, not one leg
    and a constant.
  * A split-K kernel accumulates in a different order than a non-split-K one and WILL differ slightly
    — that is expected, not a failure.

This is a NUMERIC gate, not a TASK gate. It says the op still computes the same function to within
the precision its own baseline achieves; it does not say the model's answers stayed right. Where the
correctness of the ANSWER is what matters, stack `accuracy_gate=gsm8k` on top (correctness_gates.md
"For end-to-end serving changes, add a task-level check on top of the numeric one").

Usage
-----
Run MODE=leg once per leg, each in a process carrying that leg's env (so `ref` sees the untuned
path and `cand` sees the candidate's `apply_env`), then MODE=compare to get the verdict:

  # baseline leg -- no candidate env
  python3 op_parity_probe.py --oracle <task_dir> --leg ref  --out <dir>/op_parity_ref.json
  # candidate leg -- with the candidate's apply_env exported
  python3 op_parity_probe.py --oracle <task_dir> --leg cand --out <dir>/op_parity_cand.json
  # verdict
  python3 op_parity_probe.py --compare <dir>/op_parity_ref.json <dir>/op_parity_cand.json \\
      --tol 0.01 --floor-mult 2.0 --out <dir>/op_parity.json

The leg mode prints `OP_PARITY_ERR=<max relative error vs the fp32 reference>`.
The compare mode prints `OP_PARITY=<pass|fail>` as its final line (the contract the integrator reads,
mirroring scripts/gsm8k_eval.py's `GSM8K_EXACT_MATCH=`).

`--oracle` is a capture_shapes task dir (containing reference_io.pt + meta.json) or the
reference_io.pt itself. `--target` overrides the `module:attr` recorded in the blob.
"""
import argparse
import json
import math
import importlib
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness_lib as h  # noqa: E402  (correct/flatten_outputs/load_reference_io/... live here)


def _torch():
    import torch
    return torch


def resolve_callable(target):
    """`module:attr` -> the LIVE callable in THIS process, so this leg's env governs it."""
    if ":" not in target:
        raise ValueError(f"--target must be module:attr, got {target!r}")
    mod_name, attr = target.split(":", 1)
    obj = importlib.import_module(mod_name)
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


def load_cases(oracle, device):
    """Yield {sig, regime, args, kwargs, captured_out} with args/kwargs kept SEPARATE.

    harness_lib.iter_eager_cases_from_oracle exists but folds positional args into a MoE-specific
    kwargs dict; the seam here can be any callable, so the positional/keyword split is preserved.
    """
    blob = h.load_reference_io(oracle, map_location="cpu")
    shared = blob.get("shared") or {}
    target = blob.get("target") or ""
    cases = []
    for rec in blob.get("records") or []:
        args = h.reconstruct_captured(
            h.resolve_oracle_shared(rec.get("args") or (), shared), device=device)
        kwargs = h.reconstruct_captured(
            h.resolve_oracle_shared(rec.get("kwargs") or {}, shared), device=device)
        out = h.reconstruct_captured(
            h.resolve_oracle_shared(rec.get("output"), shared), device=device)
        cases.append({"sig": rec.get("sig", ""), "regime": rec.get("regime", ""),
                      "args": list(args), "kwargs": dict(kwargs), "captured_out": out})
    return target, cases


def _to_fp32_cpu(obj):
    """Deep-copy a call tree onto CPU with every floating tensor promoted to float32.

    Integer tensors (indices, ids) are moved but NOT promoted — upcasting them would change meaning.
    """
    torch = _torch()
    if torch.is_tensor(obj):
        if obj.is_floating_point():
            return obj.detach().to("cpu", dtype=torch.float32)
        return obj.detach().to("cpu")
    if isinstance(obj, dict):
        return {k: _to_fp32_cpu(v) for k, v in obj.items()}
    if isinstance(obj, tuple) and hasattr(obj, "_fields"):
        return type(obj)(*(_to_fp32_cpu(v) for v in obj))
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_fp32_cpu(v) for v in obj)
    return obj


def _rel_err(out, ref, tol):
    """Worst relative error via harness_lib.correct -- the SAME metric the kernel lane gates on.

    `correct` floors the relative term with `tol * RMS(ref)` rather than `tol * max(|ref|)`, so a
    small element of a high-dynamic-range output cannot hide an unbounded relative error.
    """
    torch = _torch()
    po, pr = h.flatten_outputs(out), h.flatten_outputs(ref)
    if not po or len(po) != len(pr):
        return float("inf")
    # Compare in a common dtype/device: promote both sides to fp32 on the reference's device.
    po = [t.detach().to("cpu", dtype=torch.float32) for t in po]
    pr = [t.detach().to("cpu", dtype=torch.float32) for t in pr]
    _, err = h.correct(po if len(po) > 1 else po[0],
                       pr if len(pr) > 1 else pr[0], tol)
    return err


def _fingerprint(obj):
    """Order-sensitive scalar summary of a reference output, for cross-leg agreement checking.

    Rounded to 6 significant digits so that harmless fp32 nondeterminism (thread counts, CPU BLAS
    blocking) does not trip the check, while any real change to the reference function does.
    """
    torch = _torch()
    parts = []
    for t in h.flatten_outputs(obj):
        t = t.detach().to("cpu", dtype=torch.float32)
        parts.append((tuple(t.shape),
                      float(f"{t.sum().item():.6g}"),
                      float(f"{t.abs().sum().item():.6g}"),
                      float(f"{t.norm().item():.6g}")))
    return parts


def run_leg(args):
    torch = _torch()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    blob_target, cases = load_cases(args.oracle, device)
    target = args.target or blob_target
    # The reference callable is the BASELINE seam, never the leg's own. Defaulting it to --target
    # would let a candidate that changes the function itself also move its own reference.
    ref_target = args.ref_target or blob_target
    if not cases:
        raise SystemExit(f"op_parity_probe: oracle {args.oracle} has no recorded cases")
    fn = resolve_callable(target)
    ref_fn = resolve_callable(ref_target)

    per_case = []
    worst_vs_fp32 = 0.0
    worst_vs_captured = 0.0
    reference_ok = True
    ref_fp = []
    for c in cases:
        live = fn(*c["args"], **c["kwargs"])
        # fp32 CPU reference: BASELINE callable, same inputs, higher precision, env-immune.
        try:
            ref32 = ref_fn(*_to_fp32_cpu(c["args"]), **_to_fp32_cpu(c["kwargs"]))
            e32 = _rel_err(live, ref32, args.tol)
            ref_fp.append(_fingerprint(ref32))
        except Exception as e:                       # CUDA-only / fused op: no CPU fp32 path
            reference_ok = False
            e32 = None
            per_case.append({"sig": c["sig"], "regime": c["regime"], "err_vs_fp32": None,
                             "note": f"fp32 reference unavailable: {e!r}"})
        ecap = _rel_err(live, c["captured_out"], args.tol)
        if e32 is not None:
            per_case.append({"sig": c["sig"], "regime": c["regime"],
                             "err_vs_fp32": None if not math.isfinite(e32) else round(e32, 8),
                             "err_vs_captured": None if not math.isfinite(ecap) else round(ecap, 8)})
            worst_vs_fp32 = max(worst_vs_fp32, e32)
        worst_vs_captured = max(worst_vs_captured, ecap)

    out = {
        "leg": args.leg,
        "target": target,
        "ref_target": ref_target,
        "oracle": args.oracle,
        "device": str(device),
        "num_cases": len(cases),
        "tol": args.tol,
        # The gating number. None when no higher-precision reference could be built -- the compare
        # step then REFUSES to rule rather than silently falling back to a weaker comparison.
        "err_vs_fp32": round(worst_vs_fp32, 8) if reference_ok else None,
        "reference_kind": "cpu_fp32_same_callable" if reference_ok else "unavailable",
        # Report-only: how far this leg sits from the output captured off the unmodified baseline.
        # For the ref leg this is the capture-vs-replay floor; for the cand leg it is the swap's drift.
        "err_vs_captured": round(worst_vs_captured, 8) if math.isfinite(worst_vs_captured) else None,
        # Proof that both legs were scored against the SAME truth; compared, not trusted.
        "ref_fingerprint": ref_fp,
        # Proof that the two legs are DISTINCT. Measured once: a mistyped
        # PYTORCH_TUNABLEOP_FILENAME (PyTorch inserts the device ordinal before `.csv`) made the
        # candidate fall back to stock BLAS; both legs then reported the identical 3.870e-03 and the
        # gate said "pass". A verdict on two identical legs is worse than no verdict.
        "tunable": _tunable_state(),
        "per_case": per_case,
    }
    _write(args.out, out)
    print(f"OP_PARITY_ERR={out['err_vs_fp32']}")
    return 0


def _tunable_state():
    """Snapshot PyTorch TunableOp state from INSIDE the leg process (it is per-process).

    Empty/absent on a stock leg, populated on a leg whose table actually loaded -- which is exactly
    the distinction a shell-side log grep could not make.
    """
    torch = _torch()
    tun = getattr(getattr(torch, "cuda", None), "tunable", None)
    if tun is None:
        return {"available": False}
    st = {"available": True}
    for name in ("is_enabled", "tuning_is_enabled"):
        try:
            st[name] = bool(getattr(tun, name)())
        except Exception:
            pass
    try:
        res = list(tun.get_results() or [])
        st["num_results"] = len(res)
        # op+params signature only; the chosen algorithm string is what distinguishes the legs.
        st["results"] = sorted(f"{r[0]},{r[1]},{r[2]}" for r in res)[:64]
    except Exception as e:
        st["results_error"] = repr(e)[:160]
    return st


def _legs_indistinguishable(ref, cand):
    """True when nothing in the two legs' measurements shows the candidate actually did anything.

    Deliberately conservative: a candidate whose change happens to be bit-identical on these shapes
    carries no numeric risk, but this gate cannot tell that apart from a change that never loaded.
    Both deserve `unknown`, not `pass`.
    """
    def errs(d):
        return [c.get("err_vs_fp32") for c in (d.get("per_case") or [])]
    if errs(ref) != errs(cand) or not errs(ref):
        return False
    return (ref.get("tunable") or {}).get("results") == (cand.get("tunable") or {}).get("results")


def compare(args):
    ref = json.loads(open(args.compare[0]).read())
    cand = json.loads(open(args.compare[1]).read())
    e_base = ref.get("err_vs_fp32")
    e_cand = cand.get("err_vs_fp32")

    if e_base is None or e_cand is None:
        verdict, reason = "unknown", (
            "no higher-precision reference on at least one leg "
            f"(ref={ref.get('reference_kind')}, cand={cand.get('reference_kind')}) — "
            "this gate does not rule without a floor; fall back to byte-parity or accuracy_gate=gsm8k")
    elif _legs_indistinguishable(ref, cand):
        # Both legs ran the same code on the same inputs, so "the candidate matches the baseline" is
        # a tautology, not evidence. This is the mistyped-table-path failure mode, and it presents as
        # a clean pass.
        verdict, reason = "unknown", (
            "the two legs are INDISTINGUISHABLE: identical error to the last digit and no TunableOp "
            f"results loaded on the candidate (cand.tunable={cand.get('tunable')}). The candidate "
            "leg did not actually engage its change -- check PYTORCH_TUNABLEOP_FILENAME (PyTorch "
            "inserts the device ordinal before `.csv`) or the overlay path, then re-run.")
    elif ref.get("ref_fingerprint") != cand.get("ref_fingerprint"):
        # The two legs did not score against the same truth, so their errors are not comparable and
        # the candidate's number is self-referential. Refuse to rule rather than rule wrongly.
        verdict, reason = "unknown", (
            "the two legs' fp32 references DISAGREE, so they were not scored against the same truth "
            f"(ref_target ref={ref.get('ref_target')!r} cand={cand.get('ref_target')!r}) — the "
            "candidate's error is measured against a reference it moved itself; pin --ref-target to "
            "the unmodified baseline seam, or gate this change with accuracy_gate=gsm8k instead")
    else:
        # correctness_gates.md: the baseline's OWN error is the floor. A candidate passes when it is
        # not MATERIALLY worse than that floor, and in any case stays inside the absolute tolerance.
        limit = max(args.tol, args.floor_mult * e_base)
        ok = e_cand <= limit
        verdict = "pass" if ok else "fail"
        reason = (f"err_cand={e_cand:.3e} vs limit={limit:.3e} "
                  f"(= max(tol {args.tol:.3e}, {args.floor_mult}x baseline floor {e_base:.3e}))")

    out = {
        "verdict": verdict,
        "reason": reason,
        "tol": args.tol,
        "floor_mult": args.floor_mult,
        "err_base_vs_fp32": e_base,
        "err_cand_vs_fp32": e_cand,
        "err_cand_vs_captured_baseline": cand.get("err_vs_captured"),
        "err_ref_vs_captured_baseline": ref.get("err_vs_captured"),
        "reference_agreed": ref.get("ref_fingerprint") == cand.get("ref_fingerprint"),
        "target": cand.get("target") or ref.get("target"),
        "ref_target": cand.get("ref_target") or ref.get("ref_target"),
        "num_cases": cand.get("num_cases"),
        "ref_leg": args.compare[0],
        "cand_leg": args.compare[1],
    }
    _write(args.out, out)
    print(json.dumps(out, indent=2))
    print(f"OP_PARITY={verdict}")
    return 0 if verdict == "pass" else 1


def _write(path, payload):
    if not path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)


_TUNABLEOP_DTYPE = {"BFloat16": "bfloat16", "Half": "float16", "Float": "float32"}
# GemmTunableOp_BFloat16_TN,tn_6144_8_2560_ld_2560_2560_6144,Gemm_Hipblaslt_6849,0.200725
_TUNABLEOP_ROW = re.compile(r"^tn_(\d+)_(\d+)_(\d+)_ld_(\d+)_(\d+)_(\d+)$")


def synth_from_tunableop_csv(a):
    """Build a reference_io.pt for `torch.nn.functional:linear` from a PyTorch TunableOp table.

    WHY this exists instead of a capture_shapes run: `capture_shapes` refuses to wrap
    `torch.nn.functional:linear` (a C builtin -- a plain-function stand-in SIGSEGVs the server), and
    the Python-level seams around it either are registered custom ops (so patching the module
    attribute never intercepts) or take the `nn.Module` as an argument, which `_snapshot` degrades to
    `{"__repr__": ...}` -- the weights would be gone.

    For an env-only BLAS-table swap none of that is needed. The table IS the enumeration of the GEMMs
    the swap touches: every row is a shape TunableOp selected an algorithm for, so a candidate that
    changes nothing else can only differ on these. That satisfies `correctness_gates.md` L82-92
    ("test the shapes you actually deploy") more directly than sampling a live server would.

    Inputs are randn on a FIXED CPU seed and written to disk, so both legs read identical bytes rather
    than trusting RNG reproducibility across processes. randn -- not ones -- per L82-92.

    Only `GemmTunableOp_<dtype>_TN` rows whose leading dimensions confirm the N/M/K reading are used;
    anything else is listed under `skipped` so a partial oracle can never look complete.
    """
    torch = _torch()
    rows, skipped = [], []
    with open(a.from_tunableop_csv) as fh:
        for raw in fh:
            parts = [c.strip() for c in raw.strip().split(",")]
            if len(parts) < 2 or not parts[0].startswith("GemmTunableOp_"):
                if parts and parts[0] and parts[0] != "Validator":
                    skipped.append(raw.strip()[:120])
                continue
            bits = parts[0].split("_")
            dt = _TUNABLEOP_DTYPE.get(bits[1] if len(bits) > 2 else "")
            m = _TUNABLEOP_ROW.match(parts[1])
            if dt is None or not m:
                skipped.append(raw.strip()[:120])
                continue
            n, mm, k, lda, ldb, ldc = (int(g) for g in m.groups())
            # TN is what nn.Linear emits: column-major A^T(K x N) lda=K, B(K x M) ldb=K, C(N x M)
            # ldc=N. If the leading dimensions disagree, the N/M/K reading is wrong -- do not guess.
            if (lda, ldb, ldc) != (k, k, n):
                skipped.append(raw.strip()[:120] + "  [ld mismatch]")
                continue
            rows.append((dt, n, mm, k))

    if not rows:
        raise SystemExit(f"op_parity_probe: no usable GemmTunableOp TN rows in {a.from_tunableop_csv}")

    gen = torch.Generator(device="cpu").manual_seed(a.synth_seed)
    records = []
    for dt, n, mm, k in rows:
        dtype = getattr(torch, dt)
        x = torch.randn(mm, k, generator=gen, dtype=torch.float32).to(dtype)
        w = torch.randn(n, k, generator=gen, dtype=torch.float32).to(dtype)
        records.append({
            "sig": f"linear(T({mm}, {k}):{dtype}, T({n}, {k}):{dtype}, None)",
            "regime": "decode" if mm <= 32 else "prefill",
            "args": [_snap(x), _snap(w), None],
            "kwargs": {},
            # No captured baseline output exists for a synthesized oracle. The probe's gating number
            # is err_vs_fp32; err_vs_captured is report-only and degrades to None here.
            "output": None,
            "shared_keys": [],
        })

    blob = {"target": "torch.nn.functional:linear", "records": records,
            "shared": {}, "persist_policy": "full"}
    dest = a.out or os.path.join(os.path.dirname(os.path.abspath(a.from_tunableop_csv)),
                                 "reference_io.pt")
    if os.path.isdir(dest):
        dest = os.path.join(dest, "reference_io.pt")
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    torch.save(blob, dest)
    print(json.dumps({"oracle": dest, "cases": len(records),
                      "shapes": [{"dtype": d, "N": n, "M": m, "K": k} for d, n, m, k in rows],
                      "skipped": skipped, "seed": a.synth_seed,
                      "bytes": os.path.getsize(dest)}, indent=2))
    if skipped:
        print(f"OP_PARITY_ORACLE=partial ({len(skipped)} rows skipped)")
    else:
        print("OP_PARITY_ORACLE=complete")
    return 0


def _snap(t):
    """capture_shapes' on-disk tensor form, so the synthesized blob loads through the same reader."""
    return {"__tensor__": True, "data": t, "dtype": str(t.dtype), "device": "cpu",
            "shape": list(t.shape), "contiguous": True}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--oracle", help="capture_shapes task dir, or the reference_io.pt inside it")
    p.add_argument("--from-tunableop-csv", default="",
                   help="provision mode: synthesize an oracle for torch.nn.functional:linear from a "
                        "PyTorch TunableOp results CSV (the shapes the table swaps)")
    p.add_argument("--synth-seed", type=int, default=1234,
                   help="RNG seed for --from-tunableop-csv inputs")
    p.add_argument("--target", default="", help="module:attr the LEG exercises (default: the blob's own)")
    p.add_argument("--ref-target", default="",
                   help="module:attr used to build the fp32 reference; must be the UNMODIFIED "
                        "baseline seam (default: the blob's own, i.e. what the oracle was captured under)")
    p.add_argument("--leg", choices=["ref", "cand"], help="which leg this process represents")
    p.add_argument("--device", default="", help="default: cuda when available")
    p.add_argument("--tol", type=float, default=0.01,
                   help="relative tolerance; 1e-2 matches the tuning lane's rtol=atol convention")
    p.add_argument("--floor-mult", type=float, default=2.0,
                   help="candidate error may not exceed this multiple of the baseline's own error")
    p.add_argument("--compare", nargs=2, metavar=("REF_JSON", "CAND_JSON"),
                   help="verdict mode: compare two leg JSONs")
    p.add_argument("--out", default="", help="write the JSON result here")
    a = p.parse_args(argv)

    if a.compare:
        return compare(a)
    if a.from_tunableop_csv:
        return synth_from_tunableop_csv(a)
    if not a.oracle or not a.leg:
        p.error("leg mode needs --oracle and --leg (or use --compare)")
    if os.path.isdir(a.oracle):
        a.oracle = os.path.join(a.oracle, "reference_io.pt")
    if not os.path.exists(a.oracle):
        raise SystemExit(f"op_parity_probe: no oracle at {a.oracle}")
    return run_leg(a)


if __name__ == "__main__":
    sys.exit(main())
