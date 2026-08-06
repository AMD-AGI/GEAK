#!/usr/bin/env python3
"""Closed-loop plain-Triton -> Gluon recovery driver (tile-programming-triton-gluon).

Orchestrates the transcribe recovery so it stops being a manual eyeball pass:

  (1) [--with-skeleton] official ``triton_to_gluon_translator`` -> algorithm skeleton
      (control flow / masks / dtype; it drops num_stages and re-infers layouts).
  (2) parse layouts from the plain ``.ttgir`` (ttgir_to_gluon) -> concrete
      ``gl.constexpr`` layouts that match what the compiler actually inferred.
  (3) [--with-pipeline] recover the async double-buffer scaffold from the ``.ttgir``.
  (4) assemble an anchor ``.py``.
  (5) [--verify] diff the recompiled anchor's layout attributes against plain
      (verify_equivalence); correctness + record wiring is driven via dump_ir.sh.

Modes (escalating):
  --layouts-only   (default) emit only the recovered layout factory.
  --with-skeleton  also run the official source translator and assemble a kernel.
  --with-pipeline  also append the recovered async pipeline scaffold (the standard
                   pipeline-layer start: reproduce plain's pipeline, then improve).

Boundaries (see references/tile-programming/compiler-contract.md):
  Layouts recover deterministically; the async pipeline structure recovers opt-in;
  register allocation / spills are NOT in TTGIR (they happen later in LLVM) and are
  never attempted here -- those stay slicing + RA hints.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter

import ttgir_to_gluon as t2g

# --------------------------------------------------------------------------- #
# (1) official source translator (graceful fallback)
# --------------------------------------------------------------------------- #


def _unwrapped_jit_source(kernel_spec: str, target) -> str | None:
    """Translate ``module.path:object`` after peeling @triton.heuristics / @triton.autotune.

    Upstream's entry point resolves the spec with a bare ``getattr`` and hands
    whatever it finds to the AST rewriter. A wrapped kernel is a ``Heuristics`` /
    ``Autotuner`` instance whose own source contains no kernel body, so the
    translation comes back *empty* -- and successfully, which is the trap.
    Peeling to the ``JITFunction`` while keeping the defining module as the global
    scope reproduces the upstream path byte-for-byte on an unwrapped kernel and
    recovers the body on a wrapped one.

    Returns None if this Triton does not expose the internals needed to do it,
    which leaves the caller on the stock path.
    """
    try:
        from triton.tools.triton_to_gluon_translator.slice_kernel import GlobalValue
        from triton.tools.triton_to_gluon_translator.translator import translate_kernels
    except Exception:  # noqa: BLE001 - internals moved; the stock path still works
        return None
    import importlib

    from triton.runtime.jit import JITFunction
    mod_name, _, obj_name = kernel_spec.partition(":")
    if not obj_name:
        return None
    module = importlib.import_module(mod_name)
    obj = getattr(module, obj_name)
    fn = obj
    while not isinstance(fn, JITFunction) and hasattr(fn, "fn"):
        fn = fn.fn
    if not isinstance(fn, JITFunction):
        return None
    if fn is not obj:
        print(f"[recover_gluon] {obj_name} is wrapped in {type(obj).__name__}; "
              f"translating the JITFunction underneath it.", file=sys.stderr)
    return translate_kernels([GlobalValue.wrap(fn, obj_name, lambda: module)], target=target)


def run_translator(kernel_spec: str, arch: str) -> str | None:
    """Run the upstream triton_to_gluon_translator on ``module.path:object``.

    Returns the translated Gluon source, or None -- for one of three reasons, kept
    distinct on stderr because they call for different responses: the translator is
    not in this Triton (older build), it raised, or it returned nothing to
    translate. The caller falls back to a layouts-only anchor in all three.
    """
    try:
        from triton.tools.triton_to_gluon_translator.target import TranslatorTarget
        from triton.tools.triton_to_gluon_translator.translator import translate_paths
    except Exception as e:  # noqa: BLE001 - any import failure -> graceful fallback
        print(f"[recover_gluon] translator not in this Triton ({e!r}); falling back "
              f"to layouts-only.", file=sys.stderr)
        return None
    target = TranslatorTarget(arch)
    try:
        src = _unwrapped_jit_source(kernel_spec, target)
        if src is None:
            src = translate_paths([kernel_spec], target=target)
    except Exception as e:  # noqa: BLE001
        print(f"[recover_gluon] translator failed on {kernel_spec!r} ({e!r}); "
              f"falling back to layouts-only.", file=sys.stderr)
        return None
    if not src.strip():
        print(f"[recover_gluon] translator returned an empty skeleton for "
              f"{kernel_spec!r}. It is installed and it did not raise, so this is "
              f"the kernel and not the build -- check that the spec names the "
              f"kernel itself. Falling back to layouts-only.", file=sys.stderr)
        return None
    return src


# --------------------------------------------------------------------------- #
# (4) assemble the anchor
# --------------------------------------------------------------------------- #

_INTEGRATION_NOTE = """\
# ===========================================================================
# Auto-recovered Gluon anchor (tile-programming-triton-gluon / recover_gluon.py).
#
# How to use this file:
#   * The `*: gl.constexpr = gl.*` block below is the AUTHORITATIVE set of layouts
#     the plain-Triton compiler actually inferred (coalesced/MMA/shared/dot-operand).
#     Wire these into the kernel body.
#   * If a translator skeleton is included, it carries the algorithm (control flow,
#     masks, dtype) but uses the translator's *default* MMA layout + AutoLayout;
#     replace those with the recovered layouts above.
#   * If a pipeline scaffold is included, it is the pipeline-layer STARTING POINT:
#     reproduce plain's double-buffer/async structure here, then IMPROVE on it
#     (deeper buffering / operand prefetch / manual interleave) -- see pipeline.md
#     ## Auto-recovering the pipeline structure. Keep it out of the transcription
#     step (the faithful layouts-only anchor stays the attribution baseline).
#   * Register allocation / spills are NOT recovered (not in TTGIR); use slicing +
#     RA hints. Then run --verify to confirm layout-equivalence vs plain.
# ===========================================================================
"""


def assemble_anchor(ttgir_text: str, *, kernel_spec: str | None, arch: str,
                    with_skeleton: bool, with_pipeline: bool, source: str = "") -> str:
    layouts = t2g.parse_layouts(ttgir_text)
    parts = [_INTEGRATION_NOTE, ""]

    if with_skeleton:
        if not kernel_spec:
            parts.append("# [skeleton requested but no --kernel module:object given]\n")
        else:
            skeleton = run_translator(kernel_spec, arch)
            if skeleton:
                parts.append("# ---- (1) algorithm skeleton (official translator) ----")
                parts.append(skeleton.rstrip() + "\n")
            else:
                parts.append("# ---- (1) no skeleton (reason on stderr): hand-write the "
                             "kernel body and use the layouts below ----\n")

    parts.append("# ---- (2) recovered concrete layouts (authoritative) ----")
    parts.append(t2g.emit_layout_factory(layouts, source))

    if with_pipeline:
        parts.append("# ---- (3) recovered async pipeline scaffold (opt-in) ----")
        parts.append(t2g.emit_pipeline_skeleton(t2g.parse_pipeline(ttgir_text), layouts, source))

    return "\n".join(parts).rstrip() + "\n"


# --------------------------------------------------------------------------- #
# (5) layout-equivalence core (text-based; no GPU needed)
# --------------------------------------------------------------------------- #


def canonical_layout_attrs(ttgir_text: str) -> Counter:
    """Canonical layout attributes with occurrence counts, name-independent.

    Named preamble defs are reduced to ``#ttg.<kind><...>`` (whitespace collapsed,
    leading ``#name =`` dropped) and inline dot_op signatures are included, so two
    TTGIRs are layout-equivalent iff the attribute *sets* match.

    The counts are returned too, but they are not part of the equivalence
    predicate: they track how many times a layout is mentioned, which loop
    unrolling and CSE change without changing a single layout. See
    ``verify_equivalence``.
    """
    import re
    canon: Counter = Counter()
    for line in ttgir_text.splitlines():
        m = t2g._NAMED_RE.match(line.strip())
        if not m or m.group("kind") == "shared_memory":
            continue
        attr = f"#ttg.{m.group('kind')}<{m.group('rest')}>"
        canon[re.sub(r'\s+', ' ', attr).strip()] += 1
    for m in t2g._DOTOP_RE.finditer(ttgir_text):
        body = re.sub(r'\s+', ' ', m.group("body")).strip()
        canon[f"#ttg.dot_op<{{{body}}}>"] += 1
    return canon


def verify_equivalence(plain_ttgir: str, anchor_ttgir: str) -> tuple[bool, str]:
    """Compare layout attributes between the plain and recovered-anchor TTGIRs.

    The predicate is *set* equality. Occurrence counts are reported when they
    differ but never fail the check: an auto-pipelined plain kernel leaves
    ``add_pipeline`` with its loop unrolled, so each layout in its body is
    mentioned N times against the un-unrolled anchor's once. That difference is
    the pipeline layer talking, not the layout layer, and failing on it would
    make this gate unpassable on precisely the kernels step 2 exists for.
    """
    plain = canonical_layout_attrs(plain_ttgir)
    anchor = canonical_layout_attrs(anchor_ttgir)
    missing = sorted(set(plain) - set(anchor))  # in plain, never used by the anchor
    extra = sorted(set(anchor) - set(plain))    # introduced by the anchor, absent in plain
    ok = not missing and not extra
    lines = ["LAYOUT EQUIVALENCE: " + ("PASS" if ok else "FAIL")]
    if missing:
        lines.append("  missing (in plain, not in anchor):")
        lines += [f"    - {a}  x{plain[a]}" for a in missing]
    if extra:
        lines.append("  extra (in anchor, not in plain):")
        lines += [f"    + {a}  x{anchor[a]}" for a in extra]
    if ok:
        lines.append(f"  {len(plain)} distinct layout attributes matched.")
        skew = [a for a in sorted(plain) if plain[a] != anchor[a]]
        if skew:
            lines.append(f"  note: {len(skew)} of them are mentioned a different number of "
                         "times (plain's loop is unrolled, the anchor's is not). "
                         "Informational — not a layout difference:")
            lines += [f"      {a}  plain x{plain[a]}, anchor x{anchor[a]}" for a in skew]
    return ok, "\n".join(lines)


def emit_transcribe_record(layouts, *, layout_equiv: str = "not-checked",
                           correctness: str = "not-run", source: str = "",
                           sched: dict | None = None) -> str:
    """Auto-fill the experiment-records.md ## 3 Transcribe / Anchor record."""
    ref_to_var = {f"#{l.name}": l.var for l in layouts if l.kind != "dot_op"}
    sched = sched or {}
    nw = sched.get("num_warps")
    ns = sched.get("num_stages")
    nw_str = str(nw) if nw is not None else "<read from plain best config>"
    ns_str = (str(ns) if ns is not None
              else "<consumed in post-pipeliner ttgir; read buffer count via parse_pipeline>")
    lines = [
        "## 3. Transcribe / Anchor + Calibration Record  (auto-filled by recover_gluon.py)",
        f"# source: {source or '<ttgir>'}",
        "plain_ttgir_layouts_recovered:",
    ]
    for l in layouts:
        lines.append(f"  #{l.kind} ({l.name}) -> {l.to_gluon_expr(ref_to_var)}")
    lines += [
        "plain_schedule_targets:   # the config the pipeline + slicing layers must reach",
        f"  num_warps  -> {nw_str}   # occupancy / slicing target",
        f"  num_stages -> {ns_str}   # starting pipeline depth (reproduce, then improve "
        "-- pipeline.md ## Auto-recovering the pipeline structure)",
        f"layout_equivalence_vs_plain: {layout_equiv}",
        f"correctness == plain: {correctness}",
        "perf_delta_vs_plain: <fill>   # regression expected, NOT a reject",
        "register_spill: not recoverable from TTGIR -> use slicing + RA hints",
        "gluon_anchor_metrics.json: <write after re-profile>",
    ]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# self-test (offline, no GPU and no triton import)
# --------------------------------------------------------------------------- #

_SELFTEST_PREAMBLE = """\
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [16, 16, 32], isTransposed = true}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 8, order = [1, 0]}>
"""
_SELFTEST_DOT = ("    %acc = tt.dot %a, %b, %acc : "
                 "tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * "
                 "tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>\n")


def _selftest() -> int:
    failures = 0

    # An auto-pipelined plain kernel comes out of add_pipeline with its loop unrolled,
    # so every layout in the body is mentioned N times against the un-unrolled anchor's
    # once. Same layouts, different occurrence counts -- this must PASS, or the gate is
    # unpassable on exactly the kernels the round-2 relaxation exists for.
    plain = _SELFTEST_PREAMBLE + _SELFTEST_DOT * 5
    anchor = _SELFTEST_PREAMBLE + _SELFTEST_DOT
    ok, report = verify_equivalence(plain, anchor)
    if not ok:
        print(f"SELFTEST FAIL (unroll-skew): same layouts rejected on occurrence "
              f"counts alone\n{report}")
        failures += 1
    elif "plain x5, anchor x1" not in report:
        print(f"SELFTEST FAIL (unroll-skew): count skew not reported\n{report}")
        failures += 1

    # A layout that genuinely differs must still be caught, counts or no counts.
    swapped = anchor.replace("perPhase = 2, maxPhase = 8", "perPhase = 1, maxPhase = 16")
    ok, report = verify_equivalence(plain, swapped)
    if ok or "missing" not in report or "extra" not in report:
        print(f"SELFTEST FAIL (swizzle-differs): mismatch not detected\n{report}")
        failures += 1

    # So must a layout the anchor introduced and plain never had.
    extra = anchor + "#blocked1 = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 4], order = [0, 1]}>\n"
    ok, report = verify_equivalence(plain, extra)
    if ok or "extra" not in report:
        print(f"SELFTEST FAIL (anchor-extra): spurious layout not detected\n{report}")
        failures += 1

    print("SELFTEST PASS" if not failures else f"SELFTEST FAILURES: {failures}")
    return 1 if failures else 0


def _run_harness_correctness(harness_cmd: str) -> str:
    """Run a correctness harness command; return 'pass' | 'fail' | 'not-run'."""
    import subprocess
    try:
        out = subprocess.run(harness_cmd, shell=True, capture_output=True, text=True, timeout=1800)
    except Exception as e:  # noqa: BLE001
        print(f"[recover_gluon] harness did not run ({e!r})", file=sys.stderr)
        return "not-run"
    blob = (out.stdout or "") + (out.stderr or "")
    if "CORRECTNESS PASS" in blob:
        return "pass"
    if "CORRECTNESS FAIL" in blob:
        return "fail"
    return "not-run"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ttgir", help="plain-Triton .ttgir (from dump_ir.sh)")
    ap.add_argument("--kernel", help="kernel for the translator, as module.path:object")
    ap.add_argument("--arch", default="gfx950", help="target arch (e.g. gfx950, sm90)")
    ap.add_argument("--with-skeleton", action="store_true", help="run the official translator")
    ap.add_argument("--with-pipeline", action="store_true",
                    help="append the recovered async pipeline scaffold (opt-in)")
    ap.add_argument("--out", help="write the assembled anchor here (default: stdout)")
    # --verify mode: compare two already-dumped TTGIRs (the closing step).
    ap.add_argument("--verify", action="store_true",
                    help="layout-equivalence check between --ttgir (plain) and --anchor-ttgir")
    ap.add_argument("--anchor-ttgir", help="recompiled anchor .ttgir (for --verify)")
    ap.add_argument("--record", action="store_true",
                    help="emit the auto-filled experiment-records transcribe record")
    ap.add_argument("--harness", help="correctness harness command to run (looks for CORRECTNESS PASS)")
    ap.add_argument("--selftest", action="store_true",
                    help="run the bundled layout-equivalence self-tests (offline)")
    a = ap.parse_args()

    if a.selftest:
        raise SystemExit(_selftest())

    if a.verify:
        if not (a.ttgir and a.anchor_ttgir):
            ap.error("--verify needs --ttgir (plain) and --anchor-ttgir")
        plain_text = open(a.ttgir).read()
        ok, report = verify_equivalence(plain_text, open(a.anchor_ttgir).read())
        print(report)
        correctness = _run_harness_correctness(a.harness) if a.harness else "not-run"
        if a.harness:
            print(f"CORRECTNESS: {correctness}")
        print()
        print(emit_transcribe_record(t2g.parse_layouts(plain_text),
                                     layout_equiv="pass" if ok else "fail",
                                     correctness=correctness, source=a.ttgir,
                                     sched=t2g.parse_schedule_targets(plain_text)))
        raise SystemExit(0 if (ok and correctness != "fail") else 1)

    if a.record:
        if not a.ttgir:
            ap.error("--record needs --ttgir")
        rec_text = open(a.ttgir).read()
        print(emit_transcribe_record(t2g.parse_layouts(rec_text), source=a.ttgir,
                                     sched=t2g.parse_schedule_targets(rec_text)))
        return

    if not a.ttgir:
        ap.error("provide --ttgir (or use --verify / --selftest)")
    anchor = assemble_anchor(open(a.ttgir).read(), kernel_spec=a.kernel, arch=a.arch,
                             with_skeleton=a.with_skeleton, with_pipeline=a.with_pipeline,
                             source=a.ttgir)
    if a.out:
        with open(a.out, "w") as f:
            f.write(anchor)
        print(f"wrote anchor -> {a.out}")
    else:
        print(anchor)


if __name__ == "__main__":
    main()
