#!/usr/bin/env python3
"""Inventory every plain-Triton kernel by the PIPELINE FORM it can exercise.

Three mechanisms are in scope and they are not interchangeable:

  A. cross-iteration software pipeline -- `add_schedule_loops` + `add_pipeline`. Needs a
     `tl.range` whose `num_stages` resolves >= 2 (explicitly, or `None` inheriting the
     launch value). This is the one we can re-inject into Gluon.
  B. block ping-pong -- `add_block_pingpong`. Runs only AFTER the pipeliner and adds hard
     constraints of its own: exactly 1 dot-like op (2 needs numStages==4), an MFMA-layout A
     operand, >= 2 loop-carried local_loads, and tile/warp windows. So "has a dot and
     num_stages>=2" is necessary and nowhere near sufficient.
  C. async copy / direct-to-LDS -- `use_async_copy`, default ON only for gfx950/gfx1250.
     On gfx942 it is off by pass-level policy, not by hardware:
     `supportsBufferLoadToLocal()` covers CDNA3 at 32-bit vector width.

`warp_pipeline_stage` has NO plain-Triton spelling at all -- it is a Gluon-only marker that
drives `add_warp_pipeline`, which `make_ttgir` never calls. So a "plain example" of it does
not exist and the column below says so rather than guessing.

Classification is from SOURCE text, which is a screen and not a verdict: a source saying
`num_stages=2` can dispatch a branch compiled at 1, and only the dump settles it. Rows are
ranked so the compile-and-check step has somewhere to start.
"""
import json
import os
import re
import sys

# --- forms we can detect in source
RE_TLRANGE_NS = re.compile(r"tl\.range\((?:[^()]|\([^()]*\))*?num_stages\s*=\s*([A-Za-z0-9_]+)")
RE_BARE_RANGE = re.compile(r"for\s+\w+\s+in\s+range\(")
RE_CFG_NS = re.compile(r"[\"']num_stages[\"']\s*:\s*(\d+)|num_stages\s*=\s*(\d+)")
RE_DOT = re.compile(r"\btl\.dot\b|\btl\.dot_scaled\b")
RE_LOAD = re.compile(r"\btl\.load\b")
RE_JIT = re.compile(r"@triton\.jit|@triton\.autotune|@triton\.heuristics")


def classify(path: str) -> dict:
    try:
        with open(path, errors="replace") as fh:
            src = fh.read()
    except OSError:
        return {}
    if not RE_JIT.search(src):
        return {}
    ns_loop = RE_TLRANGE_NS.findall(src)
    cfg = [int(a or b) for a, b in RE_CFG_NS.findall(src)]
    explicit = sorted({v for v in ns_loop if v.isdigit()})
    symbolic = sorted({v for v in ns_loop if not v.isdigit()})
    n_dot = len(RE_DOT.findall(src))
    # Form A eligibility is conditioned on the DOT, which is the correction that cost two
    # wrong rules before it was measured. `add_schedule_loops` takes the launch
    # `options.num_stages` as the default for a loop it considers a candidate, and a loop
    # containing a dot IS one -- so a bare `range` with a dot pipelines with no annotation
    # at all (measured: loads 2->4->6, memdesc_index 0->4->6, and NO tt.num_stages in the
    # IR). A dot-free loop is not a candidate unless it is annotated: the same launch value
    # left a dot-free bare-range loop byte-identical at 1/2/3. Verified on plain AND on the
    # Gluon side under injection.
    ge2_loop = any(int(v) >= 2 for v in explicit)
    inherits = bool(symbolic) or "None" in ns_loop
    annotated = bool(ns_loop) and (ge2_loop or inherits)
    form_a = annotated or (n_dot >= 1 and RE_BARE_RANGE.search(src) is not None)
    why = ("annotated tl.range" if annotated else
           "dot loop, launch num_stages default" if form_a else "")
    return {
        "file": path,
        "tl_range_ns": ns_loop[:6],
        "explicit_ns": explicit,
        "symbolic_ns": symbolic,
        "cfg_ns": sorted(set(cfg))[:6],
        "bare_range_loops": len(RE_BARE_RANGE.findall(src)),
        "dots": n_dot,
        "loads": len(RE_LOAD.findall(src)),
        "FORM_A_swp": form_a,
        "FORM_A_why": why,
        # B is *eligible* only with a dot present and A available; the pass's own
        # constraints are checked on the dump, not here.
        "FORM_B_pingpong_eligible": bool(form_a and n_dot >= 1),
        # C needs no source pattern -- it is a backend policy switch. Any kernel with
        # loads into a dot can in principle take it; flag the dot ones.
        "FORM_C_async_candidate": bool(n_dot >= 1),
    }


def walk(roots):
    out = []
    for root in roots:
        for dirpath, dirnames, files in os.walk(root):
            dirnames[:] = [d for d in dirnames
                           if d not in ("__pycache__", "build", "3rdparty", ".git")]
            for f in files:
                if f.endswith(".py"):
                    r = classify(os.path.join(dirpath, f))
                    if r:
                        out.append(r)
    return out


def selftest():
    """Offline. Form-A eligibility turns on the DOT, and that is the rule this file got
    wrong twice before it was measured -- so pin both branches of it here."""
    import tempfile
    fails = []

    def ck(name, cond, detail=""):
        print(f"  {'ok  ' if cond else 'FAIL'} {name}" + (f"  -- {detail}" if detail and not cond else ""))
        if not cond:
            fails.append(name)

    def classify_src(src):
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(src)
            name = f.name
        try:
            return classify(name)
        finally:
            os.unlink(name)

    print("pipeline_survey selftest")
    dot_bare = classify_src(
        "@triton.jit\ndef k():\n    for k in range(0, K, BK):\n        acc = tl.dot(a, b, acc)\n")
    ck("a dot loop on a bare range IS a candidate (launch num_stages is its default)",
       dot_bare.get("FORM_A_swp") is True, json.dumps(dot_bare))
    ck("and the reason is recorded, not just the verdict",
       dot_bare.get("FORM_A_why") == "dot loop, launch num_stages default")
    free_bare = classify_src(
        "@triton.jit\ndef k():\n    for i in range(0, N, B):\n        acc += tl.load(p + i)\n")
    ck("a dot-free loop on a bare range is NOT a candidate",
       free_bare.get("FORM_A_swp") is False, json.dumps(free_bare))
    free_ann = classify_src(
        "@triton.jit\ndef k():\n    for i in tl.range(0, N, B, num_stages=2):\n        acc += tl.load(p + i)\n")
    ck("annotating that same dot-free loop makes it one",
       free_ann.get("FORM_A_swp") is True and free_ann.get("FORM_A_why") == "annotated tl.range")
    inherit = classify_src(
        "@triton.jit\ndef k():\n    for i in tl.range(0, N, B, num_stages=None):\n        acc += tl.load(p + i)\n")
    ck("num_stages=None inherits the launch value, so it counts as annotated",
       inherit.get("FORM_A_swp") is True)
    ns1 = classify_src(
        "@triton.jit\ndef k():\n    for i in tl.range(0, N, B, num_stages=1):\n        acc += tl.load(p + i)\n")
    ck("an explicit num_stages=1 does not make a dot-free loop a candidate",
       ns1.get("FORM_A_swp") is False, json.dumps(ns1))
    ck("a file with no @triton.jit is skipped entirely",
       classify_src("def k():\n    for i in range(3):\n        pass\n") == {})
    print(f"SELFTEST {'PASS' if not fails else 'FAIL'}"
          + (f" ({len(fails)} failed: {', '.join(fails)})" if fails else ""))
    return 1 if fails else 0


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--selftest" in args:
        sys.exit(selftest())
    if not args:
        print(__doc__)
        print("usage: pipeline_survey.py <root> [<root> ...]   # a tree of plain-Triton sources\n"
              "       pipeline_survey.py --selftest", file=sys.stderr)
        sys.exit(2)
    rows = walk(args)
    rows.sort(key=lambda r: (not r["FORM_A_swp"], -r["dots"], r["file"]))
    print(json.dumps(rows, indent=2))
