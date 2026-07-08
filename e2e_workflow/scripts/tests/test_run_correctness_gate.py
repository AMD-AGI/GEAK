"""Tests for the fail-closed graph-replay correctness gate (harness_lib.run_correctness).

Locks the h2 paged_attention fix: a kernel that deploys under a CUDA graph but is only checked eagerly
must NOT be allowed to pass. Covers:
  * fail-closed when no / too-few replay cases are supplied for a graph-deploy regime;
  * NOT required when the regime is enforce_eager (deployment_graph_mode == False);
  * the gate actually CATCHES a replay-only failure that eager checks miss (stale-under-replay), proving
    it would have caught the OOB class that reached e2e.
The replay-catch test needs CUDA and is skipped on eager-only boxes (there the e2e gate still catches it).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import harness_lib as h  # noqa: E402

try:
    import torch
    HAVE_TORCH = True
    HAVE_CUDA = torch.cuda.is_available()
except Exception:
    torch = None
    HAVE_TORCH = False
    HAVE_CUDA = False

TOL = 2e-2
DEV = "cuda" if HAVE_CUDA else "cpu"


def _gemm(args):
    a, b = args
    return a @ b.t()


def _tiny_case(seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    a = torch.randn(8, 16, generator=g, device=DEV)
    b = torch.randn(8, 16, generator=g, device=DEV)
    return {"args": (a, b), "ref": a @ b.t(), "sig": f"m8k16_{seed}"}


def _rand_shapes():
    def mk(rng):
        a = torch.randn(8, 16, generator=rng, device=DEV)
        b = torch.randn(8, 16, generator=rng, device=DEV)
        return (a, b)
    return [{"sig": "m8k16", "make_inputs": mk}]


def test_boundary_decode_seq_lens_spans_block_and_partition():
    geo = {"block_size": 16, "partition_size": 256}
    sl = h.boundary_decode_seq_lens(geo, ctx_max=2048)
    for x in (1, 15, 16, 17, 255, 256, 257, 2048):
        assert x in sl, (x, sl)
    assert all(1 <= x <= 2048 for x in sl)


def _expect_harness_incomplete(replay):
    """Missing / too-few replay bundle for a graph-deploy kernel must RAISE HarnessIncompleteError
    (a UT-regeneration signal), NOT return a kernel-correctness FAIL."""
    raised = None
    try:
        h.run_correctness({"cuda_graph": True}, eager_cases=[_tiny_case(1)],
                          baseline_call=_gemm, current_call=_gemm,
                          random_shapes=_rand_shapes(), tol=TOL, replay=replay, draws=1)
    except h.HarnessIncompleteError as e:
        raised = e
    assert raised is not None, "expected HarnessIncompleteError (regenerate signal), got a normal return"


def test_missing_replay_raises_regenerate_signal():
    if not HAVE_CUDA:
        print("skip (no cuda)"); return
    _expect_harness_incomplete(replay=None)


def test_single_replay_case_raises_regenerate_signal():
    if not HAVE_CUDA:
        print("skip (no cuda)"); return
    _expect_harness_incomplete(replay={"fill": lambda c: None, "run": lambda: None,
                                       "read_out": lambda: None, "cases": [_tiny_case(1)],
                                       "capture_idx": 0})


def test_not_required_when_enforce_eager():
    if not HAVE_CUDA:
        print("skip (no cuda)"); return
    ok, rep = h.run_correctness({"enforce_eager": True, "cuda_graph": True},
                                eager_cases=[_tiny_case(1)], baseline_call=_gemm, current_call=_gemm,
                                random_shapes=_rand_shapes(), tol=TOL, replay=None, draws=1)
    assert ok is True, rep
    assert "graph_replay" not in rep, rep


def test_gate_catches_replay_only_failure():
    """A kernel correct eagerly but STALE under replay (run() doesn't recompute) must FAIL — this is the
    replay-only failure class (same class as the paged-attn OOB) that eager checks cannot see."""
    if not HAVE_CUDA:
        print("skip (no cuda)"); return
    # eager legs: a correct gemm. replay legs: a buggy static kernel that only computes on capture and
    # returns a persistent (stale) buffer for later, differently-valued cases.
    big = _tiny_case(seed=10)
    small = {"args": (big["args"][0] * -1.0, big["args"][1]),
             "ref": (big["args"][0] * -1.0) @ big["args"][1].t(), "sig": "m8k16_neg"}
    static_out = torch.empty(8, 8, device=DEV)
    state = {"a": None, "b": None}

    def fill(c):
        state["a"], state["b"] = c["args"]
    captured = {"done": False}

    def run():
        if not captured["done"]:               # only computes on the capture pass -> stale afterwards
            static_out.copy_(state["a"] @ state["b"].t()); captured["done"] = True
    def read_out():
        return static_out

    replay = {"fill": fill, "run": run, "read_out": read_out,
              "cases": [big, small], "capture_idx": 0}
    ok, rep = h.run_correctness({"cuda_graph": True}, eager_cases=[_tiny_case(1)],
                                baseline_call=_gemm, current_call=_gemm,
                                random_shapes=_rand_shapes(), tol=TOL, replay=replay, draws=1)
    assert ok is False, rep
    gr = rep["graph_replay"]
    assert any(not r["correct"] for r in gr), gr   # the stale 'small' case must be caught


def test_compile_leg_absent_when_eager():
    if not HAVE_CUDA:
        print("skip (no cuda)"); return
    ok, rep = h.run_correctness({"enforce_eager": True}, eager_cases=[_tiny_case(1)],
                                baseline_call=_gemm, current_call=_gemm,
                                random_shapes=_rand_shapes(), tol=TOL, replay=None, draws=1)
    assert ok is True and "compile_parity" not in rep, rep


def _with_compiled_op(stub):
    """Temporarily replace h.compiled_op so the compile-parity logic is tested WITHOUT invoking inductor."""
    orig = h.compiled_op
    h.compiled_op = stub
    return orig


def test_compile_parity_pass_for_opaque_op():
    if not HAVE_TORCH:
        print("skip (no torch)"); return
    # opaque op: compiled == eager (compiled_op returns fn unchanged) -> parity passes.
    cur = lambda args: _gemm(args)                   # fresh callable (no leaked attrs)
    orig = _with_compiled_op(lambda fn, regime, **kw: fn)
    try:
        ok, per = h._compile_parity(cur, [_tiny_case(1)], {"compile": "torch_compile"}, TOL)
    finally:
        h.compiled_op = orig
    assert ok is True and all(r["correct"] for r in per), per


def test_compile_parity_fail_on_fusion_drift():
    if not HAVE_TORCH:
        print("skip (no torch)"); return
    # fused op whose compiled result DRIFTS from eager -> real correctness FAIL.
    cur = lambda args: _gemm(args)
    def _drift(fn, regime, **kw):
        return lambda args: fn(args) + 100.0        # simulate fusion changing the numerics
    orig = _with_compiled_op(_drift)
    try:
        ok, per = h._compile_parity(cur, [_tiny_case(1)], {"compile": "torch_compile"}, TOL)
    finally:
        h.compiled_op = orig
    assert ok is False and any(not r["correct"] for r in per), per


def test_compile_error_is_non_fatal_note():
    if not HAVE_TORCH:
        print("skip (no torch)"); return
    # compiled_op could not build (bare-op fullgraph compile failed) -> surfaced, NOT auto-rejected.
    cur = lambda args: _gemm(args)
    def _err(fn, regime, **kw):
        setattr(fn, "_geak_compile_error", "Unsupported: graph break in bare op")
        return fn
    orig = _with_compiled_op(_err)
    try:
        ok, per = h._compile_parity(cur, [_tiny_case(1)], {"compile": "torch_compile"}, TOL)
    finally:
        h.compiled_op = orig
    assert ok is True, per
    assert "NON-FATAL" in per[0]["note"], per


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print("PASS", fn.__name__)
    print(f"\nAll {len(fns)} run_correctness gate tests passed.")
