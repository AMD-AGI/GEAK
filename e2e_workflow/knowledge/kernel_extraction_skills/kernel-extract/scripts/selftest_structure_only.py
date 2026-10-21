#!/usr/bin/env python3
"""CPU self-test for the structure_only capture policy and the gates that read it.

Exercises the whole loop on a toy routed op: capture -> the gate REFUSES the unregenerated
file -> regenerate against the installed callable -> the gate accepts it, and the operands
in the regenerated file are byte-identical to what the fixed seeds produce. This is an
explicit legacy-export compatibility test; it removes all temporary tensor files on exit.

    python3 selftest_structure_only.py        # no GPU required
"""
import atexit
import os
import subprocess
import sys
from pathlib import Path
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "scripts"))

import torch  # noqa: E402

import capture_shapes as cs  # noqa: E402
import harness_lib as h  # noqa: E402


def routed_op(hidden, topk_ids, w1):
    """A toy whose OUTPUT depends on routing content, so a redrawn topk_ids is detectable."""
    return (hidden * topk_ids.float().sum(dim=-1, keepdim=True)) @ w1


class _Mod:
    __name__ = "_selftest_mod"


def _run(out_dir):
    fails = []
    # regen_reference resolves the target in a fresh process, so the toy op has to be
    # importable there too -- exactly as the real target is importable in the serving image.
    with open(os.path.join(out_dir, "_selftest_mod.py"), "w") as f:
        f.write("import torch\n\n\n" + __import__("inspect").getsource(routed_op))
    env = dict(os.environ, PYTHONPATH=out_dir + os.pathsep + os.environ.get("PYTHONPATH", ""))
    sys.path.insert(0, out_dir)
    import _selftest_mod  # noqa: E402

    os.environ["CAPTURE_PERSIST_POLICY"] = "structure_only"
    os.environ["CAPTURE_STRUCTURAL_OPERANDS"] = "topk_ids"
    os.environ["CAPTURE_SYNTH_MIN_BYTES"] = "4KiB"
    cs._STATE["installed"] = False
    cs.install(target="_selftest_mod:routed_op", out_dir=out_dir, max_cases=2)

    hidden = torch.randn(64, 2048)
    topk_ids = torch.randint(0, 8, (64, 4))
    w1 = torch.randn(2048, 512)
    _selftest_mod.routed_op(hidden, topk_ids=topk_ids, w1=w1)
    cs._flush()

    task = os.path.dirname(out_dir.rstrip("/")) or out_dir
    io_path = os.path.join(cs._STATE["out_dir"], "reference_io.pt")
    blob = h.load_reference_io(io_path)

    if blob.get("persist_policy") != "structure_only":
        fails.append("policy not recorded: %r" % blob.get("persist_policy"))
    if blob.get("oracle_status") != "needs_regen":
        fails.append("oracle_status not needs_regen: %r" % blob.get("oracle_status"))

    rec = blob["records"][0]
    if not (isinstance(rec["kwargs"]["w1"], dict) and rec["kwargs"]["w1"].get("__synth__")):
        fails.append("bulk float w1 was not reduced to a descriptor")
    if not (isinstance(rec["kwargs"]["topk_ids"], dict)
            and rec["kwargs"]["topk_ids"].get("__tensor__")):
        fails.append("structural topk_ids was NOT kept verbatim")
    if not torch.equal(rec["kwargs"]["topk_ids"]["data"], topk_ids):
        fails.append("structural topk_ids content changed")
    if not (isinstance(rec["output"], dict) and rec["output"].get("__needs_regen__")):
        fails.append("output was not marked needs_regen")

    # The seed IS the operand: materializing twice must be bit-identical.
    a = cs.materialize_synth(rec["kwargs"]["w1"])
    b = cs.materialize_synth(rec["kwargs"]["w1"])
    if not torch.equal(a, b):
        fails.append("materialize_synth is not reproducible from the recorded seed")
    if list(a.shape) != list(w1.shape) or a.dtype != w1.dtype:
        fails.append("materialized operand has the wrong shape/dtype")

    # The harness must REFUSE the placeholder rather than hand it to a correctness gate.
    try:
        h.reconstruct_captured(rec["output"])
        fails.append("reconstruct_captured accepted a needs_regen placeholder")
    except ValueError:
        pass

    # Gate 1: unregenerated capture is fatal.
    rc = subprocess.call([sys.executable, os.path.join(HERE, "oracle_power.py"), io_path],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
    if rc != 1:
        fails.append("oracle_power accepted an unregenerated capture (rc=%d)" % rc)

    # Regenerate, then the same gate must pass and the structural operand must survive.
    regen = os.path.join(cs._STATE["out_dir"], "reference_io.regen.pt")
    rc = subprocess.call([sys.executable, os.path.join(HERE, "regen_reference.py"),
                          io_path, "--out", regen, "--device", "cpu"], env=env)
    if rc != 0:
        fails.append("regen_reference failed (rc=%d)" % rc)
    else:
        rb = h.load_reference_io(regen)
        rr = rb["records"][0]
        if not torch.equal(rr["kwargs"]["topk_ids"]["data"], topk_ids):
            fails.append("regenerated file lost the structural routing")
        if not torch.equal(rr["kwargs"]["w1"]["data"], a):
            fails.append("regenerated file did not persist the materialized operand")
        want = routed_op(cs.materialize_synth(rec["args"][0]), topk_ids, a)
        if not torch.allclose(rr["output"]["data"], want):
            fails.append("regenerated output does not match the installed callable")
        rc = subprocess.call([sys.executable, os.path.join(HERE, "oracle_power.py"), regen],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
        if rc != 0:
            fails.append("oracle_power rejected a regenerated capture (rc=%d)" % rc)

        # A structural operand that is declared but absent must be fatal, not a shrug.
        rc = subprocess.call([sys.executable, os.path.join(HERE, "oracle_power.py"), regen,
                              "--structural-operand", "block_table"],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
        if rc != 1:
            fails.append("oracle_power missed an absent declared structural operand "
                         "(rc=%d)" % rc)

    for f in fails:
        print("FAIL  " + f, file=sys.stderr)
    print("%s  structure_only self-test (%d checks failed)"
          % ("FAIL" if fails else "OK", len(fails)))
    return 1 if fails else 0


def main():
    # This explicitly tests legacy export; even those small tensors are temporary.
    with tempfile.TemporaryDirectory(prefix="structonly-") as out_dir:
        try:
            return _run(out_dir)
        finally:
            # Otherwise the capture hook recreates the deleted directory at exit.
            atexit.unregister(cs._flush)
            if out_dir in sys.path:
                sys.path.remove(out_dir)


if __name__ == "__main__":
    sys.exit(main())
