#!/usr/bin/env python3
"""CPU-only coverage for tools/ep_perf_driver.py: how per-op device time is read
from the EP's Chrome trace (warmup and first-op-of-inference dropping, several
partitions, malformed traces), the summary statistics, the CPU-EP comparison,
and the MatMulNBits graph builder. The ORT/EP run itself is validated on the
gfx1151 box.

    python3 -m pytest tools/tests/test_ep_perf_driver.py
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ep_perf_driver as drv  # noqa: E402

SHAPE = "m=1,n=4096,k=4096"


def _trace(durs_per_inference, shape=SHAPE, name="matmul_nbits", t0=0.0):
    """Chrome trace shaped like the HIP EP's: per op a gpu span plus a cpu span,
    and per inference H2D/Compute/D2H io spans that must be ignored."""
    events = [{"name": "process_name", "ph": "M", "pid": 0, "tid": 0, "args": {}}]
    ts = t0
    for durs in durs_per_inference:
        events.append({"name": "H2D", "cat": "io", "ph": "X", "tid": 2, "ts": ts, "dur": 5.0,
                       "args": {"bytes": 8192}})
        for d in durs:
            events.append({"name": name, "cat": "gpu", "ph": "X", "tid": 1, "ts": ts, "dur": d,
                           "args": {"shape": shape, "bytes": 1, "GB/s": 0.0}})
            events.append({"name": name, "cat": "cpu", "ph": "X", "tid": 0, "ts": ts, "dur": 3.0,
                           "args": {"shape": shape}})
            ts += d
        ts += 100.0
    return {"displayTimeUnit": "ns", "traceEvents": events}


def test_device_samples_drop_warmup_and_first_op():
    # 3 inferences x 3 ops; warmup=1 -> keep inferences 1-2, minus their first op
    t = _trace([[900, 10, 11], [800, 20, 21], [700, 30, 31]])
    assert drv.device_samples([t], "matmul_nbits", SHAPE, runs=3, warmup=1) == [20, 21, 30, 31]


def test_device_samples_sorts_by_timestamp():
    t = _trace([[900, 10, 11], [800, 20, 21]])
    t["traceEvents"].reverse()
    assert drv.device_samples([t], "matmul_nbits", SHAPE, runs=2, warmup=1) == [20, 21]


def test_device_samples_ignore_other_ops_shapes_and_tracks():
    t = _trace([[900, 10], [800, 20]])
    other = _trace([[1, 2], [3, 4]], shape="m=1,n=1024,k=4096")["traceEvents"]
    other += _trace([[1, 2], [3, 4]], name="layer_norm")["traceEvents"]
    t["traceEvents"] += other
    assert drv.device_samples([t], "matmul_nbits", SHAPE, runs=2, warmup=1) == [20]


def test_device_samples_merge_partitions():
    # Two compiled partitions, each with its own epoch (and first op) per inference.
    a = _trace([[900, 10, 11], [800, 12, 13]])
    b = _trace([[700, 40], [600, 41]], t0=5.0)
    got = drv.device_samples([a, b], "matmul_nbits", SHAPE, runs=2, warmup=1)
    assert sorted(got) == [12, 13, 41]
    got = drv.device_samples([a, b], "matmul_nbits", SHAPE, runs=2, warmup=0)
    assert sorted(got) == [10, 11, 12, 13, 40, 41]


def test_device_samples_reject_uneven_span_counts():
    t = _trace([[900, 10, 11], [800, 20]])
    with pytest.raises(ValueError, match="not a multiple"):
        drv.device_samples([t], "matmul_nbits", SHAPE, runs=2, warmup=0)


@pytest.mark.parametrize("traces", [[], [{"traceEvents": []}], [_trace([[5], [6]])]])
def test_device_samples_require_samples(traces):
    # no spans at all, or one op per inference (only its dropped first op)
    with pytest.raises(ValueError, match="no matmul_nbits GPU spans"):
        drv.device_samples(traces, "matmul_nbits", SHAPE, runs=2, warmup=0)


def test_summarize():
    s = drv.summarize([4.0, 1.0, 3.0, 2.0, 100.0])
    assert s["median_us"] == 3.0 and s["min_us"] == 1.0 and s["samples"] == 5
    assert s["p10_us"] == 1.0 and s["p90_us"] == 100.0
    assert s["mean_us"] == pytest.approx(22.0) and s["cv"] > 1


@pytest.mark.parametrize("weight_bytes,nodes", [
    (8 * 2**20, 16), (30 * 2**20, 5), (512 * 2**20, 2), (1024, 32)])
def test_nodes_for_clamps(weight_bytes, nodes):
    assert drv.nodes_for(weight_bytes, 128 * 2**20) == nodes


def test_filtered_stderr_drops_only_profiler_output(capfd):
    with drv._filtered_stderr():
        os.write(2, b"[PERF] =====\n[PERF]  matmul_nbits 1 0.1\n"
                    b"  H2D:     1 tensors, 8192 bytes (0.0 MB), 0.004 ms\n"
                    b"  Compute: 0.861 ms\n  D2H:     16 tensors, 1 bytes (0.0 MB), 0.0 ms\n"
                    b"  Total:   0.9 ms  (H2D 1% | Compute 98% | D2H 1%)\n"
                    b"[PERF SUMMARY] gpu_ms min=0.86\n"
                    b"real warning\n  Total failure in kernel\n")
    err = capfd.readouterr().err
    assert err.splitlines() == ["real warning", "  Total failure in kernel"]


# -- numpy/onnx parts (skipped where they are not installed) -------------------

def test_compare():
    np = pytest.importorskip("numpy")
    e = [np.linspace(-2, 2, 64).astype(np.float16)]
    assert drv.compare([e[0].copy()], e, 5e-2, 1e-2, 0.999) is None
    assert "outside atol" in drv.compare([e[0] + np.float16(0.5)], e, 5e-2, 1e-2, 0.999)
    assert "cosine" in drv.compare([np.zeros_like(e[0]) + np.float16(0.01)],
                                   [np.zeros_like(e[0]) + np.float16(0.012)], 1.0, 0, 1.01)
    bad = e[0].copy()
    bad[3] = np.nan
    assert "non-finite" in drv.compare([bad], e, 5e-2, 1e-2, 0.999)
    assert "shape" in drv.compare([e[0][:10]], e, 5e-2, 1e-2, 0.999)
    assert "outputs" in drv.compare([], e, 5e-2, 1e-2, 0.999)


@pytest.mark.parametrize("zp,zp_shape,zp_dtype", [
    ("none", None, None), ("u8", (256, 2), "uint8"), ("fp16", (256, 4), "float16")])
def test_matmul_nbits_case_layout(zp, zp_shape, zp_dtype):
    pytest.importorskip("numpy")
    onnx = pytest.importorskip("onnx")
    from onnx import numpy_helper
    case = {"id": f"t_{zp}", "M": 2, "K": 512, "N": 256, "zp": zp}
    model, feeds, name, shape, nodes = drv.matmul_nbits_case(case, 2**20, seed=0)
    assert (name, shape) == ("matmul_nbits", "m=2,n=256,k=512")
    assert feeds["X"].shape == (1, 2, 512) and str(feeds["X"].dtype) == "float16"
    g = model.graph
    assert len(g.node) == nodes and len(g.output) == nodes
    assert len({n.input[1] for n in g.node}) == nodes  # distinct weights per node
    assert all(set(n.input[:1]) == {"X"} for n in g.node)
    inits = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    assert inits["W0"].shape == (256, 4, 64) and inits["S0"].shape == (256, 4)
    if zp_shape:
        assert inits["Z0"].shape == zp_shape and str(inits["Z0"].dtype) == zp_dtype
        assert len(g.node[0].input) == 4
    else:
        assert len(g.node[0].input) == 3
    attrs = {a.name: onnx.helper.get_attribute_value(a) for a in g.node[0].attribute}
    assert attrs == {"K": 512, "N": 256, "bits": 4, "block_size": 128, "accuracy_level": 4}
    assert model.ir_version == 10


def test_matmul_nbits_case_is_deterministic_and_validates():
    pytest.importorskip("numpy")
    pytest.importorskip("onnx")
    case = {"id": "d", "M": 1, "K": 256, "N": 64, "nodes": 3}
    a = drv.matmul_nbits_case(case, 2**20, seed=1)
    b = drv.matmul_nbits_case(case, 2**20, seed=1)
    assert a[4] == 3 and a[0].SerializeToString() == b[0].SerializeToString()
    for bad in ({"bits": 8}, {"zp": "int4"}):
        with pytest.raises(ValueError):
            drv.matmul_nbits_case({**case, **bad}, 2**20, seed=1)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
