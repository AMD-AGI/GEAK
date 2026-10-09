"""The Triton launch probe: what it writes into a trace, and how the mapping reads it.

A Triton kernel launched from Python has no dispatcher op, so `record_shapes` never
gives it dims (MiniMax-M3 TP8: 30 of 43 unresolved semantic rows). The probe names a
`record_function` after the launch's tensor arguments; the mapping attributes the
kernel to that annotation through its launch runtime event.
"""
import json
import os
import sys
import unittest

SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPTS)
import semantic_evidence_ledger as ledger  # noqa: E402
import semantic_kernel_mapping as mapping  # noqa: E402
import triton_launch_probe as probe  # noqa: E402


class _Tensor:
    def __init__(self, shape, dtype):
        self.shape, self.dtype = shape, dtype

    def data_ptr(self):
        return 0


ARG_NAMES = ["x_ptr", "w_ptr", "out_ptr", "n_rows", "eps"]


def _launch(name, tid=2, ts=100, correlation=7, kernel=None, kernel_ts=500):
    """A probe annotation, the launch inside it, and the kernel it produced."""
    return [
        {"cat": "user_annotation", "name": name, "pid": 1, "tid": tid,
         "ts": ts, "dur": 10, "args": {"External id": 1}},
        {"cat": "cuda_runtime", "name": "hipModuleLaunchKernel", "pid": 1,
         "tid": tid, "ts": ts + 5, "dur": 1, "args": {"correlation": correlation}},
        {"cat": "kernel", "name": kernel or "_rmsnorm_kernel.kd", "ts": kernel_ts,
         "dur": 3, "args": {"correlation": correlation, "stream": 1}},
    ]


class TritonLaunchProbeTest(unittest.TestCase):
    def _name(self):
        args = (_Tensor((32, 6144), "torch.bfloat16"), _Tensor((6144,), "torch.float32"),
                _Tensor((32, 6144), "torch.bfloat16"), 32)
        return probe.annotation_name("_rmsnorm_kernel", ARG_NAMES, args,
                                     {"eps": 1e-6})

    def test_records_tensor_arguments_in_signature_order(self):
        self.assertEqual(
            self._name(),
            "geak_triton_launch::_rmsnorm_kernel::"
            "x_ptr:32x6144:BFloat16;w_ptr:6144:float;out_ptr:32x6144:BFloat16")

    def test_the_name_survives_an_unescaping_trace_export(self):
        # Kineto writes the name between quotes verbatim: one `"` breaks the file.
        name = self._name()
        self.assertNotIn('"', name)
        self.assertNotIn("\\", name)
        json.loads('{"name": "%s"}' % name)

    def test_round_trips_through_the_parser(self):
        self.assertEqual(probe.parse_annotation(self._name()), (
            "_rmsnorm_kernel", [[32, 6144], [6144], [32, 6144]],
            ["BFloat16", "float", "BFloat16"], ["x_ptr", "w_ptr", "out_ptr"]))
        zero_d = probe.annotation_name(
            "k", ["s"], (_Tensor((), "torch.float32"),), {})
        self.assertEqual(probe.parse_annotation(zero_d)[1], [[]])
        self.assertIsNone(probe.parse_annotation("execute_32_context_0"))
        self.assertIsNone(probe.parse_annotation(probe.PREFIX + "k::x:axb:float"))

    def test_install_is_inert_unless_armed(self):
        os.environ.pop("GEAK_TRITON_LAUNCH_SHAPES", None)
        self.assertIsNone(probe.install())

    def test_mapping_attributes_the_kernel_through_its_launch(self):
        evidence = mapping._triton_launch_evidence(_launch(self._name()))
        self.assertEqual(evidence[7]["kernel_name"], "_rmsnorm_kernel")
        self.assertEqual(evidence[7]["input_dims"], [[32, 6144], [6144], [32, 6144]])

    def test_a_launch_outside_the_annotation_or_on_another_thread_is_not_attributed(self):
        events = _launch(self._name())
        events[1]["ts"] = 200
        self.assertEqual(mapping._triton_launch_evidence(events), {})
        events = _launch(self._name())
        events[1]["tid"] = 3
        self.assertEqual(mapping._triton_launch_evidence(events), {})

    def test_rows_take_the_kernels_own_arguments_as_kernel_level_shape(self):
        rows, _, _, _, _ = mapping._event_rows(_launch(self._name()), {"patterns": []})
        shape = rows[0]["shape"]
        self.assertEqual(shape["source"], "triton_launch_args")
        self.assertEqual(shape["input_dims"], [[32, 6144], [6144], [32, 6144]])
        self.assertEqual(shape["operand_names"], ["x_ptr", "w_ptr", "out_ptr"])
        self.assertEqual(ledger.shape_granularity(rows[0]), "kernel")

    def test_a_kernel_with_another_name_does_not_borrow_the_annotation(self):
        rows, _, _, _, _ = mapping._event_rows(
            _launch(self._name(), kernel="_other_kernel.kd"), {"patterns": []})
        self.assertEqual(rows[0]["shape"]["source"], "unresolved")


if __name__ == "__main__":
    unittest.main()
