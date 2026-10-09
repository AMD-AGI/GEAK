import contextlib
import os
import sys
import tempfile
import types
import unittest
from unittest import mock


SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPTS)
import semantic_runtime_capture as capture


class _Logger(object):
    def __init__(self):
        self.calls = []

    def begin_callable(self, target, args, kwargs, callable_obj):
        self.calls.append((
            "begin", target, tuple(args), dict(kwargs), callable_obj))
        return target

    def end_callable(self, entry, args, kwargs, output):
        self.calls.append(("end", entry, args, kwargs, output))


class SemanticRuntimeCaptureTest(unittest.TestCase):
    def test_all_layer_scope_ignores_representative_shape_filter(self):
        logger = capture.SemanticRuntimeLogger.__new__(
            capture.SemanticRuntimeLogger)
        logger.layer_scopes = True
        logger.layers = {3}
        logger.phases = {"DECODE"}
        logger.max_forwards = 1
        logger._bucket_forwards = {}
        logger._context = {
            "phase": "DECODE", "batch_size": 4, "input_tokens": 4}
        with mock.patch.object(logger, "active", return_value=True), \
                mock.patch.object(capture, "_stream_capturing",
                                  return_value=False):
            self.assertEqual(logger._layer_scope_source(17), "eager")
        self.assertTrue(capture._MAIN_LAYER_RE.search("model.layers.17"))
        self.assertFalse(capture._MAIN_LAYER_RE.search("model.layers.17.mlp"))

    def test_prefill_phase_filter_accepts_extend_runtime_mode(self):
        logger = capture.SemanticRuntimeLogger.__new__(
            capture.SemanticRuntimeLogger)
        logger.layers = {3}
        logger.phases = {"PREFILL"}
        logger.require_profiler = False
        logger._profile_seen = False
        logger.max_forwards = 1
        logger._bucket_forwards = {}
        logger._context = {
            "phase": "EXTEND", "batch_size": 1, "input_tokens": 8}
        with mock.patch.object(logger, "active", return_value=True):
            self.assertTrue(logger._allowed(3))

    def test_warmup_does_not_consume_bucket_before_profiler_starts(self):
        logger = capture.SemanticRuntimeLogger.__new__(
            capture.SemanticRuntimeLogger)
        logger.layers = {3}
        logger.phases = {"DECODE"}
        logger.require_profiler = True
        logger._profile_seen = False
        logger.max_forwards = 1
        logger._bucket_forwards = {}
        logger._context = {
            "phase": "DECODE", "batch_size": 4, "input_tokens": 4}
        with mock.patch.object(logger, "active", return_value=True), \
                mock.patch.object(
                    capture, "_profiler_active", return_value=False):
            self.assertFalse(logger._allowed(3))
            self.assertEqual(logger._bucket_forwards, {})
        with mock.patch.object(logger, "active", return_value=True), \
                mock.patch.object(
                    capture, "_profiler_active", return_value=True):
            self.assertTrue(logger._allowed(3))
            self.assertTrue(logger._profile_seen)
        with mock.patch.object(logger, "active", return_value=True), \
                mock.patch.object(
                    capture, "_profiler_active", return_value=False):
            self.assertTrue(logger._allowed(3))

    def test_targeted_callable_is_monkeypatched_and_logged(self):
        module = types.SimpleNamespace(launcher=lambda value: value + 1)
        logger = _Logger()
        capture._PATCHED_CALLABLES.clear()
        with mock.patch.dict(
                os.environ,
                {"GEAK_SEMANTICS_CALLABLE_TARGETS": "pkg.mod:launcher"}), \
                mock.patch.object(
                    capture.importlib, "import_module", return_value=module), \
                mock.patch.object(capture, "get_logger", return_value=logger):
            capture._install_callable_probes()
            self.assertEqual(module.launcher(4), 5)
        self.assertEqual(logger.calls[0][0:3], (
            "begin", "pkg.mod:launcher", (4,)))
        self.assertEqual(logger.calls[1][-1], 5)

    def test_dispatcher_schema_record_preserves_argument_names_and_mutation(self):
        class Alias(object):
            is_write = True
            before_set = {"a"}
            after_set = {"a"}

            def __str__(self):
                return "Tensor(a!)"

        class Argument(object):
            def __init__(self, name, value_type, default=False, alias=None):
                self.name = name
                self.type = value_type
                self.kwarg_only = False
                self.default_value = None
                self.alias_info = alias
                self._default = default

            def has_default_value(self):
                return self._default

        class Schema(object):
            name = "custom::quant"
            overload_name = ""
            arguments = [
                Argument("out", "Tensor", alias=Alias()),
                Argument("input", "Tensor"),
                Argument("scale", "Tensor"),
                Argument("shuffle", "bool", default=True),
            ]
            returns = []

            def __str__(self):
                return "custom::quant(Tensor(a!) out, Tensor input, Tensor scale, bool shuffle=False) -> ()"

        record = capture._schema_record(Schema())
        self.assertEqual(record["name"], "custom::quant")
        self.assertEqual(
            [argument["name"] for argument in record["arguments"]],
            ["out", "input", "scale", "shuffle"])
        self.assertTrue(
            record["arguments"][0]["alias_info"]["is_write"])
        self.assertTrue(record["arguments"][3]["has_default"])



class TestBucketAccountingUnderAWarmupDriver(unittest.TestCase):
    """The bucket counter must not be spent by forwards the recorder refused.

    `_allowed` blocks recording until torch.profiler is live, so a warmup forward can
    leave no trace marker to match. `mark_forward` used to increment anyway, so a driver
    that warms up BEFORE opening the profile window -- bench_e2e.sh, by design -- arrived
    at the capture window with every bucket already full and wrote an EMPTY shape log.
    Empty is indistinguishable downstream from "this phase has no shapes", so this failed
    silently; observed on the first vLLM eager probe.
    """

    def _logger(self, tmp, require_profiler=True):
        env = {
            "GEAK_SEMANTICS_CAPTURE": "1",
            "GEAK_SEMANTICS_SHAPE_LOG": os.path.join(tmp, "shape.jsonl"),
            "GEAK_SEMANTICS_FORWARDS_PER_BUCKET": "1",
            "GEAK_SEMANTICS_REQUIRE_PROFILER": "1" if require_profiler else "0",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            return capture.SemanticRuntimeLogger()

    def test_warmup_forwards_do_not_consume_the_bucket(self):
        with tempfile.TemporaryDirectory() as tmp:
            logger = self._logger(tmp)
            logger.set_context("DECODE", 8, 8)
            for _ in range(50):                 # a whole warmup + timed-repeat phase
                logger.mark_forward()
            self.assertEqual(logger._bucket_forwards, {},
                             "no bucket may be spent before the profiler is live")

            # Profiler comes up; the very next forward must still be admitted.
            logger._profile_seen = True
            logger.mark_forward()
            self.assertEqual(logger._bucket_forwards[("DECODE", 8, 8)], 1)

    def test_the_limit_still_applies_once_the_profiler_is_live(self):
        with tempfile.TemporaryDirectory() as tmp:
            logger = self._logger(tmp)
            logger._profile_seen = True
            logger.set_context("DECODE", 8, 8)
            logger.mark_forward()
            logger.mark_forward()
            self.assertEqual(logger._bucket_forwards[("DECODE", 8, 8)], 2)

    def test_counting_is_unconditional_when_no_profiler_is_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            logger = self._logger(tmp, require_profiler=False)
            logger.set_context("EXTEND", 2, 4096)
            logger.mark_forward()
            self.assertEqual(logger._bucket_forwards[("EXTEND", 2, 4096)], 1)


class _ScopeLogger(object):
    def __init__(self):
        self.scopes = []

    def layer_scope(self, layer_id, op_path):
        self.scopes.append((layer_id, op_path))
        return contextlib.nullcontext()


class _Layer(object):
    def forward_entry(self, value):
        return value + 1


class _Model(object):
    def __init__(self):
        self.layer0, self.layer1, self.mlp = _Layer(), _Layer(), object()

    def named_modules(self):
        return [("", self), ("model.layers.0", self.layer0),
                ("model.layers.1", self.layer1),
                ("model.layers.0.mlp", self.mlp)]


class LayerEntryCallablesTest(unittest.TestCase):
    def _wrap(self, env, module=None):
        model, logger = _Model(), _ScopeLogger()
        clean = {key: value for key, value in os.environ.items()
                 if not key.startswith("GEAK_SEMANTICS_LAYER_ENTRY_")}
        clean.update(env)
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.dict(os.environ, clean, clear=True))
            if module is not None:
                stack.enter_context(mock.patch.object(
                    capture.importlib, "import_module", return_value=module))
            wrapped = capture._wrap_layer_entry_functions(model, logger)
        return model, logger, wrapped

    def test_nothing_beyond_forward_is_wrapped_by_default(self):
        model, logger, wrapped = self._wrap({})
        self.assertEqual(wrapped, 0)
        self.assertEqual(model.layer0.forward_entry(1), 2)
        self.assertEqual(logger.scopes, [])

    def test_configured_method_marks_each_main_layer(self):
        model, logger, wrapped = self._wrap(
            {"GEAK_SEMANTICS_LAYER_ENTRY_METHODS": "forward_entry"})
        self.assertEqual(wrapped, 2)
        self.assertEqual(model.layer1.forward_entry(1), 2)
        self.assertEqual(logger.scopes, [(1, "model.layers.1")])

    def test_configured_function_is_replaced_in_its_calling_module(self):
        module = types.SimpleNamespace(run_layer=lambda layer, value: value * 2)
        model, logger, _ = self._wrap(
            {"GEAK_SEMANTICS_LAYER_ENTRY_FUNCTIONS": "pkg.model:run_layer"},
            module)
        self.assertEqual(module.run_layer(model.layer0, 3), 6)
        self.assertEqual(module.run_layer(object(), 3), 6)
        self.assertEqual(logger.scopes, [(0, "model.layers.0")])

    def test_misconfigured_entries_fail_loudly(self):
        with self.assertRaises(RuntimeError):
            self._wrap({"GEAK_SEMANTICS_LAYER_ENTRY_METHODS": "no_such_method"})
        with self.assertRaises(RuntimeError):
            self._wrap({"GEAK_SEMANTICS_LAYER_ENTRY_FUNCTIONS": "no_colon"})
        with self.assertRaises(RuntimeError):
            self._wrap({"GEAK_SEMANTICS_LAYER_ENTRY_FUNCTIONS": "pkg.model:missing"},
                       types.SimpleNamespace())

    def test_nested_scope_for_the_same_layer_emits_one_marker(self):
        markers = []

        class _Record(object):
            def __init__(self, name):
                self.name = name

            def __enter__(self):
                markers.append(self.name)

            def __exit__(self, *exc):
                return False

        fake_torch = types.SimpleNamespace(
            profiler=types.SimpleNamespace(record_function=_Record))
        logger = capture.SemanticRuntimeLogger.__new__(
            capture.SemanticRuntimeLogger)
        logger._scope_state = capture.threading.local()
        logger._context = {"phase": "DECODE", "batch_size": 4, "input_tokens": 4}
        with mock.patch.dict(sys.modules, {"torch": fake_torch}), \
                mock.patch.object(logger, "_layer_scope_source",
                                  return_value="capture"):
            with logger.layer_scope(3, "model.layers.3"):
                with logger.layer_scope(3, "model.layers.3"):
                    pass
                with logger.layer_scope(4, "model.layers.4"):
                    pass
            with logger.layer_scope(3, "model.layers.3"):
                pass
        self.assertEqual([m.split("|layer=")[1].split("|")[0] for m in markers],
                         ["3", "4", "3"])


if __name__ == "__main__":
    unittest.main()
