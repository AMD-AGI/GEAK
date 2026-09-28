#!/usr/bin/env python3
"""GEAK Semantics 1.2 runtime metadata and profiler-marker injection.

This module is copied into the serving runtime and installed after ModelRunner
loads the model.  It records metadata only and wraps each selected module call
in a profiler record_function marker so launched kernels can be proven to be
contained by the wrapper that supplied the Shape.
"""
import json
import functools
import importlib
import contextlib
import inspect
import os
import re
import sys
import threading


_LAYER_RE = re.compile(
    r"(?:^|\.)(?:layers|h|blocks)\.(\d+)(?:\.|$)", re.IGNORECASE)
_MAIN_LAYER_RE = re.compile(
    r"(?:^|\.)(?:layers|h|blocks)\.(\d+)$", re.IGNORECASE)
_TRUE = ("1", "true", "True", "TRUE", "yes", "on")
_LOGGER = None
_INSTALLED_CLASSES = set()
_PATCHED_CALLABLES = set()
_GRAPH_CAPTURE_PROFILER_INSTALLED = False
_OPERATOR_SCHEMA_MANIFEST_DUMPED = False


def _flag(name, default="0"):
    return os.environ.get(name, default) in _TRUE


def _csv_int(name):
    value = os.environ.get(name, "")
    return set(int(item) for item in value.split(",") if item.strip())


def _csv_upper(name):
    return set(
        item.strip().upper()
        for item in os.environ.get(name, "").split(",")
        if item.strip())


def _canonical_phase(value):
    value = str(value or "").upper()
    if value in ("EXTEND", "PREFILL", "PROMPT"):
        return "PREFILL"
    if value in ("DECODE", "GENERATION"):
        return "DECODE"
    return value


def _profiler_active():
    """Return whether record_function ranges can reach the active trace."""
    try:
        import torch
        probe = getattr(torch.autograd, "_profiler_enabled", None)
        return bool(probe and probe())
    except Exception:
        return False


def _distributed_rank():
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return int(dist.get_rank())
    except Exception:
        pass
    return 0


def _graph_capture_trace_path(rank):
    path = os.environ.get("GEAK_SEMANTICS_GRAPH_CAPTURE_TRACE", "")
    if not path:
        return ""
    try:
        return path.format(rank=rank)
    except (KeyError, ValueError):
        return path


def _operator_targets():
    return [
        item.strip()
        for item in os.environ.get(
            "GEAK_SEMANTICS_OPERATOR_TARGETS", "").split(",")
        if item.strip()]


def _value_or_call(value, default=None):
    if value is None:
        return default
    try:
        return value() if callable(value) else value
    except Exception:
        return default


def _alias_info(alias):
    if alias is None:
        return None
    return {
        "text": str(alias),
        "is_write": bool(_value_or_call(
            getattr(alias, "is_write", None), False)),
        "before_set": sorted(str(item) for item in (
            _value_or_call(getattr(alias, "before_set", None), []) or [])),
        "after_set": sorted(str(item) for item in (
            _value_or_call(getattr(alias, "after_set", None), []) or [])),
    }


def _schema_argument(argument, index):
    has_default = bool(_value_or_call(
        getattr(argument, "has_default_value", None), False))
    default = getattr(argument, "default_value", None)
    return {
        "index": index,
        "name": str(getattr(argument, "name", "arg_%d" % index)),
        "type": str(getattr(argument, "type", "unknown")),
        "kwarg_only": bool(getattr(argument, "kwarg_only", False)),
        "has_default": has_default,
        "default": str(default) if has_default else None,
        "alias_info": _alias_info(getattr(argument, "alias_info", None)),
    }


def _schema_record(schema):
    name = str(getattr(schema, "name", ""))
    overload = str(getattr(schema, "overload_name", "") or "")
    return {
        "name": name,
        "overload_name": overload,
        "qualified_name": name + ("." + overload if overload else ""),
        "schema": str(schema),
        "arguments": [
            _schema_argument(argument, index)
            for index, argument in enumerate(
                list(getattr(schema, "arguments", []) or []))],
        "returns": [
            _schema_argument(argument, index)
            for index, argument in enumerate(
                list(getattr(schema, "returns", []) or []))],
    }


def dump_operator_schema_manifest():
    """Dump registered dispatcher schemas after runtime/model initialization.

    The pre-capture operator target list is an observational prior only.  The
    complete registered schema inventory is exported so operators actually
    observed in the graph-construction trace can still be named even when the
    prior missed them.
    """
    global _OPERATOR_SCHEMA_MANIFEST_DUMPED
    if _OPERATOR_SCHEMA_MANIFEST_DUMPED:
        return
    path = os.environ.get(
        "GEAK_SEMANTICS_OPERATOR_SCHEMA_MANIFEST", "")
    if not path:
        return
    rank = _distributed_rank()
    expected_rank = int(os.environ.get("GEAK_SEMANTICS_RANK", "0"))
    if rank != expected_rank:
        return
    import torch
    getter = getattr(torch._C, "_jit_get_all_schemas", None)
    if getter is None:
        raise RuntimeError(
            "runtime torch does not expose _jit_get_all_schemas")
    schemas = sorted(
        (_schema_record(schema) for schema in getter()),
        key=lambda item: (
            item["name"], item["overload_name"], item["schema"]))
    requested = sorted(set(_operator_targets()))
    available = {item["name"] for item in schemas}
    document = {
        "schema_version": 1,
        "producer": "geak.semantic_runtime_capture",
        "rank": rank,
        "capture_timing": "after_model_and_runtime_operator_registration",
        "operator_targets_are_observational_prior_only": True,
        "requested_operators": requested,
        "requested_operator_resolution": [
            {
                "operator": name,
                "registered": name in available,
                "schema_count": sum(
                    item["name"] == name for item in schemas),
            }
            for name in requested],
        "schema_count": len(schemas),
        "schemas": schemas,
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = "%s.tmp.%d" % (path, os.getpid())
    with open(temporary, "w") as fh:
        json.dump(document, fh, indent=2, sort_keys=True)
    os.replace(temporary, path)
    _OPERATOR_SCHEMA_MANIFEST_DUMPED = True
    sys.stderr.write(
        "[GEAK_SEMANTICS] exported %d operator schemas to %s\n" %
        (len(schemas), path))


def install_graph_capture_profiler():
    """Export a rank-0 trace of CUDA/HIP graph construction.

    SGLang's ``--enable-profile-cuda-graph`` profiler normally prints only
    aggregate tables and a memory snapshot.  Semantics needs the Chrome trace
    containing GEAK record_function markers and the kernel launches nested in
    them.  Profile rank 0 only, avoid memory-history recording, and leave graph
    replay profiling to the normal clean workload profiler.
    """
    global _GRAPH_CAPTURE_PROFILER_INSTALLED
    if _GRAPH_CAPTURE_PROFILER_INSTALLED:
        return
    if not os.environ.get("GEAK_SEMANTICS_GRAPH_CAPTURE_TRACE"):
        return
    try:
        import torch
        from torch.profiler import ProfilerActivity, profile
        from sglang.srt.model_executor.cuda_graph_runner import CudaGraphRunner

        def _init_profile_context_and_memory_record(self):
            if _distributed_rank() != int(os.environ.get(
                    "GEAK_SEMANTICS_RANK", "0")):
                return contextlib.nullcontext(None)
            return profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                with_stack=False,
            )

        def _post_process_after_profile(self, prof_context):
            rank = _distributed_rank()
            if prof_context is None or rank != int(os.environ.get(
                    "GEAK_SEMANTICS_RANK", "0")):
                return
            path = _graph_capture_trace_path(rank)
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            prof_context.export_chrome_trace(path)
            sys.stderr.write(
                "[GEAK_SEMANTICS] graph-capture trace exported to %s\n" % path)

        CudaGraphRunner._init_profile_context_and_memory_record = (
            _init_profile_context_and_memory_record)
        CudaGraphRunner._post_process_after_profile = (
            _post_process_after_profile)
        _GRAPH_CAPTURE_PROFILER_INSTALLED = True
        sys.stderr.write(
            "[GEAK_SEMANTICS] installed rank-0 graph-capture profiler export\n")
    except Exception as exc:
        sys.stderr.write(
            "[GEAK_SEMANTICS] graph-capture profiler install failed: %s\n" % exc)
        raise


def _phase_of(forward_batch):
    try:
        mode = forward_batch.forward_mode
    except Exception:
        return "UNKNOWN", -1, -1
    phase = "UNKNOWN"
    for method, candidate in (
            ("is_decode", "DECODE"),
            ("is_target_verify", "TARGET_VERIFY"),
            ("is_extend", "EXTEND"),
            ("is_prefill", "PREFILL"),
            ("is_draft_extend", "DRAFT_EXTEND")):
        try:
            if getattr(mode, method, lambda: False)():
                phase = candidate
                break
        except Exception:
            pass
    if phase == "UNKNOWN":
        phase = str(getattr(mode, "name", mode)).upper()
    batch_size = -1
    for name in ("batch_size", "bs"):
        value = getattr(forward_batch, name, None)
        if isinstance(value, int):
            batch_size = value
            break
    input_tokens = -1
    try:
        input_ids = getattr(forward_batch, "input_ids", None)
        if input_ids is not None:
            input_tokens = int(input_ids.numel())
    except Exception:
        pass
    if batch_size == -1:
        try:
            batch_size = int(forward_batch.seq_lens.shape[0])
        except Exception:
            pass
    return phase, batch_size, input_tokens


def _layer_id(path):
    match = _LAYER_RE.search(path or "")
    return int(match.group(1)) if match else -1


def _op_path(path):
    if path.startswith("layers."):
        return "model." + path
    return path


def _metadata(value, aliases=None):
    import torch
    aliases = aliases if aliases is not None else {}
    if isinstance(value, torch.Tensor):
        identity = id(value)
        if identity not in aliases:
            aliases[identity] = "tensor-%d" % len(aliases)
        try:
            stride = list(value.stride())
        except Exception:
            stride = None
        return {
            "kind": "tensor",
            "alias_id": aliases[identity],
            "shape": list(value.shape),
            "dtype": str(value.dtype).replace("torch.", ""),
            "device": str(value.device),
            "stride": stride,
            "contiguous": bool(value.is_contiguous()),
            "requires_grad": bool(value.requires_grad),
        }
    if isinstance(value, tuple):
        return {"kind": "tuple", "items": [
            _metadata(item, aliases) for item in value]}
    if isinstance(value, list):
        return {"kind": "list", "items": [
            _metadata(item, aliases) for item in value]}
    if isinstance(value, dict):
        return {"kind": "dict", "items": {
            str(key): _metadata(item, aliases)
            for key, item in value.items()}}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return {
            "kind": "scalar", "type": type(value).__name__,
            "value": value,
        }
    return {
        "kind": "object",
        "type": "%s.%s" % (
            type(value).__module__, type(value).__name__),
    }


class SemanticRuntimeLogger(object):
    def __init__(self):
        self.enabled = _flag("GEAK_SEMANTICS_CAPTURE")
        self.path = os.environ.get("GEAK_SEMANTICS_SHAPE_LOG", "")
        self.rank = int(os.environ.get("GEAK_SEMANTICS_RANK", "0"))
        self.layers = _csv_int("GEAK_SEMANTICS_LAYERS")
        self.phases = _csv_upper("GEAK_SEMANTICS_PHASES")
        self.layer_scopes = _flag("GEAK_SEMANTICS_LAYER_SCOPES", "1")
        self.require_profiler = _flag(
            "GEAK_SEMANTICS_REQUIRE_PROFILER")
        self._profile_seen = False
        self.max_forwards = int(os.environ.get(
            "GEAK_SEMANTICS_FORWARDS_PER_BUCKET", "1"))
        self._context = {
            "phase": "UNKNOWN", "batch_size": -1, "input_tokens": -1}
        self._bucket_forwards = {}
        self._next_id = 0
        self._rank = None
        self._lock = threading.Lock()
        self._stacks = threading.local()
        self._fh = None
        if self.enabled and self.path:
            os.makedirs(os.path.dirname(os.path.abspath(self.path)),
                        exist_ok=True)
            self._fh = open(self.path, "a", buffering=1)

    def active(self):
        return self._fh is not None and self._rank_ok()

    def _rank_ok(self):
        if self._rank is None:
            try:
                import torch.distributed as dist
                self._rank = (
                    dist.get_rank()
                    if dist.is_available() and dist.is_initialized() else 0)
            except Exception:
                self._rank = 0
        return self._rank == self.rank

    def set_context(self, phase, batch_size, input_tokens):
        self._context = {
            "phase": phase,
            "batch_size": batch_size,
            "input_tokens": input_tokens,
        }

    def mark_forward(self):
        key = (
            self._context["phase"], self._context["batch_size"],
            self._context["input_tokens"])
        self._bucket_forwards[key] = self._bucket_forwards.get(key, 0) + 1

    def _allowed(self, layer_id):
        if not self.active():
            return False
        # Benchmark warmups run before torch.profiler starts.  Consuming the
        # one-forward bucket there records shape JSON but leaves no matching
        # trace marker, which used to remove every steady decode bucket.
        #
        # Profiler-enabled state can be thread-local: prefill may observe it
        # while the subsequent decode worker does not.  Latch the first active
        # observation for the rest of this benchmark capture.
        if self.require_profiler and not self._profile_seen:
            if not _profiler_active():
                return False
            self._profile_seen = True
        if layer_id < 0 or (self.layers and layer_id not in self.layers):
            return False
        if (self.phases
                and _canonical_phase(self._context["phase"])
                not in {_canonical_phase(item) for item in self.phases}):
            return False
        key = (
            self._context["phase"], self._context["batch_size"],
            self._context["input_tokens"])
        return self._bucket_forwards.get(key, 0) < self.max_forwards

    def _layer_scope_allowed(self, layer_id):
        """Whether a lightweight main-layer boundary marker should be emitted.

        Unlike Shape logging this intentionally ignores ``self.layers``: the
        boundary donor must cover the complete main stack.  It still observes
        rank, phase and per-bucket forward limits, and it records no tensor
        metadata for non-representative layers.
        """
        if not self.layer_scopes or not self.active() or layer_id < 0:
            return False
        if (self.phases
                and _canonical_phase(self._context["phase"])
                not in {_canonical_phase(item) for item in self.phases}):
            return False
        key = (
            self._context["phase"], self._context["batch_size"],
            self._context["input_tokens"])
        return self._bucket_forwards.get(key, 0) < self.max_forwards

    def layer_scope(self, layer_id, op_path):
        """Return a record_function context for one complete main-layer body."""
        if not self._layer_scope_allowed(layer_id):
            return contextlib.nullcontext()
        context = dict(self._context)
        marker = (
            "GEAK_LAYER_SCOPE|phase=%s|bs=%s|toks=%s|layer=%s|path=%s"
            % (context["phase"], context["batch_size"],
               context["input_tokens"], layer_id, op_path))
        try:
            import torch
            factory = getattr(
                getattr(torch, "profiler", None), "record_function", None)
            if factory is None:
                factory = torch.autograd.profiler.record_function
            return factory(marker)
        except Exception:
            return contextlib.nullcontext()

    def _stack(self):
        if not hasattr(self._stacks, "value"):
            self._stacks.value = {}
        return self._stacks.value

    def _active_modules(self):
        if not hasattr(self._stacks, "active_modules"):
            self._stacks.active_modules = []
        return self._stacks.active_modules

    def begin(self, module, layer_id, op_path):
        if not self._allowed(layer_id):
            return
        with self._lock:
            self._next_id += 1
            op_id = "geak-op-%08d" % self._next_id
        context = dict(self._context)
        marker = (
            "GEAK_SEMANTICS|op=%s|phase=%s|bs=%s|toks=%s|layer=%s|path=%s"
            % (op_id, context["phase"], context["batch_size"],
               context["input_tokens"], layer_id, op_path))
        record = None
        try:
            import torch
            factory = getattr(
                getattr(torch, "profiler", None), "record_function", None)
            if factory is None:
                factory = torch.autograd.profiler.record_function
            record = factory(marker)
            record.__enter__()
        except Exception:
            record = None
        entry = {
            "op_id": op_id,
            "marker": marker,
            "record": record,
            "context": context,
            "layer_id": layer_id,
            "op_path": op_path,
        }
        self._stack().setdefault(id(module), []).append(entry)
        self._active_modules().append(entry)

    def end(self, module, layer_id, op_name, op_type, op_path,
            args, kwargs, output):
        entries = self._stack().get(id(module), [])
        if not entries:
            return output
        entry = entries.pop()
        active = self._active_modules()
        if active and active[-1] is entry:
            active.pop()
        elif entry in active:
            active.remove(entry)
        if entry["record"] is not None:
            try:
                entry["record"].__exit__(None, None, None)
            except Exception:
                pass
        aliases = {}
        parameters = {}
        try:
            for name, value in module.named_parameters(recurse=False):
                parameters[name] = value
            for name, value in module.named_buffers(recurse=False):
                parameters[name] = value
        except Exception:
            pass
        payload = {
            "schema": "geak.semantics_runtime.v2",
            "op_instance_id": entry["op_id"],
            "marker": entry["marker"],
            "rank": self.rank,
            "layer_id": layer_id,
            "phase": entry["context"]["phase"].lower(),
            "batch_size": entry["context"]["batch_size"],
            "input_tokens": entry["context"]["input_tokens"],
            "op_name": op_name,
            "op_type": op_type,
            "op_path": op_path,
            "mapping_cardinality": "1:N",
            "evidence_level": "runtime_marker_containment",
            "inputs": _metadata(args, aliases),
            "kwargs": _metadata(kwargs, aliases),
            "parameters": _metadata(parameters, aliases),
            "output": _metadata(output, aliases),
        }
        with self._lock:
            self._fh.write(json.dumps(
                payload, sort_keys=True, separators=(",", ":")) + "\n")
        return output

    def begin_callable(self, target, args, kwargs, callable_obj):
        active = self._active_modules()
        if not active:
            return None
        parent = active[-1]
        layer_id = parent["layer_id"]
        if not self._allowed(layer_id):
            return None
        with self._lock:
            self._next_id += 1
            op_id = "geak-call-%08d" % self._next_id
        context = dict(self._context)
        op_path = "%s::launcher:%s" % (parent["op_path"], target)
        marker = (
            "GEAK_SEMANTICS|op=%s|phase=%s|bs=%s|toks=%s|layer=%s|path=%s"
            % (op_id, context["phase"], context["batch_size"],
               context["input_tokens"], layer_id, op_path))
        record = None
        try:
            import torch
            factory = getattr(
                getattr(torch, "profiler", None), "record_function", None)
            if factory is None:
                factory = torch.autograd.profiler.record_function
            record = factory(marker)
            record.__enter__()
        except Exception:
            record = None
        aliases = {}
        signature_text = None
        binding_status = "generic_args"
        try:
            signature = inspect.signature(callable_obj)
            signature_text = str(signature)
            bound = signature.bind_partial(*args, **kwargs)
            named_inputs = dict(bound.arguments)
            binding_status = "inspect_signature_bind_partial"
        except (TypeError, ValueError):
            named_inputs = {
                "args": args,
                "kwargs": kwargs,
            }
        return {
            "op_id": op_id,
            "marker": marker,
            "record": record,
            "context": context,
            "layer_id": layer_id,
            "op_path": op_path,
            "target": target,
            # Snapshot input metadata before invoking the target.  Some
            # launchers mutate output/cache arguments in-place; reading the
            # arguments in finally used to blur pre-call and post-call state.
            "aliases": aliases,
            "inputs": _metadata(named_inputs, aliases),
            "callable_signature": signature_text,
            "argument_binding": binding_status,
        }

    def end_callable(self, entry, args, kwargs, output):
        if entry is None:
            return output
        if entry["record"] is not None:
            try:
                entry["record"].__exit__(None, None, None)
            except Exception:
                pass
        aliases = entry["aliases"]
        payload = {
            "schema": "geak.semantics_runtime.v3",
            "op_instance_id": entry["op_id"],
            "marker": entry["marker"],
            "rank": self.rank,
            "layer_id": entry["layer_id"],
            "phase": entry["context"]["phase"].lower(),
            "batch_size": entry["context"]["batch_size"],
            "input_tokens": entry["context"]["input_tokens"],
            "op_name": entry["target"].split(":")[-1],
            "op_type": "targeted_python_launcher",
            "op_path": entry["op_path"],
            "mapping_cardinality": "probe_required",
            "evidence_level": "targeted_launcher_probe",
            "callable_signature": entry["callable_signature"],
            "argument_binding": entry["argument_binding"],
            "input_snapshot_timing": "before_call",
            "inputs": entry["inputs"],
            "kwargs": _metadata({}, aliases),
            "parameters": _metadata({}, aliases),
            "output": _metadata(output, aliases),
        }
        with self._lock:
            self._fh.write(json.dumps(
                payload, sort_keys=True, separators=(",", ":")) + "\n")
        return output


def get_logger():
    global _LOGGER
    if _LOGGER is None:
        _LOGGER = SemanticRuntimeLogger()
    return _LOGGER


def _callable_targets():
    return [
        item.strip()
        for item in os.environ.get(
            "GEAK_SEMANTICS_CALLABLE_TARGETS", "").split(",")
        if item.strip()]


def _install_callable_probes():
    logger = get_logger()
    for target in _callable_targets():
        if target in _PATCHED_CALLABLES:
            continue
        module_name, separator, attr_name = target.partition(":")
        if not separator:
            raise RuntimeError(
                "invalid GEAK callable target %r; expected module:attr" %
                target)
        module = importlib.import_module(module_name)
        original = getattr(module, attr_name)
        if not callable(original):
            raise RuntimeError(
                "GEAK callable target is not callable: %s" % target)

        @functools.wraps(original)
        def wrapped(*args, __original=original, __target=target, **kwargs):
            entry = logger.begin_callable(
                __target, args, kwargs, __original)
            output = None
            try:
                output = __original(*args, **kwargs)
                return output
            finally:
                logger.end_callable(entry, args, kwargs, output)

        setattr(module, attr_name, wrapped)
        _PATCHED_CALLABLES.add(target)
        sys.stderr.write(
            "[GEAK_SEMANTICS] targeted launcher wrapped %s\n" % target)


def _register_hooks(model):
    logger = get_logger()
    if not logger.active():
        return
    count = 0
    layer_scope_count = 0
    for raw_path, module in model.named_modules():
        if not raw_path:
            continue
        path = _op_path(raw_path)
        layer = _layer_id(path)
        main_layer_match = _MAIN_LAYER_RE.search(path)
        if main_layer_match and not getattr(
                module, "_geak_semantics_layer_scope_wrapped", False):
            original_forward = module.forward

            @functools.wraps(original_forward)
            def layer_scoped_forward(
                    *args, __forward=original_forward,
                    __layer=layer, __path=path, **kwargs):
                with logger.layer_scope(__layer, __path):
                    return __forward(*args, **kwargs)

            module.forward = layer_scoped_forward
            module._geak_semantics_layer_scope_wrapped = True
            layer_scope_count += 1
        if logger.layers and layer not in logger.layers:
            continue
        children = list(module.children())
        class_name = module.__class__.__name__
        is_key = any(token in class_name for token in (
            "Attention", "MoE", "Moe", "MLP", "DecoderLayer",
            "RMSNorm", "LayerNorm", "Linear", "Expert"))
        if children and not is_key:
            continue

        def make_pre(layer_id, op_path):
            def pre_hook(mod, args, kwargs=None):
                logger.begin(mod, layer_id, op_path)
            return pre_hook

        def make_post(layer_id, op_name, op_type, op_path):
            def post_hook(mod, args, kwargs, output):
                return logger.end(
                    mod, layer_id, op_name, op_type, op_path,
                    args, kwargs, output)
            return post_hook

        try:
            module.register_forward_pre_hook(
                make_pre(layer, path), with_kwargs=True)
            module.register_forward_hook(
                make_post(layer, path.split(".")[-1], class_name, path),
                with_kwargs=True)
            count += 1
        except TypeError:
            # GEAK Semantics 1.2 requires kwargs-capable hooks for reliable
            # tensor roles; fail explicitly instead of silently mislabelling.
            raise RuntimeError(
                "runtime torch lacks kwargs-capable forward hooks")
    sys.stderr.write(
        "[GEAK_SEMANTICS] registered %d all-layer scopes and %d "
        "marker+metadata hooks\n" % (layer_scope_count, count))


def install_on_model(model):
    install_graph_capture_profiler()
    dump_operator_schema_manifest()
    logger = get_logger()
    if not logger.active():
        return model
    model_class = model.__class__
    if model_class not in _INSTALLED_CLASSES:
        original_forward = model_class.forward

        def wrapped_forward(self, *args, **kwargs):
            forward_batch = args[2] if len(args) >= 3 else None
            forward_batch = kwargs.get("forward_batch", forward_batch)
            if forward_batch is not None:
                logger.set_context(*_phase_of(forward_batch))
            if not getattr(self, "_geak_semantics_hooked", False):
                _register_hooks(self)
                self._geak_semantics_hooked = True
            result = original_forward(self, *args, **kwargs)
            if forward_batch is not None:
                logger.mark_forward()
            return result

        model_class.forward = wrapped_forward
        _INSTALLED_CLASSES.add(model_class)
    elif not getattr(model, "_geak_semantics_hooked", False):
        _register_hooks(model)
        model._geak_semantics_hooked = True
    _install_callable_probes()
    sys.stderr.write(
        "[GEAK_SEMANTICS] wrapped %s.forward\n" % model_class.__name__)
    return model
