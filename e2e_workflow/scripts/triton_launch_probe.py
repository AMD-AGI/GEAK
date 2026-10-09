#!/usr/bin/env python3
"""Record the tensor arguments of every Triton launch into a torch profiler trace.

WHY THIS EXISTS
---------------
`record_shapes` fills `Input Dims` only for ops that go through the PyTorch dispatcher.
A Triton kernel launched straight from Python (`kernel[grid](...)`) has no dispatcher op
around it, so its trace row has no parent and no dims in ANY capture -- CUDA graphs on
or off. On MiniMax-M3 TP8 (vLLM) that left the gemma add+rmsnorm, swiglu and the whole
sparse-indexer chain shape-less in both phases: 30 of 43 unresolved semantic rows, and
the fusible regions around them could not get a microbench.

Every such launch goes through `triton.runtime.jit.JITFunction.run` (it is the frame
right above `hipModuleLaunchKernel` in a with_stack trace), so this wraps that one method
and, while a profiler is recording, opens a `record_function` around the launch named

    geak_triton_launch::<kernel name>::<operand>:<dims>:<dtype>;...

holding each tensor argument's parameter name, dims and dtype. The kernel
the launch produces is attributed to that annotation by External id, so
`semantic_kernel_mapping.py` reads the kernel's OWN inputs off it (shape source
`triton_launch_args`, kernel granularity). Hooks on nn.Modules cannot do this on vLLM:
they sit inside the torch.compile region and stop the engine from starting. This hook
sits below it, in the Triton runtime.

CONTRACT
--------
- Inert unless GEAK_TRITON_LAUNCH_SHAPES=1, and free unless a profiler is recording.
- Never raises into the launch: any failure to describe the arguments launches unannotated.
- Skipped while dynamo is tracing, so it never becomes part of a compiled graph.
- Idempotent. Installed as an overlay post-import hook on `triton.runtime.jit`
  (`overlay_setup.py add-triton-launch-probe`), never by editing site-packages.
"""
import os
import sys

PREFIX = "geak_triton_launch::"

# The spelling torch's own `Input type` uses, so a probed row reads like a traced one.
_DTYPE_NAMES = {
    "torch.bfloat16": "BFloat16", "torch.float16": "Half", "torch.float32": "float",
    "torch.float64": "double", "torch.int8": "signed char", "torch.uint8": "unsigned char",
    "torch.int16": "short int", "torch.int32": "int", "torch.int64": "long int",
    "torch.bool": "bool",
}

_WARNED = set()


def _armed():
    return os.environ.get("GEAK_TRITON_LAUNCH_SHAPES", "0") in ("1", "true", "True")


def _warn_once(key, message):
    if key not in _WARNED:
        _WARNED.add(key)
        sys.stderr.write("[GEAK_TRITON_LAUNCH_PROBE] %s\n" % message)


def _dtype_name(dtype):
    text = str(dtype)
    return _DTYPE_NAMES.get(text, text.replace("torch.", ""))


def describe(arg_names, args, kwargs):
    """[(name, dims, dtype)] for the tensor arguments, in signature order."""
    bound = dict(zip(arg_names, args))
    bound.update({name: value for name, value in kwargs.items() if name in arg_names})
    operands = []
    for name in arg_names:
        value = bound.get(name)
        shape = getattr(value, "shape", None)
        if shape is None or not hasattr(value, "data_ptr"):
            continue
        operands.append((name, [int(size) for size in shape],
                         _dtype_name(getattr(value, "dtype", ""))))
    return operands


# Kineto's chrome-trace export writes annotation names WITHOUT escaping, so a `"` or
# `\` in the name corrupts the whole trace file. Hence a quote-free encoding:
#     name:32x6144:BFloat16;w_ptr:6144:float     (a 0-d tensor's dims are "-")
def annotation_name(kernel_name, arg_names, args, kwargs):
    return "%s%s::%s" % (PREFIX, kernel_name, ";".join(
        "%s:%s:%s" % (name, "x".join(str(size) for size in dims) or "-", dtype)
        for name, dims, dtype in describe(arg_names, args, kwargs)))


def parse_annotation(name):
    """(kernel_name, input_dims, input_types, operand_names), or None."""
    if not isinstance(name, str) or not name.startswith(PREFIX):
        return None
    kernel_name, _, payload = name[len(PREFIX):].partition("::")
    dims, types, names = [], [], []
    try:
        for item in filter(None, payload.split(";")):
            operand, shape, dtype = item.split(":", 2)
            dims.append([] if shape == "-" else [int(size) for size in shape.split("x")])
            types.append(dtype)
            names.append(operand)
    except ValueError:
        return None
    return kernel_name, dims, types, names


def _profiling(torch):
    return (torch._C._autograd._profiler_enabled()
            and not torch.compiler.is_compiling())


def install():
    if not _armed():
        return
    # The hook fires right after triton.runtime.jit imports, which can be in the middle
    # of torch's own import, so torch is only resolved at launch time.
    try:
        from triton.runtime.jit import JITFunction
    except Exception as exc:  # noqa: BLE001 -- a probe never breaks the server
        _warn_once("import", "not installed: %s" % exc)
        return
    if getattr(JITFunction.run, "_geak_triton_launch_probe", False):
        return
    original = JITFunction.run

    def run(self, *args, **kwargs):
        torch = sys.modules.get("torch")
        try:
            name = (annotation_name(self.fn.__name__, list(self.arg_names), args, kwargs)
                    if torch is not None and _profiling(torch) else None)
        except Exception as exc:  # noqa: BLE001
            _warn_once("describe", "launch left unannotated: %s" % exc)
            name = None
        if name is None:
            return original(self, *args, **kwargs)
        with torch.autograd.profiler.record_function(name):
            return original(self, *args, **kwargs)

    run._geak_triton_launch_probe = True
    JITFunction.run = run
    _warn_once("installed", "armed on triton.runtime.jit.JITFunction.run")
