#!/usr/bin/env python3
"""Rebuild aiter's module_moe_asm JIT extension after editing its sources.

topk_softmax lives in csrc/kernels/topk_softmax_kernels.cu, which is compiled into
aiter/jit/module_moe_asm.so on first use and then cached. Editing the .cu does nothing until
the module is rebuilt, and the build arguments (include paths, flags, source list) live in
aiter's build registry rather than in a Makefile you can invoke.

build_module()'s signature is narrower than what get_args_of_build() returns, so the kwargs
have to be filtered by inspect.signature or the call raises on unexpected keywords. That
filtering is the only reason this file exists.

    python3 analysis/topk/build_moe_asm.py     # ~2 min

Then restart the server: the decode path is HIP-graph captured, so a rebuilt .so does not
take effect in a running process. Keep a copy of the stock .so before the first build
(/tmp/module_moe_asm.so.bak here) -- there is no other way back to it without a clean
container.
"""
import inspect
from aiter.jit.core import build_module, get_args_of_build
name = "module_moe_asm"
d = get_args_of_build(name)
sig = inspect.signature(build_module)
kw = {k: v for k, v in d.items() if k in sig.parameters}
print("build kwargs:", sorted(kw))
build_module(md_name=name, **{k: v for k, v in kw.items() if k != "md_name"})
print("BUILD OK")
