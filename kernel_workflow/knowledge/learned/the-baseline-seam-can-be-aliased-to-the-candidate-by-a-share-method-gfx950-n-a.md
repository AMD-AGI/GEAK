---
key: a two-module A/B harness that loads a baseline seam and a candidate seam into one process can measure the candidate against itself, because aiter's torch_compile_guard keys its custom-op registry on the bare function name and the second import silently reuses the first import's implementation
type: method
confidence: ★★★
effect: on the GLM-5.2 mxfp4 fused-MoE UT, a sentinel injected into each module proved every baseline call dispatched into kernel_src; all 7 decode buckets then read 0.997-1.005x "speedup", i.e. a thing compared to itself, and a one-line NONE_WRAPPED_OP opt-out restored the baseline to its own code
confirms_cited: 0
confirms_blind: 1
losses: 0
attempts: 1
toolchain: aiter cab9e49a / flydsl 0.3.2
last_seen: 2026-10-10
name: the-baseline-seam-can-be-aliased-to-the-candidate-by-a-share-method-gfx950-n-a
description: Two seam modules in one process can share one torch custom op, so the baseline runs the candidate's code and the A/B reads ~1.00x no matter what was edited
keywords: ['ab-protocol', 'oracle', 'harness', 'custom-op', 'torch-library', 'aiter', 'anti-pattern', 'module-aliasing', 'gfx950']
kernels: ['glm52_mxfp4_fused_moe']
platforms: ['gfx950']
kernel_class: method
regime: n/a
layer: learned
lifecycle: active
cost: L1
verified_on: 2026-10-10
levers: ['method.ab-protocol', 'method.upper-bound-oracle']
origin_kernels: ['glm52_mxfp4_fused_moe']
---
# The baseline seam can be aliased to the candidate by a shared custom-op name
- lever: Before trusting any A/B that loads a baseline module and a candidate module into ONE process, prove the baseline entry point still reaches baseline code. `importlib.util.spec_from_file_location` with two different module names is not enough isolation when the modules register into a process-global registry.
- apply: aiter's `aiter/jit/utils/torch_guard.py::torch_compile_guard` sets `loadName = calling_func.__name__` and then short-circuits with `if hasattr(torch.ops.aiter, loadName): return wrapper_custom`. `torch.ops.aiter` is process-global and the name is not namespaced by module, so whichever seam imports FIRST owns the op and the second seam's decorated function is never called. The unittests load the candidate first, so the baseline is the one that gets aliased away.
- verify: Replace one function below the registered op (`_fused_moe_impl`) with a sentinel that raises its own module's name, in BOTH modules, then call the baseline entry point and read which sentinel fires. Cheap, runs in seconds, needs no kernel. A corroborating smell: a candidate with a real edit still reads inside the null floor in every bucket.
- pitfall: The aliasing is invisible on an unedited candidate, which is exactly the configuration a harness is usually sanity-checked in -- baseline and candidate agree because they are the same bytes, not because the plumbing is sound. It is equally invisible to parity, MDE and null-floor machinery, all of which are calibrated on that same unedited run.
- caution: Loading the baseline first does not fix it, it only reverses who gets aliased. Fix it outside the task instead: append the op name to `aiter.jit.utils.torch_guard.NONE_WRAPPED_OP` BEFORE either seam is imported, which makes each module keep its own plain wrapper; or run the two arms in separate processes. Only edits ABOVE the registered op boundary (here `fused_moe`/`glm52_mxfp4_fused_moe`) ever differentiate under the leak, which is why a leaked harness can still look responsive to a top-level change.
- source: run glm52_mxfp4_fused_moe oracle bring-up, 2026-10-10, diagnosed on a throwaway copy of /sgl-workspace/ut_glm52_conc10_0917/glm52_mxfp4_fused_moe_task (evidence/_diag_oracle_identity.py); not yet director-validated
