# FlyDSL version map — where each symbol lives, per version

> **Generated file.** Rebuild with `python3 languages/flydsl/_gen_version_map.py` (see the
> script docstring for the `--root` arguments). Do not hand-edit the tables.
>
> **Unlike the other generated files here, no CI gate can hold this one current** — rebuilding
> needs four FlyDSL versions installed side by side, which no CI image has. Treat the version
> list below as the scope of what was compared, and re-run locally when a new version lands.

A recipe carries a **logic** half (tiling, what to fuse, what goes to LDS, layout, which MFMA)
and an **API** half (which module to call). The logic half is bound to the architecture and
survives a version bump; the API half is not. This file resolves the API half mechanically so
porting a recipe to another FlyDSL version is a lookup instead of a search.

**A version difference is not a reason to discard a recipe.** Reuse the logic, re-derive the
call form from the table below, then re-measure — a performance number from another version is
not evidence for this one.

## Versions scanned

Trees are the paths this table was built from, recorded for provenance. Only the installed
one is expected to persist -- the others were unpacked wheels, so the tables here are the
durable artifact, not the trees. Symbol counts include the generated `_mlir/dialects/*_ops_gen`
bindings, which is why they are large; the per-pair diffs below are the part that matters.

| version | tree | modules | public symbols |
|---|---|---|---|
| `0.2.0` | `/tmp/fly020_bak` | 92 | 3557 |
| `0.2.2` | `/tmp/geak-flydsl-0.2.2` | 92 | 3577 |
| `0.2.4` | `/tmp/fly024` | 95 | 3606 |
| `0.3.0` | `/opt/venv/lib/python3.10/site-packages` | 99 | 3616 |

## 0.2.0 → 0.2.2

4 symbol(s) gone, 7 moved, 18 new.

### Gone in 0.2.2 — no definition anywhere in the package

Code using these needs a different way to express the same operation. There is no
call-form substitution; the logic has to be re-expressed with what the version offers.

| symbol | defined in 0.2.0 |
|---|---|
| `PointerAdaptor` | `flydsl.compiler.jit_argument` |
| `TensorAdaptor` | `flydsl.compiler.jit_argument` |
| `get_c_pointers` | `flydsl.compiler.protocol` |
| `s_prefetch_inst_burst` | `flydsl.expr.rocdl.inline_asm` |

### Moved in 0.2.2 — same name, different module

A move is mechanical only when the new home is the same *kind* of module. When a helper
moved into an intrinsic namespace, the wrapper's behaviour moved to the caller: verify
what the old wrapper did before treating the new location as a drop-in.

| symbol | 0.2.0 | 0.2.2 |
|---|---|---|
| `CallState` | `flydsl.compiler.jit_function` | `flydsl.compiler.jit_executor` |
| `ballot` | `flydsl._mlir.dialects._rocdl_ops_gen` | `flydsl._mlir.dialects._rocdl_ops_gen`, `flydsl.expr.rocdl` |
| `cvt_pk_f32_fp8` | `flydsl._mlir.dialects._rocdl_ops_gen` | `flydsl._mlir.dialects._rocdl_ops_gen`, `flydsl.expr.rocdl` |
| `cvt_scalef32_pk_f32_fp4` | `flydsl._mlir.dialects._rocdl_ops_gen` | `flydsl._mlir.dialects._rocdl_ops_gen`, `flydsl.expr.rocdl` |
| `cvt_scalef32_pk_fp4_f32` | `flydsl._mlir.dialects._rocdl_ops_gen` | `flydsl._mlir.dialects._rocdl_ops_gen`, `flydsl.expr.rocdl` |
| `readlane` | `flydsl._mlir.dialects._rocdl_ops_gen` | `flydsl._mlir.dialects._rocdl_ops_gen`, `flydsl.expr.rocdl` |
| `wmma_f32_16x16x128_fp8_fp8` | `flydsl._mlir.dialects._rocdl_ops_gen` | `flydsl._mlir.dialects._rocdl_ops_gen`, `flydsl.expr.rocdl` |

<details><summary>18 symbol(s) new in 0.2.2</summary>

| symbol | 0.2.2 |
|---|---|
| `DLTensorJitArg` | `flydsl.compiler.jit_argument` |
| `Int128` | `flydsl.expr.numeric` |
| `MemRefJitArg` | `flydsl.compiler.jit_argument` |
| `MemRefSpec` | `flydsl.compiler.jit_argument` |
| `PointerJitArg` | `flydsl.compiler.jit_argument` |
| `TorchTensorJitArg` | `flydsl.compiler.jit_argument` |
| `Uint128` | `flydsl.expr.numeric` |
| `as_dsl_value` | `flydsl.expr.typing` |
| `as_ir_value` | `flydsl.expr.typing` |
| `build_abi_storage` | `flydsl.compiler.jit_executor` |
| `c_abi_spec` | `flydsl.compiler.protocol` |
| `coerce_int_tuple_args` | `flydsl.expr.primitive` |
| `dsl_loc_tracing` | `flydsl.expr.meta` |
| `dsl_math_wrap_result` | `flydsl.expr.math` |
| `dsl_wrap_result` | `flydsl.expr.meta` |
| `from_torch_tensor` | `flydsl.compiler.jit_argument` |
| `is_specializable_struct_type` | `flydsl.expr.struct` |
| `torch_dtype_to_mlir_type` | `flydsl.compiler.jit_argument` |

</details>

## 0.2.2 → 0.2.4

7 symbol(s) gone, 6 moved, 29 new.

### Gone in 0.2.4 — no definition anywhere in the package

Code using these needs a different way to express the same operation. There is no
call-form substitution; the logic has to be re-expressed with what the version offers.

| symbol | defined in 0.2.2 |
|---|---|
| `FlyDSLCompileError` | `flydsl.compiler.jit_function` |
| `FuncLocationTracker` | `flydsl.compiler.kernel_function` |
| `WrapLocations` | `flydsl.compiler.ast_rewriter` |
| `create_caller_location` | `flydsl.compiler.kernel_function` |
| `create_file_location` | `flydsl.compiler.kernel_function` |
| `get_source_location` | `flydsl.compiler.kernel_function` |
| `traced_op` | `flydsl.expr.meta` |

### Moved in 0.2.4 — same name, different module

A move is mechanical only when the new home is the same *kind* of module. When a helper
moved into an intrinsic namespace, the wrapper's behaviour moved to the caller: verify
what the old wrapper did before treating the new location as a drop-in.

| symbol | 0.2.2 | 0.2.4 |
|---|---|---|
| `barrier` | `flydsl._mlir.dialects._gpu_ops_gen`, `flydsl._mlir.dialects._rocdl_ops_gen` | `flydsl._mlir.dialects._gpu_ops_gen`, `flydsl._mlir.dialects._rocdl_ops_gen`, `flydsl.expr.gpu` |
| `block_id` | `flydsl._mlir.dialects._gpu_ops_gen` | `flydsl._mlir.dialects._gpu_ops_gen`, `flydsl.expr.gpu` |
| `gather` | `flydsl._mlir.dialects._vector_ops_gen` | `flydsl._mlir.dialects._vector_ops_gen`, `flydsl.expr.derived` |
| `maxnumf` | `flydsl._mlir.dialects._arith_ops_gen` | `flydsl._mlir.dialects._arith_ops_gen`, `flydsl.expr.arith` |
| `scatter` | `flydsl._mlir.dialects._vector_ops_gen` | `flydsl._mlir.dialects._vector_ops_gen`, `flydsl.expr.derived` |
| `thread_id` | `flydsl._mlir.dialects._gpu_ops_gen` | `flydsl._mlir.dialects._gpu_ops_gen`, `flydsl.expr.gpu` |

### Modules

Added in 0.2.4: `flydsl.compiler.diagnostics`, `flydsl.expr.enum`, `flydsl.expr.rocdl.enum`

<details><summary>29 symbol(s) new in 0.2.4</summary>

| symbol | 0.2.4 |
|---|---|
| `Basis` | `flydsl.expr.typing` |
| `DSLCompileError` | `flydsl.compiler.diagnostics` |
| `DiagRecord` | `flydsl.compiler.diagnostics` |
| `E` | `flydsl.expr.typing` |
| `FallbackLocations` | `flydsl.compiler.ast_rewriter` |
| `SourceFrame` | `flydsl.compiler.diagnostics` |
| `SyncScope` | `flydsl.expr.enum`, `flydsl.expr.rocdl.enum` |
| `TDM` | `flydsl.expr.rocdl.universal` |
| `UniversalAtomic` | `flydsl.expr.primitive` |
| `VectorAlias` | `flydsl.expr.typing` |
| `WMMAScale` | `flydsl.expr.rocdl.universal` |
| `advance_tdm_atom` | `flydsl.expr.rocdl.universal` |
| `capture_user_location` | `flydsl.expr.meta` |
| `current_fastmath` | `flydsl.expr.utils.arith` |
| `diag_record_from_diagnostic` | `flydsl.compiler.diagnostics` |
| `diag_records_from_mlir_error` | `flydsl.compiler.diagnostics` |
| `dsl_ir_diagnostics` | `flydsl.compiler.diagnostics` |
| `effective_fastmath_hint` | `flydsl.compiler.kernel_function` |
| `fastmath` | `flydsl.expr.utils.arith` |
| `file_location` | `flydsl.expr.meta` |
| `func_def_location` | `flydsl.compiler.kernel_function` |
| `install_excepthook` | `flydsl.compiler.diagnostics` |
| `lazy_classattr` | `flydsl.expr.utils` |
| `location_chain` | `flydsl.compiler.diagnostics` |
| `make_tdm_atom` | `flydsl.expr.rocdl.universal` |
| `resolve_fastmath` | `flydsl.expr.utils.arith` |
| `tracing_context` | `flydsl.expr.meta` |
| `warn_annotation_value_mismatch` | `flydsl.compiler.diagnostics` |
| `warn_invalid_annotations` | `flydsl.compiler.diagnostics` |

</details>

## 0.2.4 → 0.3.0

12 symbol(s) gone, 12 moved, 16 new.

### Known replacements — what carries the old job

Hand-written and re-validated on every regeneration against the tables below
(`version_map_notes.yaml`). These say what the mechanical diff cannot: which new
symbol does the removed one's work, and what changed about the call.

**buffer resource descriptor construction**

- Gone: `create_buffer_resource`, `create_buffer_resource_from_addr`, `BufferResourceDescriptor`
- Use: `make_buffer_ptr`, `get_buffer_rsrc`

Not a rename -- one call became two, and the input type changed. 0.2.4 `create_buffer_resource(memref_val, stride=0, max_size=True, *, num_records_bytes, base_byte_offset)` took a **memref** and returned the descriptor value directly. 0.3.0 splits it: `make_buffer_ptr(ptr: Pointer, num_records_bytes=None)` takes a typed **Pointer that must already be in the global address space** (it raises otherwise) and returns a buffer-descriptor *pointer*; `get_buffer_rsrc(ptr)` then extracts the raw ROCDL `!llvm.ptr<8>` that 0.2.4 handed back in one step. `num_records_bytes=None` reproduces the old `max_size=True` (falls back to 0xFFFFFFFF). `make_buffer_ptr` also sets descriptor flags per architecture (extra reserved bit and OOB_SELECT on RDNA), which the 0.2.4 helper did not do -- so a port inherits arch-dependent behaviour it did not have before.

  - 0.2.4: flydsl/expr/buffer_ops.py -> def create_buffer_resource(memref_val, ...)
  - 0.3.0: flydsl/expr/rocdl/universal.py:176 -> def make_buffer_ptr(ptr: Pointer, num_records_bytes=None)
  - 0.3.0: flydsl/expr/rocdl/universal.py:252 -> def get_buffer_rsrc(ptr: Pointer)

**pointer arithmetic and LLVM pointer exposure**

- Gone: `get_element_ptr`, `create_llvm_ptr`
- Use: `to_llvm_ptr`, `add_offset`

0.2.4 `get_element_ptr(base_ptr, byte_offset=None, static_byte_offset=0, elem_type=None, no_wrap_flags=None)` built an LLVM GEP from a base pointer plus byte offsets -- one call did both the offset arithmetic and the drop to LLVM pointer level. In 0.3.0 those are separate: `add_offset(ptr, offset)` does the arithmetic on a `fly.ptr` (it already existed in 0.2.4, so it is not in the "new" table) and `to_llvm_ptr(ptr)` exposes the result as an `!llvm.ptr`, resolving the address space through the active compile backend rather than taking it as an argument. Code that only needed the offset should stop at `add_offset` and never reach LLVM pointer level.

  - 0.2.4: flydsl/expr/buffer_ops.py -> def get_element_ptr(base_ptr, byte_offset, ...)
  - 0.2.4: flydsl/expr/primitive.py:1183 -> def add_offset(ptr, offset)
  - 0.3.0: flydsl/expr/primitive.py:1186 -> def to_llvm_ptr(ptr)
  - 0.3.0: flydsl/expr/primitive.py:1199 -> def add_offset(ptr, offset)

**buffer load / store**

- Gone: `buffer_load`, `buffer_store`
- Use: `buffer_load_to_lds`

`flydsl.expr.buffer_ops.buffer_load` / `buffer_store` have no single-call successor. 0.3.0 keeps `buffer_load_to_lds` in `flydsl.expr.rocdl` for the global-to-LDS path, but a plain buffer load into registers has to be re-expressed through the descriptor pointer built by `make_buffer_ptr` plus the load form the target architecture module offers. Treat this as re-implementing the access, not substituting a name, and re-measure -- the emitted instruction may differ in width and in where the wait lands.

  - 0.2.4: flydsl/expr/buffer_ops.py -> def buffer_load(...), def buffer_store(...)
  - 0.3.0: flydsl/expr/rocdl/__init__.py:626 -> def buffer_load_to_lds(rsrc, lds_ptr, voffset, size_bytes=4, soffset=0, offset=0, **kw)
  - 0.3.0: no `buffer_load` or `buffer_store` defined anywhere in the package

### Gone in 0.3.0 — no definition anywhere in the package

Code using these needs a different way to express the same operation. There is no
call-form substitution; the logic has to be re-expressed with what the version offers.

| symbol | defined in 0.2.4 |
|---|---|
| `BufferResourceDescriptor` | `flydsl.expr.buffer_ops` |
| `advance_tdm_atom` | `flydsl.expr.rocdl.universal` |
| `buffer_load` | `flydsl.expr.buffer_ops` |
| `buffer_store` | `flydsl.expr.buffer_ops` |
| `compute_mcast_masks` | `flydsl.expr.rocdl.cluster` |
| `create_buffer_resource` | `flydsl.expr.buffer_ops` |
| `create_buffer_resource_from_addr` | `flydsl.expr.buffer_ops` |
| `create_llvm_ptr` | `flydsl.expr.buffer_ops` |
| `default_f8_type` | `flydsl.expr.typing` |
| `extract_base_index` | `flydsl.expr.buffer_ops` |
| `get_element_ptr` | `flydsl.expr.buffer_ops` |
| `load_op` | `flydsl.expr.vector` |

### Moved in 0.3.0 — same name, different module

A move is mechanical only when the new home is the same *kind* of module. When a helper
moved into an intrinsic namespace, the wrapper's behaviour moved to the caller: verify
what the old wrapper did before treating the new location as a drop-in.

| symbol | 0.2.4 | 0.3.0 |
|---|---|---|
| `TDM` | `flydsl.expr.rocdl.universal` | `flydsl.expr.rocdl.cdna5` |
| `WMMAScale` | `flydsl.expr.rocdl.universal` | `flydsl.expr.rocdl.cdna5` |
| `bitcast` | `flydsl._mlir.dialects._arith_ops_gen`, `flydsl._mlir.dialects._llvm_ops_gen`, `flydsl._mlir.dialects._vector_ops_gen`, `flydsl.expr.vector` | `flydsl._mlir.dialects._arith_ops_gen`, `flydsl._mlir.dialects._llvm_ops_gen`, `flydsl._mlir.dialects._vector_ops_gen` |
| `extract` | `flydsl._mlir.dialects._vector_ops_gen`, `flydsl.expr.vector` | `flydsl._mlir.dialects._vector_ops_gen` |
| `from_elements` | `flydsl._mlir.dialects._vector_ops_gen`, `flydsl.expr.vector` | `flydsl._mlir.dialects._vector_ops_gen` |
| `make_tdm_atom` | `flydsl.expr.rocdl.universal` | `flydsl.expr.rocdl.cdna5` |
| `maximumf` | `flydsl._mlir.dialects._arith_ops_gen` | `flydsl._mlir.dialects._arith_ops_gen`, `flydsl.expr.arith` |
| `minimumf` | `flydsl._mlir.dialects._arith_ops_gen` | `flydsl._mlir.dialects._arith_ops_gen`, `flydsl.expr.arith` |
| `s_waitcnt` | `flydsl._mlir.dialects._rocdl_ops_gen` | `flydsl._mlir.dialects._rocdl_ops_gen`, `flydsl.expr.rocdl.cdna3`, `flydsl.expr.rocdl.rdna3`, `flydsl.expr.rocdl.rdna4`, `flydsl.expr.rocdl.universal` |
| `shrui` | `flydsl._mlir.dialects._arith_ops_gen` | `flydsl._mlir.dialects._arith_ops_gen`, `flydsl.expr.arith` |
| `shuffle` | `flydsl._mlir.dialects._gpu_ops_gen`, `flydsl._mlir.dialects._vector_ops_gen` | `flydsl._mlir.dialects._gpu_ops_gen`, `flydsl._mlir.dialects._vector_ops_gen`, `flydsl.expr.gpu` |
| `store` | `flydsl._mlir.dialects._llvm_ops_gen`, `flydsl._mlir.dialects._memref_ops_gen`, `flydsl._mlir.dialects._vector_ops_gen`, `flydsl.expr.vector` | `flydsl._mlir.dialects._llvm_ops_gen`, `flydsl._mlir.dialects._memref_ops_gen`, `flydsl._mlir.dialects._vector_ops_gen` |

### Modules

Removed in 0.3.0: `flydsl.expr.buffer_ops`, `flydsl.expr.vector`

Added in 0.3.0: `flydsl.expr.rocdl.cdna3`, `flydsl.expr.rocdl.cdna5`, `flydsl.expr.rocdl.rdna3`, `flydsl.expr.rocdl.rdna4`, `flydsl.expr.rocdl.utils`, `flydsl.utils.file`

<details><summary>16 symbol(s) new in 0.3.0</summary>

| symbol | 0.3.0 |
|---|---|
| `AutotuneEnvManager` | `flydsl.utils.env` |
| `GetBufferRsrcOp` | `flydsl._mlir.dialects._fly_rocdl_ops_gen` |
| `GetBufferRsrcOpAdaptor` | `flydsl._mlir.dialects._fly_rocdl_ops_gen` |
| `ToLLVMPtrOp` | `flydsl._mlir.dialects._fly_ops_gen` |
| `ToLLVMPtrOpAdaptor` | `flydsl._mlir.dialects._fly_ops_gen` |
| `atomic_write` | `flydsl.utils.file` |
| `get_buffer_rsrc` | `flydsl._mlir.dialects._fly_rocdl_ops_gen`, `flydsl.expr.rocdl.universal` |
| `make_buffer_ptr` | `flydsl.expr.rocdl.universal` |
| `merge_compile_hints` | `flydsl.compiler.kernel_function` |
| `normalize_s_waitcnt_field` | `flydsl.expr.rocdl.utils` |
| `resolve_llvm_address_space` | `flydsl.compiler.backends` |
| `shuffle_down` | `flydsl.expr.gpu` |
| `shuffle_idx` | `flydsl.expr.gpu` |
| `shuffle_up` | `flydsl.expr.gpu` |
| `shuffle_xor` | `flydsl.expr.gpu` |
| `to_llvm_ptr` | `flydsl._mlir.dialects._fly_ops_gen`, `flydsl.expr.primitive` |

</details>

## Call sites that fail on the installed FlyDSL (`0.3.0`)

Every `flydsl` import under the scanned trees, resolved **live** against the version
installed in the interpreter that generated this file. Each row raises at import time.

Scanned: `/sgl-workspace/aiter/aiter/ops/flydsl`

| file | line | import | reason |
|---|---|---|---|
| `/sgl-workspace/aiter/aiter/ops/flydsl/kernels/chunk_gated_delta_h.py` | 19 | `from flydsl.expr import vector` | name absent from module |
| `/sgl-workspace/aiter/aiter/ops/flydsl/kernels/flash_attn_func_gfx1201.py` | 39 | `from flydsl.expr import buffer_ops` | name absent from module |
| `/sgl-workspace/aiter/aiter/ops/flydsl/kernels/fused_compress_attn.py` | 66 | `from flydsl.expr import buffer_ops` | name absent from module |
| `/sgl-workspace/aiter/aiter/ops/flydsl/kernels/fused_compress_attn.py` | 66 | `from flydsl.expr import vector` | name absent from module |
| `/sgl-workspace/aiter/aiter/ops/flydsl/kernels/fused_compress_attn_hca.py` | 52 | `from flydsl.expr import buffer_ops` | name absent from module |
| `/sgl-workspace/aiter/aiter/ops/flydsl/kernels/fused_compress_attn_hca.py` | 52 | `from flydsl.expr import vector` | name absent from module |
| `/sgl-workspace/aiter/aiter/ops/flydsl/kernels/gdr_decode.py` | 15 | `from flydsl.expr import vector` | name absent from module |
| `/sgl-workspace/aiter/aiter/ops/flydsl/kernels/kernels_common.py` | 15 | `from flydsl.expr import buffer_ops` | name absent from module |
| `/sgl-workspace/aiter/aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage.py` | 28 | `from flydsl.expr import buffer_ops` | name absent from module |
| `/sgl-workspace/aiter/aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage.py` | 28 | `from flydsl.expr import vector` | name absent from module |
| `/sgl-workspace/aiter/aiter/ops/flydsl/kernels/moe_gemm_2stage.py` | 24 | `from flydsl.expr import buffer_ops` | name absent from module |
| `/sgl-workspace/aiter/aiter/ops/flydsl/kernels/moe_gemm_2stage.py` | 24 | `from flydsl.expr import vector` | name absent from module |
| `/sgl-workspace/aiter/aiter/ops/flydsl/kernels/moe_gemm_2stage.py` | 30 | `from flydsl.runtime.device import bf16_global_atomics_arch_description` | name absent from module |
| `/sgl-workspace/aiter/aiter/ops/flydsl/kernels/moe_gemm_2stage.py` | 30 | `from flydsl.runtime.device import supports_bf16_global_atomics` | name absent from module |
| `/sgl-workspace/aiter/aiter/ops/flydsl/kernels/preshuffle_gemm.py` | 9 | `from flydsl.expr import buffer_ops` | name absent from module |
| `/sgl-workspace/aiter/aiter/ops/flydsl/kernels/qk_norm_rope_quant.py` | 65 | `from flydsl.expr import buffer_ops` | name absent from module |
| `/sgl-workspace/aiter/aiter/ops/flydsl/kernels/qk_norm_rope_quant.py` | 65 | `from flydsl.expr import vector` | name absent from module |
| `/sgl-workspace/aiter/aiter/ops/flydsl/kernels/qk_norm_rope_quant.py` | 69 | `from flydsl.expr.vector import ReductionOp` | module does not import (ModuleNotFoundError) |
| `/sgl-workspace/aiter/aiter/ops/flydsl/kernels/reduce.py` | 9 | `from flydsl.dialects.ext.python_control_flow import lower_range_for_loops` | module does not import (ModuleNotFoundError) |
| `/sgl-workspace/aiter/aiter/ops/flydsl/kernels/reduce.py` | 18 | `from flydsl.dialects.ext import arith` | module does not import (ModuleNotFoundError) |
| `/sgl-workspace/aiter/aiter/ops/flydsl/kernels/reduce.py` | 30 | `from flydsl.dialects.ext import arith` | module does not import (ModuleNotFoundError) |
| `/sgl-workspace/aiter/aiter/ops/flydsl/kernels/silu_and_mul_fq.py` | 25 | `from flydsl.expr import vector` | name absent from module |
| `/sgl-workspace/aiter/aiter/ops/flydsl/kernels/silu_and_mul_fq.py` | 39 | `from flydsl.expr import buffer_ops` | name absent from module |
| `/sgl-workspace/aiter/aiter/ops/flydsl/kernels/small_m_hgemm.py` | 48 | `from flydsl.expr import vector` | name absent from module |
| `/sgl-workspace/aiter/aiter/ops/flydsl/kernels/splitk_hgemm.py` | 14 | `from flydsl.expr import buffer_ops` | name absent from module |
| `/sgl-workspace/aiter/aiter/ops/flydsl/kernels/splitk_hgemm.py` | 14 | `from flydsl.expr import vector` | name absent from module |
| `/sgl-workspace/aiter/aiter/ops/flydsl/kernels/swiglu_and_mul.py` | 31 | `from flydsl.expr import buffer_ops` | name absent from module |
| `/sgl-workspace/aiter/aiter/ops/flydsl/kernels/tensor_shim.py` | 15 | `from flydsl.expr import buffer_ops` | name absent from module |
| `/sgl-workspace/aiter/aiter/ops/flydsl/kernels/tensor_shim.py` | 15 | `from flydsl.expr import vector` | name absent from module |

## Sources

- FlyDSL `0.2.0` package tree: `/tmp/fly020_bak/flydsl/` (`__version__` read from its `__init__.py`)
- FlyDSL `0.2.2` package tree: `/tmp/geak-flydsl-0.2.2/flydsl/` (`__version__` read from its `__init__.py`)
- FlyDSL `0.2.4` package tree: `/tmp/fly024/flydsl/` (`__version__` read from its `__init__.py`)
- FlyDSL `0.3.0` package tree: `/opt/venv/lib/python3.10/site-packages/flydsl/` (`__version__` read from its `__init__.py`)
- Import call sites resolved live against installed FlyDSL `0.3.0`: `/sgl-workspace/aiter/aiter/ops/flydsl`
- Generated by `languages/flydsl/_gen_version_map.py` from the trees above.
