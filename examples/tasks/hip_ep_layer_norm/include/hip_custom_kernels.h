/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIP_CUSTOM_KERNELS_H
#define HIP_CUSTOM_KERNELS_H

/*
 * Pure C interface for HIP custom kernels.
 *
 * This header declares host-side launcher functions that are implemented
 * in .hip files compiled by hipcc. The interface uses only standard C types
 * (no HIP-specific types) so it can be included by:
 *   - Clang during bitcode compilation (lib/Runtime/real/)
 *   - MSVC for any host-side C/C++ code
 *
 * The .hip implementations (compiled by hipcc) define these functions with
 * extern "C" linkage and ship in a per-arch shared library
 * (custom_kernels_<arch>.{dll,so}); see HIP_KERNEL_API below.
 */

#include <stdint.h>

// Exports each launcher from the per-arch kernel shared library
// (custom_kernels_<arch>.{dll,so}), which the model artifact resolves at load
// (JIT dlopen, or native import). Pre-dual-format the kernels were linked into
// model.dll, so no export was needed -- hence this is new. EXPORTS is defined
// only when building that library (hip_utils.cmake); consumers leave it empty.
#if defined(_WIN32)
  #if defined(HIP_CUSTOM_KERNELS_EXPORTS)
    #define HIP_KERNEL_API __declspec(dllexport)
  #else
    #define HIP_KERNEL_API
  #endif
#elif defined(__GNUC__) || defined(__clang__)
  #define HIP_KERNEL_API __attribute__((visibility("default")))
#else
  #define HIP_KERNEL_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* =========================================================================
 * Data Type Enum
 * =========================================================================
 *
 * Unambiguous type identifiers for kernels that need to dispatch on data type.
 * Unlike element_size_bytes, this distinguishes types of the same byte size
 * (e.g. int64 vs float64, int32 vs float32).
 *
 * Existing GQA/RoPE kernels continue using element_size_bytes since they only
 * support float32/float16 (no ambiguity). New kernels should prefer hip_dtype.
 */
typedef enum {
    HIP_DTYPE_FLOAT32  = 0,
    HIP_DTYPE_FLOAT16  = 1,
    HIP_DTYPE_INT64    = 2,
    HIP_DTYPE_INT32    = 3,
    HIP_DTYPE_FLOAT64  = 4,
    HIP_DTYPE_BFLOAT16 = 5,
    HIP_DTYPE_INT16    = 6,
    HIP_DTYPE_UINT8    = 7,
    HIP_DTYPE_INT8     = 8,
} hip_dtype_t;

/* =========================================================================
 * Elementwise Subtraction
 * =========================================================================
 *
 * Computes output[i] = lhs[i] - rhs[i] for num_elements elements.
 *
 * Parameters:
 *   stream       - hipStream_t cast to void*
 *   lhs          - GPU pointer to left-hand operand
 *   rhs          - GPU pointer to right-hand operand
 *   output       - GPU pointer to output
 *   num_elements - number of elements in each tensor
 *   hip_dtype    - data type (hip_dtype_t value cast to int)
 *
 * Currently supported types: HIP_DTYPE_INT64
 * Returns: 0 on success (hipSuccess), non-zero hipError_t on failure
 */
HIP_KERNEL_API int hip_elementwise_sub(
    void* stream,
    const void* lhs,
    const void* rhs,
    void* output,
    int64_t num_elements,
    int hip_dtype);

/* =========================================================================
 * Elementwise Where (NumPy-style multidirectional broadcasting, arbitrary rank)
 * =========================================================================
 *
 * Computes output[i] = condition[idx_cond(i)] ? x[idx_x(i)] : y[idx_y(i)]
 * for each element of the output tensor. Each operand is described by its
 * own (shape_ptr, rank) pair; operand shapes are left-padded with 1s up to
 * the output rank to implement ONNX multidirectional broadcasting. Dims of
 * size 1 are broadcast against the corresponding larger output dim.
 *
 * No fixed layout is assumed; operands may have any rank <= HIP_WHERE_MAX_RANK.
 *
 * Parameters:
 *   stream      - hipStream_t cast to void*
 *   condition   - GPU pointer to bool tensor (1 byte per element)
 *   x           - GPU pointer to X tensor (selected when condition is true)
 *   y           - GPU pointer to Y tensor (selected when condition is false)
 *   output      - GPU pointer to output tensor (broadcast shape)
 *   cond_shape  - host pointer to condition shape array (length == cond_rank)
 *   cond_rank   - rank of condition tensor
 *   x_shape     - host pointer to X shape array (length == x_rank)
 *   x_rank      - rank of X tensor
 *   y_shape     - host pointer to Y shape array (length == y_rank)
 *   y_rank      - rank of Y tensor
 *   out_shape   - host pointer to output shape array (length == out_rank)
 *   out_rank    - rank of output tensor (max of input ranks)
 *   hip_dtype   - element type of x/y/output (hip_dtype_t value cast to int)
 *
 * Currently supported types for x/y/output: HIP_DTYPE_FLOAT32, HIP_DTYPE_FLOAT16,
 * HIP_DTYPE_BFLOAT16, HIP_DTYPE_INT32, HIP_DTYPE_INT64
 * Returns: 0 on success (hipSuccess), non-zero on failure (including rank > max)
 */
HIP_KERNEL_API int hip_elementwise_where(
    void* stream,
    const void* condition,
    const void* x,
    const void* y,
    void* output,
    const int64_t* cond_shape, int64_t cond_rank,
    const int64_t* x_shape,    int64_t x_rank,
    const int64_t* y_shape,    int64_t y_rank,
    const int64_t* out_shape,  int64_t out_rank,
    int hip_dtype);

/* =========================================================================
 * Elementwise Unary (Neg / Sign / Cos / Sin / Not)
 * =========================================================================
 *
 * Per-op launchers for the 5 ONNX unary ops added for the Qwen3.5 vision
 * model. All five share a single .hip translation unit
 * (lib/Runtime/Kernels/hip/elementwise_unary_kernel.hip).
 *
 * Supported hip_dtype (per op, may differ):
 *   Neg/Sign  : FLOAT16, INT32, INT64 (+ FLOAT32 for free)
 *   Cos/Sin   : FLOAT16, FLOAT32
 *   Not       : bool (i.e. 1-byte; pass element_size_bytes is unused -- the
 *               kernel reads/writes 1 byte unconditionally and ignores
 *               hip_dtype)
 * Returns: 0 on success (hipSuccess), non-zero hipError_t on failure.
 */
HIP_KERNEL_API int hip_elementwise_abs(
    void *stream, const void *input, void *output, int64_t num_elements,
    int hip_dtype);

HIP_KERNEL_API int hip_elementwise_neg(
    void* stream,
    const void* input,
    void* output,
    int64_t num_elements,
    int hip_dtype);

HIP_KERNEL_API int hip_elementwise_sign(
    void* stream,
    const void* input,
    void* output,
    int64_t num_elements,
    int hip_dtype);

HIP_KERNEL_API int hip_elementwise_cos(
    void* stream,
    const void* input,
    void* output,
    int64_t num_elements,
    int hip_dtype);

HIP_KERNEL_API int hip_elementwise_sin(
    void* stream,
    const void* input,
    void* output,
    int64_t num_elements,
    int hip_dtype);

HIP_KERNEL_API int hip_elementwise_ceil(
    void *stream, const void *input, void *output, int64_t num_elements,
    int hip_dtype);

HIP_KERNEL_API int hip_elementwise_exp(
    void* stream,
    const void* input,
    void* output,
    int64_t num_elements,
    int hip_dtype);

HIP_KERNEL_API int hip_elementwise_log(
    void* stream,
    const void* input,
    void* output,
    int64_t num_elements,
    int hip_dtype);

HIP_KERNEL_API int hip_elementwise_not(
    void* stream,
    const void* input,
    void* output,
    int64_t num_elements);

/* =========================================================================
 * Elementwise Binary (Mul / Add / Min / Max / Div / Mod / Equal / Less)
 * =========================================================================
 *
 * Same-shape binary elementwise ops. All eight share one translation unit:
 * lib/Runtime/Kernels/hip/elementwise_binary_kernel.hip.
 *
 * Mul / Add / Min / Max are reached from wrap_miopenOpTensor when MIOpen's
 * miopenOpTensor rejects the element type (notably INT32/INT64). Float
 * dtypes still use MIOpen for performance and autotuning.
 *
 * Div / Mod / Equal / Less were added for the Qwen3.5 vision path. Equal and
 * Less write bool (1 byte); their hip_dtype refers to the input element type.
 * Div and Mod preserve the input dtype.
 *
 * Broadcasting is not performed in these kernels. lhs and rhs must already
 * match in shape; upstream Expand / broadcast materialization is required.
 *
 * Returns: 0 on success (hipSuccess), non-zero hipError_t on failure.
 */
HIP_KERNEL_API int hip_elementwise_div(
    void* stream,
    const void* lhs,
    const void* rhs,
    void* output,
    int64_t num_elements,
    int hip_dtype);

HIP_KERNEL_API int hip_elementwise_mul(
    void* stream,
    const void* lhs,
    const void* rhs,
    void* output,
    int64_t num_elements,
    int hip_dtype);

HIP_KERNEL_API int hip_elementwise_add(
    void* stream,
    const void* lhs,
    const void* rhs,
    void* output,
    int64_t num_elements,
    int hip_dtype);

HIP_KERNEL_API int hip_elementwise_min(
    void* stream,
    const void* lhs,
    const void* rhs,
    void* output,
    int64_t num_elements,
    int hip_dtype);

HIP_KERNEL_API int hip_elementwise_max(
    void* stream,
    const void* lhs,
    const void* rhs,
    void* output,
    int64_t num_elements,
    int hip_dtype);

HIP_KERNEL_API int hip_elementwise_mod(
    void* stream,
    const void* lhs,
    const void* rhs,
    void* output,
    int64_t num_elements,
    int hip_dtype,
    int fmod_flag);

/* Broadcasting binary elementwise (Add / Mul / Min / Max), 4-D operands.
 *
 * Each operand is described by a 4-element int64 shape array (the lowering
 * left-pads ranks < 4 with leading 1s). NumPy/ONNX broadcasting is applied
 * per axis: any axis whose operand extent is 1 is broadcast against the
 * (larger) output extent. `out_shape4` is the broadcast result shape.
 *
 * Replaces MIOpen's miopenOpTensor for the float/half path, which is
 * pathologically slow on gfx1151 for vision-encoder elementwise shapes.
 *
 *   op: 0 = add, 1 = mul, 2 = min, 3 = max
 *
 * Supported hip_dtype: HIP_DTYPE_FLOAT32, HIP_DTYPE_FLOAT16.
 * Returns: 0 on success, -1 on unsupported dtype / launch error, -2 when the
 * output volume exceeds the 32-bit index range (caller should fall back).
 */
HIP_KERNEL_API int hip_elementwise_binary_bcast(
    void* stream,
    const void* lhs,
    const void* rhs,
    void* output,
    const int64_t* lhs_shape4,
    const int64_t* rhs_shape4,
    const int64_t* out_shape4,
    int op,
    int hip_dtype);

/*
 * Element-wise Equal with optional scalar broadcast.
 *
 * `lhs_num_elements` / `rhs_num_elements` may each be 1 (scalar broadcast)
 * or equal to `out_num_elements` (no broadcast). Other mismatches are
 * rejected by the host wrapper.
 *
 * Output type is always uint8 (1-byte bool).
 */
HIP_KERNEL_API int hip_elementwise_equal(
    void* stream,
    const void* lhs,
    const void* rhs,
    void* output,
    int64_t lhs_num_elements,
    int64_t rhs_num_elements,
    int64_t out_num_elements,
    int hip_dtype);

HIP_KERNEL_API int hip_elementwise_less(
    void* stream,
    const void* lhs,
    const void* rhs,
    void* output,
    int64_t num_elements,
    int hip_dtype);

/* And over bool (1-byte) tensors. No hip_dtype: bool is the only supported
 * input/output type (mirrors ORT v1.22.2 SPECIALIZED_BINARY_ELEMENTWISE_IMPL(And, bool)).
 */
HIP_KERNEL_API int hip_elementwise_or(
    void *stream, const void *lhs, const void *rhs, void *output,
    int64_t num_elements);

HIP_KERNEL_API int hip_elementwise_and(
    void* stream,
    const void* lhs,
    const void* rhs,
    void* output,
    int64_t num_elements);

/* =========================================================================
 * Elementwise reciprocal (1 / x)
 * =========================================================================
 *
 * ONNX Reciprocal over the full IEEE domain (including negative x and x=0
 * per IEEE rules). Implemented as a plain HIP kernel. MIOpen
 * miopenActivationPOWER (used for other wrap_power cases) does not match
 * ONNX 1/x for negative inputs; see lib/Runtime/real/power.cpp.
 *
 * Supported hip_dtype: HIP_DTYPE_FLOAT32, HIP_DTYPE_FLOAT16, HIP_DTYPE_BFLOAT16
 * Returns: 0 on success (hipSuccess), non-zero hipError_t on failure
 */
HIP_KERNEL_API int hip_elementwise_reciprocal(
    void* stream,
    const void* input,
    void* output,
    int64_t num_elements,
    int hip_dtype);

/* =========================================================================
 * Elementwise sqrt (ONNX Sqrt / IEEE)
 * =========================================================================
 *
 * Element-wise square root via HIP (sqrtf / promote half and bf16 to float).
 * Matches ONNX: negative inputs yield NaN; positive domain follows IEEE.
 * hip.sqrt lowers to wrap_power(0, 1, 0.5) which dispatches here instead of
 * MIOpen miopenActivationPOWER (see lib/Runtime/real/power.cpp).
 *
 * Supported hip_dtype: HIP_DTYPE_FLOAT32, HIP_DTYPE_FLOAT16, HIP_DTYPE_BFLOAT16
 * Returns: 0 on success (hipSuccess), non-zero hipError_t on failure
 */
HIP_KERNEL_API int hip_elementwise_sqrt(
    void* stream,
    const void* input,
    void* output,
    int64_t num_elements,
    int hip_dtype);

/* =========================================================================
 * Elementwise GELU (Gaussian Error Linear Unit)
 * =========================================================================
 *
 * Element-wise GELU activation via HIP with support for exact and approximate modes.
 *
 * Approximate mode (approximate=1, tanh):
 *   Formula: GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
 *   Standard approximation used in PyTorch, TensorFlow, and ONNX.
 *
 * Exact mode (approximate=0, erf, default):
 *   Formula: GELU(x) = x * 0.5 * (1.0 + erf(x / sqrt(2.0)))
 *   Matches ONNX Gelu operator spec exactly.
 *
 * Parameters:
 *   stream       - hipStream_t cast to void*
 *   input        - GPU pointer to input
 *   output       - GPU pointer to output
 *   num_elements - number of elements
 *   hip_dtype    - data type (HIP_DTYPE_FLOAT32, HIP_DTYPE_FLOAT16,
 *                  HIP_DTYPE_BFLOAT16, HIP_DTYPE_FLOAT64)
 *   approximate  - 0 for exact (erf), 1 for tanh approximation
 *
 * Supported data types (per ONNX spec):
 *   - HIP_DTYPE_FLOAT32 (float32)
 *   - HIP_DTYPE_FLOAT16 (float16)
 *   - HIP_DTYPE_BFLOAT16 (bfloat16)
 *   - HIP_DTYPE_FLOAT64 (double/float64)
 *
 * Returns: 0 on success (hipSuccess), non-zero hipError_t on failure
 */
HIP_KERNEL_API int hip_elementwise_gelu(
    void* stream,
    const void* input,
    void* output,
    int64_t num_elements,
    int hip_dtype,
    int64_t approximate);

/* =========================================================================
 * BiasGelu (fused bias-add + erf GELU)
 * =========================================================================
 *
 * output[i] = Gelu_erf(data[i] + bias[i % bias_len])
 */
HIP_KERNEL_API int hip_bias_gelu(void *stream, const void *data, const void *bias,
                                 void *output, int64_t num_elements,
                                 int64_t bias_len, int hip_dtype);

HIP_KERNEL_API int hip_fast_gelu(void *stream, const void *data, const void *bias,
                                 void *output, int64_t num_elements,
                                 int64_t bias_len, int hip_dtype);

/* =========================================================================
 * LeakyRelu Activation
 * =========================================================================
 *
 * Applies LeakyRelu element-wise: y = x >= 0 ? x : alpha * x
 *
 * Parameters:
 *   stream       - HIP stream (cast to hipStream_t internally)
 *   input        - Device pointer to input tensor
 *   output       - Device pointer to output tensor
 *   num_elements - Total number of elements
 *   hip_dtype    - Data type enum (HIP_DTYPE_FLOAT32, HIP_DTYPE_FLOAT16,
 *                  HIP_DTYPE_FLOAT64)
 *   alpha        - Slope for negative values (default 0.01 per ONNX spec)
 *
 * Returns: 0 on success (hipSuccess), non-zero hipError_t on failure
 */
HIP_KERNEL_API int hip_leaky_relu(
    void* stream,
    const void* input,
    void* output,
    int64_t num_elements,
    int hip_dtype,
    double alpha);

/* =========================================================================
 * Rotary Position Embedding (RoPE)
 * =========================================================================
 *
 * Applies rotary position embeddings to the input tensor.
 *
 * For each (batch, seq_pos, head, dim_pair d):
 *   cos_val = cos_cache[pos, d]
 *   sin_val = sin_cache[pos, d]
 *
 *   Non-interleaved (half-rotated):
 *     x0 = input[..., d],  x1 = input[..., d + rotary_dim/2]
 *     output[..., d]                 = x0 * cos_val - x1 * sin_val
 *     output[..., d + rotary_dim/2]  = x0 * sin_val + x1 * cos_val
 *
 *   Interleaved:
 *     x0 = input[..., 2*d],  x1 = input[..., 2*d+1]
 *     output[..., 2*d]   = x0 * cos_val - x1 * sin_val
 *     output[..., 2*d+1] = x0 * sin_val + x1 * cos_val
 *
 * When rotary_dim < head_dim, dimensions [rotary_dim, head_dim) are passed
 * through unchanged (the half-rotated kernel writes them in the d>=rotary_dim
 * branch; the interleaved path uses a separate copy kernel).
 *
 * Parameters:
 *   stream             - hipStream_t cast to void*
 *   input              - GPU pointer; layout depends on is_bnsh:
 *                          is_bnsh=0 -> BSNH [batch, seq_len, num_heads, head_dim]
 *                                       (also the 3D [B, S, num_heads*head_dim])
 *                          is_bnsh=1 -> BNSH [batch, num_heads, seq_len, head_dim]
 *   position_ids       - GPU pointer [batch, seq_len] (int64), or NULL when
 *                        cos/sin are already position-expanded (see cos_cache)
 *   cos_cache          - GPU pointer. position_ids != NULL: 2D lookup table
 *                        [max_seq, rotary_dim/2] indexed by position_ids[b,s].
 *                        position_ids == NULL: precomputed
 *                        [batch, seq_len, rotary_dim/2] indexed by the flat
 *                        token position b*seq+s.
 *   sin_cache          - GPU pointer, same shape convention as cos_cache
 *   output             - GPU pointer (same shape/layout as input)
 *   batch_size         - batch dimension
 *   seq_len            - sequence length
 *   num_heads          - number of attention heads
 *   head_dim           - dimension per head (>= rotary_dim)
 *   rotary_dim         - number of dimensions to rotate (<= head_dim)
 *   max_seq_len        - max sequence length in cos/sin cache (for bounds clamping)
 *   interleaved        - 0 = half-rotated, 1 = interleaved
 *   element_size_bytes - 2 for fp16, 4 for fp32
 *   is_bnsh            - layout flag, see input above (0 = BSNH/3D, 1 = BNSH)
 *
 * Returns: 0 on success, non-zero on error
 */
HIP_KERNEL_API int hip_rope_forward(
    void* stream,
    const void* input,
    const void* position_ids,
    const void* cos_cache,
    const void* sin_cache,
    void* output,
    int64_t batch_size,
    int64_t seq_len,
    int64_t num_heads,
    int64_t head_dim,
    int64_t rotary_dim,
    int64_t max_seq_len,
    int64_t interleaved,
    int64_t element_size_bytes,
    int64_t is_bnsh);

/* =========================================================================
 * GQA Device Kernel Launchers
 * =========================================================================
 *
 * Individual kernel launchers for the 12-step GQA pipeline (Step 0 + Steps 1-11).
 * The pure data-movement kernels (append / concat / rope / transpose / expand /
 * split) take element_size_bytes (2 = fp16, 4 = fp32) and dispatch to the
 * matching typed kernel -- this fp32-enables the decomposed GQA pipeline used
 * by the Whisper no_causal path. The fused / flash decode kernels remain FP16
 * only (Llama / gpt-oss). The orchestration (hipBLASLt GEMMs, workspace, temp
 * buffers) lives in the runtime wrapper (real/gqa.cpp).
 */

/* KV-cache element format for the fused/append/concat/decode GQA kernels. This
 * is the single ABI selector the kernels dispatch on -- callers pass ONE enum
 * instead of a per-type boolean or inferring the type from which pointers are
 * non-null. Adding a new quantized cache (INT4, FP8, ...) means: add an
 * enumerator here, add the matching kernel specialization, and add one switch
 * case in the entry -- no new parameter and no change to existing call sites'
 * shapes. Keep in sync with real/gqa.cpp::KvCacheFormat (via kv_dtype_abi). */
typedef enum {
  HIP_KV_DTYPE_FP16 = 0, /* unquantized __half cache */
  HIP_KV_DTYPE_INT8 = 1, /* symmetric per-channel int8, fp32 scale [G,d] */
} hip_kv_dtype_t;

/* KV cache append: scatter new K/V from BSHD [B,sq,G,d] into an existing
 * BNSD cache [B,G,present_seq,d] at positions [past_len .. past_len+sq).
 * present_seq is the actual sequence dimension (stride) of the present buffer,
 * which may be larger than past_len+sq if the buffer is pre-allocated.
 * Use when past and present share the same buffer (aliased / in-place).
 * seqlens_k: optional device pointer [B] int32. When non-null, past_len is
 * derived from seqlens_k[b]+1-sq (per-batch) and the host past_len is ignored.
 * Pass NULL for host-side past_len.
 * element_size_bytes: 2 = fp16, 4 = fp32 (used only by the FP16 copy path).
 * kv_dtype: hip_kv_dtype_t selecting the cache format:
 *   HIP_KV_DTYPE_FP16 -> plain elem-size copy/transpose (fp16 or fp32 cache);
 *                        scale is ignored (pass NULL).
 *   HIP_KV_DTYPE_INT8 -> symmetric per-channel INT8 quant (fp16 src, int8
 *                        cache); scale is the static fp32 per-channel table
 *                        [G,d] (no zero point), q = clamp(round(x/scale),-128,127). */
HIP_KERNEL_API int hip_gqa_kv_cache_append(
    void* stream, const void* src, void* cache,
    int batch_size, int sq, int G, int d, int present_seq, int past_len,
    const void* seqlens_k, int element_size_bytes,
    int kv_dtype, const void* scale);

/* KV cache concat: concatenate past data and new tokens into a fresh present
 * buffer.  Fills present [B,G,present_seq,d] by copying past data from
 * past [B,G,past_seq,d] at positions [0,past_len) AND transposing new tokens
 * from current BSHD [B,sq,G,d] at [past_len,past_len+sq).
 * past_seq and present_seq are the actual sequence dimensions (strides) of the
 * respective buffers.  Handles the stride mismatch (past_seq != present_seq)
 * in a single kernel launch.
 * element_size_bytes: 2 = fp16, 4 = fp32 (used only by the FP16 copy path).
 * kv_dtype / scale select the storage format exactly as in
 * hip_gqa_kv_cache_append (HIP_KV_DTYPE_FP16 -> plain copy; HIP_KV_DTYPE_INT8 ->
 * INT8 quant of the new fp16 tokens, past then read as INT8, scale = [G,d]). */
HIP_KERNEL_API int hip_gqa_kv_cache_concat(
    void* stream, const void* past, const void* current, void* present,
    int batch_size, int past_len, int sq, int G, int d,
    int past_seq, int present_seq, int element_size_bytes,
    int kv_dtype, const void* scale);

/* INT8 KV cache (symmetric per-channel, kv_cache_bit_width=8) dequant.
 * dequant_kv_i8_to_fp16: rebuilds an fp16 BNSD view [B,G,dst_seq,d] of the first
 * `total_seq` cache positions (x = q * scale) so the compute-bound prefill can
 * reuse the tuned fp16 kernel. `scale` is the static fp32 per-channel table
 * [G,d] (no zero point). Q stays fp16 throughout. (Append/concat quantization
 * is folded into hip_gqa_kv_cache_append/concat via their kv_dtype argument.) */
HIP_KERNEL_API int hip_gqa_dequant_kv_i8_to_fp16(
    void* stream, const void* src, void* dst, const void* scale,
    int batch_size, int total_seq, int G, int d, int src_seq, int dst_seq);

/* Internal GQA RoPE (half-rotated):
 * out[d] = in[d]*cos - in[d+half]*sin
 * out[d+half] = in[d+half]*cos + in[d]*sin
 * seqlens_k: optional device pointer [B] int32. When non-null, past_len is
 * derived from seqlens_k[b]+1-seq_len and the host past_len is ignored.
 * element_size_bytes: 2 = fp16, 4 = fp32. */
HIP_KERNEL_API int hip_gqa_rope(
    void* stream, const void* input, void* output,
    const void* cos_cache, const void* sin_cache,
    int batch_size, int seq_len, int num_heads,
    int head_dim, int half_rot, int past_len,
    const void* seqlens_k, int element_size_bytes);

/* Transpose middle two dims of 4D tensor:
 * [B, dim1, dim2, D] -> [B, dim2, dim1, D]
 * element_size_bytes: 2 = fp16, 4 = fp32. */
HIP_KERNEL_API int hip_gqa_transpose_mid_dims(
    void* stream, const void* src, void* dst,
    int batch_size, int dim1, int dim2, int D, int element_size_bytes);

/* KV group expansion: replicate G groups -> H heads.
 * For head h, copies from group g = h / heads_per_group.
 * element_size_bytes: 2 = fp16, 4 = fp32. */
HIP_KERNEL_API int hip_gqa_expand_kv(
    void* stream, const void* src, void* dst,
    int total_heads, int heads_per_group,
    int src_stride, int dst_stride, int copy_elems, int element_size_bytes);

/* Split packed QKV [B*S, (H+2*G)*d] into separate Q, K, V buffers.
 * Q: [B*S, H*d], K: [B*S, G*d], V: [B*S, G*d]
 * element_size_bytes: 2 = fp16, 4 = fp32. */
HIP_KERNEL_API int hip_gqa_split_qkv(
    void* stream, const void* packed, void* Q, void* K, void* V,
    int batch_size, int seq_len, int num_heads, int kv_num_heads, int head_dim,
    int element_size_bytes);

/* Causal mask (prefill only): S[k,q] = -inf where k > past_len + q.
 * When local_window_size > 0, also masks k < past_len + q - local_window_size + 1. */
HIP_KERNEL_API int hip_gqa_causal_mask(
    void* stream, void* S,
    int total_heads, int skv, int sq,
    int batch_stride, int past_len, int local_window_size);

/* Causal mask on fp32 Score matrix.  Same semantics as hip_gqa_causal_mask
 * but operates on float* and writes -INFINITY instead of -65504. */
HIP_KERNEL_API int hip_gqa_causal_mask_f32(
    void* stream, void* S,
    int total_heads, int skv, int sq,
    int batch_stride, int past_len, int local_window_size);

/* Add an attention bias onto the fp32 score matrix before softmax.
 * scores layout: [total_heads, sq, total_seq] row-major with batch_stride
 * = sq * total_seq per head.
 * bias layout: [bias_batch, bias_heads, sq, total_seq], row-major.
 * bias_batch / bias_heads may be 1 for ONNX-style broadcast. */
HIP_KERNEL_API int hip_gqa_add_attention_bias_f32(
    void* stream, void* scores, const void* bias,
    int total_heads, int num_heads, int bias_batch, int bias_heads,
    int sq, int total_seq, int score_batch_stride, int bias_element_size_bytes);

/* Column-wise softmax in-place. One threadblock per (head, query).
 * Smooth softmax is activated when head_sink is non-null OR use_smooth_softmax
 * is set.  When head_sink is non-null, uses per-head sink factors:
 *   softmax_i = exp(x_i) / (exp(head_sink[h]) + sum_j exp(x_j))
 * When head_sink is null but use_smooth_softmax is set, uses sink = 0:
 *   softmax_i = exp(x_i) / (exp(0) + sum_j exp(x_j)) */
HIP_KERNEL_API int hip_gqa_softmax_inplace(
    void* stream, void* data,
    int total_head_queries, int rows, int cols,
    int batch_stride, const void* head_sink, int num_heads,
    int use_smooth_softmax);

/* Row-wise softmax over a flattened [rows, cols] row-major fp16 buffer.
 * One block per row, softmaxes the `cols` elements of each row in-place
 * (data is overwritten with normalized probabilities). Matches ONNX
 * Softmax semantics for axis = -1 on the flattened input. Used by the
 * standalone `hip_miopen_softmax` runtime entry point. */
HIP_KERNEL_API int hip_softmax_row_2d_inplace(void* stream, void* data, int rows, int cols);

/* Column-wise softmax: fp32 input -> fp16 output.
 * Reads fp32 Score matrix (no fp16 overflow/inf), writes fp16 probabilities.
 * input_batch_stride is in float elements, output_batch_stride in half elements. */
HIP_KERNEL_API int hip_gqa_softmax_f32_to_f16(
    void* stream, const void* input_f32, void* output_f16,
    int total_head_queries, int rows, int cols,
    int input_batch_stride, int output_batch_stride,
    const void* head_sink, int num_heads, int use_smooth_softmax);

/* Column-wise softmax: fp32 input -> fp32 output.
 * Same as hip_gqa_softmax_f32_to_f16 but writes fp32 probabilities, feeding
 * the fp32 Value GEMM on the Whisper no_causal fp32 decomposed path.
 * input_batch_stride and output_batch_stride are both in float elements. */
HIP_KERNEL_API int hip_gqa_softmax_f32_to_f32(
    void* stream, const void* input_f32, void* output_f32,
    int total_head_queries, int rows, int cols,
    int input_batch_stride, int output_batch_stride,
    const void* head_sink, int num_heads, int use_smooth_softmax);

/* Legacy fast-path decode kernels (folded into gqa_kernel.hip with legacy_*
 * device kernels). The production fused decode path uses hip_gqa_flash_decode_v2
 * above; these two entries back gqa.cpp::gqa_forward_hipblaslt -- the decomposed
 * hipBLASLt fallback -- for the cases the fused path does not cover. They MUST be
 * exported (HIP_KERNEL_API) so the EP resolves them out of custom_kernels_<arch>
 * at JIT link / native import (same as every other launcher here).
 *
 * hip_gqa_fused_decode: one-block-per-(batch,head_q) decode, d in {64,128,256},
 * arbitrary HpG -- the general small-/odd-geometry decode used when flash is not
 * eligible. */
HIP_KERNEL_API int hip_gqa_fused_decode(
    void* stream, const void* Q, const void* Kcache, const void* Vcache,
    void* O, int B, int H, int G, int d, int skv, int max_seq,
    float scale, const void* seqlens_k);

/* hip_gqa_flash_decode: FA-2 split-K (scalar|WMMA), HPG in {4,8}, with
 * sliding-window + head-sink / smooth-softmax -- the fast windowed/sink decode
 * the fallback uses. partials_workspace: float scratch B*H*K_SPLITS*(d+2). */
HIP_KERNEL_API int hip_gqa_flash_decode(
    void* stream, const void* Q, const void* Kcache, const void* Vcache,
    void* O, void* partials_workspace,
    int B, int H, int G, int d, int max_seq, int K_SPLITS,
    float scale, const void* seqlens_k, int local_window_size,
    const void* head_sink, int use_smooth_softmax);

/* Optimized fused GQA prefill (sq > 1, d in {64,128}, fp16, causal, GQA) --
 * Flash-Attention-2 with WMMA tile GEMMs and intra-wave online softmax. These
 * reproduce the gqa_compare TTFT-winning kernels (OPTIMIZATION.md ch.13):
 *   v5: warp-private / register-resident (best at d == 64).
 *   v7: M-register-blocked, global-streamed K (best at d == 128).
 * Each self-tunes its launch config on the first call per (d,sq,skv,Hq,G)
 * shape and caches the winner process-wide. Layout: Q [B,sq,Hq,d];
 * K/V cache [B,G,max_seq,d] (post-RoPE, post-append). Returns 0 on success,
 * -1 if d is unsupported. No seqlens_k / sliding-window / head-sink / smooth
 * softmax / softcap -- caller must gate those out (use the decomposed path). */
HIP_KERNEL_API int hip_gqa_flash_prefill_v5_v2(
    void* stream, const void* Q, const void* Kcache, const void* Vcache,
    void* O, int B, int Hq, int G, int sq, int skv, int d, int max_seq,
    int past_len, float scale);

HIP_KERNEL_API int hip_gqa_flash_prefill_v7_v2(
    void* stream, const void* Q, const void* Kcache, const void* Vcache,
    void* O, int B, int Hq, int G, int sq, int skv, int d, int max_seq,
    int past_len, float scale);

/* Unified fused flash-prefill entry: picks v5 (d==64) or v7 (d==128) internally
 * so the runtime calls one symbol for any eligible prefill. Same layout /
 * constraints as v5/v7 above. Returns 0 on success, -1 if d is unsupported. */
HIP_KERNEL_API int hip_gqa_flash_prefill_v2(
    void* stream, const void* Q, const void* Kcache, const void* Vcache,
    void* O, int B, int Hq, int G, int sq, int skv, int d, int max_seq,
    int past_len, float scale);

/* NB: prefill is compute-bound, so there is deliberately NO separate int8
 * prefill kernel. The runtime (real/gqa.cpp) dequantizes the int8 KV cache to an
 * fp16 scratch ONCE (hip_gqa_dequant_kv_i8_to_fp16) and reuses the tuned fp16
 * hip_gqa_flash_prefill_v2 above -- ~parity with fp16. Only the bandwidth-bound
 * decode reads int8 directly (hip_gqa_flash_decode_v2 with kv_dtype=INT8). */

/* hip_mha_flash_prefill: fused non-causal FA-2 WMMA prefill for the MS
 * MultiHeadAttention contrib op (self-attention, N_q == N_kv). Replaces the
 * decomposed hipBLASLt pipeline that materializes the fp32 score matrix
 * S[B,N,sq,skv] in DRAM (~3.4 GB for the Qwen VLM vision encoder). Keeps the
 * running (m, l, O) softmax state in registers; K streamed from global, V/P
 * staged in LDS; score/value GEMMs on the RDNA3.5 WMMA unit. Head dim d need
 * not be a multiple of 16 (padded to next multiple, tail tile masked-loaded).
 * Layout: Q [B,sq,N,d] (BSND); K/V cache [B,N,max_seq,d] (BNSD); O [B,sq,N,d].
 * fp16 only, bidirectional (non-causal), past_len == 0. Returns 0 on success,
 * -1 if the (padded) head dim is unsupported (> 256). */
HIP_KERNEL_API int hip_mha_flash_prefill(
    void* stream, const void* Q, const void* Kcache, const void* Vcache,
    void* O, int B, int N, int sq, int skv, int d, int max_seq, float scale);

/* FA-2 split-K GQA decode (sq == 1, d in {64, 128}, HPG=H/G==4):
 * GQA-aware kernel that loads K/V tiles into LDS once and reuses them
 * across the 4 query heads of each KV group, then a second kernel
 * merges K_SPLITS partial (m, l, O) per query head.
 *
 * Depth-gated alternative to hip_gqa_fused_decode for skv >= ~256 where
 * Llama-3.x family shows large bandwidth headroom over the existing
 * one-block-per-head fused decode.
 *
 * Workspace: float scratch sized B*H*max_splits*(d+2)*sizeof(float) bytes.
 * Caller is responsible for allocating and passing it in.
 *
 * skv: current total KV length (host-known). Used only to autotune the split
 * count / impl and key the per-shape cache; the kernels read seqlens_k for the
 * exact per-batch length at runtime. Pass <= 0 to fall back to max_seq.
 *
 * max_splits: workspace capacity in splits. The launcher autotunes the actual
 * split count (8..max_splits, capped at 64) and the scalar-vs-WMMA impl on the
 * first call per (B,H,G,d, skv-bucket) shape, caches the winner in a
 * process-wide map (matmul_nbits-style), and reuses it on later calls.
 * Env overrides skip autotune: HIPDNN_GQA_DECODE_SCALAR / _WMMA (impl),
 * HIPDNN_GQA_DECODE_SPLITS=N (count), HIPDNN_GQA_DECODE_NOAUTOTUNE=1 (fixed).
 *
 * seqlens_k: optional device pointer [B] int32. When non-null, total_seq
 * = seqlens_k[b]+1 is read on-device (no host sync).
 *
 * local_window_size: when > 0, restricts each query to attend only to the
 * last `local_window_size` KV positions (sliding-window attention, e.g.
 * gpt-oss-20b's 128-token sliding layers). When <= 0, full attention.
 *
 * head_sink: optional device pointer [num_heads] fp16, attention-sink
 * (smooth-softmax) per-head bias. When non-null, the final softmax
 * denominator gains an exp(s_h - global_m) term per head (no V contribution
 * for the sink). When null and use_smooth_softmax != 0, behaves as if
 * s_h = 0 for all heads. This is the gpt-oss-20b / Mistral-style attention
 * sink. The partials are unaffected; the term is folded in by the reduce
 * kernel.
 *
 * KV-cache format is chosen by kv_dtype (hip_kv_dtype_t), so one entry serves
 * every format:
 *   HIP_KV_DTYPE_FP16 -> fp16 K/V cache (Kcache/Vcache are __half); k_scale/
 *     v_scale ignored (pass NULL).
 *   HIP_KV_DTYPE_INT8 -> symmetric per-channel INT8 K/V cache: Kcache/Vcache are
 *     INT8 [B,G,max_seq,d] (BNSD); k_scale/v_scale are fp32 [G,d] (one scale per
 *     (kv_head, head_dim) channel, no zero point; dequant x_fp16 = x_i8 *
 *     scale[g*d + c]). The int8 read is 1 byte/elem, halving the DRAM traffic on
 *     the bandwidth-bound decode. Both scales are required for INT8. */
HIP_KERNEL_API int hip_gqa_flash_decode_v2(
    void* stream,
    const void* Q, const void* Kcache, const void* Vcache,
    void* O,
    void* partials_workspace,
    int B, int H, int G, int d, int skv, int max_seq, int max_splits,
    float scale,
    const void* seqlens_k,
    int local_window_size,
    const void* head_sink,
    int use_smooth_softmax,
    int kv_dtype,
    const void* k_scale,
    const void* v_scale);

/* =========================================================================
 * Cast (Element Type Conversion)
 * =========================================================================
 *
 * Converts each element from input_dtype to output_dtype.
 *   output[i] = (output_type)input[i]
 *
 * Parameters:
 *   stream       - hipStream_t cast to void*
 *   input        - GPU pointer to source data
 *   output       - GPU pointer to destination
 *   num_elements - number of elements to convert
 *   input_dtype  - source data type (hip_dtype_t value cast to int)
 *   output_dtype - destination data type (hip_dtype_t value cast to int)
 *
 * Currently supported conversions: INT64 -> INT32
 * Returns: 0 on success, non-zero on failure
 */
HIP_KERNEL_API int hip_cast(
    void* stream,
    const void* input,
    void* output,
    int64_t num_elements,
    int input_dtype,
    int output_dtype);

/* =========================================================================
 * Gather (Index-Based Element Selection)
 * =========================================================================
 *
 * Gathers slices from data along the given axis using indices.
 * For axis=0 with scalar index:
 *   output[i] = data[index * output_num_elements + i]
 *
 * Parameters:
 *   stream              - hipStream_t cast to void*
 *   data                - GPU pointer to source tensor
 *   indices             - GPU pointer to index tensor (i64 values)
 *   output               - GPU pointer to output
 *   axis                 - axis along which to gather
 *   data_num_elements    - total elements in data tensor
 *   indices_num_elements - total elements in indices tensor
 *   output_num_elements  - total elements in output tensor
 *   element_size_bytes   - byte size per element (used for raw copy)
 *
 * Generic axis support. data has logical shape [outer, axis_size, inner];
 * output has logical shape [outer, indices_num, inner]. The caller computes
 * axis_size = data.shape[axis] and inner_size = product(data.shape[axis+1:]);
 * outer_size is derived as data_num / (axis_size * inner_size).
 * Supported element sizes: 2 (f16/bf16), 4 (f32/i32), 8 (i64/f64)
 * Returns: 0 on success, non-zero on failure
 */
HIP_KERNEL_API int hip_gather(
    void* stream,
    const void* data,
    const void* indices,
    void* output,
    int64_t axis,
    int64_t data_num_elements,
    int64_t indices_num_elements,
    int64_t output_num_elements,
    int64_t axis_size,
    int64_t inner_size,
    int element_size_bytes,
    int indices_element_size_bytes);

HIP_KERNEL_API int hip_gather_elements(
    void* stream,
    const void* data,
    const void* indices,
    void* output,
    int64_t axis,
    int64_t rank,
    const int64_t* data_shape,
    const int64_t* indices_shape,
    int64_t num_elements,
    int element_size_bytes,
    int indices_element_size_bytes);

HIP_KERNEL_API int hip_top_k(void* stream, const void* data, void* values,
                             void* indices, int64_t axis, int64_t largest,
                             int64_t sorted, int64_t rank,
                             const int64_t* x_shape, int64_t k,
                             int element_size_bytes);

HIP_KERNEL_API int hip_scatter_elements(
    void* stream,
    const void* data,
    const void* indices,
    const void* updates,
    void* output,
    int64_t axis,
    int64_t reduction_id,
    int64_t rank,
    const int64_t* data_shape,
    const int64_t* indices_shape,
    int64_t num_updates,
    int element_size_bytes,
    int indices_element_size_bytes);

HIP_KERNEL_API int hip_compress(
    void* stream,
    const void* input,
    const void* condition,
    void* output,
    int64_t flatten,
    int64_t axis,
    int64_t input_rank,
    int64_t output_rank,
    const int64_t* input_shape,
    const int64_t* output_shape,
    int64_t condition_len,
    int64_t num_output_elements,
    void* workspace,
    size_t workspace_bytes,
    int element_size_bytes);

HIP_KERNEL_API int hip_one_hot(
    void* stream,
    const void* indices,
    const void* depth,
    const void* values,
    void* output,
    int64_t axis,
    int64_t indices_rank,
    int64_t output_rank,
    const int64_t* indices_shape,
    const int64_t* output_shape,
    int64_t num_indices,
    int64_t num_output_elements,
    int64_t depth_scalar,
    int element_size_bytes,
    int indices_element_size_bytes);

/* =========================================================================
 * ReduceSum (Parallel Sum Reduction)
 * =========================================================================
 *
 * Reduces input by summing over contiguous blocks.
 * reduce_size = num_input_elements / num_output_elements
 * For each output element j:
 *   output[j] = sum(input[j*reduce_size .. (j+1)*reduce_size - 1])
 *
 * Parameters:
 *   stream              - hipStream_t cast to void*
 *   data                - GPU pointer to input tensor
 *   output              - GPU pointer to output tensor
 *   num_input_elements  - total input elements
 *   num_output_elements - total output elements
 *   hip_dtype           - data type (hip_dtype_t value cast to int)
 *
 * Currently supported types: HIP_DTYPE_INT64, HIP_DTYPE_INT32, HIP_DTYPE_FLOAT16
 *   - INT32 accumulates in int64 internally to avoid overflow on large slices.
 *   - FLOAT16 accumulates in float internally to preserve precision; the
 *     final result is narrowed back to half.
 * Returns: 0 on success, non-zero on failure
 */
/* `inner_size` = product of input dims AFTER the reduced axis (1 for a
 * trailing/contiguous reduce). Enables reducing a non-trailing axis (e.g.
 * channel-axis LayerNorm2d over NCHW): reduced elements are strided by
 * `inner_size`. inner_size==1 preserves the contiguous fast path. */
HIP_KERNEL_API int hip_reduce_sum(
    void* stream,
    const void* data,
    void* output,
    int64_t num_input_elements,
    int64_t num_output_elements,
    int64_t inner_size,
    int hip_dtype);

/* =========================================================================
 * ReduceMean (Parallel Mean Reduction)
 * =========================================================================
 *
 * Same layout convention and `inner_size` semantics as hip_reduce_sum, but
 * divides the float-accumulated sum of each `reduce_size`-element slice by
 * reduce_size = num_input / num_output before narrowing to the output type. The
 * division is performed in-kernel so the op needs no compile-time-static reduce
 * dim and tolerates a dynamic reduce axis.
 *
 * Supported types: HIP_DTYPE_FLOAT16 and HIP_DTYPE_FLOAT32 (ONNX ReduceMean is
 * float-domain; both the true-fp16 path and fp32-upcast RMSNorm-style paths
 * feed this). Both accumulate in float. Other dtypes return -1.
 * Returns: 0 on success, non-zero on failure
 */
HIP_KERNEL_API int hip_reduce_mean(
    void* stream,
    const void* data,
    void* output,
    int64_t num_input_elements,
    int64_t num_output_elements,
    int64_t inner_size,
    int hip_dtype);

/* =========================================================================
 * ReduceL2 (Parallel L2 Norm Reduction)
 * =========================================================================
 *
 * Same layout convention and `inner_size` semantics as hip_reduce_sum, but
 * accumulates sum(x^2) in float and writes sqrt(sum) to the output. The
 * reduction is performed in-kernel so the op needs no compile-time-static reduce
 * dim and tolerates a dynamic reduce axis.
 *
 * Supported types: HIP_DTYPE_FLOAT16 and HIP_DTYPE_FLOAT32 (ONNX ReduceL2 is
 * float-domain). Both accumulate in float. Other dtypes return -1.
 * Returns: 0 on success, non-zero on failure
 */
HIP_KERNEL_API int hip_reduce_l2(
    void* stream,
    const void* data,
    void* output,
    int64_t num_input_elements,
    int64_t num_output_elements,
    int64_t inner_size,
    int hip_dtype);

/* =========================================================================
 * Pool — MaxPool / AveragePool / LpPool (1D / 2D / 3D)
 * =========================================================================
 *
 * Generic ONNX window pooling over an `(N, C, D_1[, D_2[, D_3]])` input.
 * Lays the output `(N, C, O_1[, O_2[, O_3]])` out in row-major order matching
 * the input layout.
 *
 * `mode` selects the per-window reduction (must match HIPDNN_EP_POOL_* in
 * lib/Runtime/hipdnn_ep_runtime.h):
 *   0 (AVERAGE): Y = sum(window) / divisor
 *   1 (MAX)    : Y = max(window)
 *   2 (LP)     : Y = pow(sum(pow(|window|, p)), 1/p)
 *
 * Pad positions are never read (they fall outside the input bounds). For
 * AVERAGE, `count_include_pad` picks the divisor: 0 = number of in-bounds
 * window elements, 1 = full kernel volume (pad cells contribute 0 to the
 * sum). `p` is the LP norm exponent (>= 1); both are ignored for the modes
 * that don't use them.
 *
 * Optional `indices` (i64 buffer the same shape as the output) records the
 * row-major flat index in the *unpadded* input that each max came from —
 * MAX mode only; matches ONNX MaxPool spec for storage_order = 0. Pass NULL
 * for AVERAGE / LP.
 *
 * `spatial_rank` selects how many of the per-axis arrays are read; for
 * spatial_rank < 3 the trailing slots in `in_d`, `out_d`, `kernel`,
 * `strides`, `pads_begin`, `dilations` must be set to 1 / 0 by the caller
 * (the lowering does this).
 *
 * Supported hip_dtypes: HIP_DTYPE_FLOAT32, HIP_DTYPE_FLOAT16,
 * HIP_DTYPE_BFLOAT16, HIP_DTYPE_FLOAT64.
 * Returns: 0 on success, non-zero on failure.
 */
HIP_KERNEL_API int hip_pool(
    void* stream,
    const void* input,
    void* output,
    void* indices,            /* int64_t* — nullable, MAX only */
    int hip_dtype,
    int mode,
    int spatial_rank,
    int64_t N, int64_t C,
    int64_t in_d0, int64_t in_d1, int64_t in_d2,
    int64_t out_d0, int64_t out_d1, int64_t out_d2,
    int64_t k0, int64_t k1, int64_t k2,
    int64_t s0, int64_t s1, int64_t s2,
    int64_t p0, int64_t p1, int64_t p2,
    int64_t dil0, int64_t dil1, int64_t dil2,
    int count_include_pad,
    int p);


 /* =========================================================================
 * Resize (1D / 2D / 3D spatial)
 * =========================================================================
 *
 * Resamples the trailing spatial axes of an `(N, C, D_1, ..., D_k)` input
 * onto an `(N, C, O_1, ..., O_k)` output grid.  Per-axis scale is computed
 * inside the kernel as `scale = in_dim / out_dim`.  The (N, C) prefix is
 * pass-through.
 *
 *  mode:               0 = nearest, 1 = linear (N-linear)
 *  coord_transform:    0 = half_pixel, 1 = asymmetric, 2 = align_corners
 *  nearest_mode:       0 = round_prefer_floor (only used when mode=nearest)
 *
 * Supported hip_dtypes: HIP_DTYPE_FLOAT32, HIP_DTYPE_FLOAT16,
 * HIP_DTYPE_BFLOAT16, HIP_DTYPE_FLOAT64.
 * Returns: 0 on success, non-zero on failure.
 */
HIP_KERNEL_API int hip_resize(
    void* stream,
    const void* input,
    void* output,
    int hip_dtype,
    int spatial_rank,
    int64_t N, int64_t C,
    int64_t in_d0, int64_t in_d1, int64_t in_d2,
    int64_t out_d0, int64_t out_d1, int64_t out_d2,
    int mode,
    int coord_transform,
    int nearest_mode);

/* =========================================================================
 * Global pool (avg / max / lp)
 * =========================================================================
 *
 * Reduces each contiguous `reduce_size`-element slice into a single value.
 * Data is viewed as `[outer, reduce_size]` where
 *   outer       = N * C
 *   reduce_size = D_1 * D_2 * ... * D_k   (product of all spatial dims)
 *
 * `mode` selects the reduction (must match HIPDNN_EP_GLOBAL_POOL_* in
 * lib/Runtime/hipdnn_ep_runtime.h):
 *   0 (AVERAGE): Y = mean(slice)
 *   1 (MAX)    : Y = max(slice)
 *   2 (LP)     : Y = pow(sum(pow(|slice|, p)), 1/p)
 *
 * `p` is the LP-norm exponent; ignored for AVG / MAX. Caller must guarantee
 * `p >= 1` for LP (the runtime wrapper rejects values below that).
 *
 * One reduction block per output element (per (n, c) slice). Accumulation
 * happens in float (regardless of input dtype) to keep precision on long
 * spatial reductions of fp16 / bf16 inputs.
 *
 * Supported hip_dtypes: HIP_DTYPE_FLOAT32, HIP_DTYPE_FLOAT16,
 * HIP_DTYPE_BFLOAT16, HIP_DTYPE_FLOAT64.
 * Returns: 0 on success, non-zero on failure
 */
HIP_KERNEL_API int hip_global_pool(
    void* stream,
    const void* data,
    void* output,
    int64_t outer,
    int64_t reduce_size,
    int hip_dtype,
    int mode,
    int p);

/* =========================================================================
 * Block reductions (Max / Prod) -- same layout convention as hip_reduce_sum.
 * =========================================================================
 *
 * Both share the structure: one block reduces `reduce_size = num_input /
 * num_output` consecutive input elements into a single output. The reduce
 * axes must already be collapsed into the trailing dimension of `data`
 * (the upstream lowering arranges this for us).
 *
 * - hip_reduce_max  : Max op, init = -INF (FP) / TYPE_MIN (INT). NaN propagating
 *                     on the FP path (matches ORT _Max<float>).
 * - hip_reduce_min  : Min op, init = +INF (FP) / TYPE_MAX (INT). NaN propagating
 *                     on the FP path (matches ORT _Min<float>).
 * - hip_reduce_prod : Mul op, init = 1.
 *
 * Supported hip_dtypes: HIP_DTYPE_INT32, HIP_DTYPE_INT64, HIP_DTYPE_FLOAT16
 * (FP16 accumulates in float, narrows on write).
 */
/* `inner_size`: see hip_reduce_sum (strided non-trailing-axis support). */
HIP_KERNEL_API int hip_reduce_max(
    void* stream,
    const void* data,
    void* output,
    int64_t num_input_elements,
    int64_t num_output_elements,
    int64_t inner_size,
    int hip_dtype);

HIP_KERNEL_API int hip_reduce_min(
    void* stream,
    const void* data,
    void* output,
    int64_t num_input_elements,
    int64_t num_output_elements,
    int64_t inner_size,
    int hip_dtype);

HIP_KERNEL_API int hip_reduce_prod(
    void* stream,
    const void* data,
    void* output,
    int64_t num_input_elements,
    int64_t num_output_elements,
    int64_t inner_size,
    int hip_dtype);

/* =========================================================================
 * Tile / Expand (shape replication)
 * =========================================================================
 *
 * Both ops copy `input` into a larger `output`. Shapes are passed as
 * host-side int64 arrays from the lowering, so neither op needs to D2H
 * the GPU-side shape / repeats tensors.
 *
 * - Tile  : output_shape[d] = input_shape[d] * repeats[d].
 *           in_coord[d] = out_coord[d] % input_shape[d].
 * - Expand: output_shape[d] is the broadcast result; any input dim that is 1
 *           is replicated. in_coord[d] = (in_shape[d] == 1) ? 0
 *                                       : out_coord[d - rank_diff].
 *
 * Both kernels are bounded to kTileMaxRank = 8 input/output dimensions
 * (matches ORT's TArray<int64, 8> default).
 */
HIP_KERNEL_API int hip_tile(
    void* stream,
    const void* input,
    void* output,
    const int64_t* input_shape_host,
    const int64_t* output_shape_host,
    int rank,
    int hip_dtype);

HIP_KERNEL_API int hip_expand(
    void* stream,
    const void* input,
    void* output,
    const int64_t* input_shape_host,
    int input_rank,
    const int64_t* output_shape_host,
    int output_rank,
    int hip_dtype);

/* =========================================================================
 * GatherND
 * =========================================================================
 *
 * Pick slices of `data` along the first K = indices.shape[-1] dims (after
 * `batch_dims`), one slice per row of `indices`. INT64 indices only.
 *
 * Shapes pass as host int64 arrays (no GPU shape D2H). K and the rank
 * decomposition are computed on the host; the kernel runs one thread per
 * output element and reads K indices inline (no scratch buffer).
 */
HIP_KERNEL_API int hip_gather_nd(
    void* stream,
    const void* input,
    const void* indices,
    void* output,
    const int64_t* data_shape_host,
    int data_rank,
    const int64_t* indices_shape_host,
    int indices_rank,
    int batch_dims,
    int hip_dtype);

/* =========================================================================
 * Slice (ONNX-13+ — non-constant indices / negative-step fallback)
 * =========================================================================
 *
 * The compile-time-constant + positive-stride case is folded to
 * `tensor.extract_slice` upstream of the runtime, so this kernel only
 * services slices whose `starts` / `ends` / `axes` / `steps` are NOT
 * graph-constant (or have negative steps).
 *
 * The host wrapper D2Hs the (typically tiny) index tensors and resolves
 * them into per-axis `(start, step)` pairs in INPUT-space, one entry per
 * data dimension. Axes not listed default to `(0, 1)`. The kernel runs
 * one thread per output element and computes:
 *
 *     in_offset = sum_d ( start[d] + out_coord[d] * step[d] ) * input_stride[d]
 *     output[out_idx] = input[in_offset]
 *
 * `step[d]` may be negative; correctness relies on the host wrapper
 * having already resolved start / end to absolute positions per ONNX's
 * negative-index and clamping rules (see lib/Runtime/real/slice.cpp).
 *
 * Bounded to rank <= 8 (matches kPadMaxRank / kGatherNDMaxRank).
 *
 * Supported dtypes: f16, f32, i32, i64.
 */
HIP_KERNEL_API int hip_slice(
    void* stream,
    const void* input,
    void* output,
    const int64_t* input_shape_host,
    const int64_t* output_shape_host,     /* physical alloc shape       */
    const int64_t* logical_extent_host,   /* per-axis actual slice extent;
                                             may be NULL, in which case the
                                             kernel treats it as identical to
                                             output_shape_host (i.e. no
                                             over-alloc; entire physical
                                             buffer is filled by the slice).
                                             When set and logical[d] <
                                             output_shape[d] for some d,
                                             positions in the over-allocated
                                             tail are filled with zero — the
                                             host wrapper does not need to
                                             pre-memset the buffer.        */
    const int64_t* starts_per_axis_host,  /* length = rank */
    const int64_t* steps_per_axis_host,   /* length = rank */
    int rank,
    int hip_dtype);

/* =========================================================================
 * ScatterND (ONNX-13+ with optional `reduction`)
 * =========================================================================
 *
 * Produces an output tensor with the shape of `data` whose values are
 * `data` copied, then `updates` overwritten / reduced into at positions
 * specified by `indices`.
 *
 * The host wrapper does the data->output D2D copy first (one
 * hipMemcpyAsync). The kernel then runs one thread per (updates_slice,
 * inner) pair = num_updates_slices * slice_size threads total. Each
 * thread reads K = indices.shape[-1] int64 indices inline and writes
 * one element into output.
 *
 *   num_updates_slices = product(indices.shape[:-1])
 *                      = product(updates.shape[:indices_rank-1])
 *   slice_size         = product(data.shape[K:])
 *
 * `reduction_id`:
 *   0 = none ("replace")  — last-writer-wins for duplicate indices,
 *                           matching ONNX's "undefined" guarantee.
 *   1 = add               — atomicAdd (or CAS-emulation for fp16).
 *   2 = mul               — CAS-emulated atomic multiply.
 *   3 = min               — CAS-emulated atomic min.
 *   4 = max               — CAS-emulated atomic max.
 *
 * Bounded to rank <= 8.
 *
 * Supported dtypes: f16, f32, i32, i64. INT64 indices only.
 */
HIP_KERNEL_API int hip_scatter_nd(
    void* stream,
    const void* data,
    const void* indices,
    const void* updates,
    void* output,
    const int32_t* count_ptr,
    const int64_t* data_shape_host,
    int data_rank,
    const int64_t* indices_shape_host,
    int indices_rank,
    int reduction_id,
    int hip_dtype);

/* =========================================================================
 * NonZero
 * =========================================================================
 *
 * Single-block cooperative ordered scan: each thread counts the non-zeros in
 * its chunk, thread 0 exclusive-scans the per-chunk counts, then each thread
 * re-walks its chunk and writes coordinates into output[rank, capacity] at
 * stride = capacity, in row-major (ONNX-spec) order. Columns beyond the true
 * count are left undefined (the launcher zero-fills them defensively).
 *
 * After completion, *count_ptr (device i32) holds the actual number of
 * non-zero elements. The host reads it back via hipdnn_ep_readback_i32
 * (lowered from hip.readback_dim) and slices the output to its true extent, so
 * downstream ops and the ORT-reported shape use the count rather than the
 * worst-case capacity.
 *
 * input_dims_host: host pointer to int64_t[rank] holding the input
 * shape (copied to device internally before the kernel launch).
 *
 * Supported dtypes: f16, f32, i32, i64, i8, u8.
 * Bounded to rank <= 8.
 */
HIP_KERNEL_API int hip_nonzero(
    void* stream,
    const void* input,
    void* output,
    int* count_ptr,
    int64_t input_num_elements,
    int64_t input_rank,
    const int64_t* input_dims_host,
    int64_t output_capacity,
    int hip_dtype);

/* =========================================================================
 * CumSum
 * =========================================================================
 *
 * One thread per (outer, inner) slice; each thread sequentially scans
 * `axis_size` elements with stride `inner`. The host wrapper decomposes
 *   outer = product(shape[:axis]); axis_size = shape[axis];
 *   inner = product(shape[axis+1:])
 * and synchronously D2H-reads the axis scalar.
 *
 * FP16 accumulates in float to avoid precision loss for long axes.
 */
HIP_KERNEL_API int hip_cumsum(
    void* stream,
    const void* x,
    void* y,
    int64_t outer,
    int64_t axis_size,
    int64_t inner,
    int hip_dtype,
    int exclusive,
    int reverse);

/* =========================================================================
 * Pad (constant / reflect / edge / wrap)
 * =========================================================================
 *
 * One thread per output element. For each output coord, walk the dims and
 * either copy input or fill from the pad_value depending on mode.
 *
 * `pad_mode`:    0 = Constant, 1 = Reflect, 2 = Edge, 3 = Wrap.
 * `lower_pads_host`: per-dim begin pad (length = rank), already filtered
 *                    by the `axes` attribute (defaults to 0 for unaffected
 *                    dims). Upper bound implied by output_shape.
 * `pad_value_host` : host pointer to a scalar of the data type (used only
 *                    when pad_mode == Constant). May be null -> default 0.
 */
HIP_KERNEL_API int hip_pad(
    void* stream,
    const void* input,
    void* output,
    const int64_t* input_shape_host,
    const int64_t* output_shape_host,
    const int64_t* lower_pads_host,
    int rank,
    int hip_dtype,
    int pad_mode,
    const void* pad_value_host);

/* =========================================================================
 * LayerNormalization (ONNX-17)
 * =========================================================================
 *
 *   y = (x - mean) * rsqrt(var + epsilon) * scale + bias
 *
 * Per-row reduction with FP32 accumulators. Bias and the optional `mean` /
 * `inv_std` outputs may be null.
 *
 * `hip_dtype`   : I/O type for input/scale/bias/output -- FLOAT16 or FLOAT32.
 * `mean_dtype`  : type of mean/inv_std output buffers -- FLOAT16 or FLOAT32.
 */
HIP_KERNEL_API int hip_layer_norm(
    void* stream,
    const void* input,
    const void* scale,
    const void* bias,         // optional
    void* output,
    void* mean_out,           // optional
    void* inv_std_out,        // optional
    int64_t outer,
    int64_t norm_size,
    float epsilon,
    int hip_dtype,
    int mean_dtype);

/* =========================================================================
 * Range (1-D sequence generation)
 * =========================================================================
 *
 * Writes output[i] = start + i * delta for i in [0, output_num_elements).
 *
 * start, limit, delta are scalar pointers in device memory (limit is accepted
 * for interface symmetry and runtime validation; the kernel only needs start
 * and delta once output_num_elements is known).
 *
 * Parameters:
 *   stream              - hipStream_t cast to void*
 *   start               - GPU pointer to scalar start
 *   limit               - GPU pointer to scalar limit
 *   delta               - GPU pointer to scalar delta
 *   output              - GPU pointer to output tensor
 *   output_num_elements - total output elements
 *   hip_dtype           - element type (hip_dtype_t value)
 *   device_error_flag   - GPU pointer to int error flag (nullable)
 *
 * Supported types: HIP_DTYPE_INT16, HIP_DTYPE_INT32, HIP_DTYPE_INT64,
 *                  HIP_DTYPE_FLOAT32, HIP_DTYPE_FLOAT64
 * Returns: 0 on success, non-zero on failure
 */
HIP_KERNEL_API int hip_range(
    void* stream,
    const void* start,
    const void* limit,
    const void* delta,
    void* output,
    int64_t output_num_elements,
    int64_t hip_dtype,
    void* device_error_flag);

/* =========================================================================
 * Transpose (Generic N-D Permutation)
 * =========================================================================
 *
 * Permutes the dimensions of `input` according to `perm` and writes the
 * result to `output`.  Implements full ONNX Transpose semantics: any valid
 * permutation of [0, rank) is supported.  For each output linear index i:
 *   - decompose i into output coordinates using output shape derived from
 *     input_shape[perm[k]];
 *   - map to input coordinates via the supplied `perm`;
 *   - linearize using the row-major strides of input_shape and copy.
 *
 * Parameters:
 *   stream             - hipStream_t cast to void*
 *   input              - GPU pointer to source tensor (contiguous, row-major)
 *   output             - GPU pointer to destination tensor (contiguous,
 *                        row-major after permutation)
 *   rank               - number of dimensions (must be in [1, 8])
 *   input_shape        - host pointer to int64_t[rank] with the input shape
 *   perm               - host pointer to int64_t[rank] permutation; output
 *                        dim i corresponds to input dim perm[i]
 *   num_elements       - total elements in the tensor (product of input_shape)
 *   element_size_bytes - 1, 2, 4, or 8 (selects the typed memcpy kernel)
 *
 * Returns: 0 on success, non-zero hipError_t / -1 on failure.
 */
HIP_KERNEL_API int hip_transpose(
    void* stream,
    const void* input,
    void* output,
    int64_t rank,
    const int64_t* input_shape,
    const int64_t* perm,
    int64_t num_elements,
    int element_size_bytes);

/*
 * hip_transpose_2d_tiled: coalesced LDS-tiled fast path for a *batched
 * last-two-dim* transpose of 1/2/4/8-byte elements (transpose is pure data
 * movement, so it is dtype-agnostic given the element width). Each of `batch`
 * slices transposes a row-major [rows x cols] matrix into [cols x rows].
 *
 * Returns 0 on success, 1 if the config is declined -- unsupported element
 * width or batch > gridDim.z limit -- (caller falls back to the generic
 * hip_transpose), or a hipError_t / -1 on failure.
 */
HIP_KERNEL_API int hip_transpose_2d_tiled(
    void* stream,
    const void* input,
    void* output,
    int64_t batch,
    int64_t rows,
    int64_t cols,
    int element_size_bytes);

/* =========================================================================
 * MatMulNBits (Fused Dequant + MatMul)
 * =========================================================================
 *
 * Computes Y = A @ dequant(B)^T + bias, where B holds packed quantized
 * weights.  Supports bits=4 (packed nibbles), bits=8 (1 byte per weight),
 * bits=3 (custom continuous-bitstream packing, exploratory), and bits=2
 * (4 values per byte, LSB-first — same layout as ONNX MatMulNBits since 2
 * divides 8 evenly); other widths return an error.
 *
 * Dequantization (per-block): dequant = (quant_val - zero_point) * scale
 * For 4-bit: lower nibble = first value, upper nibble = second.
 *            Default zero_point = 8 (when zero_points is NULL).
 * For 8-bit: B is unpacked uint8 of shape [N, K]; zero_points (when
 *            provided) is uint8 [N, k_blocks]; default zero_point = 128.
 * For 2-bit: 4 values per byte, LSB-first (value k occupies bits [2k, 2k+2)
 *            of row n's byte stream), [N, ceil(K*2/8)] bytes. Because 2
 *            divides 8, this equals the ONNX MatMulNBits blockwise layout.
 *            zero_points (when provided) is [N, k_blocks] in either
 *            encoding: uint8 levels (zp_elem_size=1, not bit-packed) or
 *            fp16 offsets (zp_elem_size=2, used by symmetric QAD exports
 *            whose zero_point is 2^(bits-1) - 0.5 = 1.5 and therefore has
 *            no integer encoding). Default zero_point = 2. Same three
 *            dispatch paths as bits=3.
 * For 3-bit: NOT an ONNX MatMulNBits convention — B is a continuous
 *            per-row 3-bit bitstream, [N, ceil(K*3/8)] bytes; value k
 *            occupies bits [3k, 3k+3) of row n, LSB-first. zero_points
 *            (when provided) is uint8 [N, k_blocks] (not bit-packed);
 *            default zero_point = 4. Three dispatch paths (mirrors
 *            bits=8): an autotuned WMMA fast path for M >= 16 (batch=1,
 *            K % 32 == 0, block_size % 32 == 0 and >= 32) with the 3-bit
 *            stream decoded inline in the K-loop (no separate dequant
 *            buffer — every 8-value chunk packs into exactly 3 bytes, so
 *            the decode never crosses a byte boundary); an autotuned GEMV
 *            for decode/small-M (block_size a power of two >= 32, K % 32
 *            == 0); and a naive per-element fallback for anything else.
 *
 * Parameters:
 *   stream             - hipStream_t cast to void*
 *   A                  - GPU [batch, M, K]
 *   B                  - GPU packed weights:
 *                          bits=4: [N, k_blocks, blob_size] uint8 packed int4
 *                          bits=8: [N, K] uint8 (no packing)
 *                          bits=3: [N, ceil(K*3/8)] uint8, continuous 3-bit
 *                                  bitstream per row (see above)
 *   scales             - GPU [N, k_blocks] (same type as A)
 *   zero_points        - GPU [N, k_blocks] uint8 (nullable; default zp=8
 *                        for bits=4, zp=128 for bits=8, zp=4 for bits=3)
 *   bias               - GPU [N] (nullable, same type as A)
 *   output             - GPU [batch, M, N]
 *   M                  - rows per batch
 *   N                  - output columns
 *   K                  - inner dimension
 *   batch_count        - number of batches
 *   bits               - quantization bit-width (2, 3, 4, or 8)
 *   block_size         - quantization block size (e.g. 32)
 *   element_size_bytes - 2 for fp16, 4 for fp32
 *
 * Returns: 0 on success, non-zero on failure
 */
HIP_KERNEL_API int hip_matmul_nbits(
    void* stream,
    const void* A,
    const void* B,
    const void* scales,
    const void* zero_points,
    const void* bias,
    void* output,
    int64_t M, int64_t N, int64_t K,
    int64_t batch_count,
    int64_t bits,
    int64_t block_size,
    int64_t element_size_bytes,
    int64_t zp_elem_size,    // 1=uint8 packed nibbles, 2=fp16
    // Optional pre-unpacked zero_points buffers (matmul_nbits.cpp pointer-keyed
    // cache). When non-null, the kernel skips its own unpack/convert kernel
    // launches and reads from these directly. zp_u8 must be valid whenever
    // zero_points is non-null and zp_elem_size==1; zp_fp16 is only consumed
    // by the WMMA / col-major-GEMV (M>1) paths and may be null otherwise.
    const void* pre_unpacked_zp_u8,
    const void* pre_unpacked_zp_fp16);

/* W4A8 integer-dot-product (dp4a) GEMV for a single decode row (M==1).
 * Dynamically quantizes the fp16 activation row to per-group int8 (into
 * caller-owned scratch) and runs a `v_dot4_i32_iu8` (`__builtin_amdgcn_sudot4`)
 * GEMV, replacing the dequant-ALU-bound fp path. Requires bits==4, K%32==0.
 *   A          : fp16 activation [K]  (batch==M==1, row-major)
 *   B          : packed int4 weights  [N, K/2]
 *   scales     : fp16 [N, ceil(K/block_size)]
 *   zp_u8      : pre-unpacked uint8 zero points [N, ceil(K/block_size)] or
 *                nullptr for the symmetric (default zp=8) path
 *   out        : fp16 [N]
 *   a_qb_scratch    : >= K bytes (int8), caller/session-owned
 *   a_scale_scratch : >= ceil(K/block_size) floats, caller/session-owned
 * Returns a hipError_t (hipSuccess on success); hipErrorInvalidValue if K%32.
 */
HIP_KERNEL_API int hip_matmul_nbits_dp4a(
    void* stream,
    const void* A, const void* B, const void* scales, const void* zp_u8,
    const void* bias, void* out,
    int64_t N, int64_t K, int64_t block_size,
    void* a_qb_scratch, void* a_scale_scratch);

/* Stand-alone launchers for the zero_points unpack/convert kernels, used by
 * the asym matmul_nbits cache in lib/Runtime/real/matmul_nbits.cpp.
 *
 *   zp_packed: GPU [N, ceil(K/block_size/2)] packed nibbles
 *   dst_*:     GPU output buffer, caller-allocated
 *   N:         output rows
 *   groups_k:  K / block_size (round-up)
 */
HIP_KERNEL_API void hip_matmul_nbits_unpack_zp_u8(
    void* stream, const void* zp_packed, void* dst_u8, int N, int groups_k);
/* Same as above but for the bits=2 packing (4 group zero_points per byte,
 * [N, ceil(groups_k/4)] packed input). */
HIP_KERNEL_API void hip_matmul_nbits_unpack_zp_u8_2bit(
    void* stream, const void* zp_packed, void* dst_u8, int N, int groups_k);
/* Same as above but for the bits=3 packing (continuous per-row 3-bit stream,
 * [N, ceil(groups_k*3/8)] packed input). */
HIP_KERNEL_API void hip_matmul_nbits_unpack_zp_u8_3bit(
    void* stream, const void* zp_packed, void* dst_u8, int N, int groups_k);
HIP_KERNEL_API void hip_matmul_nbits_convert_zp_fp16(
    void* stream, const void* zp_packed, void* dst_fp16, int N, int groups_k);

/* =========================================================================
 * GatherBlockQuantized (com.microsoft)
 * =========================================================================
 *
 * Combined gather + block-wise dequantize. Equivalent to
 *   tmp        = Gather(data, indices, axis=gather_axis)   // raw quantized
 *   output[..] = (decode(tmp[..]) - zp_block[..]) * scale_block[..]
 * computed in one fused kernel: one thread per output element.
 *
 * Storage / packing:
 *   data         - packed quantized weights (uint8 storage). For bits==4,
 *                  two nibbles per byte: low nibble first, high nibble
 *                  second (matches MatMulNBits convention).
 *   bits == 4 + is_signed_data == 1  -> int4   (sign-extend the nibble)
 *   bits == 4 + is_signed_data == 0  -> uint4  (raw nibble in [0,15])
 *   bits == 8 + is_signed_data == 0  -> uint8  (raw byte  in [0,255])
 *   bits == 8 + is_signed_data == 1  -> int8   (signed byte in [-128,127])
 *
 * scales       - one per (data block) along quantize_axis: same shape as
 *                data except dim quantize_axis is data.shape[qa]/block_size.
 *                Type: T2 in {fp32, fp16, bf16}, matches output.
 * zero_points  - same logical shape as scales; same packing as data
 *                (sub-byte for bits==4). May be null; default zp is then
 *                applied to every block (caller-supplied via default_zp).
 *
 * gather_axis / quantize_axis must be normalized to [0, data_rank) by the
 * caller (negatives resolved). For uint8 data the spec requires
 * gather_axis == 0; that constraint is enforced by the host wrapper, not
 * by the kernel.
 *
 * Output rank = indices_rank + (data_rank - 1). Output shape =
 *   data.shape[0:gather_axis] ++ indices.shape ++ data.shape[gather_axis+1:]
 * out_dtype is the element type of `scales` and `output`.
 *
 * Returns: 0 on success, non-zero hipError_t on failure (incl. rank > max).
 */
HIP_KERNEL_API int hip_gather_block_quantized(
    void* stream,
    const void* data,            // packed quantized
    const void* indices,         // int32 or int64
    const void* scales,          // T2 per block
    const void* zero_points,     // packed (nullable)
    void* output,                // T2 dequantized
    const int64_t* data_shape,    int data_rank,
    const int64_t* indices_shape, int indices_rank,
    const int64_t* scales_shape,  int scales_rank,
    const int64_t* output_shape,  int output_rank,
    int bits,                    // 4 or 8
    int block_size,              // power of 2, >= 16
    int gather_axis,             // normalized
    int quantize_axis,           // normalized
    int default_zp,              // applied when zero_points == null
    int is_signed_data,          // 1 if int4 / int8, 0 if uint4 / uint8
    int indices_is_int64,        // 1 = i64, 0 = i32
    int out_dtype);              // hip_dtype_t (FLOAT16 / FLOAT32 / BFLOAT16)

/* =========================================================================
 * QMoE Sub-Kernels
 * =========================================================================
 *
 * Individual kernel launchers for QMoE (Quantized Mixture-of-Experts).
 * These only launch GPU kernels — no memory allocation, no stream sync.
 * The runtime wrapper (wrap_qmoe) orchestrates the expert loop.
 *
 * All functions take element_size_bytes: 2 for fp16, 4 for fp32.
 */

/* Top-k routing: find top-k experts per token from router_probs.
 *   router_probs   - GPU [num_tokens, num_experts]
 *   expert_indices - GPU [num_tokens, k] int32 (output)
 *   expert_weights - GPU [num_tokens, k] (output, same type as probs)
 *   normalize      - 1 to normalize selected weights (sum-to-one)
 */
HIP_KERNEL_API int hip_qmoe_topk_routing(
    void* stream,
    const void* router_probs,
    void* expert_indices,
    void* expert_weights,
    int64_t num_tokens,
    int64_t num_experts,
    int64_t k,
    int64_t normalize,
    int64_t element_size_bytes);

/* Gather rows: gathered[i,:] = input[token_ids[i],:]
 *   token_ids - GPU [count] int32
 */
HIP_KERNEL_API int hip_qmoe_gather_tokens(
    void* stream,
    const void* input,
    void* gathered,
    const void* token_ids,
    int64_t width,
    int64_t count,
    int64_t element_size_bytes);

/* In-place bias: data[i,j] += bias[j]
 *   No-op if bias is NULL.
 */
HIP_KERNEL_API int hip_qmoe_add_bias(
    void* stream,
    void* data,
    const void* bias,
    int64_t n,
    int64_t width,
    int64_t element_size_bytes);

/* SwiGLU activation (fused, swiglu_fusion=1):
 *   input  [n, 2*inter_size] -> output [n, inter_size]
 *   G = min(gate, limit)
 *   L = clamp(linear, -limit, limit)
 *   out = G * sigmoid(alpha*G) * (L + beta)
 */
HIP_KERNEL_API int hip_qmoe_swiglu(
    void* stream,
    const void* input,
    void* output,
    int64_t n,
    int64_t inter_size,
    float alpha,
    float beta,
    float limit,
    int64_t element_size_bytes);

/* Weighted scatter-add: output[token_ids[i],:] += weights[i] * expert_out[i,:]
 *   token_ids - GPU [count] int32
 *   weights   - GPU [count] (same type as output)
 */
HIP_KERNEL_API int hip_qmoe_scatter_add(
    void* stream,
    void* output,
    const void* expert_out,
    const void* token_ids,
    const void* weights,
    int64_t width,
    int64_t count,
    int64_t element_size_bytes);

/* GPU-side expert bucketing (Phase 2 foundation).
 *
 * Reorders (expert_indices, expert_weights) into per-expert contiguous slices
 * on the device, eliminating the D2H + hipStreamSynchronize that the host
 * bucket loop would otherwise need. Outputs the per-expert count and exclusive
 * prefix-sum offsets; downstream per-expert dispatch can read these directly
 * via device pointers (or as a tiny D2H of just the counts when needed).
 *
 *   expert_indices   - GPU [num_tokens * k] int32 (input from topk_routing)
 *   expert_weights   - GPU [num_tokens * k] fp16  (input from topk_routing)
 *   expert_counts    - GPU [num_experts]      int32 (output)
 *   expert_offsets   - GPU [num_experts + 1]  int32 (output, exclusive scan)
 *   sorted_token_ids - GPU [num_tokens * k]   int32 (output, grouped by eid)
 *   sorted_weights   - GPU [num_tokens * k]   fp16  (output, aligned w/ ids)
 *
 * Constraints: fp16 only; num_experts <= 1024.
 */
HIP_KERNEL_API int hip_qmoe_bucket_tokens(
    void* stream,
    const void* expert_indices,
    const void* expert_weights,
    void* expert_counts,
    void* expert_offsets,
    void* sorted_token_ids,
    void* sorted_weights,
    int64_t num_tokens,
    int64_t num_experts,
    int64_t k,
    int64_t element_size_bytes);

/* -------------------------------------------------------------------------
 * Fully fused MoE decode (num_tokens == 1).
 *
 * Replaces the multi-pass topk -> bucket -> per-expert (gather, FC1, SwiGLU,
 * FC2, scatter_add) sequence with three back-to-back kernel launches and
 * zero hipStreamSynchronize calls per layer. Caller still issues the topk
 * (hip_qmoe_topk_routing) before invoking this; the fused launcher reads
 * expert_indices/expert_weights and dispatches all k experts inline.
 *
 * Layout (single token):
 *   input            - GPU [hidden]              fp16
 *   expert_indices   - GPU [k]                   int32 (from topk_routing)
 *   expert_weights   - GPU [k]                   fp16  (from topk_routing)
 *   fc1_weights      - GPU [E, 2*inter, K_pack]  uint8 (per-expert nibbles)
 *   fc1_scales       - GPU [E, 2*inter, n_blk]   fp16
 *   fc1_zero_points  - GPU [E, 2*inter, ceil(n_blk/2)] uint8 (packed nibbles)
 *   fc1_bias         - GPU [E, 2*inter] or null  fp16
 *   fc2_weights      - GPU [E, hidden, K_pack]   uint8
 *   fc2_scales       - GPU [E, hidden, n_blk]    fp16
 *   fc2_zero_points  - GPU [E, hidden, ceil(n_blk/2)] uint8
 *   fc2_bias         - GPU [E, hidden] or null   fp16
 *   slot_buf         - GPU [k, hidden]           fp16  (transient scratch)
 *   act_out          - GPU [k, inter]            fp16  (transient scratch)
 *   output           - GPU [hidden]              fp16  (final, weighted sum)
 *
 * Constraints: fp16 only (element_size_bytes == 2); hidden_size and
 * inter_size both multiples of 32; block_size > 0 and even.
 */
HIP_KERNEL_API int hip_qmoe_decode_fused(
    void* stream,
    const void* input,
    const void* expert_indices,
    const void* expert_weights,
    const void* fc1_weights, const void* fc1_scales,
    const void* fc1_zero_points, const void* fc1_bias,
    const void* fc2_weights, const void* fc2_scales,
    const void* fc2_zero_points, const void* fc2_bias,
    void* slot_buf,
    void* act_out,
    void* output,
    int64_t hidden_size, int64_t inter_size,
    int64_t k, int64_t block_size,
    float swiglu_alpha, float swiglu_beta, float swiglu_limit,
    int64_t element_size_bytes);

// W4A8 dp4a decode variant of hip_qmoe_decode_fused (env-gated via
// HIPDNN_EP_MATMUL_DP4A). Additionally takes 4 scratch buffers: quantized
// int8 activations + per-group scales for the fc1 input ([hidden]) and the
// fc2 slot activations ([k, inter]). fp16 only; block_size a multiple of 32.
HIP_KERNEL_API int hip_qmoe_decode_fused_dp4a(
    void* stream,
    const void* input,
    const void* expert_indices,
    const void* expert_weights,
    const void* fc1_weights, const void* fc1_scales,
    const void* fc1_zero_points, const void* fc1_bias,
    const void* fc2_weights, const void* fc2_scales,
    const void* fc2_zero_points, const void* fc2_bias,
    void* slot_buf,
    void* act_out,
    void* output,
    void* a_qb_in, void* a_scale_in,
    void* a_qb_mid, void* a_scale_mid,
    int64_t hidden_size, int64_t inter_size,
    int64_t k, int64_t block_size,
    float swiglu_alpha, float swiglu_beta, float swiglu_limit,
    int64_t element_size_bytes);

/* =========================================================================
 * Linear Attention Decode (Single-Token Recurrence, Prefill-Friendly)
 * =========================================================================
 *
 * Performs one step of the linear attention recurrence for a single query
 * token. Updates state in-place and writes the attention output for that
 * token. The caller is expected to invoke this once per time step for
 * prefill (seq_len > 1); no batching across the time dimension is
 * performed inside the kernel.
 *
 * For each (batch, kv_head) pair, the recurrence is:
 *   linear:       S = S + k (x) v
 *   gated:        S = diag(exp(g)) * S + k (x) v
 *   delta:        S = S + beta * k (x) (v - S^T k)
 *   gated_delta:  S = diag(exp(g)) * S + beta * k (x) (v - diag(exp(g)) * S^T k)
 *   output_h = scale * S^T q_h   (for each query head h mapped to this KV head)
 *
 * Input tensors are views into the packed [B, T, H*D] layout of the full
 * sequence: query/key/value/output/decay/beta pointers must already point
 * at the start of the current time step (i.e. the caller has pre-advanced
 * the pointer by t * token_bytes). seq_len is the original T dimension
 * of the packed layout and is used by the kernel to compute the per-batch
 * stride (seq_len * H*D). Pass seq_len = 1 in the pure decode case where
 * the tensors are already shaped [B, 1, H*D].
 *
 * Head counts are three-way and subject to the following divisibility
 * constraints:
 *   - n_k_heads | kv_num_heads
 *       When n_k_heads < kv_num_heads multiple KV heads share the same key
 *       head (mapping: h_k = h_kv * n_k_heads / kv_num_heads).
 *   - Either q_num_heads % kv_num_heads == 0  (standard GQA, H_q >= H_kv)
 *       or   kv_num_heads % q_num_heads == 0  (inverse GQA, H_q < H_kv)
 *
 * Parameters:
 *   stream             - hipStream_t cast to void*
 *   query              - GPU [batch, T, q_num_heads * head_dim_k]
 *                        pointing at time step t
 *   key                - GPU [batch, T, n_k_heads * head_dim_k]
 *                        pointing at time step t
 *                        n_k_heads may differ from kv_num_heads; it must
 *                        divide kv_num_heads.
 *   value              - GPU [batch, T, kv_num_heads * head_dim_v]
 *                        pointing at time step t
 *   decay              - GPU decay tensor in log-space, or nullptr.
 *                        Layout is selected by decay_per_key_dim:
 *                          1 -> [batch, T, kv_num_heads * head_dim_k]
 *                               per-key-dimension decay (GLA / RWKV-6)
 *                          0 -> [batch, T, kv_num_heads]
 *                               per-head scalar decay (DeltaNet / RetNet),
 *                               broadcast across the head_dim_k axis.
 *                        Pointer must already be advanced to time step t.
 *                        Required for gated and gated_delta modes.
 *   beta               - GPU update-rate tensor, or nullptr.
 *                        Layout is selected by beta_per_head:
 *                          1 -> [batch, T, kv_num_heads]
 *                               per-head update rate.
 *                          0 -> [batch, T, 1]
 *                               single scalar update rate per (batch, T),
 *                               broadcast across all kv heads.
 *                        Pointer must already be advanced to time step t.
 *                        Required for delta and gated_delta modes.
 *   state              - GPU [batch, kv_num_heads, head_dim_k, head_dim_v]
 *                        Read/write. Must be pre-initialized (from past_state
 *                        or zeros) before the first time step.
 *   output             - GPU [batch, T, max(q_num_heads, kv_num_heads) *
 *                             head_dim_v], pointing at time step t.
 *                        Standard GQA: heads packed in Q-head order.
 *                        Inverse GQA: heads packed in KV-head order.
 *   B                  - batch dimension
 *   seq_len            - length of the T dimension in the packed layout;
 *                        used to compute per-batch stride. Use 1 when the
 *                        tensors are already shaped [B, 1, H*D].
 *   Hq                 - number of query heads
 *   Hkv                - number of key/value state heads
 *   Nk                 - number of key heads packed in the key tensor;
 *                        must divide Hkv
 *   dk                 - key dimension per head
 *   dv                 - value dimension per head
 *   scale              - output scaling factor (typically 1/sqrt(d_k))
 *   update_rule        - 0=linear, 1=gated, 2=delta, 3=gated_delta
 *   decay_per_key_dim  - decay layout flag (see `decay` above). Ignored when
 *                        decay == nullptr. Any non-zero value is treated as 1.
 *   beta_per_head      - beta  layout flag (see `beta`  above). Ignored when
 *                        beta  == nullptr. Any non-zero value is treated as 1.
 *   type               - element type enum: 0=float, 1=float16, 2=bfloat16
 *                        (HIPDNN_EP_DATATYPE_* in hipdnn_ep_runtime.h)
 *
 * Returns: 0 on success, non-zero on failure
 */
HIP_KERNEL_API int hip_linear_attention_decode(
    void* stream,
    const void* query,
    const void* key,
    const void* value,
    const void* decay,
    const void* beta,
    void* state,
    void* output,
    int64_t B,
    int64_t seq_len,
    int64_t Hq,
    int64_t Hkv,
    int64_t Nk,
    int64_t dk,
    int64_t dv,
    float scale,
    int64_t update_rule,
    int64_t decay_per_key_dim,
    int64_t beta_per_head,
    int64_t type);

// Chunked-parallel gated-delta prefill kernel (single launch, processes the
// whole sequence). Returns >0 (=1) when it declines the launch (caller must
// fall back to the per-token decode loop); 0 on success; <0 on launch error.
// Only the gated_delta rule with scalar log-decay (decay_per_key_dim==0) is
// supported; other rules/layouts/oversized smem are declined.
HIP_KERNEL_API int hip_linear_attention_prefill_chunked(
    void* stream,
    const void* query,
    const void* key,
    const void* value,
    const void* decay,
    const void* beta,
    void* state,
    void* output,
    int64_t B,
    int64_t seq_len,
    int64_t Hq,
    int64_t Hkv,
    int64_t Nk,
    int64_t dk,
    int64_t dv,
    float scale,
    int64_t update_rule,
    int64_t decay_per_key_dim,
    int64_t beta_per_head,
    int64_t type);

// Max memref rank honoured by the strided memref.copy fast path
// (hip_strided_copy) and the host per-row fallback in memrefCopy. Defined
// once here so the kernel and the runtime helper cannot drift out of sync.
#define HIPDNN_MAX_MEMREF_RANK 12

// Parallel strided device-to-device copy (one launch) for MLIR memref.copy
// where neither side is contiguous and the copy spans multiple outer dims.
// Replaces the host per-row hipMemcpyAsync loop in memrefCopy. Pointers are
// element-aligned bases; outer_sizes/strides cover the outer dims, row_elems
// is the contiguous inner suffix. Returns 0 on success, -2 if elem_size is
// unsupported (caller falls back to the host per-row path).
HIP_KERNEL_API int hip_strided_copy(void *stream, void *dst, const void *src,
                     int64_t elem_size, int outer_rank,
                     const int64_t *outer_sizes,
                     const int64_t *src_outer_strides,
                     const int64_t *dst_outer_strides, int64_t row_elems,
                     int64_t outer_total);

/* =========================================================================
 * Causal Depthwise 1D Conv -- single-step "decode" path
 * =========================================================================
 *
 * Fused fast path for the seq_len == 1 case of CausalConvWithState used by
 * Mamba / Gated DeltaNet decoders. Replaces the MIOpen virtual-buffer +
 * convolution + bias + activation chain with one compute kernel that:
 *   - reads past_state[b,c,0..k-2] (or zero if past_state==nullptr),
 *   - reads input[b,c,0],
 *   - computes the depthwise convolution dot product:
 *       output[b,c,0] = sum_{j=0..k-2} weight[c,0,j] * past_state[b,c,j]
 *                     + weight[c,0,k-1] * input[b,c,0]
 *                     + (bias ? bias[c] : 0)
 *   - applies optional SiLU (activation == 1):
 *       output[b,c,0] *= 1 / (1 + exp(-output[b,c,0]))
 *   - writes the new state by shifting forward by one step:
 *       present_state[b,c,0..k-3] = past_state[b,c,1..k-2]
 *       present_state[b,c,k-2]    = input[b,c,0]
 *
 * Bypasses hipMemcpy2DAsync entirely: at decode-shape (rows=B*C, width=k-1
 * elements) the 2D copy has thousands of pathologically thin rows and is
 * massively slower than a single launch with the same arithmetic.
 *
 * Shapes (matching wrap_causal_conv_with_state layout):
 *   input         [B, C, 1]           (past_state is [B, C, k-1])
 *   weight        [C, 1, k]           (depthwise: one k-tap filter per channel)
 *   bias          [C] or nullptr
 *   output        [B, C, 1]
 *   past_state    [B, C, k-1] or nullptr (treated as zeros)
 *   present_state [B, C, k-1]
 *
 * Constraints:
 *   - kernel_size in [1, 8]   (k-1 fits in a small register array)
 *   - element_size_bytes in {2, 4} (fp16 or fp32; matches wrapper validation)
 *   - activation in {0, 1}   (0=none, 1=SiLU)
 *
 * Returns: 0 on success, non-zero on failure.
 */
HIP_KERNEL_API int hip_causal_conv_step_decode(
    void* stream,
    const void* input,
    const void* weight,
    const void* bias,
    const void* past_state,
    void* output,
    void* present_state,
    int64_t batch_size,
    int64_t channels,
    int64_t kernel_size,
    int64_t activation,
    int64_t element_size_bytes);

// Prefill (seq_len > 1) fused causal depthwise 1D conv + bias + SiLU. One
// launch replaces the MIOpen path (Find + 3 pitched memcpys + conv + bias +
// activation + mul). fp32 accumulate; numerically matches the decode-step
// kernel at seq_len==1. Same layout/contract as hip_causal_conv_step_decode
// plus a seq_len argument. Supports kernel_size in [1,8], activation 0/1,
// element_size 2/4; caller falls back to MIOpen for anything else.
HIP_KERNEL_API int hip_causal_conv_prefill(
    void* stream,
    const void* input,
    const void* weight,
    const void* bias,
    const void* past_state,
    void* output,
    void* present_state,
    int64_t batch_size,
    int64_t channels,
    int64_t seq_len,
    int64_t kernel_size,
    int64_t activation,
    int64_t element_size_bytes);

/* =========================================================================
 * WMMA GEMM (Small-M Matrix Multiply via Wave Matrix Multiply-Accumulate)
 * =========================================================================
 *
 * Computes C[M,N] = A[M,K] * B[K,N] using RDNA 3+ WMMA instructions.
 * FP16 inputs, FP32 accumulation, FP16 output. All matrices row-major.
 *
 * Designed for M <= 512 where hipBLASLt's register-heavy tiling (256 VGPRs,
 * 4/16 occupancy) underperforms. This kernel targets ~30 VGPRs and 16/16
 * occupancy via 16x16 WMMA tiles.
 *
 * Requires K and N to be multiples of 16.
 *
 * Parameters:
 *   stream - hipStream_t cast to void*
 *   A      - GPU pointer to activation matrix [M, K] row-major (fp16)
 *   B      - GPU pointer to weight matrix [K, N] row-major (fp16)
 *   C      - GPU pointer to output matrix [M, N] (fp16)
 *   M      - number of rows in A / output
 *   K      - inner dimension (reduction axis), must be multiple of 16
 *   N      - number of columns in B / output, must be multiple of 16
 *
 * Returns: 0 on success, non-zero on failure
 */
HIP_KERNEL_API int hip_gemm_wmma_fp16(void* stream, const void* A, const void* B,
                       void* C, int M, int K, int N);

#ifdef __cplusplus
}
#endif

#endif /* HIP_CUSTOM_KERNELS_H */
