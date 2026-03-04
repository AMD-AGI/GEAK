---
Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
---

# Kernel: WKV (Weighted Key-Value for RWKV)

## Variant Context
- Input semantic type: Recurrent attention (RWKV architecture)
- Datatype(s): FP32
- Data representation: Dense state tensors
- Target architecture: Generic (NVIDIA, AMD)

## Functionality
The WKV kernel implements the core attention mechanism for RWKV (Receptance Weighted Key Value) models. Unlike transformer attention, RWKV uses a linear recurrence that can be computed efficiently in O(n) time. The kernel maintains a running state that is updated for each token.

Key features:
- Linear time complexity O(n) vs O(n²) for transformers
- Recurrent state maintenance
- Support for RWKV v6 and v7 architectures
- Efficient parallel implementation

---

## Optimization 1: RWKV v6 CUDA Implementation
- Commit ID: 2a63caaa6
- Optimization type: Algorithm (initial implementation)
- Summary: Implement RWKV WKV operation with CUDA for efficient inference
- Detailed explanation: This optimization provides the initial CUDA implementation of the WKV operation, which is the core of RWKV's attention mechanism. The implementation uses warp-level parallelism to process multiple channels simultaneously.

- Code excerpt:
    ```cpp
    // RWKV v6: RWKV_WKV op CUDA implementation
    template<int HEAD_SIZE>
    __global__ void rwkv_wkv_forward(
        const float * __restrict__ k,
        const float * __restrict__ v,
        const float * __restrict__ r,
        const float * __restrict__ w,
        const float * __restrict__ u,
        float * __restrict__ state,
        float * __restrict__ dst,
        const int T,  // Sequence length
        const int H)  // Number of heads
    {
        const int head = blockIdx.x;
        const int tid = threadIdx.x;
        
        // Each thread handles one channel
        const int c = tid;
        if (c >= HEAD_SIZE) return;
        
        // Load state for this head/channel
        float s = state[head * HEAD_SIZE + c];
        
        for (int t = 0; t < T; t++) {
            const int idx = (t * H + head) * HEAD_SIZE + c;
            
            const float kt = k[idx];
            const float vt = v[idx];
            const float rt = r[idx];
            const float wt = w[idx];
            const float ut = u[head * HEAD_SIZE + c];
            
            // WKV computation
            const float wkv = (expf(ut + kt) * vt + s) / 
                              (expf(ut + kt) + expf(-wt) * s);
            
            // Update state
            s = expf(-wt) * s + expf(kt) * vt;
            
            // Output
            dst[idx] = rt * wkv;
        }
        
        // Save state
        state[head * HEAD_SIZE + c] = s;
    }
    ```

- Evidence mapping:
  - "Linear recurrence" → sequential loop over T with state update
  - "Warp parallelism" → each thread handles one channel
  - "State maintenance" → `state` array persists across calls

---

## Optimization 2: RWKV v7 Architecture Support
- Commit ID: 7dfad387e
- Optimization type: Algorithm (new architecture)
- Summary: Add support for RWKV v7 architecture with improved attention mechanism
- Detailed explanation: RWKV v7 introduces architectural improvements over v6, including modified state update equations and additional parameters. This optimization extends the WKV kernel to support the v7 formulation.

- Code excerpt:
    ```cpp
    // llama: Add support for RWKV v7 architecture
    template<int HEAD_SIZE>
    __global__ void rwkv_wkv_v7_forward(
        const float * __restrict__ k,
        const float * __restrict__ v,
        const float * __restrict__ r,
        const float * __restrict__ w,
        const float * __restrict__ u,
        const float * __restrict__ a,  // New in v7
        const float * __restrict__ b,  // New in v7
        float * __restrict__ state,
        float * __restrict__ dst,
        const int T,
        const int H)
    {
        const int head = blockIdx.x;
        const int c = threadIdx.x;
        if (c >= HEAD_SIZE) return;
        
        float s = state[head * HEAD_SIZE + c];
        
        for (int t = 0; t < T; t++) {
            const int idx = (t * H + head) * HEAD_SIZE + c;
            
            // v7 uses modified formulation with a, b parameters
            const float kt = k[idx];
            const float vt = v[idx];
            const float rt = r[idx];
            const float wt = w[idx];
            const float at = a[idx];
            const float bt = b[idx];
            
            // v7 WKV computation (different from v6)
            const float wkv = at * s + bt * vt;
            
            // v7 state update
            s = wt * s + kt * vt;
            
            dst[idx] = rt * wkv;
        }
        
        state[head * HEAD_SIZE + c] = s;
    }
    ```

- Evidence mapping:
  - "v7 parameters" → additional `a`, `b` inputs
  - "Modified formulation" → different WKV and state equations
  - "Architecture support" → separate kernel for v7

---

## Optimization 3: QRWKV6 Model Support
- Commit ID: ee7136c6d
- Optimization type: Algorithm (quantized variant)
- Summary: Add support for QRWKV6 model architecture with GLA kernel
- Detailed explanation: QRWKV6 is a variant that uses Gated Linear Attention (GLA) instead of the standard WKV. This optimization adds the GLA kernel which provides similar functionality with different mathematical formulation.

- Code excerpt:
    ```cpp
    // llama: add support for QRWKV6 model architecture
    // GLA (Gated Linear Attention) kernel
    template<int HEAD_SIZE>
    __global__ void gla_forward(
        const float * __restrict__ q,
        const float * __restrict__ k,
        const float * __restrict__ v,
        const float * __restrict__ g,  // Gate
        float * __restrict__ state,
        float * __restrict__ dst,
        const int T,
        const int H)
    {
        const int head = blockIdx.x;
        const int c = threadIdx.x;
        if (c >= HEAD_SIZE) return;
        
        float s = state[head * HEAD_SIZE + c];
        
        for (int t = 0; t < T; t++) {
            const int idx = (t * H + head) * HEAD_SIZE + c;
            
            const float qt = q[idx];
            const float kt = k[idx];
            const float vt = v[idx];
            const float gt = g[idx];  // Gating factor
            
            // GLA computation
            const float attn = qt * s;
            
            // Gated state update
            s = gt * s + kt * vt;
            
            dst[idx] = attn;
        }
        
        state[head * HEAD_SIZE + c] = s;
    }
    ```

- Evidence mapping:
  - "GLA variant" → Gated Linear Attention formulation
  - "Gate parameter" → `g` input for gating
  - "QRWKV6 support" → alternative to standard WKV
