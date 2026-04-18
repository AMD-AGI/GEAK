# Kernel: Pooling and Elementwise Operations

## Variant Context
- Input semantic type: Pooling (spatial reduction) and elementwise operations
- Datatype(s): FP16/BF16/FP32/FP8
- Data representation: NHWC/NCHW tensor layouts
- Target architecture: gfx9 family (gfx908, gfx90a, gfx942, gfx950)

## Functionality
Pooling kernels implement spatial reduction operations (max pooling, average pooling) commonly used in CNNs. Elementwise kernels provide fused point-wise operations that can be combined with other kernels to reduce memory traffic.

## Optimization 1: Indexing Support for Pooling
- Commit ID: 3052d7c9e
- Optimization type: compute
- Summary: Added indexing support to pooling operator for max pooling with indices output.

- Detailed explanation:
  Max pooling with indices is required for unpooling in encoder-decoder architectures. The optimization tracks the index of the maximum element within each pooling window alongside the max value.

- Code excerpt:
    ```cpp
    // Max pooling with index tracking
    template <typename Problem>
    struct MaxPool2dWithIndices
    {
        using DataType = typename Problem::DataType;
        using IndexType = int32_t;
        
        CK_TILE_DEVICE void operator()(
            const DataType* input,
            DataType* output,
            IndexType* indices,
            index_t H, index_t W,
            index_t pool_h, index_t pool_w,
            index_t stride_h, index_t stride_w)
        {
            const index_t out_h = blockIdx.y;
            const index_t out_w = blockIdx.x;
            const index_t c = threadIdx.x;
            
            // Compute input window
            index_t in_h_start = out_h * stride_h;
            index_t in_w_start = out_w * stride_w;
            
            DataType max_val = -infinity;
            IndexType max_idx = 0;
            
            // Find max in pooling window
            for(index_t ph = 0; ph < pool_h; ++ph)
            {
                for(index_t pw = 0; pw < pool_w; ++pw)
                {
                    index_t in_h = in_h_start + ph;
                    index_t in_w = in_w_start + pw;
                    
                    if(in_h < H && in_w < W)
                    {
                        DataType val = input[(in_h * W + in_w) * C + c];
                        if(val > max_val)
                        {
                            max_val = val;
                            max_idx = in_h * W + in_w;
                        }
                    }
                }
            }
            
            // Store max value and index
            output[(out_h * out_W + out_w) * C + c] = max_val;
            indices[(out_h * out_W + out_w) * C + c] = max_idx;
        }
    };
    ```

- Evidence mapping:
  - "Index tracking" → `IndexType max_idx` variable
  - "Max with indices" → Both `output` and `indices` outputs
  - "Unpooling support" → Stored indices for reverse operation

## Optimization 2: 2D Multiple Reductions
- Commit ID: 4216d43da
- Optimization type: compute
- Summary: Implemented 2D multiple reductions for efficient spatial reduction operations.

- Detailed explanation:
  For operations like global average pooling, the kernel needs to reduce across both H and W dimensions. The optimization performs both reductions efficiently in a single kernel.

- Code excerpt:
    ```cpp
    // 2D multiple reductions
    template <typename Problem, typename... ReduceOps>
    struct MultiReduce2D
    {
        CK_TILE_DEVICE void operator()(
            const DataType* input,
            std::tuple<OutputType*...> outputs,
            index_t H, index_t W, index_t C)
        {
            const index_t c = blockIdx.x * blockDim.x + threadIdx.x;
            
            if(c >= C) return;
            
            // Initialize accumulators for each reduction
            auto accs = std::make_tuple(ReduceOps::Init()...);
            
            // Reduce over H and W dimensions
            for(index_t h = 0; h < H; ++h)
            {
                for(index_t w = 0; w < W; ++w)
                {
                    DataType val = input[(h * W + w) * C + c];
                    
                    // Apply each reduction
                    std::apply([&](auto&... acc) {
                        ((acc = ReduceOps::Reduce(acc, val)), ...);
                    }, accs);
                }
            }
            
            // Finalize and store results
            std::apply([&](auto*... out) {
                index_t idx = 0;
                ((out[c] = ReduceOps::Finalize(std::get<idx++>(accs), H * W)), ...);
            }, outputs);
        }
    };
    ```

- Evidence mapping:
  - "2D reduction" → Nested loops over H and W
  - "Multiple reductions" → Variadic `ReduceOps` template
  - "Single kernel" → All reductions in one pass

## Optimization 3: Unary Elementwise Operations with Naming
- Commit ID: f53d857b2
- Optimization type: code organization
- Summary: Added name member to unary elementwise ops for better debugging and profiling.

- Detailed explanation:
  Each elementwise operation now has a name string for identification in profiling tools and error messages. This aids debugging without runtime overhead (names are compile-time constants).

- Code excerpt:
    ```cpp
    // Named unary elementwise operations
    template <typename DataType>
    struct UnaryElementwiseOps
    {
        struct Relu
        {
            static constexpr const char* name = "Relu";
            
            CK_TILE_DEVICE DataType operator()(DataType x) const
            {
                return x > 0 ? x : 0;
            }
        };
        
        struct Sigmoid
        {
            static constexpr const char* name = "Sigmoid";
            
            CK_TILE_DEVICE DataType operator()(DataType x) const
            {
                return 1.0f / (1.0f + exp(-x));
            }
        };
        
        struct Gelu
        {
            static constexpr const char* name = "Gelu";
            
            CK_TILE_DEVICE DataType operator()(DataType x) const
            {
                // Approximate GELU
                return 0.5f * x * (1.0f + tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
            }
        };
    };
    ```

- Evidence mapping:
  - "Name member" → `static constexpr const char* name`
  - "Compile-time" → `constexpr` for zero runtime overhead
  - "Common ops" → Relu, Sigmoid, Gelu implementations

## Optimization 4: Fused Bias + Clamp for Convolution
- Commit ID: 5c1974065
- Optimization type: fusion
- Summary: Added fused bias addition and clamping for convolution output.

- Detailed explanation:
  After convolution, it's common to add bias and apply activation clamping (e.g., ReLU6). Fusing these operations with the convolution epilogue reduces memory traffic.

- Code excerpt:
    ```cpp
    // Fused bias + clamp epilogue
    template <typename Problem>
    struct ConvBiasClampEpilogue
    {
        CK_TILE_DEVICE void operator()(
            AccType* acc,
            const BiasType* bias,
            OutputType* output,
            AccType clamp_min,
            AccType clamp_max)
        {
            const index_t c = threadIdx.x;
            
            // Add bias
            AccType val = acc[c] + static_cast<AccType>(bias[c]);
            
            // Clamp (e.g., for ReLU6: min=0, max=6)
            val = max(clamp_min, min(clamp_max, val));
            
            // Store output
            output[c] = static_cast<OutputType>(val);
        }
    };
    ```

- Evidence mapping:
  - "Fused operations" → Bias add and clamp in one kernel
  - "Clamp range" → `clamp_min`, `clamp_max` parameters
  - "Epilogue" → Applied after convolution accumulation
