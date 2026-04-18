# Kernel: causal_conv1d_channellast (Simplified)

## Variant: fp16, AMD MI300 (gfx942)

## Functionality
Causal 1D convolution with channel-last layout for Mamba/SSM models. Supports width 2/3/4 and optional SiLU activation.

## Key Optimizations

### 1. XCD Swizzling for MI300
- Distributes blocks across 8 XCDs for load balancing
- Formula: `new_pid = (pid / 8) + ((pid % 8) * (num_blocks / 8)) % num_blocks`
- Prevents XCD imbalance from naive consecutive block assignment

### 2. Fused Multiply-Add (FMA)
- Replaced `out += weight * x` with `__fmaf_rn(weight, x, out)`
- Single instruction vs separate mul+add
- Better throughput and numerical precision

### 3. Fast Math Intrinsics
- SiLU activation uses `__expf()` instead of `expf()`
- Faster exponential suitable for NN activations

### 4. Debug Print Removal
- Eliminated all `std::cout` debug statements from launch function
- Removes I/O overhead in production

### 5. Channel-Last Kernel FMA
- Applied same FMA optimization to channel-last variant
- `out_vals[i] = __fmaf_rn(weight_vals[w], x_vals[i + w], out_vals[i])`
