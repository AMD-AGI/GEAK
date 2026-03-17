# Kernel: universal_gemm (Universal GEMM Kernel)

## Variant Context
- Input semantic type: Matrix multiplication (General Matrix Multiply)
- Datatype(s): FP16/BF16/FP32
- Data representation: Row-major or Column-major layouts
- Target architecture: gfx9 family (gfx908, gfx90a, gfx942, gfx950)

## Functionality
The Universal GEMM kernel computes C = A × B + D (with optional bias D) using a tile-based approach. It supports persistent kernel execution where workgroups loop over multiple tiles, split-K parallelism for K-dimension decomposition, and various scheduling strategies for optimal hardware utilization.

## Optimization 1: Persistent Async Input Scheduler
- Commit ID: 91b4102a5
- Optimization type: scheduling / compute
- Summary: Added signal-based synchronization for persistent GEMM kernels where input data becomes available incrementally, enabling overlap between data production and kernel consumption.

- Detailed explanation:
  The optimization introduces a `PersistentAsyncInputScheduler` that enables producer-consumer synchronization for streaming input scenarios. The scheduler divides M-dimension tiles into chunks and uses signal-based synchronization to coordinate when input data is ready.

  Key features:
  - Uses modulo wraparound for chunk index calculation: `chunk_idx = ((tile_idx + tile_idx_pivot) / tiles_per_chunk) % num_chunks`
  - Power-efficient waiting using `__builtin_amdgcn_s_sleep`
  - Pivot offset for load balancing across chunks

  This is particularly useful for scenarios like LLM inference where input tokens arrive incrementally.

- Code excerpt:
    ```cpp
    /// @brief Scheduler for persistent GEMM kernels with asynchronous input streaming.
    struct PersistentAsyncInputScheduler
    {
        /// @brief Number of M-dimension tiles grouped into each chunk.
        uint32_t tiles_per_chunk_m = 0;

        /// @brief Device pointer to array of signal values (uint32_t), one per chunk.
        uint32_t* chunk_signals = nullptr;

        /// @brief Pivot offset for rotating the chunk assignment.
        int32_t tile_idx_pivot_m = 0;

        /// @brief Number of signal chunks allocated.
        uint32_t num_chunks = 0;
    };
    
    // In kernel execution:
    // Synchronize with producer to ensure input data is ready before processing tile
    if(kargs.async_input_scheduler.chunk_signals != nullptr)
    {
        const auto tiles_per_chunk =
            amd_wave_read_first_lane(kargs.async_input_scheduler.tiles_per_chunk_m);
        const auto tile_idx_pivot =
            amd_wave_read_first_lane(kargs.async_input_scheduler.tile_idx_pivot_m);
        const auto num_chunks =
            amd_wave_read_first_lane(kargs.async_input_scheduler.num_chunks);
        if(tiles_per_chunk > 0 && num_chunks > 0)
        {
            // Pivot allows rotating chunk assignments for load balancing
            const auto chunk_idx = amd_wave_read_first_lane(
                ((iM + tile_idx_pivot) / tiles_per_chunk) % num_chunks);
            workgroup_barrier chunk_barrier(kargs.async_input_scheduler.chunk_signals);
            chunk_barrier.wait_eq_wave(/*value=*/1, /*offset=*/chunk_idx);
        }
    }
    ```

- Evidence mapping:
  - "Signal-based synchronization" → `chunk_signals` pointer and `wait_eq_wave` call
  - "Modulo wraparound" → `((iM + tile_idx_pivot) / tiles_per_chunk) % num_chunks` calculation
  - "Power-efficient waiting" → `workgroup_barrier::wait_eq_wave` using `__builtin_amdgcn_s_sleep`
  - "Load balancing" → `tile_idx_pivot_m` for rotating chunk assignments

## Optimization 2: Maximum Occupancy Grid Size for Persistent Kernels
- Commit ID: 91b4102a5
- Optimization type: launch configuration
- Summary: Added grid size calculation that maximizes hardware utilization for persistent kernels by matching grid size to hardware capacity rather than problem size.

- Detailed explanation:
  Persistent kernels loop over tiles internally, so the grid size should match the hardware's compute capacity (number of CUs × occupancy) rather than the problem size. This ensures all compute units are utilized without over-subscription.

- Code excerpt:
    ```cpp
    /**
     * @brief Calculate grid size that maximizes hardware utilization for persistent kernels.
     * @return Grid size that fills all compute units at maximum occupancy.
     * @note Persistent kernels loop over tiles, so grid size should match hardware capacity
     *       rather than problem size.
     */
    CK_TILE_HOST static auto MaxOccupancyGridSize(const stream_config& s) -> dim3
    {
        int occupancy = 0;
        hipOccupancyMaxActiveBlocksPerMultiprocessor(
            &occupancy, kernel_func, kBlockSize, GetSmemSize());
        
        int num_cu = 0;
        hipDeviceGetAttribute(&num_cu, hipDeviceAttributeMultiprocessorCount, s.device_id);
        
        return dim3(occupancy * num_cu, 1, 1);
    }
    ```

- Evidence mapping:
  - "Hardware capacity matching" → `occupancy * num_cu` calculation
  - "Persistent kernel awareness" → Comment explaining grid size should match hardware capacity

## Optimization 3: SplitK Batch Offset Load Balancing
- Commit ID: 91b4102a5
- Optimization type: scheduling
- Summary: Improved K-dimension work distribution across split-K batches using ceil division for even load balancing.

- Detailed explanation:
  The SplitKBatchOffset structure distributes K-dimension work evenly among split-K workgroups. It uses ceil division to ensure remainder work is distributed evenly rather than leaving it all for the last workgroup.

- Code excerpt:
    ```cpp
    struct SplitKBatchOffset
    {
        // Balances K-dimension work across batches to maximize parallelism while minimizing
        // load imbalance. Uses ceil division to distribute remainder work evenly.
        __device__ SplitKBatchOffset(const KernelArgs& kargs, const index_t k_id = blockIdx.z)
        {
            constexpr auto K1 = TilePartitioner::BlockGemmShape::WarpTile::at(number<2>{});
            const index_t num_k_tiles = (kargs.K + K1 - 1) / K1;
            const index_t tiles_per_batch = (num_k_tiles + kargs.k_batch - 1) / kargs.k_batch;
            
            k_start = k_id * tiles_per_batch * K1;
            k_end = min((k_id + 1) * tiles_per_batch * K1, kargs.K);
        }
    };
    ```

- Evidence mapping:
  - "Ceil division" → `(num_k_tiles + kargs.k_batch - 1) / kargs.k_batch`
  - "Even distribution" → `tiles_per_batch` calculation ensuring balanced work
