# Kernel: quick_allreduce

## Variant Context
- Input semantic type: Collective communication (AllReduce)
- Datatype(s): FP16/BF16/FP32
- Data representation: Dense tensor distributed across GPUs
- Target architecture: gfx942 (MI300X), gfx950 (MI350)

## Functionality
This kernel implements a custom AllReduce operation for multi-GPU communication. It uses a two-shot algorithm where data is first scattered to all ranks, then gathered and reduced. This is optimized for AMD GPU interconnects and avoids the overhead of standard NCCL/RCCL implementations for small to medium tensor sizes.

## Optimization 1: Local Buffer Pointer Caching
- Commit ID: ae774a370
- Optimization type: Memory
- Summary: Cached buffer pointers in local registers to reduce repeated global memory accesses.

- Detailed explanation:
  The optimization pre-loads the buffer_list pointers into a local array at the beginning of the kernel. This avoids repeated indirect memory accesses through the buffer_list array during the send and flag-setting operations. Since buffer_list is accessed multiple times per iteration (for data send and flag operations), caching these pointers in registers reduces memory traffic and latency.

- Code excerpt:
    ```cpp
    // BEFORE: Direct buffer_list access
    for(int r = 0; r < kWorldSize; r++)
    {
        int32x4_t* send_buffer = reinterpret_cast<int32x4_t*>(
            buffer_list[r] + comm_data0_offset + rank * Codec::kRankTransmittedTileSize);
        codec.send(send_buffer, &tA[r * Codec::kRankAtoms]);
    }
    if(thread < kWorldSize)
    {
        int r              = thread;
        uint32_t* flag_ptr = reinterpret_cast<uint32_t*>(buffer_list[r] + comm_flags0_offset +
                                                         rank * sizeof(uint32_t));
        set_sync_flag(flag_ptr, flag_color);
    }
    
    // AFTER: Cached buffer pointers
    uint8_t* buffer_ptr[kWorldSize];
    for (int i = 0; i < kWorldSize; ++i) {
        buffer_ptr[i] = buffer_list[i];
    }
    // ... later in the kernel
    for(int r = 0; r < kWorldSize; r++)
    {
        int32x4_t* send_buffer = reinterpret_cast<int32x4_t*>(
            buffer_ptr[r] + comm_data0_offset + rank * Codec::kRankTransmittedTileSize);
        codec.send(send_buffer, &tA[r * Codec::kRankAtoms]);
    }
    if(thread < kWorldSize)
    {
        int r              = thread;
        uint32_t* flag_ptr = reinterpret_cast<uint32_t*>(buffer_ptr[r] + comm_flags0_offset +
                                                         rank * sizeof(uint32_t));
        set_sync_flag(flag_ptr, flag_color);
    }
    ```

- Evidence mapping:
  - Pointer caching → `uint8_t* buffer_ptr[kWorldSize]` local array
  - Reduced indirection → `buffer_ptr[r]` instead of `buffer_list[r]`
  - Applied to both phases → Changes in comm_data0, comm_flags0, comm_data1, comm_flags1 accesses
