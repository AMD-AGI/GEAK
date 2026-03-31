---
tags: ["optimization", "performance", "hip", "memory", "device"]
priority: "L0-core"
source_url: "https://rocm.docs.amd.com/en/latest/how-to/hip_runtime_api/memory_management/device_memory.html"
rocm_version: "7.0+"
last_updated: 2026-01-07
---

# Device Memory

Device memory is random access memory physically located on a GPU, offering bandwidth significantly higher than host RAM. This high bandwidth is only available to on-device accesses; host or other device accesses traverse slower interfaces like PCIe or AMD Infinity Fabric.

On certain architectures like APUs, the GPU and CPU share the same physical memory. A special local data share exists on-chip, directly accessible to compute units for shared memory purposes.

## Global Memory

Global memory is general read-write memory visible to all threads on a device. Variables in global memory require the `__device__` qualifier.

### Allocating Global Memory

Memory allocation can occur via HIP runtime functions like `hipMalloc()` or using the `__device__` qualifier on variables. Memory can also be allocated within kernels using `malloc` or `new`, with each executing thread allocating the specified amount.

**Important:** Memory allocated in kernels can only be freed within kernels, not via host-side `hipFree()`. Device memory allocated on the host cannot be freed within kernels.

Example of sharing kernel-allocated memory across a block:

```c
__global__ void kernel_memory_allocation(TYPE* pointer){
  __shared__ int *memory;
  size_t blockSize = blockDim.x;
  constexpr size_t elementsPerThread = 1024;

  if(threadIdx.x == 0){
    memory = new int[blockDim.x * elementsPerThread];
  }
  __syncthreads();

  int *localPtr = memory;
  for(int i = 0; i < elementsPerThread; ++i){
    localPtr[i * blockSize + threadIdx.x] = i;
  }
  __syncthreads();

  if(threadIdx.x == 0){
    delete[] memory;
  }
}
```

### Copying Between Device and Host

Without unified memory management, explicit copying is required:

```c
size_t elements = 1 << 20;
size_t size_bytes = elements * sizeof(int);

int *host_pointer = new int[elements];
int *device_input, *device_result;
HIP_CHECK(hipMalloc(&device_input, size_bytes));
HIP_CHECK(hipMalloc(&device_result, size_bytes));

HIP_CHECK(hipMemcpy(device_input, host_pointer, size_bytes, hipMemcpyHostToDevice));

// Use memory on device

HIP_CHECK(hipMemcpy(host_pointer, device_result, size_bytes, hipMemcpyDeviceToHost));

HIP_CHECK(hipFree(device_result));
HIP_CHECK(hipFree(device_input));
delete[] host_pointer;
```

## Constant Memory

Constant memory is read-only storage visible to all device threads. It is a limited segment backed by device memory with different caching than standard device memory. The host must set values before kernel execution.

For optimal bandwidth, all warp threads must access the same memory address. Different address access serializes operations, reducing bandwidth.

### Using Constant Memory

Constant memory cannot be dynamically allocated; size must be specified at compile time:

```c
constexpr size_t const_array_size = 32;
__constant__ double const_array[const_array_size];

void set_constant_memory(double* values){
  hipMemcpyToSymbol(const_array, values, const_array_size * sizeof(double));
}

__global__ void kernel_using_const_memory(double* array){
  int warpIdx = threadIdx.x / warpSize;
  array[blockDim.x] *= const_array[warpIdx];
}
```

## Texture Memory

Texture memory is special read-only storage with performance benefits for accessing memory patterns where addresses are close in 2D or 3D representations. It provides features like filtering and boundary addressing.

To check device support, query `hipDeviceAttributeImageSupport`.

### Using Texture Memory

Textures are represented by `hipTextureObject_t`, created via `hipCreateTextureObject()`. Underlying memory is a 1D, 2D, or 3D `hipArray_t`, allocated using `hipMallocArray()`.

On-device access uses `tex1D/2D/3D` functions. Reference the [Texture management API documentation](https://rocm.docs.amd.com/en/latest/reference/hip_runtime_api/modules/memory_management/texture_management.html) for complete details.

## Surface Memory

Surface memory is a read-write version of texture memory, created via `hipCreateSurfaceObject()`. Since surfaces use the read-only texture cache, changes become visible only after launching a new kernel.

## Shared Memory

Shared memory is read-write memory visible only to threads within a block. It is allocated per block and provides low-latency, high-bandwidth access comparable to L1 cache. However, it is limited in size and allocated per block, potentially restricting concurrent block scheduling and occupancy.

### Allocate Shared Memory

Dynamic allocation uses `extern __shared__`:

```c
extern __shared__ int dynamic_shared[];

__global__ void kernel(int array1SizeX, int array1SizeY, int array2Size){
  int* array1 = dynamic_shared;
  int array1Size = array1SizeX * array1SizeY;
  int* array2 = &(array1[array1Size]);

  if(threadIdx.x < array1SizeX && threadIdx.y < array1SizeY){
    // access array1
  }
  if(threadIdx.x < array2Size){
    // access array2
  }
}
```

Static allocation declares memory directly in the kernel:

```c
__global__ void kernel(){
  __shared__ int array[128];
  __shared__ double result;
}
```
