// Copyright(C) [2025] Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>
#include <fstream>
#include <vector>
#include <hip/hip_runtime.h>

#define HIP_CHECK(expr)                                            \
    do {                                                            \
        hipError_t err = expr;                                     \
        if (err != hipSuccess) {                                   \
            std::cerr << "HIP error at " << __FILE__ << ": "       \
                      << __LINE__ << ": "                           \
                      << hipGetErrorString(err) << std::endl;      \
            std::exit(EXIT_FAILURE);                                \
        }                                                           \
    } while(0)

#define HIP_1D_KERNEL_LOOP(i, n)                              \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

template <typename T>
void loadArray(T* out_ptr, size_t size, const std::string& filename) {
  std::ifstream infile(filename, std::ios::binary);
  if (!infile) throw std::runtime_error("Cannot open file for reading.");
  
  infile.read(reinterpret_cast<char*>(out_ptr), sizeof(T) * size);
}

template <typename T_int, int BLOCK_SIZE>
__global__ void point_to_voxelidx_kernel(const T_int* __restrict__ coor,
                                         T_int* __restrict__ point_to_voxelidx,
                                         T_int* __restrict__ point_to_pointidx,
                                         const int max_points,
                                         const int max_voxels,
                                         const int num_points, const int NDim) {
  struct __align__(16) Coor
  {
    T_int x, y, z;
    T_int pad;
  };
  __shared__ Coor shared_coor[BLOCK_SIZE];

  constexpr uint32_t elements_in_128b = 16 / sizeof(T_int);
  union BLOCK_16B
  {
    T_int e[elements_in_128b];
      __uint128_t ow;
  };

  int global_loop_cnt = (num_points + blockDim.x * gridDim.x - 1) / (blockDim.x * gridDim.x);
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  for (int global_idx = 0; global_idx < global_loop_cnt; global_idx++) {
    bool is_valid = false;
    int num = 0;
    int first_match_idx = index;
    T_int coor_x = -1;
    T_int coor_y = -1;
    T_int coor_z = -1;

    if (index < num_points) {
      auto coor_offset = coor + index * NDim;
      // skip invalid points
      coor_x = __ldg(&coor_offset[0]);
      is_valid = (coor_x != -1);
      coor_y = __ldg(&coor_offset[1]);
      coor_z = __ldg(&coor_offset[2]);
    }

#pragma unroll
    for (int block_start = 0; block_start < num_points; block_start += BLOCK_SIZE) {
      // load coor to shared buffer
      // if (index >= block_start) {
        int load_pos = block_start + threadIdx.x;
        if (load_pos < num_points) {
          auto prev_coor = coor + load_pos * NDim;
          shared_coor[threadIdx.x].x = __ldg(&prev_coor[0]);
          shared_coor[threadIdx.x].y = __ldg(&prev_coor[1]);
          shared_coor[threadIdx.x].z = __ldg(&prev_coor[2]);
        }
      // }
      __syncthreads();

      // only calculate the coors before this coor[index]
      // if (is_valid && index < num_points) {
      if (is_valid) {
        BLOCK_16B v_ptr;
        // int block_end = min(block_start + BLOCK_SIZE, index);
        int block_end = min(min(block_start + BLOCK_SIZE, num_points), index);
#pragma unroll
        for (int i  = 0; i < block_end - block_start; i++) {
          // Find all previous points that have the same coors
          // if find the same coor, record it
          v_ptr.ow = *((const __uint128_t*)(shared_coor + i));
          bool is_match = (v_ptr.e[0] == coor_x) && (v_ptr.e[1] == coor_y) &&
                            (v_ptr.e[2] == coor_z);
          num += is_match ? 1 : 0;
          if (is_match && num == 1) {
            first_match_idx = block_start + i;
          } else if (is_match && num >= max_points) {
            // out of boundary
            break;
          }
        }
      }
      __syncthreads();
    }

    if (is_valid && index < num_points) {
      point_to_pointidx[index] = first_match_idx;
      if (num < max_points) {
        point_to_voxelidx[index] = num;
      }
    }

    index += blockDim.x * gridDim.x;
  }
}

int main() {
  int NDim = 3;
  int max_points = 1000;
  int max_voxels = 20000;
  int num_points = 800;

  // read temp_coors
  std::vector<int> temp_coors_size = {num_points, NDim};
  size_t temp_coors_total_size = 1;
  for (int size : temp_coors_size) {
    temp_coors_total_size *= size;
  }
  int* h_temp_coors = (int*)(malloc(temp_coors_total_size * sizeof(int)));
  loadArray(h_temp_coors, temp_coors_total_size, "temp_coors.bin");

  void* temp_coors_ptr;
  HIP_CHECK(hipMalloc(&temp_coors_ptr, temp_coors_total_size * sizeof(int)));
  int* temp_coors = reinterpret_cast<int*>(temp_coors_ptr);
  HIP_CHECK(hipMemcpy(temp_coors, h_temp_coors, temp_coors_total_size * sizeof(int), hipMemcpyHostToDevice));

  void* point_to_pointidx_ptr;
  HIP_CHECK(hipMalloc(&point_to_pointidx_ptr, num_points * sizeof(int)));
  int* point_to_pointidx = reinterpret_cast<int*>(point_to_pointidx_ptr);
  HIP_CHECK(hipMemset(point_to_pointidx, -1, num_points * sizeof(int)));
  void* point_to_voxelidx_ptr;
  HIP_CHECK(hipMalloc(&point_to_voxelidx_ptr, num_points * sizeof(int)));
  int* point_to_voxelidx = reinterpret_cast<int*>(point_to_voxelidx_ptr);
  HIP_CHECK(hipMemset(point_to_voxelidx, -1, num_points * sizeof(int)));

  // call kernel
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  dim3 map_grid(std::min((num_points + 511) / 512, 4096));
  dim3 map_block(512);
  point_to_voxelidx_kernel<int, 512><<<map_grid, map_block, 0, stream>>>(
      temp_coors,
      point_to_voxelidx,
      point_to_pointidx, max_points,
      max_voxels, num_points, NDim);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());
  int* d_point_to_pointidx = (int*)(malloc(num_points * sizeof(int)));
  HIP_CHECK(hipMemcpy(d_point_to_pointidx, point_to_pointidx, num_points * sizeof(int), hipMemcpyDeviceToHost));
  int* d_point_to_voxelidx = (int*)(malloc(num_points * sizeof(int)));
  HIP_CHECK(hipMemcpy(d_point_to_voxelidx, point_to_voxelidx, num_points * sizeof(int), hipMemcpyDeviceToHost));
    
  // check results
  int* h_point_to_pointidx = (int*)(malloc(num_points * sizeof(int)));
  loadArray(h_point_to_pointidx, num_points, "point_to_pointidx.bin");
  int* h_point_to_voxelidx = (int*)(malloc(num_points * sizeof(int)));
  loadArray(h_point_to_voxelidx, num_points, "point_to_voxelidx.bin");
  for (int i = 0; i < num_points; ++i) {
    if (h_point_to_pointidx[i] != d_point_to_pointidx[i]) {
      std::cout << "Coors: the " << i << "th element is not equal!!!" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
  for (int i = 0; i < num_points; ++i) {
    if (h_point_to_voxelidx[i] != d_point_to_voxelidx[i]) {
      std::cout << "Coors: the " << i << "th element is not equal!!!" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }

  std::cout << "\n================================================================\n"
            << "============================ PASSED ============================\n"
            << "================================================================\n";

  // release sources
  HIP_CHECK(hipFree(temp_coors));
  HIP_CHECK(hipFree(point_to_pointidx));
  HIP_CHECK(hipFree(point_to_voxelidx));
  free(h_temp_coors);
  free(d_point_to_pointidx);
  free(d_point_to_voxelidx);
  free(h_point_to_pointidx);
  free(h_point_to_voxelidx);
}
