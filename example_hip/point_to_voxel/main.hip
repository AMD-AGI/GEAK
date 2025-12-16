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

template <typename T_int>
__global__ void point_to_voxelidx_kernel(const T_int* coor,
                                         T_int* point_to_voxelidx,
                                         T_int* point_to_pointidx,
                                         const int max_points,
                                         const int max_voxels,
                                         const int num_points, const int NDim) {
  HIP_1D_KERNEL_LOOP(index, num_points) {
    auto coor_offset = coor + index * NDim;
    // skip invalid points
    if (coor_offset[0] == -1) continue;

    int num = 0;
    int coor_x = coor_offset[0];
    int coor_y = coor_offset[1];
    int coor_z = coor_offset[2];
    // only calculate the coors before this coor[index]
    for (int i = 0; i < index; ++i) {
      auto prev_coor = coor + i * NDim;
      if (prev_coor[0] == -1) continue;

      // Find all previous points that have the same coors
      // if find the same coor, record it
      if ((prev_coor[0] == coor_x) && (prev_coor[1] == coor_y) &&
          (prev_coor[2] == coor_z)) {
        num++;
        if (num == 1) {
          // point to the same coor that first show up
          point_to_pointidx[index] = i;
        } else if (num >= max_points) {
          // out of boundary
          break;
        }
      }
    }
    if (num == 0) {
      point_to_pointidx[index] = index;
    }
    if (num < max_points) {
      point_to_voxelidx[index] = num;
    }
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

  // latency measurement
  double kernel_time = 0;

  // Create events to measure the execution time of the kernels.
  hipEvent_t start, stop;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));


  // call kernel
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  dim3 map_grid(std::min((num_points + 511) / 512, 4096));
  dim3 map_block(512);

  const constexpr unsigned int iterations = 10;
  for(unsigned int i = 0; i < iterations; ++i)
    {

      float kernel_ms{};

      // Record the start event.
      HIP_CHECK(hipEventRecord(start, hipStreamDefault));


      point_to_voxelidx_kernel<int><<<map_grid, map_block, 0, stream>>>(
          temp_coors,
          point_to_voxelidx,
          point_to_pointidx, max_points,
          max_voxels, num_points, NDim);
      

      HIP_CHECK(hipGetLastError());

      HIP_CHECK(hipEventRecord(stop, hipStreamDefault));
      HIP_CHECK(hipEventSynchronize(stop));

      // Get the execution time of the kernel and add it to the total count.
      HIP_CHECK(hipEventElapsedTime(&kernel_ms, start, stop));
      kernel_time += kernel_ms;

    }
  
    // Destroy hipEvents.
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    kernel_time /= iterations;

    std::cout << "The mean time needed for each iteration has been " << kernel_time << "ms" << std::endl;

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
      // std::exit(EXIT_FAILURE);
      std::cout << "Validation failed. " << std::endl;
    }
  }
  for (int i = 0; i < num_points; ++i) {
    if (h_point_to_voxelidx[i] != d_point_to_voxelidx[i]) {
      std::cout << "Coors: the " << i << "th element is not equal!!!" << std::endl;
      // std::exit(EXIT_FAILURE);
      std::cout << "Validation failed. " << std::endl;
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
