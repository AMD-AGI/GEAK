#include "hip/hip_runtime.h"
// Optimized KNN kernel with warp-cooperative search + template parameterization
// + direct (B, K, M) output format + native transposed layout support
// Based on original from https://github.com/CVMI-Lab/PAConv/tree/main/scene_seg/lib/pointops/src/knnquery_heap

#include <cmath>
#include <cstdio>
#include <float.h>

#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))
#define WARP_SIZE 64
#define BLOCK_SIZE 256
#define WARPS_PER_BLOCK (BLOCK_SIZE / WARP_SIZE)  // 4

// Warp-cooperative KNN kernel with compile-time layout selection
// TRANSPOSED=false: xyz is (B, N, 3) with stride-3 access (AoS)
// TRANSPOSED=true:  xyz is (B, 3, N) with stride-1 per component (SoA)
// Output is always (B, K, M) format
template <int K, bool TRANSPOSED>
__global__ __launch_bounds__(BLOCK_SIZE)
void knn_kernel_warp_opt(int b, int n, int m,
                         const float *__restrict__ xyz,
                         const float *__restrict__ new_xyz,
                         int *__restrict__ idx) {
    const int warp_id = threadIdx.x >> 6;
    const int lane    = threadIdx.x & 63;
    const int bs_idx  = blockIdx.y;
    const int pt_idx  = blockIdx.x * WARPS_PER_BLOCK + warp_id;

    if (bs_idx >= b || pt_idx >= m) return;

    float qx, qy, qz;
    const float *ref;

    if constexpr (TRANSPOSED) {
        // (B, 3, N) layout: xyz[b][c][i] = xyz[b * 3 * N + c * N + i]
        const float *q_base = new_xyz + bs_idx * 3 * m;
        qx = q_base[0 * m + pt_idx];
        qy = q_base[1 * m + pt_idx];
        qz = q_base[2 * m + pt_idx];
        ref = xyz + bs_idx * 3 * n;
    } else {
        // (B, N, 3) layout: xyz[b][i][c] = xyz[b * N * 3 + i * 3 + c]
        const float *query = new_xyz + bs_idx * m * 3 + pt_idx * 3;
        qx = query[0];
        qy = query[1];
        qz = query[2];
        ref = xyz + bs_idx * n * 3;
    }

    float best_d[K];
    int   best_i[K];
    #pragma unroll
    for (int j = 0; j < K; j++) {
        best_d[j] = FLT_MAX;
        best_i[j] = 0;
    }

    for (int i = lane; i < n; i += WARP_SIZE) {
        float rx, ry, rz;
        if constexpr (TRANSPOSED) {
            rx = ref[0 * n + i];
            ry = ref[1 * n + i];
            rz = ref[2 * n + i];
        } else {
            rx = ref[i * 3 + 0];
            ry = ref[i * 3 + 1];
            rz = ref[i * 3 + 2];
        }
        float dx = qx - rx;
        float dy = qy - ry;
        float dz = qz - rz;
        float d = dx * dx + dy * dy + dz * dz;

        if (d < best_d[K - 1]) {
            best_d[K - 1] = d;
            best_i[K - 1] = i;
            #pragma unroll
            for (int p = K - 1; p > 0; p--) {
                if (best_d[p] < best_d[p - 1]) {
                    float td = best_d[p]; best_d[p] = best_d[p - 1]; best_d[p - 1] = td;
                    int   ti = best_i[p]; best_i[p] = best_i[p - 1]; best_i[p - 1] = ti;
                } else {
                    break;
                }
            }
        }
    }

    extern __shared__ char smem[];
    float *s_dist = reinterpret_cast<float*>(smem) + warp_id * 64 * K;
    int   *s_idx  = reinterpret_cast<int*>(smem + WARPS_PER_BLOCK * 64 * K * sizeof(float)) + warp_id * 64 * K;

    #pragma unroll
    for (int j = 0; j < K; j++) {
        s_dist[lane * K + j] = best_d[j];
        s_idx[lane * K + j]  = best_i[j];
    }
    __syncthreads();

    for (int stride = 1; stride < WARP_SIZE; stride *= 2) {
        if ((lane & (stride * 2 - 1)) == 0 && (lane + stride) < WARP_SIZE) {
            float *a_d = &s_dist[lane * K];
            int   *a_i = &s_idx[lane * K];
            float *b_d = &s_dist[(lane + stride) * K];
            int   *b_i = &s_idx[(lane + stride) * K];

            float merged_d[16];
            int   merged_i[16];
            int ai = 0, bi = 0;
            #pragma unroll
            for (int j = 0; j < K; j++) {
                if (ai < K && (bi >= K || a_d[ai] <= b_d[bi])) {
                    merged_d[j] = a_d[ai]; merged_i[j] = a_i[ai]; ai++;
                } else {
                    merged_d[j] = b_d[bi]; merged_i[j] = b_i[bi]; bi++;
                }
            }
            #pragma unroll
            for (int j = 0; j < K; j++) {
                a_d[j] = merged_d[j];
                a_i[j] = merged_i[j];
            }
        }
        __syncthreads();
    }

    // Write directly in (B, K, M) layout
    if (lane == 0) {
        #pragma unroll
        for (int j = 0; j < K; j++) {
            idx[bs_idx * K * m + j * m + pt_idx] = s_idx[j];
        }
    }
}


// Dispatch helper: selects transposed vs non-transposed at runtime via template function
template <int K>
static void launch_knn_dispatch(int b, int n, int m, bool transposed,
                                const float *xyz, const float *new_xyz,
                                int *idx, hipStream_t stream) {
    dim3 blocks(DIVUP(m, WARPS_PER_BLOCK), b);
    dim3 threads(BLOCK_SIZE);
    size_t smem_size = 2 * WARPS_PER_BLOCK * 64 * K * sizeof(float);
    if (transposed) {
       hipLaunchKernelGGL(( knn_kernel_warp_opt<K, true>), dim3(blocks), dim3(threads), smem_size, stream, b, n, m, xyz, new_xyz, idx);
    } else {
       hipLaunchKernelGGL(( knn_kernel_warp_opt<K, false>), dim3(blocks), dim3(threads), smem_size, stream, b, n, m, xyz, new_xyz, idx);
    }
}


// Legacy launcher compatibility - not used by optimized path but kept for ABI
void knn_kernel_launcher(int b, int n, int m, int nsample, const float *xyz, const float *new_xyz, int *idx, float *dist2, hipStream_t stream) {
    // Redirect to optimized launcher (ignore dist2)
    hipError_t err;

    switch (nsample) {
        case 3:  launch_knn_dispatch<3>(b, n, m, false, xyz, new_xyz, idx, stream); break;
        case 5:  launch_knn_dispatch<5>(b, n, m, false, xyz, new_xyz, idx, stream); break;
        case 10: launch_knn_dispatch<10>(b, n, m, false, xyz, new_xyz, idx, stream); break;
        default: launch_knn_dispatch<16>(b, n, m, false, xyz, new_xyz, idx, stream); break;
    }

    err = hipGetLastError();
    if (hipSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", hipGetErrorString(err));
        exit(-1);
    }
}

// Optimized launcher (no dist2, output in B,K,M format, supports transposed input)
void knn_kernel_launcher_opt(int b, int n, int m, int nsample, bool transposed,
                             const float *xyz, const float *new_xyz, int *idx,
                             hipStream_t stream) {
    hipError_t err;

    switch (nsample) {
        case 3:  launch_knn_dispatch<3>(b, n, m, transposed, xyz, new_xyz, idx, stream); break;
        case 5:  launch_knn_dispatch<5>(b, n, m, transposed, xyz, new_xyz, idx, stream); break;
        case 10: launch_knn_dispatch<10>(b, n, m, transposed, xyz, new_xyz, idx, stream); break;
        default: launch_knn_dispatch<16>(b, n, m, transposed, xyz, new_xyz, idx, stream); break;
    }

    err = hipGetLastError();
    if (hipSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", hipGetErrorString(err));
        exit(-1);
    }
}
