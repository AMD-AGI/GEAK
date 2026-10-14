// ============================================================
// custom_kernels MatMulNBits bits=2 vs bits=4 Verification + Benchmark
//
// Exercises the hip_matmul_nbits() API for the uint2 (2-bit packed,
// ONNX MatMulNBits convention) quantized-weight kernel and benchmarks it
// back-to-back against the existing uint4 (nibble-packed) kernel on the
// *same* (M, K, N, group_size) shape, so the two can be compared fairly.
//
//   C[M×N] = A[M×K] × dequant(B)^T
//
// Public API tensors — all ROW-MAJOR:
//   A            : FP16 [batch, M, K]                        (shared)
//   u2: B        : uint8 [N, K/4] packed 2-bit stream (4 vals/byte)
//       scales   : FP16  [N, num_groups_k]
//       zeros_u8 : uint8 [N, num_groups_k]  (optional, one byte per group)
//       zeros_pkd: uint8 [N, ceil(num_groups_k/4)] (optional, ONNX-packed)
//   u4: B_packed : uint8 [N, K/2] nibble-packed (ONNX convention)
//       scales   : FP16  [N, num_groups_k]
//       zeros_u8 : uint8 [N, num_groups_k]  (optional, per-element)
//       zeros_fp16: FP16 [N, num_groups_k]  (optional, same values as
//                                             zeros_u8, for WMMA/col-GEMV)
//   C            : FP16 row-major [batch, M, N]
//
// Two zero_points conventions for u2 are validated:
//   (a) one-byte-per-group (zeros_u8) passed directly as zero_points with
//       pre_unpacked_zp_u8 = null — the plain direct-call convention that the
//       benchmark loop uses.
//   (b) ONNX-packed (zeros_packed, 4 group-zp per byte) unpacked once via
//       hip_matmul_nbits_unpack_zp_u8_2bit() and fed back as
//       pre_unpacked_zp_u8 — the real runtime integration path. A dedicated
//       correctness check (not benchmarked) confirms the unpack reproduces
//       zeros_u8 and that the GEMM result matches the reference.
//
// Workflow:
//   1) python gen_matmul_nbits_u2_data.py MxKxN --group-size GS --dir data
//   2) build and run the test executable
// ============================================================
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "hip_custom_kernels.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <chrono>
#include <string>
#include <algorithm>
#include <functional>
#include <cstring>
#include <cstdlib>

static float half_to_float(__half h)
{
    uint16_t bits;
    std::memcpy(&bits, &h, sizeof(bits));
    uint32_t sign = (bits >> 15) & 1;
    uint32_t exp  = (bits >> 10) & 0x1F;
    uint32_t mant = bits & 0x3FF;
    uint32_t f;
    if(exp == 0)
    {
        if(mant == 0)
            f = sign << 31;
        else
        {
            exp = 1;
            while(!(mant & 0x400)) { mant <<= 1; exp--; }
            mant &= 0x3FF;
            f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
        }
    }
    else if(exp == 31)
        f = (sign << 31) | (0xFFu << 23) | (mant << 13);
    else
        f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
    float result;
    std::memcpy(&result, &f, sizeof(result));
    return result;
}

#define HIP_CHECK(call)                                                     \
    do                                                                      \
    {                                                                       \
        hipError_t err = (call);                                            \
        if(err != hipSuccess)                                               \
        {                                                                   \
            std::cerr << "HIP error at " << __FILE__ << ":" << __LINE__     \
                      << " code=" << err << " \""                           \
                      << hipGetErrorString(err) << "\"" << std::endl;       \
            exit(1);                                                        \
        }                                                                   \
    } while(0)

template <typename T>
static bool readBin(const std::string& path, std::vector<T>& data, size_t count)
{
    std::ifstream f(path, std::ios::binary);
    if(!f.is_open())
        return false;
    data.resize(count);
    f.read(reinterpret_cast<char*>(data.data()), count * sizeof(T));
    return f.good();
}

// ============================================================
// Benchmark helpers
// ============================================================
static int calibrateIters(double warmup_ms, int warmup_count,
                          double target_ms = 1500.0, int lo = 5, int hi = 200)
{
    double per_iter = warmup_ms / warmup_count;
    if(per_iter <= 0) return lo;
    return std::max(lo, std::min(hi, static_cast<int>(target_ms / per_iter)));
}

constexpr int NROUNDS = 5;

struct MeasureResult {
    double median_ms;
    double min_ms;
    double max_ms;
};

static MeasureResult measureMedian(hipStream_t stream, int niters,
                                   const std::function<void()>& launch)
{
    hipEvent_t ev0, ev1;
    hipEventCreate(&ev0);
    hipEventCreate(&ev1);
    std::vector<float> round_ms(NROUNDS);
    for(int r = 0; r < NROUNDS; r++)
    {
        hipEventRecord(ev0, stream);
        for(int i = 0; i < niters; i++)
            launch();
        hipEventRecord(ev1, stream);
        hipEventSynchronize(ev1);
        hipEventElapsedTime(&round_ms[r], ev0, ev1);
    }
    hipEventDestroy(ev0);
    hipEventDestroy(ev1);

    std::sort(round_ms.begin(), round_ms.end());
    return {round_ms[NROUNDS / 2], round_ms[0], round_ms[NROUNDS - 1]};
}

// ============================================================
// Per-kernel benchmark + verify (shared by both bits=2 and bits=4)
// ============================================================
struct KernelStat {
    std::string label;
    bool launch_ok  = false;
    double avg_ms   = 0.0;
    double gflops   = 0.0;
    double bw_gbs   = 0.0;
    double min_ms   = 0.0;
    double max_ms   = 0.0;
    bool   has_ref  = false;
    int    errors   = -1;
    int    total    = 0;
    float  max_diff = 0.0f;
    float  max_rdiff = 0.0f;
};

static int verifyAgainstRef(const std::vector<__half>& h_C,
                            const std::vector<__half>& h_C_ref, size_t countC,
                            float& max_diff_out, float& max_rdiff_out)
{
    int errors = 0;
    float max_diff = 0.0f, max_rdiff = 0.0f;
    for(size_t i = 0; i < countC; i++)
    {
        float gpu_val = half_to_float(h_C[i]);
        float ref_val = half_to_float(h_C_ref[i]);
        float diff    = std::fabs(gpu_val - ref_val);
        float rdiff   = (std::fabs(ref_val) > 1e-6f) ? diff / std::fabs(ref_val) : diff;
        if(diff > max_diff) max_diff = diff;
        if(rdiff > max_rdiff) max_rdiff = rdiff;
        float tol = std::fabs(ref_val) * 0.05f + 0.1f;
        if(diff > tol) errors++;
    }
    max_diff_out = max_diff;
    max_rdiff_out = max_rdiff;
    return errors;
}

static KernelStat benchmarkAndVerify(
    const std::string& label,
    hipStream_t stream,
    const std::function<int()>& launch_checked,
    const std::function<void()>& launch,
    int M, int N, int K,
    double mem_bytes,
    __half* d_C, size_t countC,
    const std::vector<__half>& h_C_ref, bool has_ref)
{
    KernelStat st;
    st.label   = label;
    st.has_ref = has_ref;

    std::cout << "\n  --- " << label << " ---" << std::endl;

    constexpr int PRE_WARMUP = 500;
    std::cout << "  Pre-warmup (" << PRE_WARMUP << " iters)..." << std::flush;
    auto pw0 = std::chrono::steady_clock::now();
    for(int w = 0; w < PRE_WARMUP; w++)
        launch();
    HIP_CHECK(hipStreamSynchronize(stream));
    double pw_ms = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - pw0).count() / 1000.0;
    std::cout << " done (" << std::fixed << std::setprecision(3)
              << pw_ms / PRE_WARMUP << " ms/iter)" << std::endl;

    std::cout << "  Warmup..." << std::flush;
    auto tw0 = std::chrono::steady_clock::now();
    int status = 0;
    for(int w = 0; w < 3; w++)
    {
        status = launch_checked();
        if(status != 0) break;
    }
    HIP_CHECK(hipStreamSynchronize(stream));
    double warmup_ms = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - tw0).count() / 1000.0;

    if(status != 0)
    {
        std::cout << " FAILED (status=" << status << ")" << std::endl;
        return st;
    }

    int niters = calibrateIters(warmup_ms, 3);
    std::cout << " OK (" << std::fixed << std::setprecision(2) << warmup_ms
              << " ms), iters=" << niters << std::endl;

    std::cout << "  Benchmarking (" << NROUNDS << " rounds x " << niters
              << " iters)..." << std::flush;
    auto mr = measureMedian(stream, niters, launch);
    std::cout << " done" << std::endl;

    st.launch_ok = true;
    st.avg_ms    = mr.median_ms / niters;
    st.min_ms    = mr.min_ms / niters;
    st.max_ms    = mr.max_ms / niters;
    st.gflops    = (2.0 * M * N * K) / (st.avg_ms * 1e6);
    st.bw_gbs    = mem_bytes * niters / (mr.median_ms * 1e6);

    std::cout << "  Median: " << std::setprecision(6) << st.avg_ms << " ms, "
              << std::setprecision(2) << st.gflops << " GFLOPS, "
              << st.bw_gbs << " GB/s   (range " << st.min_ms << " ~ "
              << st.max_ms << " ms)" << std::endl;

    if(has_ref)
    {
        std::vector<__half> h_C(countC);
        HIP_CHECK(hipMemcpy(h_C.data(), d_C, countC * sizeof(__half),
                            hipMemcpyDeviceToHost));
        float max_diff = 0.0f, max_rdiff = 0.0f;
        int errors = verifyAgainstRef(h_C, h_C_ref, countC, max_diff, max_rdiff);

        st.errors    = errors;
        st.total     = static_cast<int>(countC);
        st.max_diff  = max_diff;
        st.max_rdiff = max_rdiff;

        std::cout << "  Verify: " << (st.total - errors) << "/" << st.total
                  << " OK, max_abs_diff=" << std::setprecision(4) << max_diff
                  << ", max_rel_diff=" << (max_rdiff * 100.0f) << "%   "
                  << (errors == 0 ? "PASS" : "FAIL") << std::endl;
    }

    return st;
}

static void printComparison(const KernelStat& u2, const KernelStat& u4,
                            size_t bytes_u2, size_t bytes_u4)
{
    std::cout << "\n  === u2 vs u4 comparison ===" << std::endl;
    std::cout << "  " << std::left << std::setw(10) << "Kernel"
              << std::right << std::setw(14) << "Median(ms)"
              << std::setw(12) << "GFLOPS"
              << std::setw(10) << "GB/s"
              << std::setw(14) << "B size(KB)"
              << std::setw(10) << "Verify" << std::endl;

    auto row = [&](const KernelStat& s, size_t bytes) {
        std::cout << "  " << std::left << std::setw(10) << s.label
                  << std::right << std::setw(14);
        if(s.launch_ok)
            std::cout << std::fixed << std::setprecision(6) << s.avg_ms;
        else
            std::cout << "FAILED";
        std::cout << std::setw(12) << std::setprecision(2)
                  << (s.launch_ok ? s.gflops : 0.0)
                  << std::setw(10) << (s.launch_ok ? s.bw_gbs : 0.0)
                  << std::setw(14) << std::setprecision(1) << (bytes / 1024.0)
                  << std::setw(10)
                  << (!s.has_ref ? "N/A" : (s.errors == 0 ? "PASS" : "FAIL"))
                  << std::endl;
    };
    row(u2, bytes_u2);
    row(u4, bytes_u4);

    if(u2.launch_ok && u4.launch_ok)
    {
        double speedup = u4.avg_ms / u2.avg_ms;
        double size_ratio = 100.0 * bytes_u2 / bytes_u4;
        std::cout << "  u2/u4 weight size: " << std::fixed << std::setprecision(1)
                  << size_ratio << "%   u2 vs u4 speed: "
                  << std::setprecision(2) << speedup << "x "
                  << (speedup >= 1.0 ? "(u2 faster)" : "(u4 faster)")
                  << std::endl;
    }
}

// ============================================================
// Packed-zero_points real-model path: correctness + PERFORMANCE (bits=2)
//
// Runs the exact steady-state path the runtime uses for an asym 2-bit ONNX
// model, and compares its performance head-to-head against u4 WITH zp:
//   1. hip_matmul_nbits_unpack_zp_u8_2bit() unpacks the 4-per-byte packed
//      zero_points blob to one byte per group ONCE (byte-exact check vs the
//      generator's zeros_u8), mirroring the wrapper's pointer-keyed cache.
//   2. The GEMM is benchmarked + verified with the raw packed blob as
//      zero_points and the unpacked buffer as pre_unpacked_zp_u8. The unpack
//      is NOT in the timed loop (once-per-pointer cache in the real runtime),
//      so the number reflects steady-state decode cost, which is what should
//      be compared against u4's with-zp throughput.
// ============================================================
static bool benchmarkPackedZpPath(
    hipStream_t stream,
    const __half* d_A, const uint8_t* d_B_u2, const __half* d_S_u2,
    const uint8_t* d_Z_packed, const std::vector<uint8_t>& h_Z_u8_ref,
    int M, int N, int K, int group_size, int num_groups_k,
    __half* d_C, size_t countC, double mem_bytes,
    const std::vector<__half>& h_C_ref, bool has_ref,
    const KernelStat& u4_stat)
{
    std::cout << "\n  --- u2 packed-zp real-model path (vs u4 with zp) ---" << std::endl;

    size_t countZ = static_cast<size_t>(N) * num_groups_k;
    uint8_t* d_Z_unpacked = nullptr;
    HIP_CHECK(hipMalloc(&d_Z_unpacked, countZ));
    HIP_CHECK(hipMemset(d_Z_unpacked, 0xEE, countZ));

    // Unpack ONCE (as the wrapper's cache does), then check byte-exactness.
    hip_matmul_nbits_unpack_zp_u8_2bit(stream, d_Z_packed, d_Z_unpacked,
                                       N, num_groups_k);
    HIP_CHECK(hipStreamSynchronize(stream));

    std::vector<uint8_t> h_Z_unpacked(countZ);
    HIP_CHECK(hipMemcpy(h_Z_unpacked.data(), d_Z_unpacked, countZ,
                        hipMemcpyDeviceToHost));
    size_t zp_mismatches = 0;
    for(size_t i = 0; i < countZ; i++)
        if(h_Z_unpacked[i] != h_Z_u8_ref[i]) zp_mismatches++;
    bool unpack_ok = (zp_mismatches == 0);
    std::cout << "  Unpack check: " << (countZ - zp_mismatches) << "/" << countZ
              << " zero_points match one-byte-per-group   "
              << (unpack_ok ? "PASS" : "FAIL") << std::endl;

    // Benchmark + verify the steady-state GEMM: packed blob as zero_points,
    // pre-unpacked buffer as pre_unpacked_zp_u8 (unpack excluded from timing).
    auto launch_checked = [&]() -> int {
        return hip_matmul_nbits(
            stream, d_A, d_B_u2, d_S_u2,
            d_Z_packed,        // zero_points (raw ONNX-packed blob)
            nullptr,           // bias
            d_C,
            M, N, K, 1, 2, group_size, 2, 1,
            d_Z_unpacked,      // pre_unpacked_zp_u8 (runtime cache output)
            nullptr);          // pre_unpacked_zp_fp16 (u2 WMMA consumes u8)
    };
    auto launch = [&]() { launch_checked(); };

    KernelStat st = benchmarkAndVerify(
        "u2(pkd-zp)", stream, launch_checked, launch, M, N, K,
        mem_bytes, d_C, countC, h_C_ref, has_ref);

    if(st.launch_ok && u4_stat.launch_ok)
    {
        double speedup = u4_stat.avg_ms / st.avg_ms;
        std::cout << "  u2(pkd-zp) vs u4(zp) speed: " << std::fixed
                  << std::setprecision(2) << speedup << "x "
                  << (speedup >= 1.0 ? "(u2 faster)" : "(u4 faster)")
                  << "   [u2 " << st.gflops << " GFLOPS / " << st.bw_gbs
                  << " GB/s  vs  u4 " << u4_stat.gflops << " GFLOPS / "
                  << u4_stat.bw_gbs << " GB/s]" << std::endl;
    }

    hipFree(d_Z_unpacked);
    return unpack_ok && st.launch_ok && (!has_ref || st.errors == 0);
}

// ============================================================
// Single-shape comparison test
// ============================================================
bool testCompareShape(int M, int N, int K, int group_size,
                      const std::string& data_dir, bool use_zeros)
{
    int num_groups_k = (K + group_size - 1) / group_size;

    std::cout << "\n=== MatMulNBits u2 vs u4  M=" << M << " N=" << N
              << " K=" << K << " group_size=" << group_size
              << (use_zeros ? "" : " (no zeros)") << " ===" << std::endl;

    size_t countA          = static_cast<size_t>(M) * K;
    size_t countC          = static_cast<size_t>(M) * N;
    size_t rowBytesU2      = static_cast<size_t>(K) / 4;
    size_t countB_u2       = static_cast<size_t>(N) * rowBytesU2;
    size_t countB_u4       = static_cast<size_t>(N) * K / 2;
    size_t countS          = static_cast<size_t>(N) * num_groups_k;
    size_t packedZpCols    = (static_cast<size_t>(num_groups_k) + 3) / 4;
    size_t countZ_packed   = static_cast<size_t>(N) * packedZpCols;

    // ---- Load shared A ----
    std::vector<__half> h_A;
    if(!readBin(data_dir + "/matmul_nbits_A.bin", h_A, countA))
    {
        std::cerr << "  ERROR: cannot read " << data_dir << "/matmul_nbits_A.bin" << std::endl;
        std::cerr << "  Run: python gen_matmul_nbits_u2_data.py " << M << "x" << K << "x" << N
                  << " --group-size " << group_size
                  << (use_zeros ? "" : " --no-zeros") << " --dir " << data_dir << std::endl;
        return false;
    }

    // ---- Load u2 data ----
    std::vector<uint8_t> h_B_u2;
    std::vector<__half>  h_S_u2;
    std::vector<uint8_t> h_Z_u2_u8;
    std::vector<uint8_t> h_Z_u2_packed;
    std::vector<__half>  h_Cref_u2;
    bool ok_u2 = readBin(data_dir + "/matmul_nbits_u2_B.bin", h_B_u2, countB_u2)
              && readBin(data_dir + "/matmul_nbits_u2_scales.bin", h_S_u2, countS);
    if(use_zeros)
        ok_u2 = ok_u2
              && readBin(data_dir + "/matmul_nbits_u2_zeros_u8.bin", h_Z_u2_u8, countS)
              && readBin(data_dir + "/matmul_nbits_u2_zeros_packed.bin", h_Z_u2_packed, countZ_packed);
    bool has_ref_u2 = readBin(data_dir + "/matmul_nbits_u2_C_ref.bin", h_Cref_u2, countC);

    // ---- Load u4 data ----
    std::vector<uint8_t> h_B_u4;
    std::vector<__half>  h_S_u4;
    std::vector<uint8_t> h_Zu8_u4;
    std::vector<__half>  h_Zfp16_u4;
    std::vector<__half>  h_Cref_u4;
    bool ok_u4 = readBin(data_dir + "/matmul_nbits_u4_B_packed.bin", h_B_u4, countB_u4)
              && readBin(data_dir + "/matmul_nbits_u4_scales.bin", h_S_u4, countS);
    if(use_zeros)
        ok_u4 = ok_u4
              && readBin(data_dir + "/matmul_nbits_u4_zeros_u8.bin", h_Zu8_u4, countS)
              && readBin(data_dir + "/matmul_nbits_u4_zeros_fp16.bin", h_Zfp16_u4, countS);
    bool has_ref_u4 = readBin(data_dir + "/matmul_nbits_u4_C_ref.bin", h_Cref_u4, countC);

    if(!ok_u2 || !ok_u4)
    {
        std::cerr << "  ERROR: failed to read u2/u4 input data from " << data_dir << "/" << std::endl;
        std::cerr << "  Run: python gen_matmul_nbits_u2_data.py " << M << "x" << K << "x" << N
                  << " --group-size " << group_size
                  << (use_zeros ? "" : " --no-zeros") << " --dir " << data_dir << std::endl;
        return false;
    }
    std::cout << "  Loaded shared A + u2/u4 weights from " << data_dir << "/" << std::endl;

    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    // ---- Device buffers: shared A + per-kernel C ----
    __half* d_A = nullptr;
    __half* d_C_u2 = nullptr;
    __half* d_C_u4 = nullptr;
    HIP_CHECK(hipMalloc(&d_A, countA * sizeof(__half)));
    HIP_CHECK(hipMalloc(&d_C_u2, countC * sizeof(__half)));
    HIP_CHECK(hipMalloc(&d_C_u4, countC * sizeof(__half)));
    HIP_CHECK(hipMemcpy(d_A, h_A.data(), countA * sizeof(__half), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_C_u2, 0, countC * sizeof(__half)));
    HIP_CHECK(hipMemset(d_C_u4, 0, countC * sizeof(__half)));

    // ---- u2 device buffers ----
    uint8_t* d_B_u2 = nullptr;
    __half*  d_S_u2 = nullptr;
    uint8_t* d_Z_u2_u8 = nullptr;
    uint8_t* d_Z_u2_packed = nullptr;
    HIP_CHECK(hipMalloc(&d_B_u2, countB_u2));
    HIP_CHECK(hipMalloc(&d_S_u2, countS * sizeof(__half)));
    HIP_CHECK(hipMemcpy(d_B_u2, h_B_u2.data(), countB_u2, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_S_u2, h_S_u2.data(), countS * sizeof(__half), hipMemcpyHostToDevice));
    if(use_zeros)
    {
        HIP_CHECK(hipMalloc(&d_Z_u2_u8, countS));
        HIP_CHECK(hipMemcpy(d_Z_u2_u8, h_Z_u2_u8.data(), countS, hipMemcpyHostToDevice));
        HIP_CHECK(hipMalloc(&d_Z_u2_packed, countZ_packed));
        HIP_CHECK(hipMemcpy(d_Z_u2_packed, h_Z_u2_packed.data(), countZ_packed, hipMemcpyHostToDevice));
    }

    // ---- u4 device buffers ----
    uint8_t* d_B_u4 = nullptr;
    __half*  d_S_u4 = nullptr;
    uint8_t* d_Zu8_u4 = nullptr;
    __half*  d_Zfp16_u4 = nullptr;
    HIP_CHECK(hipMalloc(&d_B_u4, countB_u4));
    HIP_CHECK(hipMalloc(&d_S_u4, countS * sizeof(__half)));
    HIP_CHECK(hipMemcpy(d_B_u4, h_B_u4.data(), countB_u4, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_S_u4, h_S_u4.data(), countS * sizeof(__half), hipMemcpyHostToDevice));
    if(use_zeros)
    {
        HIP_CHECK(hipMalloc(&d_Zu8_u4, countS));
        HIP_CHECK(hipMalloc(&d_Zfp16_u4, countS * sizeof(__half)));
        HIP_CHECK(hipMemcpy(d_Zu8_u4, h_Zu8_u4.data(), countS, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_Zfp16_u4, h_Zfp16_u4.data(), countS * sizeof(__half), hipMemcpyHostToDevice));
    }

    // ---- Launchers ----
    // u2 benchmark path: one-byte-per-group zero_points passed directly
    // (pre_unpacked_zp_u8 = null). The packed-zp real-model path is validated
    // separately below via validatePackedZpPath().
    auto launch_u2_checked = [&]() -> int {
        return hip_matmul_nbits(
            stream, d_A, d_B_u2, d_S_u2,
            use_zeros ? d_Z_u2_u8 : nullptr,
            nullptr,           // bias
            d_C_u2,
            M, N, K,
            1,                 // batch_count
            2,                 // bits
            group_size,        // block_size
            2,                 // element_size_bytes (fp16)
            1,                 // zp_elem_size (uint8, one byte per group)
            nullptr,           // pre_unpacked_zp_u8 (direct convention)
            nullptr);          // pre_unpacked_zp_fp16 (unused for bits=2)
    };
    auto launch_u2 = [&]() { launch_u2_checked(); };

    auto launch_u4_checked = [&]() -> int {
        return hip_matmul_nbits(
            stream, d_A, d_B_u4, d_S_u4,
            use_zeros ? d_Zu8_u4 : nullptr,
            nullptr,           // bias
            d_C_u4,
            M, N, K,
            1,                 // batch_count
            4,                 // bits
            group_size,        // block_size
            2,                 // element_size_bytes (fp16)
            1,                 // zp_elem_size (uint8, pre-unpacked)
            use_zeros ? d_Zu8_u4    : nullptr,   // pre_unpacked_zp_u8
            use_zeros ? d_Zfp16_u4  : nullptr);  // pre_unpacked_zp_fp16
    };
    auto launch_u4 = [&]() { launch_u4_checked(); };

    double mem_bytes_u2 = static_cast<double>(countA) * 2 + static_cast<double>(countB_u2)
                        + static_cast<double>(countS) * 2
                        + (use_zeros ? static_cast<double>(countS) : 0.0)
                        + static_cast<double>(countC) * 2;
    double mem_bytes_u4 = static_cast<double>(countA) * 2 + static_cast<double>(countB_u4)
                        + static_cast<double>(countS) * 2
                        + (use_zeros ? static_cast<double>(countS) * 2 : 0.0)
                        + static_cast<double>(countC) * 2;

    KernelStat stat_u2 = benchmarkAndVerify(
        "u2", stream, launch_u2_checked, launch_u2, M, N, K,
        mem_bytes_u2, d_C_u2, countC, h_Cref_u2, has_ref_u2);
    KernelStat stat_u4 = benchmarkAndVerify(
        "u4", stream, launch_u4_checked, launch_u4, M, N, K,
        mem_bytes_u4, d_C_u4, countC, h_Cref_u4, has_ref_u4);

    printComparison(stat_u2, stat_u4, countB_u2, countB_u4);

    // ---- Packed-zp real-model path (only meaningful with zeros) ----
    bool packed_ok = true;
    if(use_zeros)
    {
        packed_ok = benchmarkPackedZpPath(
            stream, d_A, d_B_u2, d_S_u2, d_Z_u2_packed, h_Z_u2_u8,
            M, N, K, group_size, num_groups_k,
            d_C_u2, countC, mem_bytes_u2, h_Cref_u2, has_ref_u2, stat_u4);
    }

    bool pass = stat_u2.launch_ok && stat_u4.launch_ok
             && (!stat_u2.has_ref || stat_u2.errors == 0)
             && (!stat_u4.has_ref || stat_u4.errors == 0)
             && packed_ok;

    hipFree(d_A);
    hipFree(d_C_u2); hipFree(d_C_u4);
    hipFree(d_B_u2); hipFree(d_S_u2);
    if(d_Z_u2_u8) hipFree(d_Z_u2_u8);
    if(d_Z_u2_packed) hipFree(d_Z_u2_packed);
    hipFree(d_B_u4); hipFree(d_S_u4);
    if(d_Zu8_u4) hipFree(d_Zu8_u4);
    if(d_Zfp16_u4) hipFree(d_Zfp16_u4);
    hipStreamDestroy(stream);

    return pass;
}

// ============================================================
// Model sweep: parse model config and iterate all shapes
// ============================================================
static std::string slurpFile(const std::string& path)
{
    std::ifstream f(path);
    return std::string((std::istreambuf_iterator<char>(f)),
                        std::istreambuf_iterator<char>());
}

static std::vector<int> jsonIntArray(const std::string& s, const std::string& key)
{
    std::vector<int> result;
    std::string needle = "\"" + key + "\"";
    auto pos = s.find(needle);
    if(pos == std::string::npos) return result;
    auto bracket = s.find('[', pos);
    if(bracket == std::string::npos) return result;
    auto end_bracket = s.find(']', bracket);
    if(end_bracket == std::string::npos) return result;
    std::string arr = s.substr(bracket + 1, end_bracket - bracket - 1);
    size_t i = 0;
    while(i < arr.size())
    {
        while(i < arr.size() && !std::isdigit(static_cast<unsigned char>(arr[i])) && arr[i] != '-')
            i++;
        if(i >= arr.size()) break;
        result.push_back(std::stoi(arr.substr(i)));
        while(i < arr.size() && (std::isdigit(static_cast<unsigned char>(arr[i])) || arr[i] == '-'))
            i++;
    }
    return result;
}

static std::vector<int> jsonNestedIntArray(const std::string& s,
                                           const std::string& objKey,
                                           const std::string& arrKey)
{
    auto opos = s.find("\"" + objKey + "\"");
    if(opos == std::string::npos) return {};
    auto brace = s.find('{', opos);
    if(brace == std::string::npos) return {};
    int depth = 1;
    auto bend = brace + 1;
    while(bend < s.size() && depth > 0)
    {
        if(s[bend] == '{') depth++;
        if(s[bend] == '}') depth--;
        bend++;
    }
    return jsonIntArray(s.substr(brace, bend - brace), arrKey);
}

struct ModelConfig {
    std::vector<int> M_array;
    std::vector<int> K_array;
    std::vector<int> N_array;
};

static ModelConfig parseModelConfig(const std::string& path)
{
    auto json = slurpFile(path);
    if(json.empty())
    {
        std::cerr << "ERROR: cannot read " << path << std::endl;
        exit(1);
    }
    ModelConfig cfg;
    cfg.M_array = jsonIntArray(json, "M_array");
    cfg.K_array = jsonNestedIntArray(json, "KN_pairs", "K");
    cfg.N_array = jsonNestedIntArray(json, "KN_pairs", "N");
    if(cfg.K_array.size() != cfg.N_array.size())
    {
        std::cerr << "ERROR: K and N arrays in KN_pairs must have the same length" << std::endl;
        exit(1);
    }
    std::cout << "  Model: " << cfg.M_array.size() << " M values x "
              << cfg.K_array.size() << " KN pairs = "
              << cfg.M_array.size() * cfg.K_array.size() << " shapes" << std::endl;
    return cfg;
}

struct SweepRow {
    int M, K, N;
    KernelStat u2, u4;
    size_t bytesB_u2 = 0, bytesB_u4 = 0;
    bool skipped = false;
};

static int runModelSweep(const std::string& json_path, int group_size,
                         bool use_zeros, const std::string& data_root)
{
    std::cout << "\nModel sweep (u2 vs u4): " << json_path
              << "  (group_size=" << group_size
              << (use_zeros ? ", with zeros)" : ", no zeros)")
              << "\n  Data root: " << data_root << std::endl;

    auto model = parseModelConfig(json_path);
    if(model.M_array.empty() || model.K_array.empty())
    {
        std::cerr << "ERROR: empty model config" << std::endl;
        return 1;
    }

    std::vector<SweepRow> rows;
    int shape_idx = 0;
    int total_shapes = static_cast<int>(model.K_array.size() * model.M_array.size());

    for(size_t ki = 0; ki < model.K_array.size(); ki++)
    {
        int K = model.K_array[ki];
        int N = model.N_array[ki];
        for(size_t mi = 0; mi < model.M_array.size(); mi++)
        {
            int M = model.M_array[mi];
            shape_idx++;
            SweepRow row{M, K, N};

            if(K % 32 != 0)
            {
                std::cout << "\n=== [" << shape_idx << "/" << total_shapes
                          << "] M=" << M << " K=" << K << " N=" << N
                          << "  SKIPPED (K not %32) ===" << std::endl;
                row.skipped = true;
                rows.push_back(row);
                continue;
            }

            std::string shape_dir = data_root + "/"
                + std::to_string(M) + "x" + std::to_string(K) + "x" + std::to_string(N);

            std::cout << "\n=== [" << shape_idx << "/" << total_shapes << "] "
                      << shape_dir << " ===" << std::endl;

            bool pass = testCompareShape(M, N, K, group_size, shape_dir, use_zeros);
            (void)pass;
            rows.push_back(row);
        }
    }

    std::cout << "\nDone: " << rows.size() << " shapes attempted "
              << "(see per-shape sections above for pass/fail and timings)" << std::endl;
    return 0;
}

// ============================================================

int main(int argc, char* argv[])
{
    std::cout << "custom_kernels MatMulNBits bits=2 (uint2) vs bits=4 (uint4) Verification" << std::endl;
    std::cout << "==========================================================================" << std::endl;

    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    std::cout << "GPU: " << prop.name << " (arch: " << prop.gcnArchName << ")" << std::endl;

    // --- Check for --model mode ---
    std::string model_json;
    std::string model_data_root = "data_model";
    int model_gs = 128;
    bool model_use_zeros = true;
    for(int i = 1; i < argc; i++)
    {
        if(std::string(argv[i]) == "--model" && i + 1 < argc)
            model_json = argv[++i];
        else if(std::string(argv[i]) == "--data-root" && i + 1 < argc)
            model_data_root = argv[++i];
        else if(std::string(argv[i]) == "--group-size" && i + 1 < argc)
            model_gs = std::atoi(argv[++i]);
        else if(std::string(argv[i]) == "--no-zeros")
            model_use_zeros = false;
    }

    if(!model_json.empty())
        return runModelSweep(model_json, model_gs, model_use_zeros, model_data_root);

    // --- Random-data single-shape mode (default) ---
    int M = 128, N = 128, K = 128, gs = 128;
    std::string data_dir = "data";
    bool use_zeros = true;

    for(int i = 1; i < argc; i++)
    {
        if(std::string(argv[i]) == "--no-zeros")
            use_zeros = false;
    }

    if(argc >= 2 && std::string(argv[1]) != "--no-zeros")
    {
        if(sscanf(argv[1], "%dx%dx%d", &M, &K, &N) != 3)
        {
            std::cerr << "Usage: " << argv[0]
                      << " [MxKxN] [group_size] [data_dir] [--no-zeros]" << std::endl;
            std::cerr << "       " << argv[0]
                      << " --model <json> [--data-root DIR] [--group-size GS] [--no-zeros]" << std::endl;
            return 1;
        }
    }
    if(argc >= 3 && std::string(argv[2]) != "--no-zeros") gs = atoi(argv[2]);
    if(argc >= 4 && std::string(argv[3]) != "--no-zeros") data_dir = argv[3];

    std::cout << "Data dir: " << data_dir << std::endl;
    if(!use_zeros)
        std::cout << "Zero points: disabled (--no-zeros)" << std::endl;

    bool all_pass = testCompareShape(M, N, K, gs, data_dir, use_zeros);

    std::cout << "\n==========================================================================" << std::endl;
    std::cout << "Overall: " << (all_pass ? "ALL PASSED" : "SOME FAILED") << std::endl;

    return all_pass ? 0 : 1;
}
