#!/usr/bin/env python3
"""
MatMulNBits bits=2 vs bits=4 test data generator + NumPy reference

Generates a SHARED FP16 A, then two independently-quantized B matrices of
the SAME (M, K, N, group_size) shape:
  - uint2, packed 4-per-byte, LSB-first (this IS the ONNX MatMulNBits
    blockwise convention for bits=2 — blob_size = block_size/4 bytes — and,
    because 2 divides 8 evenly, simultaneously a plain continuous per-row
    2-bit stream; the two are byte-identical when block_size % 4 == 0)
  - uint4, nibble-packed (the existing ONNX MatMulNBits convention, same
    format gen_matmul_nbits_data.py in ../gemm_fp16u4 produces)

so the C++ test binary can benchmark hip_matmul_nbits(bits=2) and
hip_matmul_nbits(bits=4) back-to-back on identical shapes and report a fair
side-by-side comparison.

uint2 packing: value k occupies bits [2k, 2k+2) of row n's bitstream,
LSB-first. Row byte stride = K/4 bytes (K must be a multiple of 4; the GEMV
fast path in the kernel additionally requires K % 32 == 0).

zero_points, TWO layouts (both emitted when zeros are enabled):
  - *_zeros_u8      : uint8 [N, num_groups_k]  one byte per group (0..3).
                      This is what the u2 kernels index directly, and what
                      the runtime's zp-unpack kernel produces.
  - *_zeros_packed  : uint8 [N, ceil(num_groups_k/4)]  the SAME values,
                      packed 4-per-byte exactly like the ONNX bits=2
                      zero_points blob a real model ships. The test binary
                      runs hip_matmul_nbits_unpack_zp_u8_2bit() on this and
                      checks it reproduces *_zeros_u8, then feeds the result
                      through the kernel as pre_unpacked_zp_u8 — i.e. the
                      real runtime integration path.

Data layout (all row-major):
  A               : FP16  [M, K]                    (shared by both kernels)
  u2_B            : uint8 [N, K/4]        packed 2-bit stream
  u2_scales       : FP16  [N, num_groups_k]
  u2_zeros_u8     : uint8 [N, num_groups_k]          (optional)
  u2_zeros_packed : uint8 [N, ceil(num_groups_k/4)]  (optional)
  u2_C_ref        : FP16  [M, N]
  u4_B_packed     : uint8 [N, K/2]        nibble-packed
  u4_scales       : FP16  [N, num_groups_k]
  u4_zeros_u8     : uint8 [N, num_groups_k]          (optional, per-element)
  u4_zeros_fp16   : FP16  [N, num_groups_k]          (optional, same values)
  u4_C_ref        : FP16  [M, N]

Usage:
    python3 gen_matmul_nbits_u2_data.py [MxKxN] [--group-size GS] [--dir DIR]
"""

import numpy as np
import argparse
import os
import time


def pack_2bit(vals):
    """vals: uint8 [N, L] with values 0..3 -> uint8 [N, ceil(L/4)], packing
    4 values per byte LSB-first: byte = v0 | v1<<2 | v2<<4 | v3<<6.

    Used for both the weight stream (L=K, K%4==0) and the packed zero_points
    blob (L=num_groups_k, padded with zeros to a multiple of 4)."""
    N, L = vals.shape
    Lpad = (L + 3) // 4 * 4
    if Lpad != L:
        pad = np.zeros((N, Lpad - L), dtype=np.uint8)
        vals = np.concatenate([vals, pad], axis=1)
    v = vals.reshape(N, Lpad // 4, 4).astype(np.uint32)
    packed = v[:, :, 0] | (v[:, :, 1] << 2) | (v[:, :, 2] << 4) | (v[:, :, 3] << 6)
    return packed.astype(np.uint8)


def unpack_2bit(packed, L):
    """packed: uint8 [N, ceil(L/4)] -> uint8 [N, L] values 0..3 (inverse of
    pack_2bit; used only to self-check the round-trip)."""
    N = packed.shape[0]
    out = np.zeros((N, packed.shape[1] * 4), dtype=np.uint8)
    for j in range(4):
        out[:, j::4] = (packed >> (2 * j)) & 0x3
    return out[:, :L]


def main():
    parser = argparse.ArgumentParser(
        description='Generate MatMulNBits bits=2 vs bits=4 comparison test data')
    parser.add_argument('size', nargs='?', type=str, default='128x128x128',
                        help='Matrix size MxKxN (default: 128x128x128)')
    parser.add_argument('--group-size', type=int, default=128,
                        help='Quantization group size along K (default: 128)')
    parser.add_argument('--no-ref', action='store_true',
                        help='Skip computing reference C')
    parser.add_argument('--no-zeros', action='store_true',
                        help='Skip generating zeros (use default zero point)')
    parser.add_argument('--dir', type=str, default='data',
                        help='Output directory (default: data/)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    args = parser.parse_args()

    parts = args.size.split('x')
    if len(parts) != 3:
        parser.error(f"Size must be MxKxN (got '{args.size}')")
    M, K, N = int(parts[0]), int(parts[1]), int(parts[2])
    group_size = args.group_size
    num_groups_k = (K + group_size - 1) // group_size

    if K % 32 != 0:
        parser.error(f"K ({K}) must be a multiple of 32 (uint2 GEMV fast "
                     f"path + uint4 packing both require it)")

    out_dir = args.dir
    os.makedirs(out_dir, exist_ok=True)

    zp_str = "no-zeros" if args.no_zeros else "with-zeros"
    print(f"Generating M={M} N={N} K={K} gs={group_size} groups={num_groups_k} "
          f"({zp_str}, seed={args.seed})")

    np.random.seed(args.seed)

    # ---- Shared A ----
    A = np.random.uniform(-0.5, 0.5, (M, K)).astype(np.float16)
    A.flatten(order='C').tofile(os.path.join(out_dir, "matmul_nbits_A.bin"))

    group_idx = np.arange(K) // group_size

    # ================= uint2 (2-bit packed, ONNX convention) =================
    B_u2 = np.random.randint(0, 4, (N, K), dtype=np.uint8)
    B_u2_packed = pack_2bit(B_u2)

    # Self-check: pack/unpack round-trip before trusting the reference.
    B_u2_roundtrip = unpack_2bit(B_u2_packed, K)
    if not np.array_equal(B_u2, B_u2_roundtrip):
        raise RuntimeError("uint2 pack/unpack round-trip mismatch -- "
                           "pack_2bit has a bug")

    scales_u2 = np.random.uniform(0.01, 0.05, (N, num_groups_k)).astype(np.float16)
    zeros_u2 = None
    zeros_u2_packed = None
    if not args.no_zeros:
        # 2-bit zero_points are in 0..3; default (symmetric) zp is 2.
        zeros_u2 = np.random.randint(1, 3, (N, num_groups_k)).astype(np.uint8)
        zeros_u2_packed = pack_2bit(zeros_u2)

    B_u2_packed.flatten(order='C').tofile(
        os.path.join(out_dir, "matmul_nbits_u2_B.bin"))
    scales_u2.flatten(order='C').tofile(
        os.path.join(out_dir, "matmul_nbits_u2_scales.bin"))
    if zeros_u2 is not None:
        zeros_u2.flatten(order='C').tofile(
            os.path.join(out_dir, "matmul_nbits_u2_zeros_u8.bin"))
        zeros_u2_packed.flatten(order='C').tofile(
            os.path.join(out_dir, "matmul_nbits_u2_zeros_packed.bin"))

    # ================= uint4 (nibble-packed, ONNX convention) =================
    B_u4 = np.random.randint(0, 16, (N, K), dtype=np.uint8)
    B_u4_even = B_u4[:, 0::2]
    B_u4_odd = B_u4[:, 1::2]
    B_u4_packed = (B_u4_even | (B_u4_odd << 4)).astype(np.uint8)

    scales_u4 = np.random.uniform(0.01, 0.05, (N, num_groups_k)).astype(np.float16)
    zeros_u4_u8 = None
    zeros_u4_fp16 = None
    if not args.no_zeros:
        zeros_u4_u8 = np.random.randint(7, 10, (N, num_groups_k), dtype=np.uint8)
        zeros_u4_fp16 = zeros_u4_u8.astype(np.float16)

    B_u4_packed.flatten(order='C').tofile(
        os.path.join(out_dir, "matmul_nbits_u4_B_packed.bin"))
    scales_u4.flatten(order='C').tofile(
        os.path.join(out_dir, "matmul_nbits_u4_scales.bin"))
    if zeros_u4_u8 is not None:
        zeros_u4_u8.flatten(order='C').tofile(
            os.path.join(out_dir, "matmul_nbits_u4_zeros_u8.bin"))
        zeros_u4_fp16.flatten(order='C').tofile(
            os.path.join(out_dir, "matmul_nbits_u4_zeros_fp16.bin"))

    # ================= References =================
    if not args.no_ref:
        print("Computing references...", end=" ", flush=True)
        t0 = time.time()

        scales_u2_f32 = scales_u2.astype(np.float32)
        if zeros_u2 is not None:
            zp_u2_f32 = zeros_u2.astype(np.float32)
            B_u2_dq = (B_u2.astype(np.float32) - zp_u2_f32[:, group_idx]) \
                * scales_u2_f32[:, group_idx]
        else:
            B_u2_dq = (B_u2.astype(np.float32) - 2.0) * scales_u2_f32[:, group_idx]
        C_ref_u2 = (A.astype(np.float32) @ B_u2_dq.T).astype(np.float16)
        C_ref_u2.flatten(order='C').tofile(
            os.path.join(out_dir, "matmul_nbits_u2_C_ref.bin"))

        scales_u4_f32 = scales_u4.astype(np.float32)
        if zeros_u4_u8 is not None:
            zp_u4_f32 = zeros_u4_u8.astype(np.float32)
            B_u4_dq = (B_u4.astype(np.float32) - zp_u4_f32[:, group_idx]) \
                * scales_u4_f32[:, group_idx]
        else:
            B_u4_dq = (B_u4.astype(np.float32) - 8.0) * scales_u4_f32[:, group_idx]
        C_ref_u4 = (A.astype(np.float32) @ B_u4_dq.T).astype(np.float16)
        C_ref_u4.flatten(order='C').tofile(
            os.path.join(out_dir, "matmul_nbits_u4_C_ref.bin"))

        elapsed = time.time() - t0
        print(f"done ({elapsed:.2f}s)")
    else:
        print("Skipping reference (--no-ref)")

    row_bytes_u2 = K // 4
    row_bytes_u4 = K // 2
    print(f"B size: uint2={N * row_bytes_u2} bytes, uint4={N * row_bytes_u4} "
          f"bytes ({100.0 * row_bytes_u2 / row_bytes_u4:.1f}% of uint4)")

    meta_file = os.path.join(out_dir, "matmul_nbits_meta.txt")
    with open(meta_file, 'w') as f:
        f.write(f"M={M}\nN={N}\nK={K}\ngroup_size={group_size}\n")
        f.write(f"num_groups_k={num_groups_k}\nseed={args.seed}\n")
        f.write(f"use_zeros={'true' if not args.no_zeros else 'false'}\n")

    print(f"Data saved to {out_dir}/")


if __name__ == '__main__':
    main()
