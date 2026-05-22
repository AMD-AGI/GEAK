import tilelang
import tilelang.language as T

BLOCK_M = 128
BLOCK_N = 256
BLOCK_K = 64
NUM_STAGES = 4
THREADS = 256
ENABLE_RASTERIZATION = True
K_PACK = 2


@tilelang.jit(out_idx=[2], target="hip")
def matmul_kernel(
    M: int,
    N: int,
    K: int,
    block_M: int = BLOCK_M,
    block_N: int = BLOCK_N,
    block_K: int = BLOCK_K,
    num_stages: int = NUM_STAGES,
    threads: int = THREADS,
    enable_rasterization: bool = ENABLE_RASTERIZATION,
    k_pack: int = K_PACK,
):
    dtype = "float16"
    accum_dtype = "float"

    @T.prim_func
    def main(
        A: T.Buffer((M, K), dtype),
        B: T.Buffer((N, K), dtype),
        C: T.Buffer((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_N, block_K), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            C_shared = T.alloc_shared((block_M, block_N), dtype)

            T.use_swizzle(panel_size=10, enable=enable_rasterization)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, k * block_K], A_shared, coalesced_width=4 * k_pack)
                T.copy(B[bx * block_N, k * block_K], B_shared, coalesced_width=4 * k_pack)
                T.gemm(A_shared, B_shared, C_local, transpose_B=True, k_pack=k_pack)
            T.copy(C_local, C_shared)
            T.copy(C_shared, C[by * block_M, bx * block_N])

    return main
