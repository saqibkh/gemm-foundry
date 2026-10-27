#pragma once
#include<cuda_runtime.h>
#include<cuda_fp16.h>
#include <mma.h>
#include<cstdint>
#include "../kernels/gemm_precision.cuh"

namespace gemm_foundry::detail{
namespace wmma = nvcuda::wmma;

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

template <typename TC>
__global__ void gemm_wmma_fp16_kernel(
    const __half* A, const __half* B , TC* C ,
    std::int64_t M, std::int64_t N , std::int64_t K ,
    float alpha , float beta , float scale_a , float scale_b ,
    std::int64_t lda , std::int64_t ldb , std::int64_t ldc,
    bool transpose_a , bool transpose_b ,
    std::int64_t stride_a , std::int64_t stride_b, std::int64_t stride_c){
        const std::int64_t row_base = static_cast<std::int64_t>(blockIdx.y) *WMMA_M;
        const std::int64_t col_base = static_cast<std::int64_t>(blockIdx.x) *WMMA_N;
        const std::int64_t batch = static_cast<std::int64_t>(blockIdx.z) ;

        const std::int64_t a_base = batch * stride_a;
        const std::int64_t b_base = batch * stride_b;
        const std::int64_t c_base = batch * stride_c;

        __shared__ __align__(32) __half tile_a[WMMA_M * WMMA_K];
        __shared__ __align__(32) __half tile_b[WMMA_K * WMMA_N];
        __shared__ __align__(32) float accumulator_tile[WMMA_M * WMMA_N];

        // declaring register based small tile container
        wmma::fragment< wmma::matrix_a , WMMA_M , WMMA_N, WMMA_K , __half , wmma::row_major> a_frag;
        wmma::fragment< wmma::matrix_b , WMMA_M , WMMA_N, WMMA_K , __half , wmma::row_major> b_frag;
        wmma::fragment< wmma::accumulator , WMMA_M , WMMA_N, WMMA_K , float > c_frag;

        wmma::fill_fragment(c_frag, 0.0f);

        // Iterates through K axis of the matrices in steps of 16 elements at a time.
        for(std::int64_t base_k = 0 ; base_k < K; base_k += WMMA_K){

            // strided grid loop where all threads cooperatively load global memory into the shared memory tile structures (tile_a and tile_b)
            for(int idx = threadIdx.x ; idx < WMMA_M * WMMA_K ; idx += blockDim.x){
                const int local_row = idx / WMMA_K;
                const int local_k = idx % WMMA_K;

                // Cooperatively load A's tile
                const std::int64_t global_row = row_base + local_row;
                const std::int64_t global_k = base_k + local_k;
                if(global_row < M && global_k < K){
                    const std::int64_t a_idx = transpose_a ? (a_base + global_k * lda + global_row) : (a_base + global_row * lda + global_k);
                    tile_a[idx] = A[a_idx];
                }else{
                    tile_a[idx] = __float2half_rn(0.0f);
                }

                // Cooperatively load B's tile
                const std::int64_t b_global_k = local_row + base_k;
                const std::int64_t b_global_col = col_base + local_k;
                if(b_global_k < K && b_global_col < N){
                    const std::int64_t b_idx = transpose_b ? (b_base + b_global_col * ldb + b_global_k) : (b_base + b_global_k * ldb + b_global_col);
                    tile_b[idx] = B[b_idx];
                }else{
                    tile_b[idx] = __float2half_rn(0.0f);
                }

            }
            __syncthreads();

            // shared memory --> register
            wmma::load_matrix_sync(a_frag , tile_a , WMMA_K , wmma::row_major);
            wmma::load_matrix_sync(b_frag , tile_b , WMMA_N , wmma::row_major);

            // Tensore core operations & FP32 accumulation
            wmma::mma_sync(c_frag , a_frag , b_frag , c_frag);

            __syncthreads();    
        }
        
        // it synchronize all the threads in a warp(32 threads)
        // writing it back to shared memory
        wmma::store_matrix_sync(accumulator_tile , c_frag , WMMA_N , wmma::row_major);

        // synchronizes all warps inside the entire thread block
        __syncthreads();

        // Apply alpha, beta and output conversion using  CUDA threads
        for(int idx = threadIdx.x ; idx < WMMA_M * WMMA_N ; idx += blockDim.x){
            const int local_row = idx / WMMA_N;
            const int local_col = idx % WMMA_N;

            const std::int64_t row = row_base + local_row;
            const std::int64_t col = col_base + local_col;

            if(row < M && col < N){
                const std::int64_t c_idx = c_base + row * ldc + col;
                const float scaled_sum = scale_a * scale_b * accumulator_tile[idx];
                float result = scaled_sum * alpha;

                if(beta != 0.0f){
                    result += beta * to_accum(C[c_idx]);
                }
                C[c_idx] = from_accum<TC>(result);
            }
        }
    }
}// namespace gemm_foundry::detail
