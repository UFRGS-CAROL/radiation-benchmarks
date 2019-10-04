#include "cuda_fp16.h"
#include "stdint.h"
#include <iostream>

#define HALF_ROUND_STYLE 1  // 1: nearest, -1: truncate (fastest, default)
#include "half.hpp"
using half_float::half;
using namespace half_float::literal;

typedef float real_t;
typedef half_float::half half_t;
typedef __half half_t_device;

#define BLOCK_SIZE 16

#ifndef ZERO_FLOAT
    #define ZERO_FLOAT 2.2e-20
#endif

#ifndef ZERO_DOUBLE
    #define ZERO_DOUBLE 1.4e-40
#endif

#ifndef ZERO_HALF
    #define ZERO_HALF 4.166e-05
#endif


__device__ __forceinline__ void axpy__(const double a, const double b, double &c) {
    c = __fma_rn(a, b, c);
}
__device__ __forceinline__ void axpy__(const float a, const float b, float &c) {
    c = __fmaf_rn(a, b, c);
}
__device__ __forceinline__ void axpy__(const double a, const double b, float &c) {
    c = __fmaf_rn(__double2float_rn(a), __double2float_rn(b), c);
}
__device__ __forceinline__ void axpy__(const float a, const float b, __half &c) {
    c = __hfma(__float2half(a), __float2half(b), c);
}

__device__ unsigned long long errors = 0;

template<const uint32_t THRESHOLD_uint32_t>
__device__ void check_bit_error(const __half &lhs, const float &rhs) {
	const uint32_t lhs_data = __half2uint_rn(lhs);
	const uint32_t rhs_data = __float_as_uint(rhs);
	uint32_t sub_res;
	if (lhs_data > rhs_data) {
		sub_res = lhs_data - rhs_data;
	} else {
		sub_res = rhs_data - lhs_data;
	}

	if (sub_res > THRESHOLD_uint32_t) {
		atomicAdd(&errors, 1);
	}
}

template<const uint32_t THRESHOLD_uint32_t>
__device__ void check_bit_error(const float &lhs, const float &rhs) {
	float diff = fabs(lhs - rhs);
	if (diff > ZERO_FLOAT) {
		atomicAdd(&errors, 1);
	}
}

// template<const uint32_t THRESHOLD, const uint32_t COUNT, typename real_t, typename half_t>
// __global__ void matrix_mult_dmr_kernel(real_t *A, real_t *B, int M, int N, int K, real_t *C, half_t *C_h) {

//     int row = blockIdx.x * blockDim.x + threadIdx.x;
//     int col = blockIdx.y * blockDim.y + threadIdx.y;
    
//     if (row < M && col < N) {
//         register real_t acc_real_t = 0.0;
// 	    register half_t acc_half_t = 0.0;

//         #pragma unroll COUNT
//         for (int i = 0; i < K; i++) {
//             axpy__(A[row * M + i], B[col * N + i], acc_real_t);
//             axpy__(A[row * M + i], B[col * N + i], acc_half_t);

//             if ((i % COUNT) == 0) {
//                 check_bit_error<THRESHOLD>(acc_half_t, acc_real_t);
//                 acc_half_t = half_t(acc_real_t);
//             }
//         }

//         C[row * M + col] = acc_real_t;
//         C_h[row * M + col] = acc_half_t;
//     }

// }

template<const uint32_t THRESHOLD, const uint32_t COUNT, typename real_t, typename half_t>
void matrix_mult_dmr(real_t *A, real_t *B, int M, int N, int K, real_t *D, half_t *D_h, real_t alpha, real_t beta, real_t *C) {
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    matrix_mult_dmr_kernel<THRESHOLD, COUNT,half_t, real_t><<<dimGrid,dimBlock>>>(D,(__half*)D_h, C, A, B, alpha, beta, M, N);
}

template<const uint32_t THRESHOLD, const uint32_t COUNT, typename real_t, typename half_t>
__global__ void matrix_mult_dmr_kernel(real_t *D_r, half_t *D_h, real_t *C,
        real_t *A, real_t *B, real_t alpha, real_t beta, int wA, int wB) {
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;


    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Index of the first sub-matrix of A processed by the block
    int aBegin = wA * BLOCK_SIZE * by;

    // Index of the last sub-matrix of A processed by the block
    int aEnd = aBegin + wA - 1;

    // Step size used to iterate through the sub-matrices of A
    int aStep = BLOCK_SIZE;

    // Index of the first sub-matrix of B processed by the block
    int bBegin = BLOCK_SIZE * bx;

    // Step size used to iterate through the sub-matrices of B
    int bStep = BLOCK_SIZE * wB;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    real_t Csub = 0;
    half_t Csub_half = 0;
    //double threshold = -2222;
    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
        // Declaration of the shared memory array As used to
        // store the sub-matrix of A
        __shared__ real_t As[BLOCK_SIZE][BLOCK_SIZE];

        // Declaration of the shared memory array Bs used to
        // store the sub-matrix of B
        __shared__ real_t Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load the matrices from device memory
        // to shared memory; each thread loads
        // one element of each matrix
        As[ty][tx] = A[a + wA * ty + tx];
        Bs[ty][tx] = B[b + wB * ty + tx];

        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
#pragma unroll COUNT

        for (int k = 0; k < BLOCK_SIZE; ++k) {
//          Csub += As[ty][k] * Bs[k][tx];
//          Csub_half += half_t(As[ty][k]) * half_t(Bs[k][tx]);
            axpy__(As[ty][k], Bs[k][tx], Csub);
            axpy__(half_t(As[ty][k]), half_t(Bs[k][tx]), Csub_half);

#if CHECKBLOCK == 1
            check_bit_error<THRESHOLD>(Csub_half, Csub);

            Csub_half = half_t(Csub);
//#elif CHECKBLOCK >= 1
//          if((k % CHECKBLOCK) == 0) {
//              check_relative_error(Csub_half, Csub);
//
//              double diff = fabs(double(Csub) - double(Csub_half));
//              threshold = fmax(threshold, diff);
//
//              Csub_half = half_t(Csub);
//          }
#endif

        }

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }
    const int index = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx + wB * ty + tx;

    const real_t d_r = alpha * Csub + beta * C[index];
    const half_t d_h = half_t(alpha) * Csub_half
            + half_t(beta) * half_t(C[index]);

//#if CHECKBLOCK == 0
//  check_relative_error(d_h, d_r);
//  threshold = fabs(double(Csub) - double(Csub_half));
//#endif

   

// Write the block sub-matrix to device memory;
// each thread writes one element
    D_r[index] = d_r;
    D_h[index] = d_h;
}
