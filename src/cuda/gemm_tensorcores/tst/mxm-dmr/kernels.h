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

template<const uint32_t THRESHOLD, const uint32_t COUNT, typename real_t, typename half_t>
__global__ void matrix_mult_dmr_kernel(real_t *A, real_t *B, int M, int N, int K, real_t *C, half_t *C_h) {

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < M && col < N) {
        register real_t acc_real_t = 0.0;
	    register half_t acc_half_t = 0.0;

        #pragma unroll COUNT
        for (int i = 0; i < K; i++) {
            axpy__(A[row * M + i], B[col * N + i], acc_real_t);
            axpy__(A[row * M + i], B[col * N + i], acc_half_t);

            if ((i % COUNT) == 0) {
                check_bit_error<THRESHOLD>(acc_half_t, acc_real_t);
                acc_half_t = half_t(acc_real_t);
            }
        }

        C[row * M + col] = acc_real_t;
        C_h[row * M + col] = acc_half_t;
    }

}

template<const uint32_t THRESHOLD, const uint32_t COUNT, typename real_t, typename half_t>
void matrix_mult_dmr(real_t *A, real_t *B, int M, int N, int K, real_t *C, half_t *C_h) {
    unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    matrix_mult_dmr_kernel<THRESHOLD, COUNT><<<dimGrid,dimBlock>>>(A, B, M, N, K, C, (__half*)C_h);
}