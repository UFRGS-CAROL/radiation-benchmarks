/*
 * type.cu
 *
 *  Created on: 01/10/2018
 *      Author: fernando
 */

#include "type.h"
#include "cuda_fp16.h"

#include "darknet.h"

#include <assert.h>

extern "C" {
#include "cuda.h"
}

//#if __CUDA_ARCH__ > 600
//
//extern void hgemm(int b_operation, int a_operation, int N, int M, int K,
//		half *alpha, half* b_gpu, int ldb, half* a_gpu, int lda, half* beta,
//		half* c_gpu, int ldc);
//#endif

typedef half real_t_fp16;

//extern "C" void check_error(cudaError_t status);
//extern "C" dim3 cuda_gridsize(size_t n);
//extern void check_error(cudaError_t status);

__global__ void cuda_f32_to_f16(real_t *X, size_t N, real_t_fp16 *Y) {
	size_t i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
	if (i < N) {
		Y[i] = __float2half(X[i]);
	}
}

__global__ void cuda_f16_to_f32(real_t_fp16 *X, size_t N, real_t *Y) {
	size_t i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
	if (i < N) {
		Y[i] = __half2float(X[i]);
	}
}

//	run_cuda_gemm_half(TA, TB, M, N, K, ALPHA, A_gpu, lda, B_gpu, ldb, BETA, C_gpu, ldc);
void run_cuda_gemm_half(cublasHandle_t handle, int TA, int TB, int M, int N,
		int K, real_t ALPHA, real_t *A_gpu, int lda, real_t *B_gpu, int ldb,
		real_t BETA, real_t *C_gpu, int ldc, cudaStream_t st) {
	real_t_fp16 *a = nullptr;
	real_t_fp16 *b = nullptr;
	real_t_fp16 *c = nullptr;

	int siz_a = M * K;
	int siz_b = K * N;
	int siz_c = M * N;

//	convert_and_push_3_arrays(A_gpu, B_gpu, C_gpu,
//			a, M * K, b, K * N, c, M * N);
	check_error(cudaMalloc((void**) &a, sizeof(real_t_fp16) * siz_a));

	cuda_f32_to_f16<<<cuda_gridsize(siz_a), BLOCK, 0, st>>>(A_gpu, siz_a, a);
	check_error(cudaPeekAtLastError());

	check_error(cudaMalloc((void**) &b, sizeof(real_t_fp16) * siz_b));
	cuda_f32_to_f16<<<cuda_gridsize(siz_b), BLOCK, 0, st>>>(B_gpu, siz_b, b);
	check_error(cudaPeekAtLastError());

	if (BETA != 0) {
		check_error(cudaMalloc((void**) &c, sizeof(real_t_fp16) * siz_c));
		cuda_f32_to_f16<<<cuda_gridsize(siz_c), BLOCK, 0, st>>>(C_gpu, siz_c,
				c);
		check_error(cudaPeekAtLastError());
	}

//#if __CUDA_ARCH__ > 600
	real_t_fp16 alpha = real_t_fp16(ALPHA);
	real_t_fp16 beta = real_t_fp16(BETA);
//#ifndef OPENGEMM

	cudaError_t status = (cudaError_t) cublasHgemm(handle,
			(TB ? CUBLAS_OP_T : CUBLAS_OP_N), (TA ? CUBLAS_OP_T : CUBLAS_OP_N),
			N, M, K, &alpha, b, ldb, a, lda, &beta, c, ldc);
	check_error(status);
//#else
//	hgemm((TB ? CUBLAS_OP_T : CUBLAS_OP_N), (TA ? CUBLAS_OP_T : CUBLAS_OP_N), N, M, K, &alpha, b, ldb,
//			a, lda, &beta, c, ldc);
//#endif
//#endif

//	printf("Executed the hgemm\n");
	cuda_f16_to_f32<<<cuda_gridsize(siz_c), BLOCK, 0, st>>>(c, siz_c, C_gpu);
	check_error(cudaPeekAtLastError());

	//free the three half arrays
	check_error(cudaFree(a));

	check_error(cudaFree(b));

	check_error(cudaFree(c));
	check_error(cudaPeekAtLastError());
}
