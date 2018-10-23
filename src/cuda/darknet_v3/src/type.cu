/*
 * type.cu
 *
 *  Created on: 01/10/2018
 *      Author: fernando
 */

#include "type.h"
#include "cuda_fp16.h"
#include "cuda.h"
#include <assert.h>

extern void hgemm(int b_operation, int a_operation, int N, int M, int K,
		half *alpha, half* b_gpu, int ldb, half* a_gpu, int lda, half* beta,
		half* c_gpu, int ldc);

/**
 * Read a file for all precisions
 */
int fread_float_to_real_t(real_t* dst, size_t siz, size_t times, FILE* fp) {
	float* temp = (float*) calloc(times, sizeof(float));
	if (temp == NULL) {
		return -1;
	}
	size_t fread_result = fread(temp, sizeof(float), times, fp);
	if (fread_result != times) {
		free(temp);
		return -1;
	}

	for (size_t i = 0; i < times; i++) {
		//TODO: make ready for half
		dst[i] = real_t(temp[i]);
	}
	free(temp);
	return fread_result;

}

typedef half real_t_fp16;

void check_error(cudaError_t status) {
	//cudaDeviceSynchronize();
	cudaError_t status2 = cudaGetLastError();
	if (status != cudaSuccess) {
		const char *s = cudaGetErrorString(status);
		char buffer[256];
		printf("CUDA Error: %s\n", s);
		assert(0);
		snprintf(buffer, 256, "CUDA Error: %s", s);
		error(buffer);
	}
	if (status2 != cudaSuccess) {
		const char *s = cudaGetErrorString(status);
		char buffer[256];
		printf("CUDA Error Prev: %s\n", s);
		assert(0);
		snprintf(buffer, 256, "CUDA Error Prev: %s", s);
		error(buffer);
	}
}

dim3 cuda_gridsize(size_t n) {
	unsigned k = (n - 1) / BLOCK + 1;
	unsigned x = k;
	unsigned y = 1;
	unsigned z = 1;

	if (x > 65535) {
		x = ceil(sqrt(k));
		y = (n - 1) / (x * BLOCK) + 1;
	}

	dim3 d = { x, y, z };
	//printf("%ld %ld %ld %ld\n", n, x, y, x*y*BLOCK);
	return d;
}

__global__ void cuda_f32_to_f16(real_t *X, size_t N, real_t_fp16 *Y) {
	size_t i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
	if (i < N) {
		Y[i] = __float2half(X[i]);
	}
}

//__global__ void cuda_f32_to_f16(real_t* input_f32, size_t size,
//		real_t_fp16 *output_f16) {
//	int idx = blockIdx.x * blockDim.x + threadIdx.x;
//	if (idx < size)
//	output_f16[idx] = __float2half(input_f32[idx]);
//}

__global__ void cuda_f16_to_f32(real_t_fp16 *X, size_t N, real_t *Y) {
	size_t i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
	if (i < N) {
		Y[i] = __half2float(X[i]);
	}
}
//__global__ void cuda_f16_to_f32(real_t_fp16* input_f16, size_t size,
//		float *output_f32) {
//	int idx = blockIdx.x * blockDim.x + threadIdx.x;
//	if (idx < size)
//	output_f32[idx] = __half2float(input_f16[idx]);
//}

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

	real_t_fp16 alpha = real_t_fp16(ALPHA);
	real_t_fp16 beta = real_t_fp16(BETA);

#ifndef OPENGEMM
	cudaError_t status = (cudaError_t) cublasHgemm(handle,
			(TB ? CUBLAS_OP_T : CUBLAS_OP_N), (TA ? CUBLAS_OP_T : CUBLAS_OP_N),
			N, M, K, &alpha, b, ldb, a, lda, &beta, c, ldc);
	check_error(status);
#else
	hgemm((TB ? CUBLAS_OP_T : CUBLAS_OP_N), (TA ? CUBLAS_OP_T : CUBLAS_OP_N), N, M, K, &alpha, b, ldb,
			a, lda, &beta, c, ldc);
#endif
//	printf("Executed the hgemm\n");
	cuda_f16_to_f32<<<cuda_gridsize(siz_c), BLOCK, 0, st>>>(c, siz_c, C_gpu);
	check_error(cudaPeekAtLastError());

	//free the three half arrays
	check_error(cudaFree(a));

	check_error(cudaFree(b));

	check_error(cudaFree(c));
	check_error(cudaPeekAtLastError());
}

//
//#ifdef __NVCC__
//
//__device__                __forceinline__ real_t_device exp_real(real_t_device x) {
//#if REAL_TYPE == HALF
//	return expf(x);
//#elif REAL_TYPE == FLOAT
//	return expf(x);
//#elif REAL_TYPE == DOUBLE
//	return exp(x);
//#endif
//}
//
//__device__                __forceinline__ real_t_device floor_real(real_t_device x) {
//#if REAL_TYPE == HALF
//	return floorf(half(x));
//#elif REAL_TYPE == FLOAT
//	return floorf(x);
//#elif REAL_TYPE == DOUBLE
//	return floor(x);
//#endif
//}
//
//__device__                __forceinline__ real_t_device pow_real(real_t_device x,
//		real_t_device y) {
//#if REAL_TYPE == HALF
//	return powf(x, y);
//#elif REAL_TYPE == FLOAT
//	return powf(x, y);
//#elif REAL_TYPE == DOUBLE
//	return pow(x, y);
//#endif
//}
//
//__device__                __forceinline__ real_t_device sqrt_real(real_t_device x) {
//#if REAL_TYPE == HALF
//	return sqrtf(x);
//#elif REAL_TYPE == FLOAT
//	return sqrtf(x);
//#elif REAL_TYPE == DOUBLE
//	return sqrt(x);
//#endif
//}
//
//__device__                __forceinline__ real_t_device fabs_real(real_t_device x) {
//#if REAL_TYPE == HALF
//	return fabsf(x);
//#elif REAL_TYPE == FLOAT
//	return fabsf(x);
//#elif REAL_TYPE == DOUBLE
//	return fabs(x);
//#endif
//}
//
//__device__                __forceinline__ real_t_device log_real(real_t_device x) {
//#if REAL_TYPE == HALF
//	return hlog(x);
//#elif REAL_TYPE == FLOAT
//	return logf(x);
//#elif REAL_TYPE == DOUBLE
//	return log(x);
//#endif
//}
//
//__device__         __forceinline__ real_t_device atomic_add_real(real_t_device *x,
//		real_t_device val) {
//#if REAL_TYPE == HALF
//#if __CUDA_ARCH__ > 700
//	return atomicAdd((half*)x, (half)val);
//#endif
//
//	half old = *x;
//	*x += val;
//	return old;
//#else
//	return atomicAdd(x, val);
//#endif
//}
//
//__device__        __forceinline__ real_t_device cos_real(real_t_device x) {
//#if REAL_TYPE == HALF
//	return hcos(x);
//#elif REAL_TYPE == FLOAT
//	return cosf(x);
//#elif REAL_TYPE == DOUBLE
//	return cos(x);
//#endif
//}
//
//__device__        __forceinline__ real_t_device sin_real(real_t_device x) {
//#if REAL_TYPE == HALF
//	return hsin(x);
//#elif REAL_TYPE == FLOAT
//	return sinf(x);
//#elif REAL_TYPE == DOUBLE
//	return sin(x);
//#endif
//}

//#endif
