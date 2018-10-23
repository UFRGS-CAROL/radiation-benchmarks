/**
 * GEMM kernels open source
 */
#include "gemm_kernels.h"
#include "stdio.h"

#include "cuda_fp16.h"

#define BLOCK_SIZE 32

extern void check_error(cudaError_t status);
//{
//	//cudaDeviceSynchronize();
//	cudaError_t status2 = cudaGetLastError();
//	if (status != cudaSuccess) {
//		const char *s = cudaGetErrorString(status);
//		char buffer[256];
//		printf("CUDA Error: %s\n", s);
//		snprintf(buffer, 256, "CUDA Error: %s", s);
//		printf("%s", buffer);
//		exit(1);
//	}
//	if (status2 != cudaSuccess) {
//		const char *s = cudaGetErrorString(status);
//		char buffer[256];
//		printf("CUDA Error Prev: %s\n", s);
//		snprintf(buffer, 256, "CUDA Error Prev: %s", s);
//		printf("%s", buffer);
//		exit(1);
//	}
//}

//__global__ void MatrixMulKernel(half *d_A0, half *d_B0, half *d_C0, int n,
//		int m, int k) {
//	int tx = blockIdx.x * blockDim.x + threadIdx.x;
//	int ty = blockIdx.y * blockDim.y + threadIdx.y;
//
//	if (m < ty || n < tx)
//		return;
//
//	half acc = 0.0;
//	for (int i = 0; i < k; i++) {
//		acc = __hfma(d_A0[ty * m + i], d_B0[i * n + tx], acc);
//	}
//
//	d_C0[ty * m + tx] = acc;
//}

template<class tested_type>
__global__ void MatrixMulKernel(tested_type *d_A0, tested_type *d_B0,
		tested_type *d_C0, int N, int M, int K, int lda, int ldb, int ldc) {

	//N
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	//M
	int ty = blockIdx.y * blockDim.y + threadIdx.y;

	if (M < ty || N < tx){
		return;
	}
	tested_type acc = 0.0;
	for (int k = 0; k < K; k++) {
		acc = d_A0[ty * lda + k] * d_B0[k * ldb + tx] + acc;
	}

	d_C0[ty * ldc + tx] = acc;
}

void hgemm(int b_operation, int a_operation, int N, int M, int K,
		half *alpha, half* b_gpu, int ldb, half* a_gpu, int lda, half* beta,
		half* c_gpu, int ldc) {
	int gridsize_n = ceil(N / BLOCK_SIZE);
	int gridsize_m = ceil(M / BLOCK_SIZE);

	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid(gridsize_n, gridsize_m);

//	printf("M %d N %d K %d lda %d, ldb %d, ldc %d, x %d y %d\n", M, N, K, lda, ldb, ldc, grid.x, grid.y);
	MatrixMulKernel<half> <<<grid, threads>>>(a_gpu, b_gpu, c_gpu, N, M, K, lda, ldb, ldc);
	check_error(cudaError_t(cudaPeekAtLastError()));
}

void sgemm(int b_operation, int a_operation, int N, int M, int K,
		float *alpha, float* b_gpu, int ldb, float* a_gpu, int lda, float* beta,
		float* c_gpu, int ldc) {
	int gridsize_n = ceil(N / BLOCK_SIZE);
	int gridsize_m = ceil(M / BLOCK_SIZE);

	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid(gridsize_n, gridsize_m);

	MatrixMulKernel<float> <<<grid, threads>>>(a_gpu, b_gpu, c_gpu, N, M, K, lda, ldb, ldc);
	check_error(cudaError_t(cudaPeekAtLastError()));
}

void dgemm(int b_operation, int a_operation, int N, int M, int K,
		double *alpha, double* b_gpu, int ldb, double* a_gpu, int lda,
		double* beta, double* c_gpu, int ldc) {
	int gridsize_n = ceil(N / BLOCK_SIZE);
	int gridsize_m = ceil(M / BLOCK_SIZE);

	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid(gridsize_n, gridsize_m);

	MatrixMulKernel<double> <<<grid, threads>>>(a_gpu, b_gpu, c_gpu, N, M, K, lda, ldb, ldc);
	check_error(cudaError_t(cudaPeekAtLastError()));
}
