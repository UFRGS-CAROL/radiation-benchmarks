#include "gemm.h"
#include "utils.h"
#include "cuda.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>

#ifdef OPENGEMM
#include "gemm_kernels.h"
#endif

void gemm_bin(int M, int N, int K, real_t ALPHA, char *A, int lda, real_t *B,
		int ldb, real_t *C, int ldc) {
	int i, j, k;
	for (i = 0; i < M; ++i) {
		for (k = 0; k < K; ++k) {
			char A_PART = A[i * lda + k];
			if (A_PART) {
				for (j = 0; j < N; ++j) {
					C[i * ldc + j] += B[k * ldb + j];
				}
			} else {
				for (j = 0; j < N; ++j) {
					C[i * ldc + j] -= B[k * ldb + j];
				}
			}
		}
	}
}

real_t *random_matrix(int rows, int cols) {
	int i;
	real_t *m = calloc(rows * cols, sizeof(real_t));
	for (i = 0; i < rows * cols; ++i) {
		m[i] = (real_t) rand() / RAND_MAX;
	}
	return m;
}

void time_random_matrix(int TA, int TB, int m, int k, int n) {
	real_t *a;
	if (!TA)
		a = random_matrix(m, k);
	else
		a = random_matrix(k, m);
	int lda = (!TA) ? k : m;
	real_t *b;
	if (!TB)
		b = random_matrix(k, n);
	else
		b = random_matrix(n, k);
	int ldb = (!TB) ? n : k;

	real_t *c = random_matrix(m, n);
	int i;
	clock_t start = clock(), end;
	for (i = 0; i < 10; ++i) {
		gemm_cpu(TA, TB, m, n, k, 1, a, lda, b, ldb, 1, c, n);
	}
	end = clock();
	printf("Matrix Multiplication %dx%d * %dx%d, TA=%d, TB=%d: %lf ms\n", m, k,
			k, n, TA, TB, (real_t)(end - start) / CLOCKS_PER_SEC);
	free(a);
	free(b);
	free(c);
}

void gemm(int TA, int TB, int M, int N, int K, real_t ALPHA, real_t *A, int lda,
		real_t *B, int ldb, real_t BETA, real_t *C, int ldc) {
	gemm_cpu(TA, TB, M, N, K, ALPHA, A, lda, B, ldb, BETA, C, ldc);
}

void gemm_nn(int M, int N, int K, real_t ALPHA, real_t *A, int lda, real_t *B,
		int ldb, real_t *C, int ldc) {
	int i, j, k;
#pragma omp parallel for
	for (i = 0; i < M; ++i) {
		for (k = 0; k < K; ++k) {
			register real_t A_PART = ALPHA * A[i * lda + k];
			for (j = 0; j < N; ++j) {
				C[i * ldc + j] += A_PART * B[k * ldb + j];
			}
		}
	}
}

void gemm_nt(int M, int N, int K, real_t ALPHA, real_t *A, int lda, real_t *B,
		int ldb, real_t *C, int ldc) {
	int i, j, k;
#pragma omp parallel for
	for (i = 0; i < M; ++i) {
		for (j = 0; j < N; ++j) {
			register real_t sum = 0;
			for (k = 0; k < K; ++k) {
				sum += ALPHA * A[i * lda + k] * B[j * ldb + k];
			}
			C[i * ldc + j] += sum;
		}
	}
}

void gemm_tn(int M, int N, int K, real_t ALPHA, real_t *A, int lda, real_t *B,
		int ldb, real_t *C, int ldc) {
	int i, j, k;
#pragma omp parallel for
	for (i = 0; i < M; ++i) {
		for (k = 0; k < K; ++k) {
			register real_t A_PART = ALPHA * A[k * lda + i];
			for (j = 0; j < N; ++j) {
				C[i * ldc + j] += A_PART * B[k * ldb + j];
			}
		}
	}
}

void gemm_tt(int M, int N, int K, real_t ALPHA, real_t *A, int lda, real_t *B,
		int ldb, real_t *C, int ldc) {
	int i, j, k;
#pragma omp parallel for
	for (i = 0; i < M; ++i) {
		for (j = 0; j < N; ++j) {
			register real_t sum = 0;
			for (k = 0; k < K; ++k) {
				sum += ALPHA * A[i + k * lda] * B[k + j * ldb];
			}
			C[i * ldc + j] += sum;
		}
	}
}

void gemm_cpu(int TA, int TB, int M, int N, int K, real_t ALPHA, real_t *A,
		int lda, real_t *B, int ldb, real_t BETA, real_t *C, int ldc) {
	//printf("cpu: %d %d %d %d %d %f %d %d %f %d\n",TA, TB, M, N, K, ALPHA, lda, ldb, BETA, ldc);
	int i, j;
	for (i = 0; i < M; ++i) {
		for (j = 0; j < N; ++j) {
			C[i * ldc + j] *= BETA;
		}
	}
	if (!TA && !TB)
		gemm_nn(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
	else if (TA && !TB)
		gemm_tn(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
	else if (!TA && TB)
		gemm_nt(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
	else
		gemm_tt(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
}

#ifdef GPU

void gemm_gpu(int TA, int TB, int M, int N, int K, real_t ALPHA, real_t *A_gpu,
		int lda, real_t *B_gpu, int ldb, real_t BETA, real_t *C_gpu, int ldc,
		unsigned char use_tensor_cores, cudaStream_t st) {
	cublasHandle_t handle = blas_handle(use_tensor_cores);
	cublasSetStream(handle, st);
//	if(TB || TA)
//	{
//		printf("Matrix need to be transposed %d %d\n", TB, TA);
//	}
#ifndef OPENGEMM

#if REAL_TYPE == HALF
	//run_cuda_gemm_half(int TA, int TB, int M, int N, int K, real_t ALPHA, real_t *A_gpu,
//	int lda, real_t *B_gpu, int ldb, real_t BETA, real_t *C_gpu, int ldc)
	run_cuda_gemm_half(handle, TA, TB, M, N, K, ALPHA, A_gpu, lda, B_gpu, ldb, BETA, C_gpu, ldc, st);
#elif REAL_TYPE == FLOAT
	cudaError_t status = (cudaError_t) cublasSgemm(handle, (TB ? CUBLAS_OP_T : CUBLAS_OP_N),
			(TA ? CUBLAS_OP_T : CUBLAS_OP_N), N, M, K, &ALPHA, B_gpu, ldb,
			A_gpu, lda, &BETA, C_gpu, ldc);
	check_error(status);
#elif REAL_TYPE == DOUBLE
	cudaError_t status = (cudaError_t) cublasDgemm(handle, (TB ? CUBLAS_OP_T : CUBLAS_OP_N),
			(TA ? CUBLAS_OP_T : CUBLAS_OP_N), N, M, K, &ALPHA, B_gpu, ldb,
			A_gpu, lda, &BETA, C_gpu, ldc);
	check_error(status);
#endif

#else

#if REAL_TYPE == HALF
	run_cuda_gemm_half(handle, TA, TB, M, N, K, ALPHA, A_gpu, lda, B_gpu, ldb, BETA, C_gpu, ldc, st);
#elif REAL_TYPE == FLOAT
	sgemm((TB ? CUBLAS_OP_T : CUBLAS_OP_N),
			(TA ? CUBLAS_OP_T : CUBLAS_OP_N), N, M, K, &ALPHA, B_gpu, ldb,
			A_gpu, lda, &BETA, C_gpu, ldc);

#elif REAL_TYPE == DOUBLE
	dgemm((TB ? CUBLAS_OP_T : CUBLAS_OP_N),
			(TA ? CUBLAS_OP_T : CUBLAS_OP_N), N, M, K, &ALPHA, B_gpu, ldb,
			A_gpu, lda, &BETA, C_gpu, ldc);

#endif

#endif

//	cublasHandle_t handle = blas_handle();
//	cudaError_t status = cublasSgemm(handle, (TB ? CUBLAS_OP_T : CUBLAS_OP_N),
//			(TA ? CUBLAS_OP_T : CUBLAS_OP_N), N, M, K, &ALPHA, B_gpu, ldb,
//			A_gpu, lda, &BETA, C_gpu, ldc);

}

void time_gpu_random_matrix(int TA, int TB, int m, int k, int n) {
	real_t *a;
	if (!TA)
		a = random_matrix(m, k);
	else
		a = random_matrix(k, m);
	int lda = (!TA) ? k : m;
	real_t *b;
	if (!TB)
		b = random_matrix(k, n);
	else
		b = random_matrix(n, k);
	int ldb = (!TB) ? n : k;

	real_t *c = random_matrix(m, n);
	int i;
	clock_t start = clock(), end;
	for (i = 0; i < 32; ++i) {
		gemm_gpu(TA, TB, m, n, k, 1, a, lda, b, ldb, 1, c, n, 0, 0x0);
	}
	end = clock();
	printf("Matrix Multiplication %dx%d * %dx%d, TA=%d, TB=%d: %lf s\n", m, k,
			k, n, TA, TB, (real_t)(end - start) / CLOCKS_PER_SEC);
	free(a);
	free(b);
	free(c);
}

void time_gpu(int TA, int TB, int m, int k, int n) {
	int iter = 10;
	real_t *a = random_matrix(m, k);
	real_t *b = random_matrix(k, n);

	int lda = (!TA) ? k : m;
	int ldb = (!TB) ? n : k;

	real_t *c = random_matrix(m, n);

	real_t *a_cl = cuda_make_array(a, m * k);
	real_t *b_cl = cuda_make_array(b, k * n);
	real_t *c_cl = cuda_make_array(c, m * n);

	int i;
	clock_t start = clock(), end;
	for (i = 0; i < iter; ++i) {
		gemm_gpu(TA, TB, m, n, k, 1, a_cl, lda, b_cl, ldb, 1, c_cl, n, 0, 0x0);
		cudaDeviceSynchronize();
	}
	double flop = ((double) m) * n * (2. * k + 2.) * iter;
	double gflop = flop / pow(10., 9);
	end = clock();
	double seconds = sec(end - start);
	printf(
			"Matrix Multiplication %dx%d * %dx%d, TA=%d, TB=%d: %lf s, %lf GFLOPS\n",
			m, k, k, n, TA, TB, seconds, gflop / seconds);
	cuda_free(a_cl);
	cuda_free(b_cl);
	cuda_free(c_cl);
	free(a);
	free(b);
	free(c);
}

void test_gpu_accuracy(int TA, int TB, int m, int k, int n) {
	srand(0);
	real_t *a;
	if (!TA)
		a = random_matrix(m, k);
	else
		a = random_matrix(k, m);
	int lda = (!TA) ? k : m;
	real_t *b;
	if (!TB)
		b = random_matrix(k, n);
	else
		b = random_matrix(n, k);
	int ldb = (!TB) ? n : k;

	real_t *c = random_matrix(m, n);
	real_t *c_gpu = random_matrix(m, n);
	memset(c, 0, m * n * sizeof(real_t));
	memset(c_gpu, 0, m * n * sizeof(real_t));
	int i;
	//pm(m,k,b);
	gemm_gpu(TA, TB, m, n, k, 1, a, lda, b, ldb, 1, c_gpu, n, 0, 0x0);
	//printf("GPU\n");
	//pm(m, n, c_gpu);

	gemm_cpu(TA, TB, m, n, k, 1, a, lda, b, ldb, 1, c, n);
	//printf("\n\nCPU\n");
	//pm(m, n, c);
	double sse = 0;
	for (i = 0; i < m * n; ++i) {
		//printf("%f %f\n", c[i], c_gpu[i]);
		sse += pow(c[i] - c_gpu[i], 2);
	}
	printf("Matrix Multiplication %dx%d * %dx%d, TA=%d, TB=%d: %g SSE\n", m, k,
			k, n, TA, TB, sse / (m * n));
	free(a);
	free(b);
	free(c);
	free(c_gpu);
}

int test_gpu_blas() {
	/*
	 test_gpu_accuracy(0,0,10,576,75);

	 test_gpu_accuracy(0,0,17,10,10);
	 test_gpu_accuracy(1,0,17,10,10);
	 test_gpu_accuracy(0,1,17,10,10);
	 test_gpu_accuracy(1,1,17,10,10);

	 test_gpu_accuracy(0,0,1000,10,100);
	 test_gpu_accuracy(1,0,1000,10,100);
	 test_gpu_accuracy(0,1,1000,10,100);
	 test_gpu_accuracy(1,1,1000,10,100);

	 test_gpu_accuracy(0,0,10,10,10);

	 time_gpu(0,0,64,2916,363);(cudaError_t) cublasDgemm
	 time_gpu(0,0,64,2916,363);
	 time_gpu(0,0,64,2916,363);
	 time_gpu(0,0,192,729,1600);
	 time_gpu(0,0,384,196,1728);
	 time_gpu(0,0,256,196,3456);
	 time_gpu(0,0,256,196,2304);
	 time_gpu(0,0,128,4096,12544);
	 time_gpu(0,0,128,4096,4096);
	 */
	time_gpu(0, 0, 64, 75, 12544);
	time_gpu(0, 0, 64, 75, 12544);
	time_gpu(0, 0, 64, 75, 12544);
	time_gpu(0, 0, 64, 576, 12544);
	time_gpu(0, 0, 256, 2304, 784);
	time_gpu(1, 1, 2304, 256, 784);
	time_gpu(0, 0, 512, 4608, 196);
	time_gpu(1, 1, 4608, 512, 196);

	return 0;
}
#endif

