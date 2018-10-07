#ifndef GEMM_H
#define GEMM_H

#include "type.h"

void gemm_bin(int M, int N, int K, real_t ALPHA, char *A, int lda, real_t *B,
		int ldb, real_t *C, int ldc);

void gemm(int TA, int TB, int M, int N, int K, real_t ALPHA, real_t *A, int lda,
		real_t *B, int ldb, real_t BETA, real_t *C, int ldc);

void gemm_cpu(int TA, int TB, int M, int N, int K, real_t ALPHA, real_t *A,
		int lda, real_t *B, int ldb, real_t BETA, real_t *C, int ldc);

#ifdef GPU
void gemm_gpu(int TA, int TB, int M, int N, int K, real_t ALPHA,
		real_t *A_gpu, int lda,
		real_t *B_gpu, int ldb,
		real_t BETA,
		real_t *C_gpu, int ldc, unsigned char use_tensor_cores,
		cudaStream_t st);

//void gemm_gpu(int TA, int TB, int M, int N, int K, real_t ALPHA,
//        real_t *A, int lda,
//        real_t *B, int ldb,
//        real_t BETA,
//        real_t *C, int ldc);
#endif
#endif
