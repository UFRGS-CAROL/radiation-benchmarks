/*
 * gemm_kernels.h
 *
 *  Created on: Oct 22, 2018
 *      Author: carol
 */

#ifndef GEMM_KERNELS_H_
#define GEMM_KERNELS_H_

#ifdef __cplusplus
extern "C" {
#endif

void sgemm(int b_operation,
		int a_operation, int N, int M, int K,
		float *alpha, float* b_gpu, int ldb, float* a_gpu, int lda,
		float* beta, float* c_gpu, int ldc) ;
void dgemm(int b_operation,
		int a_operation, int N, int M, int K,
		double *alpha, double* b_gpu, int ldb, double* a_gpu, int lda,
		double* beta, double* c_gpu, int ldc);


#ifdef __cplusplus
}
#endif

#endif /* GEMM_KERNELS_H_ */
