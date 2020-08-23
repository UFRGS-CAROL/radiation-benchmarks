/*
 * parse_gemm_layer.h
 *
 *  Created on: Apr 11, 2020
 *      Author: fernando
 */

#ifndef PARSE_GEMM_LAYER_H_
#define PARSE_GEMM_LAYER_H_

#ifdef __cplusplus
extern "C" {
#endif

void parse_entry_cpu(int TA, int TB, int M, int N, int K, float ALPHA, float *A,
		int lda, float *B, int ldb, float BETA, float *C, int ldc);

void parse_entry_gpu(int TA, int TB, int M, int N, int K, float ALPHA, float *A,
		int lda, float *B, int ldb, float BETA, float *C, int ldc);

void inject_fault(int TA, int TB, int M, int N, int K, float *C);

#ifdef __cplusplus
}
#endif

#endif /* PARSE_GEMM_LAYER_H_ */
