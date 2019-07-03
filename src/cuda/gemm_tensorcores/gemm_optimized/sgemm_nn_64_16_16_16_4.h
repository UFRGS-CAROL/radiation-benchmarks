/*
 * sgemm_nn_64_16_16_16_4.h
 *
 *  Created on: 30/06/2019
 *      Author: fernando
 */

#ifndef SGEMM_NN_64_16_16_16_4_H_
#define SGEMM_NN_64_16_16_16_4_H_



void sgemm(cudaStream_t stream, float *C, const float *A, const float *B,
		int32_t m, int32_t n, int32_t k, int32_t lda, int32_t ldb, int32_t ldc,
		float alpha, float beta);

// template<typename real_t, typename half_real_t>
void sgemm_dmr(cudaStream_t stream, real_t *C, half_real_t *C_inc, const real_t *A, const real_t *B,
		int32_t m, int32_t n, int32_t k, int32_t lda, int32_t ldb, int32_t ldc,
		real_t alpha, real_t beta);


void sgemm(cudaStream_t stream, double *C, const double *A, const double *B,
		int32_t m, int32_t n, int32_t k, int32_t lda, int32_t ldb, int32_t ldc,
		double alpha, double beta);

#endif /* SGEMM_NN_64_16_16_16_4_H_ */
