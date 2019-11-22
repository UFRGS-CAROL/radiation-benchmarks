/*
 * tensor_kernels.h
 *
 *  Created on: Nov 22, 2019
 *      Author: fernando
 */

#ifndef TENSOR_KERNELS_H_
#define TENSOR_KERNELS_H_

#include <mma.h>

#include "common.h"

template<typename half_t, typename real_t>
__global__ void matrix_mult_kernel_wmma_unhardened(half_t *a, half_t *b,
		real_t *c, real_t *d, real_t alpha, real_t beta, int m, int n, int k) {
	// Leading dimensions. Packed with no transpositions.
	int lda = m;
	int ldb = n;
	int ldc = m;

	// Tile using a 2D grid
	int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
	int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

	// Declare the fragments
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
			half_t, nvcuda::wmma::col_major> a_frag;
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
			half_t, nvcuda::wmma::col_major> b_frag;
	nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K,
			real_t> acc_frag;
	nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K,
			real_t> c_frag;

	nvcuda::wmma::fill_fragment(acc_frag, 0.0f);

	// Loop over k
	for (int i = 0; i < k; i += WMMA_K) {
		int aRow = warpM * WMMA_M;
		int aCol = i;

		int bRow = i;
		int bCol = warpN * WMMA_N;

		// Bounds checking
		if (aRow < M && aCol < K && bRow < K && bCol < N) {
			// Load the inputs
			nvcuda::wmma::load_matrix_sync(a_frag, a + aRow + aCol * lda, lda);
			nvcuda::wmma::load_matrix_sync(b_frag, b + bRow + bCol * ldb, ldb);

			// Perform the matrix multiplication
			nvcuda::wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);

		}
	}

	// Load in the current value of c, scale it by beta, and add this our result scaled by alpha
	int cRow = warpM * WMMA_M;
	int cCol = warpN * WMMA_N;

	if (cRow < M && cCol < N) {
		nvcuda::wmma::load_matrix_sync(c_frag, c + cRow + cCol * ldc, ldc,
				nvcuda::wmma::mem_col_major);

		for (int i = 0; i < c_frag.num_elements; i++) {
			c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
		}

		// Store the output
		nvcuda::wmma::store_matrix_sync(d + cRow + cCol * ldc, c_frag, ldc,
				nvcuda::wmma::mem_col_major);
	}
}

template<typename half_t, typename real_t>
__global__ void matrix_mult_kernel_wmma_dmr(half_t *a, half_t *b, real_t *c,
		real_t *d, real_t *d_dmr, real_t alpha, real_t beta, int m, int n,
		int k) {
	// Leading dimensions. Packed with no transpositions.
	int lda = m;
	int ldb = n;
	int ldc = m;

	// Tile using a 2D grid
	int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
	int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

	// Declare the fragments
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
			half_t, nvcuda::wmma::col_major> a_frag;
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
			half_t, nvcuda::wmma::col_major> b_frag;
	nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K,
			real_t> acc_frag;
	nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K,
			real_t> c_frag;

	nvcuda::wmma::fill_fragment(acc_frag, 0.0f);

	half_t acc_sw = 0;

	// Loop over k
	for (int i = 0; i < k; i += WMMA_K) {
		int aRow = warpM * WMMA_M;
		int aCol = i;

		int bRow = i;
		int bCol = warpN * WMMA_N;

		// Bounds checking
		if (aRow < M && aCol < K && bRow < K && bCol < N) {
			// Load the inputs
			nvcuda::wmma::load_matrix_sync(a_frag, a + aRow + aCol * lda, lda);
			nvcuda::wmma::load_matrix_sync(b_frag, b + bRow + bCol * ldb, ldb);

			// Perform the matrix multiplication
			nvcuda::wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
#pragma unroll
			for (int internal = 0; internal < WMMA_K; internal++) {
				half_t a_it = a[aRow + aCol * lda + internal];
				half_t b_it = a[aRow + aCol * lda + internal];
				acc_sw += a_it * b_it;
			}

		}
	}

	// Load in the current value of c, scale it by beta, and add this our result scaled by alpha
	int cRow = warpM * WMMA_M;
	int cCol = warpN * WMMA_N;

	if (cRow < M && cCol < N) {
		nvcuda::wmma::load_matrix_sync(c_frag, c + cRow + cCol * ldc, ldc,
				nvcuda::wmma::mem_col_major);

		for (int i = 0; i < c_frag.num_elements; i++) {
			c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
		}

		// Store the output
		nvcuda::wmma::store_matrix_sync(d + cRow + cCol * ldc, c_frag, ldc,
				nvcuda::wmma::mem_col_major);
	}
}

#endif /* TENSOR_KERNELS_H_ */
