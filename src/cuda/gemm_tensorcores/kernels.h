/*
 * kernels.h
 *
 *  Created on: 05/10/2018
 *      Author: fernando
 */

#ifndef KERNELS_H_
#define KERNELS_H_

#include <mma.h>
//#include "const_matrices.h"
#include <cuda_fp16.h>

using namespace nvcuda;

// The only dimensions currently supported by WMMA
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

#define WARP_SIZE 32

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif

__global__ void simple_wmma_gemm(float *d0, float *d1, float *d2, size_t m_ld, size_t n_ld, float alpha, float beta) {

	printf("D0 = %f\n", d0[1]);
}

__global__ void simple_wmma_gemm(half *d0, half *d1, half *d2, size_t m_ld, size_t n_ld, half alpha, half beta) {
	// Leading dimensions. Packed with no transpositions.
	int lda = WMMA_M;
	int ldb = WMMA_K;
	int ldc = WMMA_M;
	int ldc_c = m_ld;

	// Tile using a 2D grid
	int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
	int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

	// Declare the fragments
	wmma::fragment < wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major
			> a_frag;
	wmma::fragment < wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major
			> b_frag;
	wmma::fragment < wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half > acc_frag;
	wmma::fragment < wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half > c_frag;

	wmma::fill_fragment(a_frag, __float2half(1.0f));
	wmma::fill_fragment(b_frag, __float2half(2.0f));
	wmma::fill_fragment(c_frag, __float2half(5.0f));
	wmma::fill_fragment(acc_frag, __float2half(0.0f));

	// Loop over k
//	   for (int i = 0; i < K; i += WMMA_K) {
//	      int aRow = warpM * WMMA_M;
//	      int aCol = i;
//
//	      int bRow = i;
//	      int bCol = warpN * WMMA_N;

// Bounds checking
//	      if (aRow < M && aCol < K && bRow < K && bCol < N) {
	// Load the inputs
//	wmma::load_matrix_sync(a_frag, mat_a_16x16_half, lda);
//	wmma::load_matrix_sync(b_frag, mat_b_16x16_half, ldb);

	// Perform the matrix multiplication
	wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);

//	      }
//	   }

	// Load in the current value of c, scale it by beta, and add this our result scaled by alpha
	int cRow = warpM * WMMA_M;
	int cCol = warpN * WMMA_N;

	if (cRow < m_ld && cCol < n_ld) {
//		wmma::load_matrix_sync(c_frag, mat_c_16x16_half, ldc,
//				wmma::mem_col_major);

		for (int i = 0; i < c_frag.num_elements; i++) {
			c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
		}

		// Store the output
		wmma::store_matrix_sync(d0 + cRow + cCol * ldc, c_frag, ldc_c,
				wmma::mem_col_major);
		wmma::store_matrix_sync(d1 + cRow + cCol * ldc, c_frag, ldc_c,
				wmma::mem_col_major);
		wmma::store_matrix_sync(d2 + cRow + cCol * ldc, c_frag, ldc_c,
				wmma::mem_col_major);

	}
}

//-------------------------------------------------------------------------------------------------
// Performs an MxNxK GEMM (C=alpha*A*B + beta*C) assuming:
//  1) Matrices are packed in memory.
//  2) M, N and K are multiples of 16.
//  3) Neither A nor B are transposed.
// Note: This is a less performant version of the compute_gemm kernel. It is designed for
//       demonstration purposes only to show the CUDA WMMA API use without relying on
//       availability of the shared memory.
//template<class half_t, class real_t>
//__global__ void simple_wmma_gemm(half_t *a0, half_t *a1, half_t *a2,
//		half_t *b0, half_t *b1, half_t *b2,
//		real_t *c0, real_t *c1, real_t *c2,
//		real_t *d0, real_t *d1, real_t *d2,
//		size_t m_ld, size_t n_ld, size_t k_ld, real_t alpha, real_t beta, unsigned long long int* is_memory_bad) {
//	// Leading dimensions. Packed with no transpositions.
//	size_t lda = m_ld;
//	size_t ldb = k_ld;
//	size_t ldc = n_ld;
//
//	// Tile using a 2D grid
//	int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
//	int warpN = (blockIdx.y * blockDim.y + threadIdx.y);
//
//	// Declare the fragments
//	wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half_t,
//			wmma::row_major> a0_frag;
////	wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half_t,
////			wmma::row_major> a1_frag;
////	wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half_t,
////			wmma::row_major> a2_frag;
//	wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half_t,
//			wmma::col_major> b0_frag;
////	wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half_t,
////			wmma::col_major> b1_frag;
////	wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half_t,
////			wmma::col_major> b2_frag;
//
////
//	wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, real_t> acc_frag;
//
//	wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, real_t> c0_frag;
////	wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, real_t> c1_frag;
////	wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, real_t> c2_frag;
//
//	wmma::fill_fragment(acc_frag, 0.0f);
//
//	// Loop over k
//	for (int i = 0; i < k_ld; i += WMMA_K) {
//		int aCol = i;
//		int aRow = warpM * WMMA_M;
//
//		int bCol = i;
//		int bRow = warpN * WMMA_N;
//
//		// Bounds checking
//		if (aRow < m_ld && aCol < k_ld && bRow < k_ld && bCol < n_ld) {
//			// Load the inputs
//			wmma::load_matrix_sync(a0_frag, a0 + aCol + aRow * lda, lda);
////			wmma::load_matrix_sync(a1_frag, a1 + aCol + aRow * lda, lda);
////			wmma::load_matrix_sync(a2_frag, a2 + aCol + aRow * lda, lda);
//
//			wmma::load_matrix_sync(b0_frag, b0 + bCol + bRow * ldb, ldb);
////			wmma::load_matrix_sync(b1_frag, b1 + bCol + bRow * ldb, ldb);
////			wmma::load_matrix_sync(b2_frag, b2 + bCol + bRow * ldb, ldb);
//			// Perform the matrix multiplication
//			//~ wmma::mma_sync(acc_frag,
//			//~ (read_voter(a0_frag, a1_frag, a2_frag, (aCol + aRow * lda))),
//			//~ (read_voter(b0_frag, b1_frag, b2_frag, (bCol + bRow * ldb))),
//			//~ acc_frag);
//
//			wmma::mma_sync(acc_frag, a0_frag, b0_frag, acc_frag);
//
//		}
//	}
//
//	// Load in the current value of c, scale it by beta, and add this our result scaled by alpha
//	int cCol = warpN * WMMA_N;
//	int cRow = warpM * WMMA_M;
//
//	if (cRow < m_ld && cCol < n_ld) {
//		wmma::load_matrix_sync(c0_frag, c0 + cCol + cRow * ldc, ldc,
//				wmma::mem_row_major);
////		wmma::load_matrix_sync(c1_frag, c1 + cCol + cRow * ldc, ldc,
////				wmma::mem_row_major);
////		wmma::load_matrix_sync(c2_frag, c2 + cCol + cRow * ldc, ldc,
////				wmma::mem_row_major);
//
//		//~ for (int i = 0; i < c0_frag.num_elements; i++) {
//		//~ read_voter(c0_frag, c1_frag, c2_frag, (cCol + cRow * ldc)).x[i] =
//		//~ alpha * acc_frag.x[i]
//		//~ + beta
//		//~ * read_voter(c0_frag, c1_frag, c2_frag,
//		//~ (cCol + cRow * ldc)).x[i];
//		//~ }
//
//		// Store the output
//		//wmma::store_matrix_sync(d + cCol + cRow * ldc, c_frag, ldc, wmma::mem_row_major);
//
//		wmma::store_matrix_sync(d0 + cCol + cRow * ldc, c0_frag, ldc,
//				wmma::mem_row_major);
//		wmma::store_matrix_sync(d1 + cCol + cRow * ldc, c0_frag, ldc,
//				wmma::mem_row_major);
//		wmma::store_matrix_sync(d2 + cCol + cRow * ldc, c0_frag, ldc,
//				wmma::mem_row_major);
//
//	}
//}


template<class real_t>
__device__ real_t inline read_voter(real_t *v1, real_t *v2, real_t *v3,
		int offset, unsigned long long int* is_memory_bad) {

	register real_t in1 = v1[offset];
	register real_t in2 = v2[offset];
	register real_t in3 = v3[offset];

	if (in1 == in2 || in1 == in3) {
		return in1;
	}

	if (in2 == in3) {
		return in2;
	}

	if (in1 != in2 && in2 != in3 && in1 != in3) {
		atomicAdd(is_memory_bad, 1);
	}

	return in1;
}

template<class half_t, class real_t>
__global__ void matrix_mul(half_t *a0, half_t *a1, half_t *a2, half_t *b0,
		half_t *b1, half_t *b2, real_t *c0, real_t *c1, real_t *c2, real_t*d0,
		real_t *d1, real_t *d2, size_t M, size_t N, size_t K, real_t alpha,
		real_t beta, unsigned long long int* is_memory_bad) {

	register int tx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
	register int ty = blockIdx.y * BLOCK_SIZE + threadIdx.y;
	register int k;

	if (tx * ty > M * N)
		return;

	register real_t acc = 0.0;
	for (k = 0; k < N; k++) {

		half_t tmp = read_voter<half_t>(a0, a1, a2, ty * N + k, is_memory_bad)
				* read_voter<half_t>(b0, b1, b2, k * N + tx, is_memory_bad);
		acc = real_t(tmp) + acc;

	}

	acc = alpha * acc
			+ beta * read_voter<real_t>(c0, c1, c2, ty * N + tx, is_memory_bad);

	d0[ty * N + tx] = (real_t) acc;
	d1[ty * N + tx] = (real_t) acc;
	d2[ty * N + tx] = (real_t) acc;

}



#endif /* KERNELS_H_ */
