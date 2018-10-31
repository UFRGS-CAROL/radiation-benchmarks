/*
 * kernels.h
 *
 *  Created on: 05/10/2018
 *      Author: fernando
 */

#ifndef KERNELS_H_
#define KERNELS_H_

#include <mma.h>
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

//-------------------------------------------------------------------------------------------------
// Performs an MxNxK GEMM (C=alpha*A*B + beta*C) assuming:
//  1) Matrices are packed in memory.
//  2) M, N and K are multiples of 16.
//  3) Neither A nor B are transposed.
// Note: This is a less performant version of the compute_gemm kernel. It is designed for
//       demonstration purposes only to show the CUDA WMMA API use without relying on
//       availability of the shared memory.
template<class half_t, class real_t>
__global__ void simple_wmma_gemm(half_t *a, half_t *b, real_t *c, real_t *d,
		int m_ld, int n_ld, int k_ld, real_t alpha, real_t beta) {
	// Leading dimensions. Packed with no transpositions.
	int lda = m_ld;
	int ldb = k_ld;
	int ldc = n_ld;

	// Tile using a 2D grid
	int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
	int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

	// Declare the fragments
	wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half_t,
			wmma::row_major> a_frag;
	wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half_t,
			wmma::col_major> b_frag;
	wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, real_t> acc_frag;
	wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, real_t> c_frag;

	wmma::fill_fragment(acc_frag, 0.0f);

	// Loop over k
	for (int i = 0; i < k_ld; i += WMMA_K) {
		int aCol = i;
		int aRow = warpM * WMMA_M;

		int bCol = i;
		int bRow = warpN * WMMA_N;

		// Bounds checking
		if (aRow < m_ld && aCol < k_ld && bRow < k_ld && bCol < n_ld) {
			// Load the inputs
			wmma::load_matrix_sync(a_frag, a + aCol + aRow * lda, lda);
			wmma::load_matrix_sync(b_frag, b + bCol + bRow * ldb, ldb);

			// Perform the matrix multiplication
			wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);

		}
	}

	// Load in the current value of c, scale it by beta, and add this our result scaled by alpha
	int cCol = warpN * WMMA_N;
	int cRow = warpM * WMMA_M;

	if (cRow < m_ld && cCol < n_ld) {
		wmma::load_matrix_sync(c_frag, c + cCol + cRow * ldc, ldc,
				wmma::mem_row_major);

		for (int i = 0; i < c_frag.num_elements; i++) {
			c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
		}

		// Store the output
		wmma::store_matrix_sync(d + cCol + cRow * ldc, c_frag, ldc,
				wmma::mem_row_major);
	}
}

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

template<class half_t, class real_t>
__global__ void matrix_mul(half_t *a0, half_t *b0, real_t *c0, real_t*d0,
		size_t M, size_t N, size_t K, real_t alpha, real_t beta) {

	register int tx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
	register int ty = blockIdx.y * BLOCK_SIZE + threadIdx.y;
	register int k;

	if (tx * ty > M * N)
		return;

	register real_t acc = 0.0;
	for (k = 0; k < N; k++) {
		acc = real_t(a0[ty * N + k] * b0[k * N + tx]) + acc;
	}

	acc = alpha * acc
			+ beta * c0[ty * N + tx];

	d0[ty * N + tx] = (real_t) acc;
}

#endif /* KERNELS_H_ */
