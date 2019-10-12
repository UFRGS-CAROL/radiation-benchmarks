/*
 * nondmr_kernels.h
 *
 *  Created on: Jul 20, 2019
 *      Author: fernando
 */

#ifndef NONDMR_KERNELS_H_
#define NONDMR_KERNELS_H_

#include <mma.h>
#include <cuda_fp16.h>
#include <assert.h>

#include "device_functions.h"
#include "common.h"

template<class T>
__global__ void hw_mxm_kernel(T* D, T *C, float *A, float *B, T alpha, T beta,
		int wA, int wB) {
	assert(0);
}

template<class T>
__global__ void hw_mxm_kernel(T* D, T *C, double *A, double *B, T alpha, T beta,
		int wA, int wB) {
	assert(0);
}

template<class half_t, class real_t>
__global__ void hw_mxm_kernel(real_t *D, real_t *C, half_t *A, half_t *B,
		real_t alpha, real_t beta, int wA, int wB) {
	// Leading dimensions. Packed with no transpositions.
	int m_ld = wA;
	int k_ld = wB;
	int n_ld = wA;
	int lda = m_ld;
	int ldb = k_ld;
	int ldc = n_ld;

	// Tile using a 2D grid
	int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
	int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

	// Declare the fragments
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
			half_t, nvcuda::wmma::row_major> a_frag;
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
			half_t, nvcuda::wmma::col_major> b_frag;
	nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K,
			real_t> acc_frag;
	nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K,
			real_t> c_frag;

	nvcuda::wmma::fill_fragment(acc_frag, 0.0f);

	// Loop over k
	for (int i = 0; i < k_ld; i += WMMA_K) {
		int aCol = i;
		int aRow = warpM * WMMA_M;

		int bCol = i;
		int bRow = warpN * WMMA_N;

		// Bounds checking
		if (aRow < m_ld && aCol < k_ld && bRow < k_ld && bCol < n_ld) {
			// Load the inputs
			nvcuda::wmma::load_matrix_sync(a_frag, A + aCol + aRow * lda, lda);
			nvcuda::wmma::load_matrix_sync(b_frag, B + bCol + bRow * ldb, ldb);

			// Perform the matrix multiplication
			nvcuda::wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);

		}
	}

	// Load in the current value of c, scale it by beta, and add this our result scaled by alpha
	int cCol = warpN * WMMA_N;
	int cRow = warpM * WMMA_M;

	if (cRow < m_ld && cCol < n_ld) {
		nvcuda::wmma::load_matrix_sync(c_frag, C + cCol + cRow * ldc, ldc,
				nvcuda::wmma::mem_row_major);

		for (int i = 0; i < c_frag.num_elements; i++) {
			c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
		}

		// Store the output
		nvcuda::wmma::store_matrix_sync(D + cCol + cRow * ldc, c_frag, ldc,
				nvcuda::wmma::mem_row_major);
	}

}

template<class real_t>
__global__ void sw_mxm_kernel(real_t *D, real_t *C, real_t *A, real_t *B,
		real_t alpha, real_t beta, int wA, int wB) {
	// Block index
	int bx = blockIdx.x;
	int by = blockIdx.y;

	// Thread index
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	// Index of the first sub-matrix of A processed by the block
	int aBegin = wA * BLOCK_SIZE * by;

	// Index of the last sub-matrix of A processed by the block
	int aEnd = aBegin + wA - 1;

	// Step size used to iterate through the sub-matrices of A
	int aStep = BLOCK_SIZE;

	// Index of the first sub-matrix of B processed by the block
	int bBegin = BLOCK_SIZE * bx;

	// Step size used to iterate through the sub-matrices of B
	int bStep = BLOCK_SIZE * wB;

	// Csub is used to store the element of the block sub-matrix
	// that is computed by the thread
	real_t Csub = 0;

	// Loop over all the sub-matrices of A and B
	// required to compute the block sub-matrix
	for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
		// Declaration of the shared memory array As used to
		// store the sub-matrix of A
		__shared__ real_t As[BLOCK_SIZE][BLOCK_SIZE];

		// Declaration of the shared memory array Bs used to
		// store the sub-matrix of B
		__shared__ real_t Bs[BLOCK_SIZE][BLOCK_SIZE];

		// Load the matrices from device memory
		// to shared memory; each thread loads
		// one element of each matrix
		As[ty][tx] = A[a + wA * ty + tx];
		Bs[ty][tx] = B[b + wB * ty + tx];

		// Synchronize to make sure the matrices are loaded
		__syncthreads();

		// Multiply the two matrices together;
		// each thread computes one element
		// of the block sub-matrix
#pragma unroll
		for (int k = 0; k < BLOCK_SIZE; k++) {
			fma_dmr(As[ty][k], Bs[k][tx], Csub);
		}

		// Synchronize to make sure that the preceding
		// computation is done before loading two new
		// sub-matrices of A and B in the next iteration
		__syncthreads();
	}

	// Write the block sub-matrix to device memory;
	// each thread writes one element
	int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
	D[c + wB * ty + tx] = alpha * Csub + beta * C[c + wB * ty + tx];
}

//__global__ void sw_mxm_kernel(half *D, half *C, half *A, half *B, half alpha,
//		half beta, int wA, int wB) {
//	// Block index
//	int bx = blockIdx.x;
//	int by = blockIdx.y;
//
//	// Thread index
//	int tx = threadIdx.x;
//	int ty = threadIdx.y;
//
//	// Index of the first sub-matrix of A processed by the block
//	int aBegin = wA * BLOCK_SIZE * by;
//
//	// Index of the last sub-matrix of A processed by the block
//	int aEnd = aBegin + wA - 1;
//
//	// Step size used to iterate through the sub-matrices of A
//	int aStep = BLOCK_SIZE;
//
//	// Index of the first sub-matrix of B processed by the block
//	int bBegin = BLOCK_SIZE * bx;
//
//	// Step size used to iterate through the sub-matrices of B
//	int bStep = BLOCK_SIZE * wB;
//
//	// Csub is used to store the element of the block sub-matrix
//	// that is computed by the thread
//	half Csub = 0;
//
//	// Loop over all the sub-matrices of A and B
//	// required to compute the block sub-matrix
//	for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
//		// Declaration of the shared memory array As used to
//		// store the sub-matrix of A
//		__shared__ half As[BLOCK_SIZE][BLOCK_SIZE];
//
//		// Declaration of the shared memory array Bs used to
//		// store the sub-matrix of B
//		__shared__ half Bs[BLOCK_SIZE][BLOCK_SIZE];
//
//		// Load the matrices from device memory
//		// to shared memory; each thread loads
//		// one element of each matrix
//		As[ty][tx] = A[a + wA * ty + tx];
//		Bs[ty][tx] = B[b + wB * ty + tx];
//
//		// Synchronize to make sure the matrices are loaded
//		__syncthreads();
//
//		half2 acc(0, 0);
//		// Multiply the two matrices together;
//		// each thread computes one element
//		// of the block sub-matrix
//#pragma unroll
//		for (int k = 0; k < BLOCK_SIZE; k += 2) {
//			half2 a = __halves2half2(As[ty][k + 0], As[ty][k + 1]);
//			half2 b = __halves2half2(Bs[k + 0][tx], Bs[k + 1][tx]);
//
//			fma_dmr(a, b, acc);
//		}
//
//		Csub += acc.x + acc.y;
//		// Synchronize to make sure that the preceding
//		// computation is done before loading two new
//		// sub-matrices of A and B in the next iteration
//		__syncthreads();
//	}
//
//	// Write the block sub-matrix to device memory;
//	// each thread writes one element
//	int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
//
//	D[c + wB * ty + tx] = alpha * Csub + beta * C[c + wB * ty + tx];
//}

#endif /* NONDMR_KERNELS_H_ */
