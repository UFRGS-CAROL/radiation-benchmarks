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

using namespace nvcuda;

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif

#ifndef SHARED_MEMORY_LIMIT_64K
// Set this to 0 to use more than 64 Kb of shared memory to cache data, to
// improve the performance of the computations on GPU.
// Note that you need a GPU that can have more than 64 Kb of shared memory
// per multiprocessor.
#define SHARED_MEMORY_LIMIT_64K 1
#endif

// GPU configuration.

#define WARP_SIZE 32

// MMA matrix tile dimensions.

#define M 16
#define N 16
#define K 16

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// GEMM configuration.

#define M_TILES 256
#define N_TILES 256
#define K_TILES 256

#define M_GLOBAL (M * M_TILES)
#define N_GLOBAL (N * N_TILES)
#define K_GLOBAL (K * K_TILES)

#define C_LAYOUT nvcuda::wmma::mem_row_major

// Implementation constants.

#define WARPS_PER_BLOCK 8
#define THREADS_PER_BLOCK (WARP_SIZE * WARPS_PER_BLOCK)

#if SHARED_MEMORY_LIMIT_64K
// With only 64 Kb shared memory available, we can fit two 8-tile chunks of
// the A and B matrix data, that are 16 * 16 * 8 * 8 * 2 = 32 Kb each
// (i.e. two 8x8 arrays of tiles of 16x16 half-typed elements per CTA).
// But we cannot account the 8 Kb total skew overhead, without which the performance
// would be severely impacted. So we choose to reduce the chunk size in half,
// i.e. the amount of A and B matrix data we cache in shared memory.
// Accordingly, this doubles the number of outer iterations across the global K
// dimension, which only slightly impacts the performance.
#define CHUNK_K 4
#else
#define CHUNK_K 8
#endif

#define CHUNK_LINE_BYTES (CHUNK_K * K * sizeof(half))
#define WARP_COPY_BYTES (WARP_SIZE * sizeof(int4))
#define CHUNK_COPY_LINES_PER_WARP (WARP_COPY_BYTES / CHUNK_LINE_BYTES)
#define CHUNK_COPY_LINE_LANES (WARP_SIZE / CHUNK_COPY_LINES_PER_WARP)

#define BLOCK_ROW_WARPS 2
#define BLOCK_COL_WARPS 4

#define WARP_ROW_TILES 4
#define WARP_COL_TILES 2

#define BLOCK_ROW_TILES (WARP_ROW_TILES * BLOCK_ROW_WARPS)
#define BLOCK_COL_TILES (WARP_COL_TILES * BLOCK_COL_WARPS)

#define GLOBAL_MEM_STRIDE N_GLOBAL

#define SHMEM_STRIDE (N * BLOCK_ROW_TILES)
#define SHMEM_OFFSET (N * WARP_ROW_TILES)

// The macro below is used to shift rows of the A matrix and columns of the B matrix
// in shared memory to minimize possible bank conflicts.
// Before performing the nvcuda::nvcuda::wmma::mma_sync operation, the warp must load the matrix
// data using the nvcuda::nvcuda::wmma::load_matrix_sync operation. Although the memory access pattern
// is not specified for that function, each lane in the warp can read one or multiple matrix
// elements from different matrix rows or columns.
// For shared memory, such access can result in bank conflicts if different rows / columns
// of the matrix map to the same bank. By shifting each row and column by a few bytes, we
// make sure that they map to different banks, thus reducing the number of possible bank
// conflicts.
// The number of 8 two-byte "half" elements is chosen as the minimum possible shift because
// we must keep each row and column 128-bit aligned, as required by nvcuda::nvcuda::wmma::load_matrix_sync.
#define SKEW_HALF 8

template<class T>
__device__ __forceinline__ void hw_mxm_device(T* D, T *C, float *A, float *B,
		T alpha, T beta, int wA, int wB) {
	assert(0 && "DOES NOT SUPPORT FLOAT");
}

template<class T>
__device__ __forceinline__ void hw_mxm_device(T* D, T *C, double *A, double *B,
		T alpha, T beta, int wA, int wB) {
	assert(0 && "DOES NOT SUPPORT DOUBLE");
}

template<class half_t, class real_t>
__device__ __forceinline__ void hw_mxm_device(real_t* d, const real_t *c,
		const half_t *a, const half_t *b, real_t alpha, real_t beta, int wA,
		int wB) {
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
__device__ __forceinline__ void sw_mxm_device(real_t* D, real_t *C, real_t *A,
		real_t *B, real_t alpha, real_t beta, int wA, int wB) {
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

		for (int k = 0; k < BLOCK_SIZE; ++k) {
			Csub += As[ty][k] * Bs[k][tx];
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

template<class half_t, class real_t>
__global__ void hw_mxm_kernel(real_t *D, real_t *C, half_t *A, half_t *B,
		real_t alpha, real_t beta, int wA, int wB) {
	hw_mxm_device(D, C, A, B, alpha, beta, wA, wB);

}

template<class real_t>
__global__ void sw_mxm_kernel(real_t *D, real_t *C, real_t *A, real_t *B,
		real_t alpha, real_t beta, int wA, int wB) {
	sw_mxm_device(D, C, A, B, alpha, beta, wA, wB);
}

#endif /* NONDMR_KERNELS_H_ */
