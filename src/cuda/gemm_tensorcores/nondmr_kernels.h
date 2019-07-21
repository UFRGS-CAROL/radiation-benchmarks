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

template<class real_t>
__device__ __forceinline__ void hw_mxm_device(real_t* D, real_t *C, real_t *A,
		real_t *B, real_t alpha, real_t beta, int wA, int wB) {
	extern __shared__ half shmem[][CHUNK_K * K + SKEW_HALF];

	// Warp and lane identification.
	const unsigned int warpId = threadIdx.x / WARP_SIZE;
	const unsigned int laneId = threadIdx.x % WARP_SIZE;

	// Offset in shared memory from which the B matrix is stored.
	const size_t shmem_idx_b_off = BLOCK_COL_TILES * M;

	// This pointer is used to access the C and D matrix tiles this warp computes.
	real_t *shmem_warp_tile_ptr = (real_t*) &shmem[0][0]
			+ (warpId / 2) * SHMEM_STRIDE * K * 2+ (warpId % 2) * SHMEM_OFFSET;

	// This pointer is used to stream the C and D matrices block-wide tile to and from shared memory.
	real_t *shmem_warp_stream_ptr = (real_t*) &shmem[0][0]
			+ warpId * SHMEM_STRIDE * K;

	// Adjust the beta scaler, as it'll be multiplied by alpha at the end of
	// each tile computation. Technically this is not generally correct (may result
	// in a loss of precision). Zero still needs to be specially handled though.
	beta /= alpha;

	// Each CTA slides along the 128 x 128 tiles from the top left corner of the matrix to the
	// right and down, and selects the next tile to compute. Once there's no such tile,
	// all warps in this CTA exit.
	for (unsigned int block_pos = blockIdx.x;; block_pos += gridDim.x) {
		const unsigned int block_tile_i = ((block_pos * BLOCK_ROW_TILES)
				/ N_TILES) * (BLOCK_COL_TILES);
		const unsigned int block_tile_j = (block_pos * BLOCK_COL_TILES)
				% N_TILES;

		// Stop when there are no more D matrix tiles to compute in this CTA.
		if (block_tile_i >= M_TILES) {
			break;
		}

		// This warp's pointer to the C matrix data to copy memory from to shared memory.
		const size_t gmem_idx = (block_tile_i + warpId) * M * GLOBAL_MEM_STRIDE
				+ block_tile_j * N;
		const real_t *src_gmem_warp_stream_ptr = &C[gmem_idx];

		// Stream multiple C tiles to shared memory.
#pragma unroll
		for (int i = 0; i < K; i++) {
			typedef int4 copy_t;

			*((copy_t *) (shmem_warp_stream_ptr + SHMEM_STRIDE * i) + laneId) =
					*((copy_t *) (src_gmem_warp_stream_ptr
							+ GLOBAL_MEM_STRIDE * i) + laneId);
		}

		__syncthreads();

		// These fragments will accumulate the result of A and B matrix fragment multiplications
		// along the K_GLOBAL dimension.
		nvcuda::wmma::fragment<nvcuda::wmma::accumulator, M, N, K, real_t> c[WARP_COL_TILES][WARP_ROW_TILES];

		// Load the C matrix tiles into fragments from shared memory.
#pragma unroll
		for (int i = 0; i < WARP_COL_TILES; i++) {
#pragma unroll
			for (int j = 0; j < WARP_ROW_TILES; j++) {
				const real_t *tile_ptr = shmem_warp_tile_ptr
						+ i * SHMEM_STRIDE * K + j * N;

				nvcuda::wmma::load_matrix_sync(c[i][j], tile_ptr, SHMEM_STRIDE,
				C_LAYOUT);
			}
		}

		__syncthreads();

		// Scale the C matrix.
#pragma unroll
		for (int i = 0; i < WARP_COL_TILES; i++) {
#pragma unroll
			for (int j = 0; j < WARP_ROW_TILES; j++) {
#pragma unroll
				for (int t = 0; t < c[i][j].num_elements; t++) {
					c[i][j].x[t] *= beta;
				}
			}
		}

		// Select what warp copies what matrix to shared memory.
		// Warps 0-3 copy the A matrix, warps 4-7 copy the B matrix.
		const half *warp_ptr =
				(warpId < 4) ?
						(&A[block_tile_i * M * K_GLOBAL]
								+ M * K_GLOBAL * (warpId % 4) * 2) :
						(&B[block_tile_j * N * K_GLOBAL]
								+ N * K_GLOBAL * (warpId % 4) * 2);

		// Go through the global K dimension by a fixed step at a time.
#pragma unroll
		for (int tile_k = 0; tile_k < K_TILES; tile_k += CHUNK_K) {
			// Copy slices of the A and B matrices to shared memory.
			// The first half of the warps in the CTA copy the A matrix, the rest copy the B matrix.
			size_t shmem_idx =
					warpId < (WARPS_PER_BLOCK / 2) ?
							(M * (warpId % (WARPS_PER_BLOCK / 2)) * 2) :
							(N * (warpId % (WARPS_PER_BLOCK / 2)) * 2
									+ shmem_idx_b_off);

			// First half of the warp copies the first row / column of the matrix,
			// the second half of the warp copies the next.
			int4 *lane_ptr = (int4*) (warp_ptr + tile_k * K
					+ (laneId / CHUNK_COPY_LINE_LANES) * K_GLOBAL)
					+ (laneId % CHUNK_COPY_LINE_LANES);

			// Shift the second half of the warp to the next row / column in the shared memory.
			shmem_idx += laneId / CHUNK_COPY_LINE_LANES;

#pragma unroll
			for (int i = 0;
					i < ((WARP_SIZE / 2) / CHUNK_COPY_LINES_PER_WARP) * 2;
					i++) {
				// Copy 16 bytes at once in each lane.
				*((int4*) &shmem[shmem_idx][0]
						+ (laneId % CHUNK_COPY_LINE_LANES)) = *lane_ptr;

				// Advance the global memory pointer and the shared memory index.
				lane_ptr = (int4*) ((half*) lane_ptr
						+ K_GLOBAL * CHUNK_COPY_LINES_PER_WARP);
				shmem_idx += CHUNK_COPY_LINES_PER_WARP;
			}

			__syncthreads();

			// Compute a grid of C matrix tiles in each warp.
#pragma unroll
			for (int k_step = 0; k_step < CHUNK_K; k_step++) {
				nvcuda::wmma::fragment < nvcuda::wmma::matrix_a, M, N, K, half, nvcuda::wmma::row_major
						> a[WARP_COL_TILES];
				nvcuda::wmma::fragment < nvcuda::wmma::matrix_b, M, N, K, half, nvcuda::wmma::col_major
						> b[WARP_ROW_TILES];

#pragma unroll
				for (int i = 0; i < WARP_COL_TILES; i++) {
					size_t shmem_idx_a = (warpId / 2) * M * 2 + (i * M);
					const half *tile_ptr = &shmem[shmem_idx_a][k_step * K];

					nvcuda::wmma::load_matrix_sync(a[i], tile_ptr,
					K * CHUNK_K + SKEW_HALF);

#pragma unroll
					for (int j = 0; j < WARP_ROW_TILES; j++) {
						if (i == 0) {
							// Load the B matrix fragment once, because it is going to be reused
							// against the other A matrix fragments.
							size_t shmem_idx_b = shmem_idx_b_off
									+ (WARP_ROW_TILES * N) * (warpId % 2)
									+ (j * N);
							const half *tile_ptr = &shmem[shmem_idx_b][k_step
									* K];

							nvcuda::wmma::load_matrix_sync(b[j], tile_ptr,
							K * CHUNK_K + SKEW_HALF);
						}

						nvcuda::wmma::mma_sync(c[i][j], a[i], b[j], c[i][j]);
					}
				}
			}

			__syncthreads();
		}

		// Store the D fragments to shared memory.
#pragma unroll
		for (int i = 0; i < WARP_COL_TILES; i++) {
#pragma unroll
			for (int j = 0; j < WARP_ROW_TILES; j++) {
#pragma unroll
				// Uniform, point-wise transformations of ALL fragment elements by ALL threads in the
				// warp are well-defined even though element indices within fragment storage are not defined.
				for (int t = 0; t < c[i][j].num_elements; t++)
					c[i][j].x[t] *= alpha;

				real_t *tile_ptr = shmem_warp_tile_ptr + i * SHMEM_STRIDE * K
						+ j * N;

				nvcuda::wmma::store_matrix_sync(tile_ptr, c[i][j], SHMEM_STRIDE,
				C_LAYOUT);
			}
		}

		__syncthreads();

		// Now that shared memory contains all the D tiles, stream them to global memory.
		real_t *dst_gmem_warp_stream_ptr = &D[gmem_idx];

#pragma unroll
		for (int i = 0; i < K; i++) {
			*((int4*) (dst_gmem_warp_stream_ptr + GLOBAL_MEM_STRIDE * i)
					+ laneId) = *((int4*) (shmem_warp_stream_ptr
					+ SHMEM_STRIDE * i) + laneId);
		}

		__syncthreads();
	}
}

template<class real_t>
__device__ __forceinline__ void sw_mxm_device(real_t* D, real_t *C, real_t *A,
		real_t *B,  real_t alpha, real_t beta, int wA, int wB) {
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

template<class real_t>
__global__ void hw_mxm_kernel(real_t *D, real_t *C, real_t *A, real_t *B,
		 real_t alpha,  real_t beta, int wA, int wB) {
	hw_mxm_device(D, C, A, B, alpha, beta, wA, wB);

}

template<class real_t>
__global__ void sw_mxm_kernel(real_t *D, real_t *C, real_t *A, real_t *B,
		 real_t alpha, real_t beta, int wA, int wB) {
	sw_mxm_device(D, C, A, B, alpha, beta, wA, wB);
}

#endif /* NONDMR_KERNELS_H_ */
