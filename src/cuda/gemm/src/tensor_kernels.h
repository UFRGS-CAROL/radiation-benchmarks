/*
 * tensor_kernels.h
 *
 *  Created on: Nov 22, 2019
 *      Author: fernando
 */

#ifndef TENSOR_KERNELS_H_
#define TENSOR_KERNELS_H_

#include <cuda.h>

#include <mma.h>

#include "common.h"

template<typename half_t, typename real_t> __device__
void call_mxm_wmma_unhardened(const half_t* A, const half_t* B, const real_t* C,
		real_t* D, real_t alpha, real_t beta) {
	extern __shared__ half_t shmem[][CHUNK_K * K + SKEW_HALF];
	// Warp and lane identification.
	const unsigned int warpId = threadIdx.x / WARP_SIZE;
	const unsigned int laneId = threadIdx.x % WARP_SIZE;
	// Offset in shared memory from which the B matrix is stored.
	const size_t shmem_idx_b_off = BLOCK_COL_TILES * M;
	// This pointer is used to access the C and D matrix tiles this warp computes.
	real_t *shmem_warp_tile_ptr = (real_t*) &shmem[0][0]
			+ (warpId / 2) * SHMEM_STRIDE * K * 2+ (warpId%2) * SHMEM_OFFSET;
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
		const half_t *warp_ptr =
				(warpId < 4) ?
						(&A[block_tile_i * M * K_GLOBAL]
								+ M * K_GLOBAL * (warpId % 4) * 2) :
						(&B[block_tile_j * N * K_GLOBAL]
								+ N * K_GLOBAL * (warpId % 4) * 2);

		// Go through the global K dimension by a fixed step at a time.
#pragma unroll
		for (int tile_k = 0; tile_k < K_TILES; tile_k += CHUNK_K) {
			// Copy slices of the A and B matrices to shared memory.
			// The first half_t of the warps in the CTA copy the A matrix, the rest copy the B matrix.
			size_t shmem_idx =
					warpId < (WARPS_PER_BLOCK / 2) ?
							(M * (warpId % (WARPS_PER_BLOCK / 2)) * 2) :
							(N * (warpId % (WARPS_PER_BLOCK / 2)) * 2
									+ shmem_idx_b_off);

			// First half_t of the warp copies the first row / column of the matrix,
			// the second half_t of the warp copies the next.
			int4 *lane_ptr = (int4*) (warp_ptr + tile_k * K
					+ (laneId / CHUNK_COPY_LINE_LANES) * K_GLOBAL)
					+ (laneId % CHUNK_COPY_LINE_LANES);

			// Shift the second half_t of the warp to the next row / column in the shared memory.
			shmem_idx += laneId / CHUNK_COPY_LINE_LANES;

#pragma unroll
			for (int i = 0;
					i < ((WARP_SIZE / 2) / CHUNK_COPY_LINES_PER_WARP) * 2;
					i++) {
				// Copy 16 bytes at once in each lane.
				*((int4*) &shmem[shmem_idx][0]
						+ (laneId % CHUNK_COPY_LINE_LANES)) = *lane_ptr;

				// Advance the global memory pointer and the shared memory index.
				lane_ptr = (int4*) ((half_t*) lane_ptr
						+ K_GLOBAL * CHUNK_COPY_LINES_PER_WARP);
				shmem_idx += CHUNK_COPY_LINES_PER_WARP;
			}

			__syncthreads();

			// Compute a grid of C matrix tiles in each warp.
#pragma unroll
			for (int k_step = 0; k_step < CHUNK_K; k_step++) {
				nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, M, N, K, half_t,
						nvcuda::wmma::row_major> a[WARP_COL_TILES];
				nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, M, N, K, half_t,
						nvcuda::wmma::col_major> b[WARP_ROW_TILES];

#pragma unroll
				for (int i = 0; i < WARP_COL_TILES; i++) {
					size_t shmem_idx_a = (warpId / 2) * M * 2 + (i * M);
					const half_t *tile_ptr = &shmem[shmem_idx_a][k_step * K];

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
							const half_t *tile_ptr = &shmem[shmem_idx_b][k_step
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

template<typename half_t, typename real_t>
__global__ void matrix_mult_kernel_wmma_unhardened(const half_t *A,
		const half_t *B, const real_t *C, real_t *D, real_t alpha,
		real_t beta) {
#if __CUDA_ARCH__ >= 700
	call_mxm_wmma_unhardened(A, B, C, D, alpha, beta);
#endif /* END CUDA ARCH CHECK */
}

//template<typename half_t, typename real_t>
//__global__ void matrix_mult_kernel_wmma_unhardened(half_t *a, half_t *b,
//		real_t *c, real_t *d, real_t alpha, real_t beta, int m, int n, int k) {
//	// Leading dimensions. Packed with no transposition.
//	int lda = m;
//	int ldb = n;
//	int ldc = m;
//
//	// Tile using a 2D grid
//	int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
//	int warpN = (blockIdx.y * blockDim.y + threadIdx.y);
//
//	// Declare the fragments
//	nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
//			half_t, nvcuda::wmma::col_major> a_frag;
//	nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
//			half_t, nvcuda::wmma::col_major> b_frag;
//	nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K,
//			real_t> acc_frag;
//	nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K,
//			real_t> c_frag;
//
//	nvcuda::wmma::fill_fragment(acc_frag, 0.0f);
//
//	// Loop over k
//	for (int i = 0; i < k; i += WMMA_K) {
//		int aRow = warpM * WMMA_M;
//		int aCol = i;
//
//		int bRow = i;
//		int bCol = warpN * WMMA_N;
//
//		// Bounds checking
//		if (aRow < M && aCol < K && bRow < K && bCol < N) {
//			// Load the inputs
//			nvcuda::wmma::load_matrix_sync(a_frag, a + aRow + aCol * lda, lda);
//			nvcuda::wmma::load_matrix_sync(b_frag, b + bRow + bCol * ldb, ldb);
//
//			// Perform the matrix multiplication
//			nvcuda::wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
//
//		}
//	}
//
//	// Load in the current value of c, scale it by beta, and add this our result scaled by alpha
//	int cRow = warpM * WMMA_M;
//	int cCol = warpN * WMMA_N;
//
//	if (cRow < M && cCol < N) {
//		nvcuda::wmma::load_matrix_sync(c_frag, c + cRow + cCol * ldc, ldc,
//				nvcuda::wmma::mem_col_major);
//
//		for (int i = 0; i < c_frag.num_elements; i++) {
//			c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
//		}
//
//		// Store the output
//		nvcuda::wmma::store_matrix_sync(d + cRow + cCol * ldc, c_frag, ldc,
//				nvcuda::wmma::mem_col_major);
//	}
//}

#endif /* TENSOR_KERNELS_H_ */
