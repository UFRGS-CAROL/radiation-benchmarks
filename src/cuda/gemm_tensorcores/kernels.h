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


// helper functions and utilities to work with CUDA
#include <helper_cuda.h>
#include <helper_functions.h>

using namespace nvcuda;

// MMA matrix tile dimensions.

#define M 16
#define N 16
#define K 16

// The only dimensions currently supported by WMMA
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

#define C_LAYOUT wmma::mem_row_major


#define WARP_SIZE 32

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif

#ifndef SHARED_MEMORY_LIMIT_64K
// Set this to 0 to use more than 64 Kb of shared memory to cache data, to
// improve the performance of the computations on GPU.
// Note that you need a GPU that can have more than 64 Kb of shared memory
// per multiprocessor.
#define SHARED_MEMORY_LIMIT_64K 0
#endif

// Implementation constants.

#define WARPS_PER_BLOCK 8
#define THREADS_PER_BLOCK (WARP_SIZE * WARPS_PER_BLOCK)

#if SHARED_MEMORY_LIMIT_64K
// With only 64 Kb shared memory available, we can fit two 8-tile chunks of
// the A and B matrix data, that are 16 * 16 * 8 * 8 * 2 = 32 Kb each
// (i.e. two 8x8 arrays of tiles of 16x16 half-typed elements per CTA).
// But we cannot account the 8 Kb total skew overhead, without which the
// performance would be severely impacted. So we choose to reduce the chunk size
// in half, i.e. the amount of A and B matrix data we cache in shared memory.
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

#define SKEW_HALF 8

namespace experimental { 
    namespace precision { 
        struct u4; // 4-bit unsigned 
        struct s4; // 4-bit signed 
        struct b1; // 1-bit 
     } 
    enum bmmaBitOp { bmmaBitOpXOR = 1 }; 
    enum bmmaAccumulateOp { bmmaAccumulateOpPOPC = 1 }; 
} 

__device__ float errors = 0;


//-------------------------------------------------------------------------------------------------
//FULL gemm function
//-------------------------------------------------------------------------------------------------
 template<class half_t, class real_t>
 __global__ void compute_gemm(const half_t *A, const half_t *B, const real_t *C, real_t *D, float alpha, float beta)
 {
 	extern __shared__ half shmem[][CHUNK_K * K + SKEW_HALF];

 	// Warp and lane identification.
 	const unsigned int warpId = threadIdx.x / WARP_SIZE;
 	const unsigned int laneId = threadIdx.x % WARP_SIZE;

 	// Offset in shared memory from which the B matrix is stored.
 	const size_t shmem_idx_b_off = BLOCK_COL_TILES * M;

 	// This pointer is used to access the C and D matrix tiles this warp computes.
 	float *shmem_warp_tile_ptr = (float*)&shmem[0][0] + (warpId/2) * SHMEM_STRIDE * K * 2 + (warpId%2) * SHMEM_OFFSET;

 	// This pointer is used to stream the C and D matrices block-wide tile to and from shared memory.
 	float *shmem_warp_stream_ptr = (float*)&shmem[0][0] + warpId * SHMEM_STRIDE * K;

 	// Adjust the beta scaler, as it'll be multiplied by alpha at the end of
 	// each tile computation
 	// Technically this is not generally correct (may result
 	// in a loss of precision). Zero still needs to be specially handled though.
 	beta /= alpha;

 	// Each CTA slides along the 128 x 128 tiles from the top left corner of the matrix to the
 	// right and down, and selects the next tile to compute. Once there's no such tile,
 	// all warps in this CTA exit.
 	for(unsigned int block_pos = blockIdx.x;; block_pos += gridDim.x) {
 		const unsigned int block_tile_i = ((block_pos * BLOCK_ROW_TILES) / N_TILES) * (BLOCK_COL_TILES);
 		const unsigned int block_tile_j = (block_pos * BLOCK_COL_TILES) % N_TILES;

 		// Stop when there are no more D matrix tiles to compute in this CTA.
 		if (block_tile_i >= M_TILES) {
 			break;
 		}

 		// This warp's pointer to the C matrix data to copy memory from to shared memory.
 		const size_t gmem_idx = (block_tile_i + warpId) * M * GLOBAL_MEM_STRIDE + block_tile_j * N;
 		const real_t *src_gmem_warp_stream_ptr = &C[gmem_idx];

 		// Stream multiple C tiles to shared memory.
 #pragma unroll
 		for (int i = 0; i < K; i++) {
 			typedef int4 copy_t;

 			*((copy_t *)(shmem_warp_stream_ptr + SHMEM_STRIDE * i) + laneId) = 
 					*((copy_t *)(src_gmem_warp_stream_ptr + GLOBAL_MEM_STRIDE * i) + laneId);
 		}

 		__syncthreads();

 		// These fragments will accumulate the result of A and B matrix fragment multiplications
 		// along the K_GLOBAL dimension.
 		wmma::fragment<wmma::accumulator, M, N, K, float> c[WARP_COL_TILES][WARP_ROW_TILES];

 		// Load the C matrix tiles into fragments from shared memory.
 #pragma unroll
 		for (int i = 0; i < WARP_COL_TILES; i++) {
 #pragma unroll
 			for (int j = 0; j < WARP_ROW_TILES; j++) {
 				const float *tile_ptr = shmem_warp_tile_ptr + i * SHMEM_STRIDE * K + j * N;

 				wmma::load_matrix_sync(c[i][j], tile_ptr, SHMEM_STRIDE, C_LAYOUT);
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
 		const half *warp_ptr = (warpId < 4) ? (&A[block_tile_i * M * K_GLOBAL] + M * K_GLOBAL * (warpId % 4) * 2) :
 				(&B[block_tile_j * N * K_GLOBAL] + N * K_GLOBAL * (warpId % 4) * 2);

 		// Go through the global K dimension by a fixed step at a time.
 #pragma unroll
 		for (int tile_k = 0; tile_k < K_TILES; tile_k += CHUNK_K) {
 			// Copy slices of the A and B matrices to shared memory.
 			// The first half of the warps in the CTA copy the A matrix, the rest copy the B matrix.
 			size_t shmem_idx = warpId < (WARPS_PER_BLOCK/2) ? (M * (warpId % (WARPS_PER_BLOCK/2)) * 2) : 
 					(N * (warpId % (WARPS_PER_BLOCK/2)) * 2 + shmem_idx_b_off);

 			// First half of the warp copies the first row / column of the matrix,
 			// the second half of the warp copies the next.
 			int4 *lane_ptr = (int4*)(warp_ptr + tile_k * K + (laneId / CHUNK_COPY_LINE_LANES) * K_GLOBAL) + (laneId % CHUNK_COPY_LINE_LANES);

 			// Shift the second half of the warp to the next row / column in the shared memory.
 			shmem_idx += laneId / CHUNK_COPY_LINE_LANES;

 #pragma unroll
 			for(int i = 0; i < ((WARP_SIZE/2) / CHUNK_COPY_LINES_PER_WARP) * 2; i++) {
 				// Copy 16 bytes at once in each lane.
 				*((int4*)&shmem[shmem_idx][0] + (laneId % CHUNK_COPY_LINE_LANES)) = *lane_ptr;

 				// Advance the global memory pointer and the shared memory index.
 				lane_ptr = (int4*)((half*)lane_ptr + K_GLOBAL * CHUNK_COPY_LINES_PER_WARP);
 				shmem_idx += CHUNK_COPY_LINES_PER_WARP;
 			}

 			__syncthreads();

 			// Compute a grid of C matrix tiles in each warp.
 #pragma unroll
 			for (int k_step = 0; k_step < CHUNK_K; k_step++) {
 				wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> a[WARP_COL_TILES];
 				wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::col_major> b[WARP_ROW_TILES];

 #pragma unroll
 				for (int i = 0; i < WARP_COL_TILES; i++) {
 					size_t shmem_idx_a = (warpId/2) * M * 2 + (i * M);
 					const half *tile_ptr = &shmem[shmem_idx_a][k_step * K];

 					wmma::load_matrix_sync(a[i], tile_ptr, K * CHUNK_K + SKEW_HALF);

 #pragma unroll
 for (int j = 0; j < WARP_ROW_TILES; j++) {
 	if (i == 0) {
 		// Load the B matrix fragment once, because it is going to be reused
 		// against the other A matrix fragments.
 		size_t shmem_idx_b = shmem_idx_b_off + (WARP_ROW_TILES * N) * (warpId%2) + (j * N);
 		const half *tile_ptr = &shmem[shmem_idx_b][k_step * K];

 		wmma::load_matrix_sync(b[j], tile_ptr, K * CHUNK_K + SKEW_HALF);
 	}

 	wmma::mma_sync(c[i][j], a[i], b[j], c[i][j]);
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

 				float *tile_ptr = shmem_warp_tile_ptr + i * SHMEM_STRIDE * K + j * N;

 				wmma::store_matrix_sync(tile_ptr, c[i][j], SHMEM_STRIDE, C_LAYOUT);
 			}
 		}

 		__syncthreads();

 		// Now that shared memory contains all the D tiles, stream them to global memory.
 		real_t *dst_gmem_warp_stream_ptr = &D[gmem_idx];

 #pragma unroll
 		for (int i = 0; i < K; i++) {
 			*((int4*)(dst_gmem_warp_stream_ptr + GLOBAL_MEM_STRIDE * i) + laneId) =
 					*((int4*)(shmem_warp_stream_ptr + SHMEM_STRIDE * i) + laneId);
 		}

 		__syncthreads();
 	}
 }
template<class real_t>
__device__ real_t inline read_voter_wmma(real_t *v1, real_t *v2, real_t *v3, int offset, unsigned long long int* is_memory_bad) {
	
	
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
__global__ void simple_wmma_gemm(real_t *d0, real_t *d1, real_t *d2,
		int m_ld, int n_ld, int k_ld, real_t alpha, real_t beta) {
	

	// Leading dimensions. Packed with no transpositions.
//	int lda = m_ld;
//	int ldb = k_ld;
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
	wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, real_t> cd_frag;

	// if ((threadIdx.x | threadIdx.y ) == 0 ){
	__shared__ half_t a_shared[WMMA_M][WMMA_N];
	__shared__ half_t b_shared[WMMA_M][WMMA_N];
	__shared__ real_t c_shared[WMMA_M][WMMA_N];
	__shared__ real_t d_shared[WMMA_M][WMMA_N];

	a_shared[threadIdx.x][threadIdx.y] = half_t(2.0f);

	b_shared[threadIdx.x][threadIdx.y] = half_t(2.0f);

	c_shared[threadIdx.x][threadIdx.y] = real_t(2.0f);

	d_shared[threadIdx.x][threadIdx.y] = real_t(0.0f);
	real_t acc = 0;

	__syncthreads();
	

	for(int i = 0; i < WMMA_N; i++){
		 acc += real_t(a_shared[threadIdx.y][i] * b_shared[i][threadIdx.x]);

	}

	d_shared[threadIdx.x][threadIdx.y] = alpha * acc + beta * c_shared[threadIdx.x][threadIdx.y];

	// }
	


		wmma::fill_fragment(acc_frag, 0.0f);
		wmma::fill_fragment(a_frag, 2.0f);
		wmma::fill_fragment(b_frag, 2.0f);
		wmma::fill_fragment(c_frag, 2.0f);
		
		

		// Loop over k
		for (int i = 0; i < k_ld; i += WMMA_K) {
			int aCol = i;
			int aRow = warpM * WMMA_M;

			int bCol = i;
			int bRow = warpN * WMMA_N;

			// Bounds checking
			if (aRow < m_ld && aCol < k_ld && bRow < k_ld && bCol < n_ld) {
				// Load the inputs
	//			wmma::load_matrix_sync(a_frag, a + aCol + aRow * lda, lda);
	//			wmma::load_matrix_sync(b_frag, b + bCol + bRow * ldb, ldb);
		
				// Perform the matrix multiplication
				wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);

			}
		}

		// Load in the current value of c, scale it by beta, and add this our result scaled by alpha
		int cCol = warpN * WMMA_N;
		int cRow = warpM * WMMA_M;

		if (cRow < m_ld && cCol < n_ld) {
	//		wmma::load_matrix_sync(c_frag, c + cCol + cRow * ldc, ldc,
	//				wmma::mem_row_major);

			for (int i = 0; i < c_frag.num_elements; i++) {
				c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
			}
			error_voter(c_frag);

			// Store the output
			wmma::store_matrix_sync(d0 + cCol + cRow * ldc, c_frag, ldc,
					wmma::mem_row_major);
			// Store the output
			wmma::store_matrix_sync(d1 + cCol + cRow * ldc, c_frag, ldc,
					wmma::mem_row_major);
			// Store the output
			wmma::store_matrix_sync(d2 + cCol + cRow * ldc, c_frag, ldc,
					wmma::mem_row_major);
		}
	// }
	
	
}	

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

// template<class real_t>
// __device__ void inline error_voter (real_t d_shared, wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, real_t,
// wmma::row_major> &acc_frag){
	
// 	register real_t error_checker = d_shared - acc_frag;
// 	if (error_checker > 0) {
// 		atomicAdd(&errors, 1);
// 		return errors;
// 	}
// 	return 0;
// }

template<class real_t>
__device__ void inline error_voter (wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, real_t> &c_frag){
	
	register real_t error_checker = c_frag;
	//if (error_checker > 0) {
	//	atomicAdd(&errors, 1);		
	// }
	
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
		real_t *d1, real_t *d2, size_t mul_M, size_t mul_N, size_t mul_K, real_t alpha,
		real_t beta, unsigned long long int* is_memory_bad) {

	register int tx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
	register int ty = blockIdx.y * BLOCK_SIZE + threadIdx.y;
	register int k;

	if (tx * ty > mul_M * mul_N)
		return;

	register real_t acc = 0.0;
	for (k = 0; k < mul_N; k++) {

		half_t tmp = read_voter<half_t>(a0, a1, a2, ty * mul_N + k, is_memory_bad)
				* read_voter<half_t>(b0, b1, b2, k * mul_N + tx, is_memory_bad);
		acc = real_t(tmp) + acc;

	}

	acc = alpha * acc
			+ beta * read_voter<real_t>(c0, c1, c2, ty * mul_N + tx, is_memory_bad);

	d0[ty * mul_N + tx] = (real_t) acc;
	d1[ty * mul_N + tx] = (real_t) acc;
	d2[ty * mul_N + tx] = (real_t) acc;

}

template<class half_t, class real_t>
__global__ void matrix_mul(half_t *a0, half_t *b0, real_t *c0, real_t*d0,
		size_t mul_M, size_t mul_N, size_t mul_K, real_t alpha, real_t beta) {

	register int tx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
	register int ty = blockIdx.y * BLOCK_SIZE + threadIdx.y;
	register int k;

	if (tx * ty > mul_M * mul_N)
		return;

	register real_t acc = 0.0;
	for (k = 0; k < mul_N; k++) {
		acc = real_t(a0[ty * mul_N + k] * b0[k * mul_N + tx]) + acc;
	}

	acc = alpha * acc
			+ beta * c0[ty * mul_N + tx];

	d0[ty * mul_N + tx] = (real_t) acc;
}

#endif /* KERNELS_H_ */
