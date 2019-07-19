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

#define M_O 8192
#define N_O 8192
#define K_O 8192

#define LDA 8192
#define LDB 8192
#define LDC 8192

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

__device__ __forceinline__ void fma__(const double a, const double b,
    float& c) {
  c = __fmaf_rn(__double2float_rn(a), __double2float_rn(b), c);
}

__device__ __forceinline__ void fma__(const float a, const float b,
    float& c) {
  c = __fmaf_rn(a, b, c);
}


__device__ __forceinline__ void fma__(const double a, const double b,
    double& c) {
  c = __fma_rn(a, b, c);
}

template<typename real_t, typename half_real_t>
__device__ void saxpy(real_t a, real_t *b, real_t *c, half_real_t *c_inc) {
  fma__(a, b[0], c[0]);
  fma__(a, b[1], c[1]);
  fma__(a, b[2], c[2]);
  fma__(a, b[3], c[3]);
  fma__(a, b[4], c[4]);
  fma__(a, b[5], c[5]);
  fma__(a, b[6], c[6]);
  fma__(a, b[7], c[7]);
  fma__(a, b[8], c[8]);
  fma__(a, b[9], c[9]);
  fma__(a, b[10], c[10]);
  fma__(a, b[11], c[11]);
  fma__(a, b[12], c[12]);
  fma__(a, b[13], c[13]);
  fma__(a, b[14], c[14]);
  fma__(a, b[15], c[15]);

  fma__(a, b[0], c_inc[0]);
  fma__(a, b[1], c_inc[1]);
  fma__(a, b[2], c_inc[2]);
  fma__(a, b[3], c_inc[3]);
  fma__(a, b[4], c_inc[4]);
  fma__(a, b[5], c_inc[5]);
  fma__(a, b[6], c_inc[6]);
  fma__(a, b[7], c_inc[7]);
  fma__(a, b[8], c_inc[8]);
  fma__(a, b[9], c_inc[9]);
  fma__(a, b[10], c_inc[10]);
  fma__(a, b[11], c_inc[11]);
  fma__(a, b[12], c_inc[12]);
  fma__(a, b[13], c_inc[13]);
  fma__(a, b[14], c_inc[14]);
  fma__(a, b[15], c_inc[15]);
}

  //OPTIMIZED GEMM USING TENSOR CORES, UNHARDENED  
  template<class half_t, class real_t>
 __global__ void op_tensor_gemm(const half_t *A, const half_t *B, const real_t *C, real_t *D, float alpha, float beta)
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

// DMR - OPTIMIZED GEMM USING TENSOR CORES + GEMM SW
// D - tensor output
// d - sw output

template<class half_t, class real_t>
 __global__ void op_tensor_gemm_DMR( half_t *A, half_t *B,  real_t *C, real_t *D, real_t *d, float alpha, float beta)
 {
 	// Block index
	  int bx = blockIdx.x;
  	int by = blockIdx.y;

  	// Thread index
  	int tx = threadIdx.x;
  	int ty = threadIdx.y;

  	// Index of the first sub-matrix of A processed by the block
  	int aBegin = M * BLOCK_SIZE * by;

  	// Index of the last sub-matrix of A processed by the block
  	int aEnd   = aBegin + M - 1;



  	// Step size used to iterate through the sub-matrices of A
  	int aStep  = BLOCK_SIZE;

  	// Index of the first sub-matrix of B processed by the block
  	int bBegin = BLOCK_SIZE * bx;

  	// Step size used to iterate through the sub-matrices of B
  	int bStep  = BLOCK_SIZE * N;



 	//volatile half_t Csub = 0;
  	half_t Csub = 0;
	// Loop over all the sub-matrices of A and B
  	// required to compute the block sub-matrix
  	for (int a = aBegin, b = bBegin; a <= aEnd;  a += aStep, b += bStep) {
    

    	__shared__ half_t As[BLOCK_SIZE][BLOCK_SIZE];

    	__shared__ half_t Bs[BLOCK_SIZE][BLOCK_SIZE];

    	As[ty][tx] = A[a + M * ty + tx];
    	Bs[ty][tx] = B[b + N * ty + tx];

    	// Synchronize to make sure the matrices are loaded
    	__syncthreads();

	#pragma unroll

    	for (int k = 0; k < BLOCK_SIZE; ++k) {
      	
      		Csub = fma__(As[ty][k], Bs[k][tx],Csub);
    	}

   	 	// Synchronize to make sure that the preceding
   	 	// computation is done before loading two new
    	// sub-matrices of A and B in the next iteration
    	__syncthreads();
  	}
  	Csub = fma__((Csub*(half_t)alpha), (Csub*(half_t)beta), Csub);
  	// Write the block sub-matrix to device memory;
  	// each thread writes one element
  	int c_p = N * BLOCK_SIZE * by + BLOCK_SIZE * bx;
  	d[c_p + N * ty + tx] = Csub;

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



__device__ __forceinline__ double abs_(double a) {
        return fabs(a);
}

__device__ __forceinline__ float abs_(float a) {
        return fabsf(a);
}

__device__    __forceinline__ half abs_(half a) {
        return fabsf(a);
}




//TRIPLICATE SIMPLE GEMM USING TENSOR CORES
template<class half_t, class real_t>
__global__ void s_tensor_gemm_triplicate(real_t *d0, real_t *d1, real_t *d2,
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
	
	
}

//UNHARDENED SIMPLE GEMM USING TENSOR CORES
template<class half_t, class real_t>
__global__ void s_tensor_gemm(half_t *a, half_t *b, real_t *c, real_t *d,
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

//DMR - SIMPLE GEMM USING TENSOR CORES + GEMM SW
template<class half_t, class real_t>
__global__ void s_tensor_gemm_DMR(half_t *a, half_t *a1, half_t *b, real_t *c, half_t *d, real_t *d_frag, 
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
	// Block index
	int bx = blockIdx.x;
  	int by = blockIdx.y;

  	// Thread index
  	int tx = threadIdx.x;
  	int ty = threadIdx.y;

  	// Index of the first sub-matrix of A processed by the block
  	int aBegin = m_ld * BLOCK_SIZE * by;

  	// Index of the last sub-matrix of A processed by the block
  	int aEnd   = aBegin + m_ld - 1;



  	// Step size used to iterate through the sub-matrices of A
  	int aStep  = BLOCK_SIZE;

  	// Index of the first sub-matrix of B processed by the block
  	int bBegin = BLOCK_SIZE * bx;

  	// Step size used to iterate through the sub-matrices of B
  	int bStep  = BLOCK_SIZE * n_ld;



 	volatile half_t Csub = 0;
  
	// Loop over all the sub-matrices of A and B
  	// required to compute the block sub-matrix
  	for (int A = aBegin, B = bBegin; A <= aEnd;  A += aStep, B += bStep) {
    

    	__shared__ half_t As[BLOCK_SIZE][BLOCK_SIZE];

    	__shared__ half_t Bs[BLOCK_SIZE][BLOCK_SIZE];

    	As[ty][tx] = a1[A + m_ld * ty + tx];
    	Bs[ty][tx] = b[B + n_ld * ty + tx];

    	// Synchronize to make sure the matrices are loaded
    	__syncthreads();

	#pragma unroll

    	for (int k = 0; k < BLOCK_SIZE; ++k) {
      	
      		Csub = fma__(As[ty][k], Bs[k][tx],Csub);
    	}

   	 	// Synchronize to make sure that the preceding
   	 	// computation is done before loading two new
    	// sub-matrices of A and B in the next iteration
    	__syncthreads();
  	}

  	// Write the block sub-matrix to device memory;
  	// each thread writes one element
  	int c_p = n_ld * BLOCK_SIZE * by + BLOCK_SIZE * bx;
  	d[c_p + n_ld * ty + tx] = Csub;
 

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

		
		// for(int i = 0; i < WMMA_N; i++)
		// 	for(int j = 0; j < WMMA_M; j++){
		// 		register real_t error_checker = abs_(d_shared[i][j] - acc_frag.x[i * WMMA_M + j]);
		// 		if (error_checker > real_t(0.0)) {
		// 			atomicAdd(&errors, 1);
		// 			//return errors;
		// 		}
		// }			

		// Store the output
		wmma::store_matrix_sync(d_frag + cCol + cRow * ldc, c_frag, ldc,
				wmma::mem_row_major);
	}
}



//DMR KERNEL
template<typename real_t, typename half_real_t>
__global__ void s_gemm_DMR(half_real_t* C_inc, real_t *C, const real_t *A, const real_t *B, int m,
    int n, int k, int lda, int ldb, int ldc, real_t alpha, real_t beta) {

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;

  const int ibx = blockIdx.x * 64;
  const int iby = blockIdx.y * 16;

  const int idt = ty * 16 + tx;

  B += tx + __mul24(iby + ty, ldb);
  A += ibx + idt;
  C += ibx + idt + __mul24(iby, ldc);

  C_inc += ibx + idt + __mul24(iby, ldc);

  const real_t *Bend = B + k;

  real_t Cb[16] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
  half_real_t Cb_inc[16] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

  m = 2 * lda;
  n = 3 * lda;

  do {
    //float Ab[4] = {A[0], A[lda], A[2*lda], A[3*lda]};
    real_t Ab[4] = { A[0], A[lda], A[m], A[n] };
    __shared__ real_t Bb[16][17];
    Bb[tx][ty + 0] = B[0];
    Bb[tx][ty + 4] = B[4 * ldb];
    Bb[tx][ty + 8] = B[8 * ldb];
    Bb[tx][ty + 12] = B[12 * ldb];

    __syncthreads();

    A += 4 * lda;
    saxpy(Ab[0], &Bb[0][0], Cb, Cb_inc);

    Ab[0] = A[0];
    saxpy(Ab[1], &Bb[1][0], Cb, Cb_inc);

    Ab[1] = A[lda];
    saxpy(Ab[2], &Bb[2][0], Cb, Cb_inc);

    Ab[2] = A[m];
    saxpy(Ab[3], &Bb[3][0], Cb, Cb_inc);
    Ab[3] = A[n];

    A += 4 * lda;
    saxpy(Ab[0], &Bb[4][0], Cb, Cb_inc);

    Ab[0] = A[0];
    saxpy(Ab[1], &Bb[5][0], Cb, Cb_inc);

    Ab[1] = A[lda];
    saxpy(Ab[2], &Bb[6][0], Cb, Cb_inc);

    Ab[2] = A[m];
    saxpy(Ab[3], &Bb[7][0], Cb, Cb_inc);

    Ab[3] = A[n];

    A += 4 * lda;
    saxpy(Ab[0], &Bb[8][0], Cb, Cb_inc);

    Ab[0] = A[0];
    saxpy(Ab[1], &Bb[9][0], Cb, Cb_inc);

    Ab[1] = A[lda];
    saxpy(Ab[2], &Bb[10][0], Cb, Cb_inc);

    Ab[2] = A[m];
    saxpy(Ab[3], &Bb[11][0], Cb, Cb_inc);

    Ab[3] = A[n];

    A += 4 * lda;
    saxpy(Ab[0], &Bb[12][0], Cb, Cb_inc);

    saxpy(Ab[1], &Bb[13][0], Cb, Cb_inc);

    saxpy(Ab[2], &Bb[14][0], Cb, Cb_inc);

    saxpy(Ab[3], &Bb[15][0], Cb, Cb_inc);

    B += 16;

    __syncthreads();
  } while (B < Bend);

  half_real_t alpha_inc = half_real_t(alpha);
  half_real_t beta_inc = half_real_t(beta);
#pragma unroll 16
  for (int i = 0; i < 16; i++, C += ldc, C_inc += ldc) {
    C[0] = alpha * Cb[i] + beta * C[0];
    C_inc[0] =  alpha_inc * Cb_inc[i] + beta_inc * C_inc[0];
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



// __device__ __forceinline__ float mul_(float a, float b ) {
//         return __fmul_ru(a,b);
// }

// __device__    __forceinline__ half mul_(half a, half b) {
//         return __hmul(a, b);
// }

// __device__ __forceinline__ float sum_(float a, float b ) {
//         return __fadd_ru(a, b);
// }

// __device__    __forceinline__ half sum_(half a, half b) {
//         return __hadd(a, b);
// }


// __device__ __forceinline__ float fma_(half a, half b, float c ) {
//         return fmaf(a, b,c);
// }

// __device__    __forceinline__ half fma_(half a, half b, half c) {
//         return __hfma(a, b, c);
// }


#endif /* KERNELS_H_ */
