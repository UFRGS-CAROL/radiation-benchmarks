/*
 * common.h
 *
 *  Created on: 11/10/2019
 *      Author: fernando
 */

#ifndef COMMON_H_
#define COMMON_H_

#include <stdexcept>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif

#define SUB_ABS(lhs, rhs) ((lhs > rhs) ? (lhs - rhs) : (rhs - lhs))

#define ZERO_FULL 1e-13

#ifndef ZERO_FLOAT
#define ZERO_FLOAT 2.2e-05
#endif

#ifndef ZERO_HALF
#define ZERO_HALF 4.166E-4
#endif

#ifndef ZERO_DOUBLE
#define ZERO_DOUBLE 1.0e-15
#endif

// Ninety percent
#define MIN_PERCENTAGE 0.85f

// Hundred percent
// I keep 10% in each direction 0.9 to 1.1
#define MAX_PERCENTAGE 1.15f

//Threshold for one operation
#define THRESHOLD_1 26

//Threshold for 16 operation
#define THRESHOLD_31 40

//Threshold for one operation
#define THRESHOLD_32 100

void exception(std::string msg, std::string file, int line);

#define throw_line(msg) exception(msg, __FILE__, __LINE__)

#define WARP_SIZE 32

#define WARPS_PER_BLOCK 8
#define THREADS_PER_BLOCK (WARP_SIZE * WARPS_PER_BLOCK)

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

#define BLOCK_ROW_WARPS 2
#define BLOCK_COL_WARPS 4

#define WARP_ROW_TILES 4
#define WARP_COL_TILES 2

#define BLOCK_ROW_TILES (WARP_ROW_TILES * BLOCK_ROW_WARPS)
#define BLOCK_COL_TILES (WARP_COL_TILES * BLOCK_COL_WARPS)

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

// The macro below is used to shift rows of the A matrix and columns of the B matrix
// in shared memory to minimize possible bank conflicts.
// Before performing the nvcuda::wmma::mma_sync operation, the warp must load the matrix
// data using the nvcuda::wmma::load_matrix_sync operation. Although the memory access pattern
// is not specified for that function, each lane in the warp can read one or multiple matrix
// elements from different matrix rows or columns.
// For shared memory, such access can result in bank conflicts if different rows / columns
// of the matrix map to the same bank. By shifting each row and column by a few bytes, we
// make sure that they map to different banks, thus reducing the number of possible bank
// conflicts.
// The number of 8 two-byte "half" elements is chosen as the minimum possible shift because
// we must keep each row and column 128-bit aligned, as required by nvcuda::wmma::load_matrix_sync.
#define SKEW_HALF 8
// GPU configuration.

#endif /* COMMON_H_ */
