/*
 * lud_half_kernel.h
 *
 *  Created on: 16/01/2019
 *      Author: fernando
 */

#ifndef LUD_HALF_KERNEL_H_
#define LUD_HALF_KERNEL_H_

#include "lud_kernel.h"

/**
 * For float and double precision
 * for half precision see lud_half_kernel.h
 */
__global__ void lud_diagonal(half *m, int matrix_dim, int offset) {
	int i, j;
	//We need half number of threads
	const int half_block_size = BLOCK_SIZE / 2;
	__shared__ half2 shadow[half_block_size][half_block_size];

	int array_offset = offset * matrix_dim + offset;

	if (threadIdx.x % 2 == 0) {
		for (i = 0; i < half_block_size; i++) {
//		shadow[i][threadIdx.x] = m[array_offset + threadIdx.x];
			shadow[i][threadIdx.x] = __halves2half2(
					m[array_offset + threadIdx.x],
					m[array_offset + threadIdx.x + 1]);

			array_offset += matrix_dim;
		}
		__syncthreads();

		for (i = 0; i < half_block_size - 1; i++) {

			if (threadIdx.x > i) {
				for (j = 0; j < i; j++) {
					shadow[threadIdx.x][i] -= shadow[threadIdx.x][j]
							* shadow[j][i];
				}
				shadow[threadIdx.x][i] /= shadow[i][i];
			}

			__syncthreads();
			if (threadIdx.x > i) {

				for (j = 0; j < i + 1; j++)
					shadow[i + 1][threadIdx.x] -= shadow[i + 1][j]
							* shadow[j][threadIdx.x];
			}
			__syncthreads();
		}

		/*
		 The first row is not modified, it
		 is no need to write it back to the
		 global memory

		 */
		array_offset = (offset + 1) * matrix_dim + offset;
		for (i = 1; i < half_block_size; i++) {
			m[array_offset + threadIdx.x] = __low2half(shadow[i][threadIdx.x]);
			m[array_offset + threadIdx.x + 1] = __high2half(
					shadow[i][threadIdx.x]);

			array_offset += matrix_dim;
		}
	}
}

#endif /* LUD_HALF_KERNEL_H_ */
