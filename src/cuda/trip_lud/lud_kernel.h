/*
 * lud_kernel.h
 *
 *  Created on: 14/01/2019
 *      Author: fernando
 */

#ifndef LUD_KERNEL_H_
#define LUD_KERNEL_H_

#include <cuda.h>
#include <cuda_fp16.h>

#include "half.hpp"

#if PRECISION == 16
#define REAL_T_DEVICE half
#define REAL_T_HOST half_float::half
#define PRECISION_STR "Half"
#endif

#if PRECISION == 32
#define REAL_T_DEVICE float
#define REAL_T_HOST REAL_T_DEVICE
#define PRECISION_STR "Float"
#endif

#if PRECISION == 64
#define REAL_T_DEVICE double
#define REAL_T_HOST REAL_T_DEVICE
#define PRECISION_STR "Double"
#endif

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif


template<typename real_t_device>
void lud_cuda(real_t_device *m, int matrix_dim);

#define max(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a > _b ? _a : _b; })

#define min(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a < _b ? _a : _b; })



/**
 * For float and double precision
 * for half precision see lud_half_kernel.h
 */

template<typename real_t_device>
__global__ void lud_diagonal(real_t_device *m, int matrix_dim, int offset) {
	int i, j;
	__shared__ real_t_device shadow[BLOCK_SIZE][BLOCK_SIZE];

	int array_offset = offset * matrix_dim + offset;
	for (i = 0; i < BLOCK_SIZE; i++) {
		shadow[i][threadIdx.x] = m[array_offset + threadIdx.x];
		array_offset += matrix_dim;
	}
	__syncthreads();
	for (i = 0; i < BLOCK_SIZE - 1; i++) {

		if (threadIdx.x > i) {
			for (j = 0; j < i; j++)
				shadow[threadIdx.x][i] -= shadow[threadIdx.x][j] * shadow[j][i];
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
	for (i = 1; i < BLOCK_SIZE; i++) {
		m[array_offset + threadIdx.x] = shadow[i][threadIdx.x];
		array_offset += matrix_dim;
	}
}

template<typename real_t_device>
__global__ void lud_perimeter(real_t_device *m, int matrix_dim, int offset) {
	__shared__ real_t_device dia[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ real_t_device peri_row[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ real_t_device peri_col[BLOCK_SIZE][BLOCK_SIZE];

	int i, j, array_offset;
	int idx;

	if (threadIdx.x < BLOCK_SIZE) {
		idx = threadIdx.x;

		array_offset = offset * matrix_dim + offset;
		for (i = 0; i < BLOCK_SIZE / 2; i++) {
			dia[i][idx] = m[array_offset + idx];
			array_offset += matrix_dim;
		}

		array_offset = offset * matrix_dim + offset;
		for (i = 0; i < BLOCK_SIZE; i++) {
			peri_row[i][idx] = m[array_offset + (blockIdx.x + 1) * BLOCK_SIZE
					+ idx];
			array_offset += matrix_dim;
		}

	} else {
		idx = threadIdx.x - BLOCK_SIZE;

		array_offset = (offset + BLOCK_SIZE / 2) * matrix_dim + offset;
		for (i = BLOCK_SIZE / 2; i < BLOCK_SIZE; i++) {
			dia[i][idx] = m[array_offset + idx];
			array_offset += matrix_dim;
		}

		array_offset = (offset + (blockIdx.x + 1) * BLOCK_SIZE) * matrix_dim
				+ offset;
		for (i = 0; i < BLOCK_SIZE; i++) {
			peri_col[i][idx] = m[array_offset + idx];
			array_offset += matrix_dim;
		}

	}
	__syncthreads();

	/* this version works ok on hardware, but not gpgpusim
	 **************************************************************
	 if (threadIdx.x < BLOCK_SIZE) { //peri-row
	 idx=threadIdx.x;
	 for(i=1; i < BLOCK_SIZE; i++){
	 for (j=0; j < i; j++)
	 peri_row[i][idx]-=dia[i][j]*peri_row[j][idx];
	 }


	 array_offset = (offset+1)*matrix_dim+offset;
	 for(i=1; i < BLOCK_SIZE; i++){
	 m[array_offset+(blockIdx.x+1)*BLOCK_SIZE+idx] = peri_row[i][idx];
	 array_offset += matrix_dim;
	 }
	 } else { //peri-col
	 idx=threadIdx.x - BLOCK_SIZE;
	 for(i=0; i < BLOCK_SIZE; i++){
	 for(j=0; j < i; j++)
	 peri_col[idx][i]-=peri_col[idx][j]*dia[j][i];
	 peri_col[idx][i] /= dia[i][i];
	 }

	 __syncthreads();

	 array_offset = (offset+(blockIdx.x+1)*BLOCK_SIZE)*matrix_dim+offset;
	 for(i=0; i < BLOCK_SIZE; i++){
	 m[array_offset+idx] =  peri_col[i][idx];
	 array_offset += matrix_dim;
	 }
	 }
	 ***************************************************************/
	if (threadIdx.x < BLOCK_SIZE) { //peri-row
		idx = threadIdx.x;
		for (i = 1; i < BLOCK_SIZE; i++) {
			for (j = 0; j < i; j++)
				peri_row[i][idx] -= dia[i][j] * peri_row[j][idx];
		}
	} else { //peri-col
		idx = threadIdx.x - BLOCK_SIZE;
		for (i = 0; i < BLOCK_SIZE; i++) {
			for (j = 0; j < i; j++)
				peri_col[idx][i] -= peri_col[idx][j] * dia[j][i];
			peri_col[idx][i] /= dia[i][i];
		}
	}

	__syncthreads();

	if (threadIdx.x < BLOCK_SIZE) { //peri-row
		idx = threadIdx.x;
		array_offset = (offset + 1) * matrix_dim + offset;
		for (i = 1; i < BLOCK_SIZE; i++) {
			m[array_offset + (blockIdx.x + 1) * BLOCK_SIZE + idx] =
					peri_row[i][idx];
			array_offset += matrix_dim;
		}
	} else { //peri-col
		idx = threadIdx.x - BLOCK_SIZE;
		array_offset = (offset + (blockIdx.x + 1) * BLOCK_SIZE) * matrix_dim
				+ offset;
		for (i = 0; i < BLOCK_SIZE; i++) {
			m[array_offset + idx] = peri_col[i][idx];
			array_offset += matrix_dim;
		}
	}

}

template<typename real_t_device>
__global__ void lud_internal(real_t_device *m, int matrix_dim, int offset) {
	__shared__ real_t_device peri_row[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ real_t_device peri_col[BLOCK_SIZE][BLOCK_SIZE];

	int i;
	real_t_device sum;

	int global_row_id = offset + (blockIdx.y + 1) * BLOCK_SIZE;
	int global_col_id = offset + (blockIdx.x + 1) * BLOCK_SIZE;

	peri_row[threadIdx.y][threadIdx.x] = m[(offset + threadIdx.y) * matrix_dim
			+ global_col_id + threadIdx.x];
	peri_col[threadIdx.y][threadIdx.x] = m[(global_row_id + threadIdx.y)
			* matrix_dim + offset + threadIdx.x];

	__syncthreads();

	sum = 0;
	for (i = 0; i < BLOCK_SIZE; i++)
		sum += peri_col[threadIdx.y][i] * peri_row[i][threadIdx.x];
	m[(global_row_id + threadIdx.y) * matrix_dim + global_col_id + threadIdx.x] -=
			sum;

}

template<typename real_t_device>
void lud_cuda(real_t_device *m, int matrix_dim) {
	int i = 0;
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	real_t_device *m_debug = (real_t_device*) malloc(
			matrix_dim * matrix_dim * sizeof(real_t_device));

	for (i = 0; i < matrix_dim - BLOCK_SIZE; i += BLOCK_SIZE) {
		lud_diagonal<real_t_device> <<<1, BLOCK_SIZE>>>(m, matrix_dim, i);

		lud_perimeter<real_t_device> <<<(matrix_dim - i) / BLOCK_SIZE - 1, BLOCK_SIZE * 2>>>(m,
				matrix_dim, i);

		dim3 dimGrid((matrix_dim - i) / BLOCK_SIZE - 1,
				(matrix_dim - i) / BLOCK_SIZE - 1);

		lud_internal<real_t_device> <<<dimGrid, dimBlock>>>(m, matrix_dim, i);
	}
	lud_diagonal<real_t_device> <<<1, BLOCK_SIZE>>>(m, matrix_dim, i);
	cudaDeviceSynchronize();
}



#endif /* LUD_KERNEL_H_ */
