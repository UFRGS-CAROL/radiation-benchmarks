#include <stdio.h>

#include "lud_kernel.h"
#include "lud_half_kernel.h"

/**
 * For float and double precision
 * for half precision see lud_half_kernel.h
 */

template<typename real_t>
__global__ void lud_diagonal(real_t *m, int matrix_dim, int offset) {
	int i, j;
	__shared__ real_t shadow[BLOCK_SIZE][BLOCK_SIZE];

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

template<typename real_t>
__global__ void lud_perimeter(real_t *m, int matrix_dim, int offset) {
	__shared__ real_t dia[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ real_t peri_row[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ real_t peri_col[BLOCK_SIZE][BLOCK_SIZE];

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

template<typename real_t>
__global__ void lud_internal(real_t *m, int matrix_dim, int offset) {
	__shared__ real_t peri_row[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ real_t peri_col[BLOCK_SIZE][BLOCK_SIZE];

	int i;
	float sum;

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

template<typename real_t>
void lud_cuda(real_t *m, int matrix_dim) {
	int i = 0;
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	real_t *m_debug = (real_t*) malloc(
			matrix_dim * matrix_dim * sizeof(real_t));

	for (i = 0; i < matrix_dim - BLOCK_SIZE; i += BLOCK_SIZE) {
		lud_diagonal<real_t> <<<1, BLOCK_SIZE>>>(m, matrix_dim, i);

		lud_perimeter<real_t> <<<(matrix_dim - i) / BLOCK_SIZE - 1, BLOCK_SIZE * 2>>>(m,
				matrix_dim, i);

		dim3 dimGrid((matrix_dim - i) / BLOCK_SIZE - 1,
				(matrix_dim - i) / BLOCK_SIZE - 1);

		lud_internal<real_t> <<<dimGrid, dimBlock>>>(m, matrix_dim, i);
	}
	lud_diagonal<real_t> <<<1, BLOCK_SIZE>>>(m, matrix_dim, i);
	cudaDeviceSynchronize();
}


void lud_cuda_float(float *m, int matrix_dim){
	lud_cuda<float>(m, matrix_dim);
}

void lud_cuda_double(double *m, int matrix_dim){
	lud_cuda<double>(m, matrix_dim);
}
//
//void lud_cuda_half(half *m, int matrix_dim){
//	lud_cuda<half>(m, matrix_dim);
//}
