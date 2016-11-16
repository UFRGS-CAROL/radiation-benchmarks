
extern "C" {
#include "abft.h"

}
__device__ ErrorReturn err_count;

__global__ void check_col(float *mat, long rows, long cols) {
	long i = blockIdx.x * blockDim.x + threadIdx.x;

	long k;
	float acc = 0;
	//must be less one
	for (k = 0; k < cols - 1; k++) {
		acc += mat[i * cols + k];
	}
	long b_index = i * cols + cols - 1;
	//printf("b_index %ld acc %lf \n", b_index, acc);
	float diff = fabs(fabs(mat[b_index]) - fabs(acc));
	if (diff >= MAX_THRESHOLD) {
		atomicAdd(&err_count.col_detected_errors, 1);
		//printf("passou no col mat[%ld] = %lf diff %lf read %lf calc %lf \n",
		//		b_index, mat[b_index], mat[b_index], acc, diff);
	}
	//__syncthreads();
}

__global__ void check_row(float *mat, long rows, long cols) {
	long j = blockIdx.x * blockDim.x + threadIdx.x;

	long k;
	float acc = 0;
	//must be less one
	for (k = 0; k < rows - 1; k++) {
		acc += mat[k * cols + j];
	}
	//printf("a_index %ld acc %lf \n", rows_a * cols_a + j, acc);
	long a_index = (rows - 1) * cols + j;
	float diff = fabs(fabs(mat[a_index]) - fabs(acc));
	if (diff >= MAX_THRESHOLD) {
		atomicAdd(&err_count.row_detected_errors, 1);
		//printf("passou no col mat[%ld] = %lf diff %lf read %lf calc %lf \n",
		//		a_index, mat[a_index], mat[a_index], acc, diff);
	}
	//__syncthreads();
}

//DYNAMIC PARALLELISM ONLY TO CALL NEW KERNELS, ARE FUCK KIDDING???
//man, I am so lazy
__global__ void check_checksums(float *c, long rows_c, long cols_c) {
	long i = blockIdx.x * blockDim.x + threadIdx.x;
	//printf("i value %ld\n", i);
	//rows
	if (i == 0) {
		long blocks = ceil(cols_c / float(BLOCK_SIZE));
		long threads = ceil(cols_c / float(blocks));
//		printf("passou no row\n");
		check_row<<<blocks, threads>>>(c, rows_c, cols_c);
	}
	//cols
	if (i == 1) {
//		printf("passou no col\n");
		long blocks = ceil(rows_c / float(BLOCK_SIZE));
		long threads = ceil(rows_c / float(blocks));
		check_col<<<blocks, threads>>>(c, rows_c, cols_c);
	}
	//printf("passou aqui foi\n");

	__syncthreads();
	//printf("values %d %d\n ", row_detected_errors, col_detected_errors);
}

//since dgemm is optimized for square matrices I'm going to use
//first ABRAHAM operation
//	for (j = 0; j < col_a; j++) {
//		acc = 0;
//		for (i = 0; i < lin_a; i++)
//
//			acc += a[i * col_a + j];
//
//        a[lin_a * col_a + j] = acc;
//	}
//rows_b MUST BE THE SAME OF cols_a
__global__ void first_abraham_op(float *a, long rows_a, long cols_a) {
	long j = blockIdx.x * blockDim.x + threadIdx.x;

	long k;
	float acc = 0;
	for (k = 0; k < rows_a - 1; k++) {
		acc += a[k * cols_a + j];
	}

	long a_index = (rows_a - 1) * cols_a + j;
	//printf("a_index %ld acc %lf \n", a_index, acc);
	a[a_index] = acc;
}

/**
 * 	for (i = 0; i < lin_b; i++) {
 acc = 0;
 for (j = 0; j < col_b; j++)
 acc += b[i * (col_b + 1) + j];
 //printf("i * col_b %ld col b %ld  acc %lf\n", i * col_b, col_b, acc);
 b[i * (col_b + 1) + col_b] = acc;
 }
 */
__global__ void second_abraham_op(float *b, long rows_b, long cols_b) {
	long i = blockIdx.x * blockDim.x + threadIdx.x;

	long k;
	float acc = 0;
	for (k = 0; k < cols_b - 1; k++) {
		acc += b[i * cols_b + k];
	}
	long b_index = i * cols_b + cols_b - 1;
	//if (i == 0)	b[1] = 9999; //pseudo fault injection

	//printf("b_index %ld acc %lf \n", b_index, acc);

	b[b_index] = acc;
}

__global__ void calc_checksums(float *a, float *b, long rows_a, long cols_a,
		long rows_b, long cols_b) {
	long i = blockIdx.x * blockDim.x + threadIdx.x;
	//printf("i value %ld\n", i);
	//rows
	if (i == 0) {
		//1d grid for abft operations
		long blocks_abft_first = ceil((cols_a + 1) / float(BLOCK_SIZE));
		long threads_abft_first = ceil((cols_a + 1) / float(blocks_abft_first));
		first_abraham_op<<<blocks_abft_first, threads_abft_first>>>(a,
				rows_a + 1, cols_a + 1);
	}

	if (i == 1) {
		//second
		long blocks_abft_second = ceil((rows_b + 1) / float(BLOCK_SIZE));
		long threads_abft_second = ceil(
				(rows_b + 1) / float(blocks_abft_second));
		second_abraham_op<<<blocks_abft_second, threads_abft_second>>>(b,
				rows_b + 1, cols_b + 1);
	}
	__syncthreads();
}

extern "C" void abraham_sum(float *a, float *b, long rows_a, long cols_a, long rows_b,
		long cols_b) {
	//these variables will be live only for abft
//	cudaMalloc()

	//-----------------------------------------
	calc_checksums<<<1, 2>>>(a, b, rows_a, cols_a, rows_b, cols_b);
	//gpuErrchk(cudaPeekAtLastError());
}


extern "C" ErrorReturn abraham_check(float *c, long rows, long cols) {
//	printf("passou why\n");
	ErrorReturn ret;
	ret.col_detected_errors = 0;
	ret.row_detected_errors = 0;
	cudaMemcpyToSymbol(&err_count, ret, sizeof(ErrorReturn));
	check_checksums<<<1, 2>>>(c, rows, cols);
	//gpuErrchk(cudaPeekAtLastError());

	cudaMemcpyFromSymbol(&ret, err_count,
			sizeof(ErrorReturn));
	return ret;
}

