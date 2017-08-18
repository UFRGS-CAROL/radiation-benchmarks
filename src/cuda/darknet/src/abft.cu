extern "C" {
#include "abft.h"
#include <stdio.h>
}

extern "C" {

static int use_abft = 0;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort =
		true) {
	if (code != cudaSuccess) {
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
				line);
		if (abort)
			exit(code);
	}
}

void set_abft_gemm(int n) {
	use_abft = n;
}

int get_use_abft_gemm() {
	return use_abft;
}
}

__device__ long get_index(float *mat, long i, long j, long n) {
	return i * n + j;
}

__device__ error_return err_count;

__global__ void check_col(float *mat, long rows, long cols) {
	long i = blockIdx.x * blockDim.x + threadIdx.x;

	long k;
	double acc = 0;
	//must be less one
//	if (cols == 1) {
//		//acc = (mat[i * cols]);
//		return;
//	} else {
	for (k = 0; k < cols - 1; k++) {
		acc += (mat[i * cols + k]);// / DIV_VALUE);
	}
//	}
	long b_index = i * cols + cols - 1;
	//printf("b_index %ld acc %lf \n", b_index, acc);
	float diff = fabs(fabs(mat[b_index]) - fabs(acc));
	if (diff >= MAX_THRESHOLD) {
		atomicAdd(&err_count.col_detected_errors, 1);
		//printf("passou no col mat[%ld] = %ld diff %ld calc %ld i %ld\n",
		//		b_index, (long) mat[b_index], (long) diff, (long) acc, i);
	}
	//__syncthreads();
}

__global__ void check_row(float *mat, long rows, long cols) {
	long j = blockIdx.x * blockDim.x + threadIdx.x;

	long k;
	double acc = 0;
	//must be less one
//	if (rows == 1) {
//		acc = (mat[j]);
//	} else {
	for (k = 0; k < rows - 1; k++) {
		acc += (mat[k * cols + j]);// / DIV_VALUE);
	}
//	}
	//printf("a_index %ld acc %lf \n", rows_a * cols_a + j, acc);
	long a_index = (rows - 1) * cols + j;
	float diff = fabs(fabs(mat[a_index]) - fabs(acc));
	if (diff >= MAX_THRESHOLD) {
		atomicAdd(&err_count.row_detected_errors, 1);
		//printf("passou no row mat[%ld] = %lf diff %lf calc %lf i value %ld\n",
		//		a_index, mat[a_index - 1], diff, acc, j);
	}
	//__syncthreads();
}

__global__ void first_abraham_op(float *a, long rows_a, long cols_a) {
	long j = blockIdx.x * blockDim.x + threadIdx.x;

	long k;
	double acc = 0;
//	if (rows_a == 1) {
//		return;
//	} else {
	for (k = 0; k < rows_a - 1; k++) {
		acc += (a[k * cols_a + j]);// / DIV_VALUE);
	}
//	}
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
	double acc = 0;
//	if (cols_b == 1) {
//		return;
//	} else {
	for (k = 0; k < cols_b - 1; k++) {
		acc += (b[i * cols_b + k]);// / DIV_VALUE);
	}
//	}

	long b_index = i * cols_b + cols_b - 1;
	//if (i == 0)	b[1] = 9999; //pseudo fault injection

	//printf("b_index %ld acc %lf \n", b_index, acc);

	b[b_index] = acc;
}

__global__ void fault_injection(float *mat, int pos) {
	mat[pos] = (pos * 5000);
}

void calc_checksums_from_host(float *a, float *b, long rows_a, long cols_a,
		long rows_b, long cols_b) {
	//1d grid for abft operations
	long blocks = ceil(float(cols_a) / float(BLOCK_SIZE));
	long threads = ceil(float(cols_a) / float(blocks));
	if (rows_a > 1)
		first_abraham_op<<<blocks, threads>>>(a, rows_a, cols_a);
	gpuErrchk(cudaPeekAtLastError());
	//second
	blocks = ceil(float(rows_b) / float(BLOCK_SIZE));
	threads = ceil(float(rows_b) / float(blocks));
	if (cols_b > 1)
		second_abraham_op<<<blocks, threads>>>(b, rows_b, cols_b);
	gpuErrchk(cudaPeekAtLastError());
}

void check_checksums_from_host(float *c, long rows_c, long cols_c) {
	long blocks = ceil(float(cols_c) / float(BLOCK_SIZE));
	long threads = ceil(float(cols_c) / float(blocks));
	if (rows_c > 1)
		check_row<<<blocks, threads>>>(c, rows_c, cols_c);
	gpuErrchk(cudaPeekAtLastError());
	blocks = ceil(float(rows_c) / float(BLOCK_SIZE));
	threads = ceil(float(rows_c) / float(blocks));

	if (cols_c > 1)
		check_col<<<blocks, threads>>>(c, rows_c, cols_c);
	gpuErrchk(cudaPeekAtLastError());
}

extern "C" void abraham_sum(float *a, float *b, long rows_a, long cols_a,
		long rows_b, long cols_b) {
	calc_checksums_from_host(a, b, rows_a, cols_a, rows_b, cols_b);
	//fault_injection<<<1,1>>>(b, cols_b * rows_b / 100);
}

extern "C" error_return abraham_check(float *c, long rows, long cols) {
//	printf("passou why\n");
	error_return ret;
	ret.col_detected_errors = 0;
	ret.row_detected_errors = 0;

	cudaMemcpyToSymbol(err_count, &ret, sizeof(error_return));
	check_checksums_from_host(c, rows, cols);

	cudaMemcpyFromSymbol(&ret, err_count, sizeof(error_return));
	return ret;
}

__global__ void calc_checksums(float *a, float *b, long rows_a, long cols_a,
		long rows_b, long cols_b) {
	long i = blockIdx.x * blockDim.x + threadIdx.x;
	//printf("i value %ld\n", i);
	//rows
	if (i == 0) {
		//1d grid for abft operations
		long blocks = ceil(float(cols_a) / float(BLOCK_SIZE));
		long threads = ceil(float(cols_a) / float(blocks));
		first_abraham_op<<<blocks, threads>>>(a, rows_a, cols_a);
	}

	if (i == 1) {
		//second
		long blocks = ceil(float(rows_b) / float(BLOCK_SIZE));
		long threads = ceil(float(rows_b) / float(blocks));
		second_abraham_op<<<blocks, threads>>>(b, rows_b, cols_b);
	}
	__syncthreads();
}

//man, I am so lazy
__global__ void check_checksums(float *c, long rows_c, long cols_c) {
	long i = blockIdx.x * blockDim.x + threadIdx.x;
	//printf("i value %ld\n", i);
	//rows
	if (i == 0) {
		long blocks = ceil(float(cols_c) / float(BLOCK_SIZE));
		long threads = ceil(float(cols_c) / float(blocks));
		check_row<<<blocks, threads>>>(c, rows_c, cols_c);
//		printf("cols %d blocks %ld threads %ld\n", cols_c, blocks, threads);
	}
	//cols
	if (i == 1) {
		long blocks = ceil(float(rows_c) / float(BLOCK_SIZE));
		long threads = ceil(float(rows_c) / float(blocks));
		check_col<<<blocks, threads>>>(c, rows_c, cols_c);
//		printf("blocks %ld threads %ld\n", blocks, threads);
	}
	//printf("passou aqui foi\n");

	__syncthreads();
	//printf("values %d %d\n ", row_detected_errors, col_detected_errors);
}

