/*
 * cudaUtil.cpp
 *
 *  Created on: Jun 6, 2017
 *      Author: carol
 */

#include "cudaUtil.h"


__global__ void fill(float *input, float t) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	input[x] = t;
}


//1D=(65536), 2D=(65536, 65536), 3D=(4096, 4096, 4096)
void cuda_gridsize(dim3 *threads, dim3 *blocks, size_t x, size_t y,
		size_t z) {

	long blocks_x = ceil(float(x) / float(BLOCK_SIZE));
	long threads_x = ceil(float(x) / float(blocks_x));
	long blocks_y = ceil(float(y) / float(BLOCK_SIZE));
	long threads_y = ceil(float(y) / float(blocks_y));
	long blocks_z = ceil(float(z) / float(BLOCK_SIZE));
	long threads_z = ceil(float(z) / float(blocks_z));

	*blocks = dim3(blocks_x, blocks_y, blocks_z);
	*threads = dim3(threads_x, threads_y, threads_z);

	printf("b_x %d b_y %d b_z %d\nt_x %d t_y %d t_z %d\n", blocks->x, blocks->y,
			blocks->z, threads->x, threads->y, threads->z);

}



__device__ float sigmod_gpu(float in) {
	return 1.0 / (1.0 + exp(-in));
}

__device__ float df_sigmod_gpu(float f_x) {
	return f_x * (1.0 - f_x);
}


__device__ float dot_gpu(float *x, int x_size, float *w) {
//	assert(x.size() == w.size());
	float sum = 0;
#pragma unroll
	for (int i = 0; i < x_size; i++) {
		sum += x[i] * w[i];
	}
	return sum;
}



//Copied from Ugo Varetto github
//SEE https://github.com/ugovaretto/cuda-training/blob/master/src/004_3_parallel-dot-product-atomics-portable-optimized.cu
//for more information
//------------------------------------------------------------------------------

//Full on-gpu reduction

// each block atomically increments this variable when done
// performing the first reduction step
__device__ unsigned int count = 0;
// shared memory used by partial_dot and sum functions
// for temporary partial reductions; declare as global variable
// because used in more than one function
__shared__ float cache[BLOCK_M];

// partial dot product: each thread block produces a single value
__device__ float partial_dot(const float* v1, const float* v2, int N,
		float* out) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= N)
		return float(0);
	cache[threadIdx.x] = 0.f;
	// the threads in the thread block iterate over the entire domain; iteration happens
	// whenever the total number of threads is lower than the domain size
	while (i < N) {
		cache[threadIdx.x] += v1[i] * v2[i];
		i += gridDim.x * blockDim.x;
	}
	__syncthreads(); // required because later on the current thread is accessing
					 // data written by another thread
	i = BLOCK_M / 2;
	while (i > 0) {
		if (threadIdx.x < i)
			cache[threadIdx.x] += cache[threadIdx.x + i];
		__syncthreads();
		i /= 2;
	}
	return cache[0];
}

// sum all elements in array; array size assumed to be equal to number of blocks
__device__ float sum(const float* v) {
	cache[threadIdx.x] = 0.f;
	int i = threadIdx.x;
	// the threads in the thread block iterate oevr the entire domain
	// of size == gridDim.x == total number of blocks; iteration happens
	// whenever the number of threads in a thread block is lower than
	// the total number of thread blocks
	while (i < gridDim.x) {
		cache[threadIdx.x] += v[i];
		i += blockDim.x;
	}
	__syncthreads(); // required because later on the current thread is accessing
					 // data written by another thread
	i = BLOCK_M / 2;
	while (i > 0) {
		if (threadIdx.x < i)
			cache[threadIdx.x] += cache[threadIdx.x + i];
		__syncthreads();
		i /= 2;
	}
	return cache[0];
}

// perform parallel dot product in two steps:
// 1) each block computes a single value and stores it into an array of size == number of blocks
// 2) the last block to finish step (1) performs a reduction on the array produced in the above step
// parameters:
// v1 first input vector
// v2 second input vector
// N  size of input vector
// out output vector: size MUST be equal to the number of GPU blocks since it us used
//     for partial reduction; result is at position 0
__global__ void full_dot(const float* v1, const float* v2, int N, float* out) {
	// true if last block to compute value
	__shared__ bool lastBlock;
	// each block computes a value
	float r = partial_dot(v1, v2, N, out);
	if (threadIdx.x == 0) {
		// value is stored into output array by first thread of each block
		out[blockIdx.x] = r;
		// wait for value to be available to all the threads on the device
		__threadfence();
		// increment atomic counter and retrieve value
		const unsigned int v = atomicInc(&count, gridDim.x);
		// check if last block to perform computation
		lastBlock = (v == gridDim.x - 1);
	}
	// the code below is executed by *all* threads in the block:
	// make sure all the threads in the block access the correct value
	// of the variable 'lastBlock'
	__syncthreads();

	// last block performs a the final reduction steps which produces one single value
	if (lastBlock) {
		r = sum(out);
		if (threadIdx.x == 0) {
			out[0] = r;
			count = 0;
		}
	}
}
