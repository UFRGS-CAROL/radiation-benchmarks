/*
 * debug.cu
 *
 *  Created on: Jun 20, 2017
 *      Author: carol
 *
 *      this file is only for debugging
 *      classes
 */
#include <iostream>
#include <vector>
#include <cstdio>

//Copied from Ugo Varetto github
//SEE https://github.com/ugovaretto/cuda-training/blob/master/src/004_3_parallel-dot-product-atomics-portable-optimized.cu
//for more information
const size_t BLOCK_SIZE = 16;

//------------------------------------------------------------------------------

//Full on-gpu reduction

// each block atomically increments this variable when done
// performing the first reduction step
__device__ unsigned int count = 0;
// shared memory used by partial_dot and sum functions
// for temporary partial reductions; declare as global variable
// because used in more than one function
__shared__ float cache[BLOCK_SIZE];

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
	i = BLOCK_SIZE / 2;
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
	i = BLOCK_SIZE / 2;
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

__global__ void fill(float *input) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	input[x] = x;
}

// initialization function run on the GPU
__global__ void init_vector(float* v, int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	while (i < N) {
		v[i] = 1.0f;	//float( i ) / 1000000.f;
		i += gridDim.x * blockDim.x;
	}
}

// cpu implementation of dot product
float dot(const float* v1, const float* v2, int N) {
	float s = 0;
	for (int i = 0; i != N; ++i) {
		s += v1[i] * v2[i];
	}
	return s;
}

void print_matrix(float *m, size_t h, size_t w) {
	printf("matxix\n");
	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			printf("%f ", m[i * w + j]);
		}
		printf("\n");
	}

}

void test_dot_product() {
	const size_t ARRAY_SIZE = 1024;	//1024 * 1024; //1Mi elements
	const int BLOCKS = 64;	//512;
	const int THREADS_PER_BLOCK = BLOCK_SIZE;//256; // total threads = 512 x 256 = 128ki threads;
	const size_t SIZE = ARRAY_SIZE * sizeof(float);
	float *dev_v1;
	float *dev_v2; // vector 2
	float* dev_out; // result array, final result is at position 0;
	cudaMallocManaged(&dev_v1, SIZE);
	cudaMallocManaged(&dev_v2, SIZE);
	cudaMallocManaged(&dev_out, sizeof(float) * BLOCKS);

	// host storage
	std::vector<float> host_v1(ARRAY_SIZE);
	std::vector<float> host_v2(ARRAY_SIZE);

	init_vector<<<1024, 256>>>(dev_v1, ARRAY_SIZE);
	cudaMemcpy(&host_v1, dev_v1, SIZE, cudaMemcpyDeviceToHost);

	// initialize vector 2 with kernel; much faster than using for loops on the cpu
	init_vector<<<1024, 256>>>(dev_v2, ARRAY_SIZE);
	cudaMemcpy(&host_v2, dev_v2, SIZE, cudaMemcpyDeviceToHost);

	full_dot<<<BLOCKS, THREADS_PER_BLOCK>>>(dev_v1, dev_v2, ARRAY_SIZE,
			dev_out);
	std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl;

	std::cout << "GPU: " << dev_out[0] << std::endl;
	std::cout << "CPU: " << dot(&host_v1[0], &host_v2[0], ARRAY_SIZE) << std::endl;
	free(dev_v1);
	free(dev_v2);
	free(dev_out);
}

void forward_maxpool_layer_gpu() {
//
////
//	size_t out_width = 2;
//	size_t out_height = 2;
//	size_t out_depth = 1;
//	size_t in_height = 8;
//	size_t in_width = 8;
//	size_t bytes = sizeof(float);
//
//	float *input, *output, *max_loc;
//	cudaMalloc(&input, bytes * in_height * in_width);
//	cudaMalloc(&output, bytes * out_depth * out_height * out_width);
//	cudaMalloc(&max_loc, bytes * in_height * in_width);
//
//	dim3 blocks, threads;
//	cuda_gridsize(&threads, &blocks, in_width, in_height, out_depth);
//
//	//fill first
//	fill<<<1, in_height * in_width>>>(input);
//
//	float host_input[in_height * in_width];
//	cudaMemcpy(host_input, input, bytes * in_height * in_width,
//			cudaMemcpyDeviceToHost);
//	print_matrix(host_input, in_height, in_width);
//
//	forward_maxpool_layer_kernel<<<blocks, threads>>>(input, max_loc, output,
//			out_width, out_height, out_depth, in_height, in_width);
//
//	float host_out[out_width * out_height * out_depth];
//
//	cudaMemcpy(host_out, output, bytes * out_depth * out_height * out_width,
//			cudaMemcpyDeviceToHost);
//
//	print_matrix(host_out, out_height, out_width);
//
//	cudaError_t ret = cudaDeviceSynchronize();
//	CUDA_CHECK_RETURN(ret);
//
//	cudaFree(input);
//	cudaFree(output);
//	cudaFree(max_loc);
}

int main(int argc, char **argv) {

	std::string opt(argv[1]);
	test_dot_product();
//	if (opt == "maxpool") {
//		forward_maxpool_layer_gpu();
//	} else if (opt == "device_vector") {
//
//	}

	return 0;
}

