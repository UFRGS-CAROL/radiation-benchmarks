/**
 * Here all kernels which were in kernels.ocl were translated
 */

#include "ConvolutionalLayerKernel.h"
#include "cudaUtil.h"
#include <cstdio>


__device__ float sigmod(float in) {
	return 1.0 / (1.0 + exp(-in));
}

__device__ inline int get_global_id(int index) {
	switch (index) {
	case 0:
		return blockIdx.x * blockDim.x + threadIdx.x;
	case 1:
		return blockIdx.y * blockDim.y + threadIdx.y;
	case 2:
		return blockIdx.z * blockDim.z + threadIdx.z;
	};
	return -1;
}

__global__ void forward_parallel(float* input_buf, float* weight_buf,
		float* b_buf, float* output_buf, int in_width, int in_height,
		int in_depth, int out_width, int out_height, int out_depth,
		int kernel_size) {

	if (get_global_id(0) > out_depth * out_width)
		return;
	if (get_global_id(1) > out_height)
		return;

	int out = get_global_id(0) / out_width;
	int w_index = get_global_id(0) % out_width;
	int h_index = get_global_id(1);

	float sum = 0;
	int size = kernel_size * kernel_size;

	for (unsigned int in = 0; in < in_depth; in++) {
		float weight_buf_sub[25]; // Set by brute force, NEED to be changed
		float input_buf_sub[25]; // Set by brute force, NEED to be changed
		// load input and weight for this sub area
		for (unsigned int y = 0; y < kernel_size; y++) {
			for (unsigned int x = 0; x < kernel_size; x++) {
				input_buf_sub[y * kernel_size + x] = input_buf[in
						* (in_width * in_height) + (h_index + y) * in_width + x
						+ w_index];
				weight_buf_sub[y * kernel_size + x] = weight_buf[in * out_depth
						* size + out * size + y * kernel_size + x];
			}
		}

		// compute the convolution
		for (unsigned int i = 0; i < size; i++) {
			sum += input_buf_sub[i] * weight_buf_sub[size - i - 1];
		}
	}

	unsigned int out_index = out * out_width * out_height + h_index * out_width
			+ w_index;
	unsigned int b_index = out_index;
	output_buf[out_index] = sigmod(sum + b_buf[b_index]);
}

void call_foward_parallel(float* input_buf, float* weight_buf, float* b_buf,
		float* output_buf, int in_width, int in_height, int in_depth,
		int out_width, int out_height, int out_depth, int kernel_size) {

	//PEDRO if these are the right dimentions
	long blocks_rows = ceil(float(out_height) / float(BLOCK_SIZE));
	long threads_rows = ceil(float(out_height) / float(blocks_rows));
	long blocks_cols = ceil(float(out_width) / float(BLOCK_SIZE));
	long threads_cols = ceil(float(out_width) / float(blocks_cols));

	dim3 blocks(blocks_rows, blocks_cols);
	dim3 threads(threads_rows, threads_cols);

	forward_parallel<<<blocks, threads>>>(input_buf,
			weight_buf, b_buf, output_buf, in_width, in_height, in_depth,
			out_width, out_height, out_depth, kernel_size);

	cudaError_t ret = cudaDeviceSynchronize();	
	CUDA_CHECK_RETURN(ret);
}
