/*
 * MaxpoolingLayerKernel.cu
 *
 *  Created on: Jun 6, 2017
 *      Author: carol
 */

#include "cudaUtil.h"
//#include "MaxpoolingLayerKernel.h"

#define MAXPOOL_SIZE 2

__device__   inline size_t get_out_index(size_t out_width, size_t out_height,
		size_t out, size_t h_, size_t w_) {
	return out * out_width * out_height + h_ / 2 * out_width + (w_ / 2);
}

__device__ inline float max_in_(float_t *input_, float_t *max_loc,
		size_t in_width_, size_t in_height_, size_t in_index, size_t h_,
		size_t w_, size_t out_index) {
	float_t max_pixel = 0;
	size_t tmp;

#pragma unroll
	for (size_t x = 0; x < MAXPOOL_SIZE; x++) {
#pragma unroll
		for (size_t y = 0; y < MAXPOOL_SIZE; y++) {
			tmp = (in_index * in_width_ * in_height_) + ((h_ + y) * in_width_)
					+ (w_ + x);
			if (max_pixel < input_[tmp]) {
				max_pixel = input_[tmp];
				max_loc[out_index] = tmp;
			}
		}
	}
	return max_pixel;
}

/**
 * void MaxpoolingLayer::forward_cpu() {
 for (size_t out = 0; out < out_depth_; out++) {
 for (size_t h_ = 0; h_ < in_height_; h_ += 2) {
 for (size_t w_ = 0; w_ < in_width_; w_ += 2) {
 output_[getOutIndex(out, h_, w_)] = max_In_(out, h_, w_,
 getOutIndex(out, h_, w_));
 }
 }
 }
 }
 */
__global__ void forward_maxpool_layer_kernel(float_t *input_, float_t *max_loc,
		float_t *output_, size_t out_width, size_t out_height,
		size_t out_depth_, size_t in_height, size_t in_width) {

	int h_ = blockIdx.y * blockDim.y + threadIdx.y;
	int w_ = blockIdx.x * blockDim.x + threadIdx.x;
	int out = blockIdx.z * blockDim.z + threadIdx.z;
	//	for (size_t out = 0; out < out_depth_; out++) {
	//		for (size_t h_ = 0; h_ < in_height_; h_ += 2) {
	//			for (size_t w_ = 0; w_ < in_width_; w_ += 2) {
	if ((out < out_depth_) && (h_ < in_height) && (w_ < in_width) && !(h_ % 2)
			&& !(w_ % 2)) {
		size_t index = get_out_index(out_width, out_height, out, h_, w_);
		output_[index] = max_in_(input_, max_loc, in_width, in_height, out, h_,
				w_, index);

	}
}

__global__ void fill(float *input) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	input[x] = x;
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

//void backward_maxpool_layer_gpu(MaxpoolingLayer l) {
////	size_t n = layer.h * layer.w * layer.c * layer.batch;
//
////	backward_maxpool_layer_kernel<<<cuda_gridsize(n), BLOCK>>>(n, layer.h,
////			layer.w, layer.c, layer.stride, layer.size, layer.pad,
////			layer.delta_gpu, net.delta_gpu, layer.indexes_gpu);
//	cudaError_t ret = cudaDeviceSynchronize();
//	CUDA_CHECK_RETURN(ret);
//}

void forward_maxpool_layer_gpu(float_t *input, float_t *output,
		float_t *max_loc, size_t out_width, size_t out_height, size_t out_depth,
		size_t in_height, size_t in_width) {

	dim3 blocks, threads;
	cuda_gridsize(&threads, &blocks, in_width, in_height, out_depth);

	forward_maxpool_layer_kernel<<<blocks, threads>>>(input, max_loc, output,
			out_width, out_height, out_depth, in_height, in_width);
	cudaError_t ret = cudaDeviceSynchronize();
	CUDA_CHECK_RETURN(ret);
}


//void forward_maxpool_layer_gpu() {
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
//	cudaMemcpy(host_input, input, bytes * in_height * in_width, cudaMemcpyDeviceToHost);
//	print_matrix(host_input, in_height, in_width);
//
//	forward_maxpool_layer_kernel<<<blocks, threads>>>(input, max_loc, output,
//			out_width, out_height, out_depth, in_height, in_width);
//
//	float host_out[out_width * out_height * out_depth];
//
//	cudaMemcpy (host_out, output, bytes * out_depth * out_height * out_width, cudaMemcpyDeviceToHost);
//
//	print_matrix(host_out, out_height, out_width);
//
//	cudaError_t ret = cudaDeviceSynchronize();
//	CUDA_CHECK_RETURN(ret);
//
//	cudaFree(input);
//	cudaFree(output);
//	cudaFree(max_loc);
//}
//
//int main() {
//	forward_maxpool_layer_gpu();
//}
//
