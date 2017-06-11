/*
 * MaxpoolingLayerKernel.cu
 *
 *  Created on: Jun 6, 2017
 *      Author: carol
 */

#include "cudaUtil.h"
#include "MaxpoolingLayerKernel.h"

__device__    inline size_t get_out_index(size_t out_width, size_t out_height,
		size_t out, size_t h_, size_t w_) {
	return out * out_width_ * out_height_ + h_ / 2 * out_width_ + (w_ / 2);
}

__device__ inline float max_in_(float_t *input_, float_t *max_loc,
		size_t in_width_, size_t in_index, size_t h_, size_t w_,
		size_t out_index) {
	float_t max_pixel = 0;
	size_t tmp;
	for (size_t x = 0; x < 2; x++) {
		for (size_t y = 0; y < 2; y++) {
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
	int out = blockIdx.x * blockDim.x + threadIdx.x;
	int w_ = blockIdx.z * blockDim.z + threadIdx.z;
	//	for (size_t out = 0; out < out_depth_; out++) {
	//		for (size_t h_ = 0; h_ < in_height_; h_ += 2) {
	//			for (size_t w_ = 0; w_ < in_width_; w_ += 2) {
	if ((out < out_depth_) && (h_ < in_height_) && (w_ < in_width_) && !(h_ % 2)
			&& !(w_ % 2)) {
		size_t index = get_OutIndex(out_width, out_height, out, h_, w_);
		output_[index] = max_in_(input_, max_loc, output_, out, h_, w_, index);

	}
}

void forward_maxpool_layer_gpu(MaxpoolingLayer l) {
	float *input_ = l.get_raw_vector(l.input_buf);
	float *output

	cudaError_t
	ret = cudaDeviceSynchronize();
	CUDA_CHECK_RETURN(ret);
}

void backward_maxpool_layer_gpu(MaxpoolingLayer l) {
//	size_t n = layer.h * layer.w * layer.c * layer.batch;

//	backward_maxpool_layer_kernel<<<cuda_gridsize(n), BLOCK>>>(n, layer.h,
//			layer.w, layer.c, layer.stride, layer.size, layer.pad,
//			layer.delta_gpu, net.delta_gpu, layer.indexes_gpu);
	cudaError_t ret = cudaDeviceSynchronize();
	CUDA_CHECK_RETURN(ret);
}
