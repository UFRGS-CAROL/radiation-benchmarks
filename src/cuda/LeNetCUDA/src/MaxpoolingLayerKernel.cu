/*
 * MaxpoolingLayerKernel.cu
 *
 *  Created on: Jun 6, 2017
 *      Author: carol
 */

#include "cudaUtil.h"
#include "MaxpoolingLayerKernel.h"


#define MAXPOOL_SIZE 2

__device__    inline size_t get_out_index(size_t out_width, size_t out_height,
		size_t out, size_t h_, size_t w_) {
	return out * out_width * out_height + h_ / 2 * out_width + (w_ / 2);
}


__device__ inline Pair get_max_loc_pair(size_t first, size_t second) {
	Pair ret;
	ret.first = first;
	ret.second = second;
	return ret;
}

__device__ inline float max_in_(float_t *input_, Pair *max_loc,
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
				max_loc[out_index] = get_max_loc_pair(out_index, tmp);
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
__global__ void forward_maxpool_layer_kernel(float_t *input_, Pair *max_loc,
		float_t *output_, size_t out_width, size_t out_height,
		size_t out_depth_, size_t in_height, size_t in_width) {

	int h_ = blockIdx.y * blockDim.y + threadIdx.y;
	int w_ = (blockIdx.x * blockDim.x + threadIdx.x) / out_depth_;
	int out = (blockIdx.x * blockDim.x + threadIdx.x) % out_depth_;

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


void call_forward_maxpool_layer_gpu(float_t *input, float_t *output,
		Pair *max_loc, size_t out_width, size_t out_height, size_t out_depth,
		size_t in_height, size_t in_width) {

	dim3 blocks, threads;

	cuda_gridsize(&threads, &blocks, in_width * out_depth, in_height);

	//printf("in_height %d in_width * out_depth %d threads x %d threads y %d\n", in_height, in_width * out_depth, threads.x, threads.y);

	forward_maxpool_layer_kernel<<<blocks, threads>>>(input, max_loc, output,
			out_width, out_height, out_depth, in_height, in_width);
	CudaCheckError();

}

__global__ void backpropagation_maxpool(Pair *max_loc, float *g_, float *g_next,
		size_t max_size, size_t g_max_size) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x > max_size)
		return;

	Pair p = max_loc[x];
	if (p.first != MAX && p.second != MAX && p.second < g_max_size && p.first < g_max_size) {
		g_[p.second] = g_next[p.first];
	}
}

void call_backpropagation_maxpool(Pair *max_loc, float *g_, float *g_next, size_t max_size, size_t g_max_size) {
	dim3 blocks, threads;
	cuda_gridsize(&threads, &blocks, max_size);
//	for(int i = 0; i < max_size; i++){
//		auto p = max_loc[i];
//		if(p.first != MAX && p.second == MAX){
//			std::cout << p.first << " " << p.second << " " << i << "\n";
//		}
//	}
	assert(g_max_size != 0);
	backpropagation_maxpool<<<blocks, threads>>>(max_loc, g_, g_next, max_size, g_max_size);
	CudaCheckError();
}

