/*
 * FullyConnectedLayerKernel.cu
 *
 *  Created on: Jun 6, 2017
 *      Author: carol
 */

#include "FullyConnectedLayerKernel.h"
#include "cudaUtil.h"

__device__ float_t sigmod(float_t in) {
	return 1.0 / (1.0 + exp(-in));
}


__device__ float dot(float *x, int x_size, float *w) {
//	assert(x.size() == w.size());
	float sum = 0;
#pragma unroll
	for (int i = 0; i < x_size; i++) {
		sum += x[i] * w[i];
	}
	return sum;
}

/*
 * original function
vec_host get_W(size_t index) {
	vec_host v;
	for (int i = 0; i < in_depth_; i++) {
		v.push_back(W_[index * in_depth_ + i]);
	}
	return v;
}
*/
__device__ void get_W(int index, int in_depth_, float *W_, float *v_output){
#pragma unroll
	for (int i = 0; i < in_depth_; i++){
		v_output[i] = W_[index * in_depth_ + i];
	}
}

__global__ void forward_gpu_kernel(float *output_, float *input_, float *b_,
		float **v_output, int max_size, int in_depth_) {

	int out = blockIdx.x * blockDim.x + threadIdx.x;

	if (out > max_size)
		return;

//	for (size_t out = 0; out < out_depth_; out++) {
	get_W(out, in_depth_, W_, v_output[out]);
	float dot_result = dot(input_, v_output[out]);

	output_[out] = sigmod(dot_result + b_[out]);
//	}
}

void call_forward_fully_connected() {

}
