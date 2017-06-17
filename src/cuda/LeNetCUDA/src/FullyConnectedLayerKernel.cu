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


float_t dot(vec_host x, vec_host w) {
	assert(x.size() == w.size());
	float_t sum = 0;
	for (size_t i = 0; i < x.size(); i++) {
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
__global__ void get_W(int index, float *W_, float *v_output, int in_depth_){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i > in_depth_)
		return;
	v_output[i] = W_[index * in_depth_ + i];
}

__global__ void forward_gpu_kernel(float *output_, float *input_, float *b_,
		int max_size) {

	int out = blockIdx.x * blockDim.x + threadIdx.x;

	if (out > max_size)
		return;

//	for (size_t out = 0; out < out_depth_; out++) {
	vec_host get_W_result =  get_W(out);
	float dot_result = dot(input_, get_W_result);

	output_[out] = sigmod(dot_result + b_[out]);
//	}
}

void call_forward_fully_connected() {

}
