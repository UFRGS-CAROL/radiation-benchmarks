/*
 * OutputLayerKernel.cu
 *
 *  Created on: Jun 20, 2017
 *      Author: carol
 */

#include "cudaUtil.h"

__global__ void forward_output_layer_kernel(float *err, float *exp_y_vec, float *input_, int in_depth_, int exp_y) {
	*err = 0;
//	FERNANDO CHECK IT
//	exp_y_vec.clear();
//	exp_y_vec.resize(in_depth_);

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i > in_depth_)
		return;

	if (i == 0)
		exp_y_vec[exp_y] = 1;
	__syncthreads();

//	for (size_t i = 0; i < in_depth_; i++) {
	*err += 0.5 * (exp_y_vec[i] - input_[i]) * (exp_y_vec[i] - input_[i]);
//	}
	output_ = input_;
}

__global__ void backprop_output_layer_kernel() {
	/* compute err terms of output layers */
//	g_.clear();

	for (size_t i = 0; i < in_depth_; i++) {
		g_.push_back((exp_y_vec[i] - input_[i]) * df_sigmod(input_[i]));

	}
}



void call_forward_output_layer(){

}
