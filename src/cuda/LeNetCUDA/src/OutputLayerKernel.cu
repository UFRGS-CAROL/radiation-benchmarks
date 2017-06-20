/*
 * OutputLayerKernel.cu
 *
 *  Created on: Jun 20, 2017
 *      Author: carol
 */

#include "cudaUtil.h"

__global__ void forward_output_layer_kernel() {
	this->err = 0;
	exp_y_vec.clear();
	exp_y_vec.resize(in_depth_);
	exp_y_vec[this->exp_y] = 1;
	for (size_t i = 0; i < in_depth_; i++) {
		err += 0.5 * (exp_y_vec[i] - input_[i]) * (exp_y_vec[i] - input_[i]);
	}
	output_ = input_;
}

__global__ void backprop_output_layer_kernel() {
	/* compute err terms of output layers */
//	g_.clear();

	for (size_t i = 0; i < in_depth_; i++) {
		g_.push_back((exp_y_vec[i] - input_[i]) * df_sigmod(input_[i]));

	}
}


