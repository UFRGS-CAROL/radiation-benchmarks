/*
 * OutputLayerKernel.cu
 *
 *  Created on: Jun 20, 2017
 *      Author: carol
 */

#include "cudaUtil.h"
#include "OutputLayerKernel.h"

__device__ float df_sigmod_gpu_output(float f_x) {
	return f_x * (1.0 - f_x);
}

__global__ void forward_output_layer_kernel(float *exp_y_vec, float *input_,
		float *reduce_output, float *output_, int in_depth_, int exp_y) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i > in_depth_)
		return;

//	for (size_t i = 0; i < in_depth_; i++) {
	reduce_output[i] = 0.5 * (exp_y_vec[i] - input_[i])
			* (exp_y_vec[i] - input_[i]);
//	}
	//copy that was done in the host before
	output_[i] = input_[i];
}

void call_forward_output_layer(float *err, float *exp_y_vec, float *input_,
		float *reduce_output, float *output_, int in_depth_, int exp_y) {
	//hahahah nvidia managed
	exp_y_vec[exp_y] = 1;

	dim3 blocks, threads;
	cuda_gridsize(&threads, &blocks, in_depth_);

	forward_output_layer_kernel<<<blocks, threads>>>(exp_y_vec, input_,
			reduce_output, output_, in_depth_, exp_y);

	CudaCheckError();
	*err = 0;
	for (int i = 0; i < in_depth_; i++) {
		*err += reduce_output[i];
	}
}

__global__ void backprop_output_layer_kernel(float *exp_y_vec, float *input_,
		float *g_, int in_depth_) {
	/* compute err terms of output layers */
//	g_.clear();
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i > in_depth_)
		return;
//	for (size_t i = 0; i < in_depth_; i++) {
	g_[i] = ((exp_y_vec[i] - input_[i]) * df_sigmod_gpu_output(input_[i]));

//	}
}

void call_backpropagation_output_layer(float *exp_y_vec, float *input_,
		float *g_, int in_depth_) {
	dim3 blocks, threads;
	cuda_gridsize(&threads, &blocks, in_depth_);
	backprop_output_layer_kernel<<<blocks, threads>>>(exp_y_vec, input_, g_,
			in_depth_);
	CudaCheckError();
}
