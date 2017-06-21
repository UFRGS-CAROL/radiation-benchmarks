/*
 * OutputLayerKernel.cu
 *
 *  Created on: Jun 20, 2017
 *      Author: carol
 */

#include "cudaUtil.h"

__global__ void forward_output_layer_kernel(float *exp_y_vec, float *input_,
		float *output_, int in_depth_, int exp_y) {
//	*err = 0;
//	FERNANDO CHECK IT
//	exp_y_vec.clear();
//	exp_y_vec.resize(in_depth_);

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i > in_depth_)
		return;

	if (i == 0)
		exp_y_vec[exp_y] = 1;
	__syncthreads();

//	for (size_t i = 0; i < in_depth_; i++) {
	output_[i] = 0.5 * (exp_y_vec[i] - input_[i]) * (exp_y_vec[i] - input_[i]);
//	}

}

__global__ void backprop_output_layer_kernel(float *exp_y_vec, float *input_,
		float *g_, int in_depth_) {
	/* compute err terms of output layers */
//	g_.clear();
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i > in_depth_)
		return;
//	for (size_t i = 0; i < in_depth_; i++) {
	g_[i] = ((exp_y_vec[i] - input_[i]) * df_sigmod_gpu(input_[i]));

//	}
}

void call_forward_output_layer(float *err, float *exp_y_vec, float *input_,
		float *output_, float *one_vector, float *output_dot, int in_depth_,
		int exp_y) {
	dim3 blocks, threads;
	cuda_gridsize(&thread, &blocks, in_depth_);

	forward_output_layer_kernel<<<blocks, threads>>>(err, exp_y_vec, input_,
			output_, in_depth_, exp_y);
	cudaError_t ret = cudaDeviceSynchronize();
	CUDA_CHECK_RETURN(ret);

	full_dot<<<blocks, threads>>>(output_, one_vector, in_depth_, output_dot);
	ret = cudaDeviceSynchronize();
	CUDA_CHECK_RETURN(ret);
	cudaMemcpy(err, output_dot, sizeof(float), cudaMemcpyDeviceToHost);
	ret = cudaDeviceSynchronize();
	CUDA_CHECK_RETURN(ret);
}

void call_backpropagation_output_layer(float *exp_y_vec, float *input_,
		float *g_, int in_depth_) {
	dim3 blocks, threads;
	cuda_gridsize(&thread, &blocks, in_depth_);
	backprop_output_layer_kernel<<<blocks, threads>>>(exp_y_vec, input_, g_, in_depth_);
	cudaError_t ret = cudaDeviceSynchronize();
	CUDA_CHECK_RETURN(ret);
}
