/*
 * FullyConnectedLayerKernel.cu
 *
 *  Created on: Jun 6, 2017
 *      Author: carol
 */

#include "FullyConnectedLayerKernel.h"
#include "cudaUtil.h"

//__device__ inline float *get_W_gpu(int index, int in_depth_, float *W_) {
//#pragma unroll
//	for (int i = 0; i < in_depth_; i++) {
//		v_output[i] = W_[index * in_depth_ + i];
//	}
//	return v_output;
//}

__device__ float sigmod_gpu_fully(float in) {
	return 1.0 / (1.0 + exp(-in));
}

__device__ float df_sigmod_gpu_fully(float f_x) {
	return f_x * (1.0 - f_x);
}

__device__ float dot_gpu_fully(float *x, int x_size, float *w) {
//	assert(x.size() == w.size());
	double sum = 0;
#pragma unroll
	for (int i = 0; i < x_size; i++) {
		sum += x[i] * w[i];
	}
	return sum;
}

__global__ void forward_gpu_kernel(float *output_, float *input_, float *b_,
		float *W_, int out_depth_, int in_depth_, int input_size) {

	int out = blockIdx.x * blockDim.x + threadIdx.x;

	if (out > out_depth_)
		return;

//	 original for was like this for (size_t out = 0; out < out_depth_; out++)
//	 get_W_gpu(out, in_depth_, W_, &v_output[out * in_depth_]);
	float *v = &W_[out * in_depth_];
	float dot = dot_gpu_fully(input_, input_size, v);

	output_[out] = sigmod_gpu_fully(dot + b_[out]);
}

void call_forward_fully_connected(float *output_, float *input_, float *b_,
		float *W_, int out_depth_, int in_depth_, int input_size) {

	dim3 blocks, threads;
	cuda_gridsize(&threads, &blocks, out_depth_);
	forward_gpu_kernel<<<blocks, threads>>>(output_, input_, b_, W_, out_depth_, in_depth_, input_size);
	CudaCheckError();
}

__device__ void get_W_step(float *r_output, float *W_, int in, int out_depth_,
		int in_depth_) {
//	vec_host r;
//	for (size_t i = in; i < out_depth_ * in_depth_; i += in_depth_) {
//		r.push_back(W_[i]);
//	}
//	return r;
	for (int i = in, j = 0; i < out_depth_ * in_depth_; i += in_depth_, j++) {
		r_output[j] = W_[i];
	}
}

__global__ void backpropagation_gpu_err_terms(float *g_, float *g_next,
		float *input_, float *r_output, float *W_, int out_depth_,
		int in_depth_, int g_next_size) {
	/*
	 Compute the err terms;
	 */
	int in = blockIdx.x * blockDim.x + threadIdx.x;
	if (in_depth_ < in)
		return;

//	for (size_t in = 0; in < in_depth_; in++) {
	int r_index = out_depth_ * in;
	get_W_step(&r_output[r_index], W_, in, out_depth_, in_depth_);
	float dot_result = dot_gpu_fully(g_next, g_next_size, &r_output[r_index]);
	g_[in] = df_sigmod_gpu_fully(input_[in]) * dot_result;
//	}

}

__global__ void backpropagation_gpu_update_weights(float *input_, float *g_next,
		float *deltaW_, float *W_, float *b_, float alpha_, float lambda_,
		int in_depth_, int out_depth_) {
	/*
	 Update weights.
	 */
	int out = blockIdx.x * blockDim.x + threadIdx.x;
	if (out > out_depth_)
		return;

//	for (size_t out = 0; out < out_depth_; out++) {
	for (int in = 0; in < in_depth_; in++) {
		auto delta = alpha_/*learning rate*/
		* input_[in] * g_next[out]/*err terms*/
		/*+ lambda_ weight decay*/
		+ lambda_ * deltaW_[out * in_depth_ + in];
		W_[out * in_depth_ + in] += delta;
		deltaW_[out * in_depth_ + in] = delta;
	}
	__syncthreads();
	atomicAdd(&b_[out], alpha_ * g_next[out]);
//	b_[out] += alpha_ * g_next[out];
//	}
}

void call_backpropagation_fully_connected(float *input_, float *g_,
		float *g_next, float *deltaW_, float *W_, float *b_, float *r_output,
		float alpha_, float lambda_, int in_depth_, int out_depth_,
		int g_next_size) {

	dim3 blocks, threads;
	cuda_gridsize(&threads, &blocks, in_depth_);
	backpropagation_gpu_err_terms<<<blocks, threads>>>(g_, g_next, input_,
			r_output, W_, out_depth_, in_depth_, g_next_size);
	CudaCheckError();

	cuda_gridsize(&threads, &blocks, out_depth_);

	backpropagation_gpu_update_weights<<<blocks, threads>>>(input_, g_next,
			deltaW_, W_, b_, alpha_, lambda_, in_depth_, out_depth_);
	CudaCheckError();
}
