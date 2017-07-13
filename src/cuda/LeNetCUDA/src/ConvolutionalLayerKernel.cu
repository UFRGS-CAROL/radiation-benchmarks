/**
 * Here all kernels which were in kernels.ocl were translated
 */

#include "ConvolutionalLayerKernel.h"
#include "cudaUtil.h"
#include <cstdio>

__device__ float sigmod_gpu_conv(float in) {
	return 1.0 / (1.0 + expf(-in));
}

__device__ float df_sigmod_gpu_conv(float f_x) {
	return f_x * (1.0 - f_x);
}

__device__    inline size_t get_b_(size_t out, size_t h_, size_t w_,
		size_t out_width_, size_t out_height_) {
	return out * out_width_ * out_height_ + h_ * out_height_ + w_;
}

__device__    inline size_t get_out_index(size_t out, size_t h_, size_t w_,
		size_t out_width_, size_t out_height_) {
	return out * out_height_ * out_width_ + h_ * out_width_ + w_;
}

__device__ inline int get_global_id(int index) {
	switch (index) {
	case 0:
		return blockIdx.x * blockDim.x + threadIdx.x;
	case 1:
		return blockIdx.y * blockDim.y + threadIdx.y;
	case 2:
		return blockIdx.z * blockDim.z + threadIdx.z;
	};
	return -1;
}

__device__ inline float conv(int kernel_size, unsigned int in, int in_width,
		int in_height, int h_index, int w_index, int out_depth, int size,
		int out, float* input_buf, float* weight_buf) {
	float weight_buf_sub[CONV_KERNEL_SIZE]; // Set by brute force, NEED to be changed
	float input_buf_sub[CONV_KERNEL_SIZE]; // Set by brute force, NEED to be changed
	float sum = 0;
	// load input and weight for this sub area
	// Aqui ele faz o que a funcao conv faz
	for (unsigned int y = 0; y < kernel_size; y++) {
		for (unsigned int x = 0; x < kernel_size; x++) {
			input_buf_sub[y * kernel_size + x] = input_buf[in
					* (in_width * in_height) + (h_index + y) * in_width + x
					+ w_index];
			weight_buf_sub[y * kernel_size + x] = weight_buf[in * out_depth
					* size + out * size + y * kernel_size + x];
		}
	}
	// compute the convolution
	for (unsigned int i = 0; i < size; i++) {
		sum += input_buf_sub[i] * weight_buf_sub[size - i - 1];
	}
	return sum;
}

__global__ void forward_parallel(float* input_buf, float* weight_buf,
		float* b_buf, float* output_buf, int in_width, int in_height,
		int in_depth, int out_width, int out_height, int out_depth,
		int kernel_size) {

	if (get_global_id(0) > out_depth * out_width) {
		return;
	}
	if (get_global_id(1) > out_height) {
		return;
	}

	int out = get_global_id(0) / out_width;
	int w_index = get_global_id(0) % out_width;
	int h_index = get_global_id(1);

	float sum = 0;
	int size = kernel_size * kernel_size;
	if (size != 25) {
		printf("\n\n\nPau no kernel\n\n\n");
		return;
	}

	for (unsigned int in = 0; in < in_depth; in++) {
		// load input and weight for this sub area
		// Aqui ele faz o que a funcao conv faz
		sum += conv(kernel_size, in, in_width, in_height, h_index, w_index,
				out_depth, size, out, input_buf, weight_buf);
	}

	unsigned int out_index = get_out_index(out, h_index, w_index, out_width,
			out_height);
	unsigned int b_index = get_b_(out, h_index, w_index, out_width, out_height);
	output_buf[out_index] = sigmod_gpu_conv(sum + b_buf[b_index]);
}

void call_foward_parallel(float* input_buf, float* weight_buf, float* b_buf,
		float* output_buf, int in_width, int in_height, int in_depth,
		int out_width, int out_height, int out_depth, int kernel_size) {

	dim3 blocks, threads;

//	cuda_gridsize(&threads, &blocks, out_width, out_height, out_depth);
	cuda_gridsize(&threads, &blocks, out_width * out_depth, out_height);

	//I need check it yet
	forward_parallel<<<blocks, threads>>>(input_buf, weight_buf, b_buf,
			output_buf, in_width, in_height, in_depth, out_width, out_height,
			out_depth, kernel_size);
	CudaCheckError();

}

__global__ void backpropagation_update_err(float *W_, //weights
		float *g_, //err array
		float *g_next, //b_next from this->next->g_
		float *input_, //input array
		int out_depth_, //size of the first for loop
		int in_depth_, //size of the second for loop
		int out_width_, //size of the third for loop
		int out_height_, //out height
		int kernel_size_, //kernel size
		int in_width_,  //in width
		int in_height_ //in height
		) {

//	int out = get_global_id(0) / out_width;
//		int w_index = get_global_id(0) % out_width;
//		int h_index = get_global_id(1);

	int out = get_global_id(0) / out_width_; //out iterator, comes from the first for loop, < out_depth
	int in = get_global_id(1); //in iterator, comes from the second for loop, < in_depth
//	int w_ = get_global_id(2); //w_ iterator, comes from the third for loop, < out_width
	int w_ = get_global_id(0) % out_width_; //w_ iterator, comes from the third for loop, < out_width

	if ((out >= out_depth_ || in >= in_depth_ || w_ >= out_width_)
			|| (in_width_ * in_height_ * in_depth_) < (out * in * w_))
		return;

	/*update err terms of this layer.*/
//	for (size_t out = 0; out < out_depth_; out++) {
//		for (size_t in = 0; in < in_depth_; in++) {
//			for (size_t w_ = 0; w_ < out_width_; w_++) {
	for (size_t h_ = 0; h_ < out_height_; h_++) {
		for (size_t y_ = 0; y_ < kernel_size_; y_++) {
			for (size_t x_ = 0; x_ < kernel_size_; x_++) {
				int ff = in * in_width_ * in_height_ + (h_ + y_) * in_width_
						+ (x_ + w_);

				float temp_g = /*next layer err terms*/
				g_next[out * out_width_ * out_height_ + h_ * out_width_ + w_]
						*
						/*weight*/
						W_[in * out_depth_ * kernel_size_ * kernel_size_
								+ out * kernel_size_ * kernel_size_
								+ kernel_size_ * (kernel_size_ - y_ - 1)
								+ (kernel_size_ - 1 - x_)] *
						/*df of input*/
						df_sigmod_gpu_conv(input_[ff]);

				__syncthreads();
				atomicAdd(&g_[ff], temp_g);
			}
		}
	}
//			}
//		}
//	}

}

__global__ void backpropagation_update_weights(float *W_, //weights
		float *b_, //b_ array
		float *g_next, //b_next from this->next->g_
		float *input_, //input_ array
		float *deltaW_, //deltaW array
		float alpha_, //alpha value
		float lambda_, //lambda
		int out_height_, //out_height
		int kernel_size_, //kernel size
		int in_width_, //in width
		int in_height_, //in height
		int out_width_, // out width
		int out_depth_, // out depth
		int in_depth_) {

	int out = get_global_id(0) / out_height_; //out iterator, comes from the first for loop, < out_depth
	int in = get_global_id(1); //in iterator, comes from the second for loop, < in_depth
//	int h_ = get_global_id(2); //h_ iterator, comes from the third for loop, < out_height_
	int h_ = get_global_id(0) % out_height_;

	if (out >= out_depth_ || in >= in_depth_ || h_ >= out_height_)
		return;

	/*update weight*/
//	for (size_t out = 0; out < out_depth_; out++) {
//		for (size_t in = 0; in < in_depth_; in++) {
//			for (size_t h_ = 0; h_ < out_height_; h_++) {
	for (size_t w_ = 0; w_ < out_height_; w_++) {
		auto tt = get_b_(out, h_, w_, out_width_, out_height_);
		for (size_t y_ = 0; y_ < kernel_size_; y_++) {
			for (size_t x_ = 0; x_ < kernel_size_; x_++) {
				/*find update pixel*/
				auto target = in * out_depth_ * kernel_size_ * kernel_size_
						+ out * kernel_size_ * kernel_size_
						+ kernel_size_ * (kernel_size_ - y_ - 1)
						+ (kernel_size_ - 1 - x_);
				/*cal delta*/
				auto delta =
				/*learning rate*/
				alpha_
						*
						/*input*/
						input_[in * in_width_ * in_height_
								+ (h_ + y_) * in_width_ + (x_ + w_)] *
						/*next layer err terms*/
						g_next[tt]
				/*weight momentum*/
				+ lambda_ * deltaW_[target];

				__syncthreads();
//				W_[target] += delta;
				atomicAdd(&W_[target], delta);
				/*update momentum*/
//				deltaW_[target] = delta;
				atomicExch(&deltaW_[target], delta);
			}
		}
		__syncthreads();
		b_[tt] += alpha_ * g_next[tt];
	}
//			}
//		}
//	}
}

void call_backpropagation_parallel(float *W_, //weights
		float *g_, //err array
		float *input_, //input array
		float *g_next, //b_next from this->next->g_
		float *deltaW, //deltaW array
		float *b_,  //b_ vector
		float alpha, //alpha value
		float lambda, //lambda value
		int out_depth, //size of the first for loop
		int in_depth_, //size of the second for loop
		int out_width, //size of the third for loop
		int out_height_, // size of loop
		int kernel_size_, //size of loop
		int in_width_, //width size
		int in_height_ //in height
		) {
	dim3 blocks;
	dim3 threads;

	cuda_gridsize(&threads, &blocks, out_depth * out_width, in_depth_);
//	for (size_t out = 0; out < out_depth_; out++) {
//		for (size_t in = 0; in < in_depth_; in++) {
//			for (size_t w_ = 0; w_ < out_width_; w_++) {

	backpropagation_update_err<<<blocks, threads>>>(W_, g_, g_next, input_,
			out_depth, in_depth_, out_width, out_height_, kernel_size_,
			in_width_, in_height_);
	CudaCheckError();

//	for (size_t out = 0; out < out_depth_; out++) {
//		for (size_t in = 0; in < in_depth_; in++) {
//			for (size_t h_ = 0; h_ < out_height_; h_++) {
	cuda_gridsize(&threads, &blocks, out_depth * out_height_, in_depth_);

	backpropagation_update_weights<<<blocks, threads>>>(W_, b_, g_next, input_,
			deltaW, alpha, lambda, out_height_, kernel_size_, in_width_,
			in_height_, out_width, out_depth, in_depth_);
	CudaCheckError();
}

