/*
 * MaxpoolingLayerKernel.cu
 *
 *  Created on: Jun 6, 2017
 *      Author: carol
 */

#include "cudaUtil.h"
#include "MaxpoolingLayer.h"


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

__global__ void forward_maxpool_layer_kernel(float_t *input_, Pair *max_loc,
		float_t *output_, size_t out_width, size_t out_height,
		size_t out_depth_, size_t in_height, size_t in_width) {

	int h_ = blockIdx.y * blockDim.y + threadIdx.y;
	int w_ = (blockIdx.x * blockDim.x + threadIdx.x) / out_depth_;
	int out = (blockIdx.x * blockDim.x + threadIdx.x) % out_depth_;

	if ((out < out_depth_) && (h_ < in_height) && (w_ < in_width) && !(h_ % 2)
			&& !(w_ % 2)) {
		size_t index = get_out_index(out_width, out_height, out, h_, w_);
		output_[index] = max_in_(input_, max_loc, in_width, in_height, out, h_,
				w_, index);

	}
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

__global__ void forward_maxpool_layer_kernel_darknet(int n, int in_h, int in_w,
		int in_c, int stride, int size, int pad, float *input, float *output,
		size_t *indexes) {
	int h = (in_h + 2 * pad) / stride;
	int w = (in_w + 2 * pad) / stride;
	int c = in_c;

	int id = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
	if (id >= n)
		return;

	int j = id % w;
	id /= w;
	int i = id % h;
	id /= h;
	int k = id % c;
	id /= c;
	int b = id;

	int w_offset = -pad;
	int h_offset = -pad;

	int out_index = j + w * (i + h * (k + c * b));
	float max = -INFINITY;
	int max_i = -1;
	int l, m;
	for (l = 0; l < size; ++l) {
		for (m = 0; m < size; ++m) {
			int cur_h = h_offset + i * stride + l;
			int cur_w = w_offset + j * stride + m;
			int index = cur_w + in_w * (cur_h + in_h * (k + b * in_c));
			int valid = (cur_h >= 0 && cur_h < in_h && cur_w >= 0
					&& cur_w < in_w);
			float val = (valid != 0) ? input[index] : -INFINITY;
			max_i = (val > max) ? index : max_i;
			max = (val > max) ? val : max;
		}
	}
	output[out_index] = max;
	indexes[out_index] = max_i;
}

__global__ void backward_maxpool_layer_kernel(int n, int in_h, int in_w,
		int in_c, int stride, int size, int pad, float *delta,
		float *prev_delta, size_t *indexes) {
	int h = (in_h + 2 * pad) / stride;
	int w = (in_w + 2 * pad) / stride;
	int c = in_c;
	int area = (size - 1) / stride;

	int id = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
	if (id >= n)
		return;

	int index = id;
	int j = id % in_w;
	id /= in_w;
	int i = id % in_h;
	id /= in_h;
	int k = id % in_c;
	id /= in_c;
	int b = id;

	int w_offset = -pad;
	int h_offset = -pad;

	float d = 0;
	int l, m;
	for (l = -area; l < area + 1; ++l) {
		for (m = -area; m < area + 1; ++m) {
			int out_w = (j - w_offset) / stride + m;
			int out_h = (i - h_offset) / stride + l;
			int out_index = out_w + w * (out_h + h * (k + c * b));
			int valid = (out_w >= 0 && out_w < w && out_h >= 0 && out_h < h);
			d += (valid && indexes[out_index] == index) ? delta[out_index] : 0;
		}
	}
	prev_delta[index] += d;
}
//
void call_forward_maxpool_layer_gpu(float_t *input, float_t *output,
//		Pair *max_loc,
		size_t *indexes, size_t out_width, size_t out_height, size_t out_depth,
		size_t in_height, size_t in_width, size_t in_depth_, size_t stride, size_t pad, size_t size, size_t batch) {

//	dim3 blocks, threads;
//
//	cuda_gridsize(&threads, &blocks, in_width * out_depth, in_height);
//
//	forward_maxpool_layer_kernel<<<blocks, threads>>>(input, max_loc, output,
//			out_width, out_height, out_depth, in_height, in_width);
//	CudaCheckError();

	// Trying darknet approach
	int h = out_height;
	int w = out_width;
	int c = out_depth;

	size_t n = h * w * c * batch;
	dim3 blocks = cuda_gridsize(n);
	forward_maxpool_layer_kernel_darknet<<<blocks, BLOCK_SIZE_FULL>>>(n, in_height,
			in_width, in_depth_, stride, size, pad,
			input, output, indexes);
	CudaCheckError();
}

void call_backpropagation_maxpool(size_t batch, size_t in_height, size_t in_width, size_t in_depth, size_t stride,
		size_t size, size_t pad, float* delta_gpu, float* net_delta_gpu, size_t* indexes_gpu) {
//	dim3 blocks, threads;
//	cuda_gridsize(&threads, &blocks, max_size);
//
//	assert(g_max_size != 0);
//	backpropagation_maxpool<<<blocks, threads>>>(max_loc, g_, g_next, max_size, g_max_size);
//	CudaCheckError();
	//	size_t n = layer.h * layer.w * layer.c * layer.batch;

	int h = in_height, w = in_width, c = in_depth;
	size_t n = h * w * c * batch;
	dim3 blocks = cuda_gridsize(n);

	backward_maxpool_layer_kernel<<<cuda_gridsize(n), BLOCK_SIZE_FULL>>>(n, h, w, c, stride,
			size, pad, delta_gpu, net_delta_gpu, indexes_gpu);
	CudaCheckError();
}


void MaxpoolingLayer::forward() {
// execute the code on the device
		float_t *input = this->input_.d_data();
		float_t *output = this->output_.d_data();
		Pair *max_loc_buf = this->max_loc.d_data();
		size_t out_width = this->out_width_;
		size_t out_height = this->out_height_;
		size_t out_depth = this->out_depth_;
		size_t in_height = this->in_height_;
		size_t in_width = this->in_width_;
		size_t *indexes = this->indexes.d_data();
		size_t in_depth = this->in_depth_;

		call_forward_maxpool_layer_gpu(input, output,indexes, out_width,
				out_height, out_depth, in_height, in_width, in_depth, 2, 0, 2, this->batch);
}


void MaxpoolingLayer::back_prop() {
	g_.clear();
	g_.resize(this->in_width_ * this->in_height_ * this->in_depth_);

	Pair *max_loc = this->max_loc.d_data();
	float *g_ = this->g_.d_data();
	float *g_next = this->next->g_.d_data();
	size_t max_size = this->max_loc.size();
	size_t g_max_size = this->g_.size();
	size_t *indexes = this->indexes.d_data();

	//call_backpropagation_maxpool(max_loc, g_, g_next, max_size, g_max_size);
//	void call_backpropagation_maxpool(size_t batch, size_t in_height, size_t in_width, size_t in_depth, size_t stride,
//			sizet_t size, size_t pad, float* delta_gpu, float* net_delta_gpu, float* indexes_gpu)
	call_backpropagation_maxpool(this->batch, this->in_height_, this->in_width_, this->in_depth_, this->stride, 2, this->pad, g_next,
			g_, indexes);

}
//
//int main(){
//	int N = 8;
//	int N_OUT = 4;
//	float *h, *h_out, *d, *d_out;
//	int *indexes;
//	h = (float*) malloc(sizeof(float) * N * N);
//	h_out = (float*) malloc(sizeof(float) * N_OUT * N_OUT);
//	cudaMalloc(&d, N * N * sizeof(float));
//	cudaMalloc(&d_out, N_OUT * N_OUT * sizeof(float));
//	cudaMalloc(&indexes, N_OUT * N_OUT * sizeof(int));
//
//
//	for(int i = 0; i < N; i++){
//		for(int j = 0; j < N; j++){
//			printf("%f ", float(i * N + j));
//			h[i * N + j] = i * N + j;
//		}
//		printf("\n");
//	}
//	cudaMemcpy(d, h, sizeof(float) * N * N, cudaMemcpyHostToDevice);
//
//	int n = N_OUT * N_OUT;
//
////	forward_maxpool_layer_kernel_darknet(int n, int in_h, int in_w,
////			int in_c, int stride, int size, int pad, float *input, float *output,
////			int *indexes);
//	dim3 blocks = cuda_gridsize(n);
//	forward_maxpool_layer_kernel_darknet<<<blocks, BLOCK_SIZE_FULL>>>(n, N, N, 1, 2, 2, 0, d, d_out, indexes);
//	cudaError_t s = cudaDeviceSynchronize();
//	printf("\n%d\n", s);
//
//	cudaMemcpy(h_out, d_out, sizeof(float) * N_OUT * N_OUT, cudaMemcpyDeviceToHost);
//
//	for(int i = 0; i < N_OUT * N_OUT; i++)
//		printf("%f ", h_out[i]);
//	printf("\n");
//
//	cudaFree(d);
//	cudaFree(d_out);
//	cudaFree(indexes);
//	free(h);
//	free(h_out);
//
//
//}
