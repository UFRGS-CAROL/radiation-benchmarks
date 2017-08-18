#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "maxpool_layer.h"
#include "cuda.h"
}

#define FACTOR 10.0

unsigned long long *error_detected = NULL;

float LOOK_UP_TABLE[] = { //for hardened maxpool
		23.2265, //layer 0
		23.2265, //layer 1
		17.821, //layer 2
		17.821, //layer 3
		24.3061, //layer 4
		13.2396, //layer 5
		17.4524, //layer 6
		10.5258, //layer 7
		10.5258, //layer 8
		21.5326, //layer 9
		14.2375, //layer 10
		28.7749, //layer 11
		17.065, //layer 12
		29.3857, //layer 13
		13.5177, //layer 14
		23.2387, //layer 15
		11.1505, //layer 16
		12.7082, //layer 17
		10.9391, //layer 18
		10.9391, //layer 19
		25.4447, //layer 20
		22.5485, //layer 21
		51.8862, //layer 22
		238.784, //layer 23
		56.2831, //layer 24
		49.6482, //layer 25
		49.1465, //layer 26
		40.7246, //layer 27
		1.07145, //layer 28
		1.07145, //layer 29
		1.18859, //layer 30
		1.18859, //layer 31
		};

int maxpool_iterator = 0;

__global__ void forward_maxpool_layer_kernel_hardened(int n, int in_h, int in_w,
		int in_c, int stride, int size, int pad, float *input, float *output,
		int *indexes, float max_value_allowed,
		unsigned long long *error_detected, int maxp) {
	int h = (in_h + 2 * pad - size + 1) / stride + 1;
	int w = (in_w + 2 * pad - size + 1) / stride + 1;
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

	//temp matrix
	float old_max = max;
	int old_max_i = max_i;

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

			//hardened trick
			if (max > max_value_allowed) {
				max = old_max;
				max_i = old_max_i;

				//count how many errors
				atomicAdd(&error_detected[maxp], 1);
			} else {
				old_max = max;
				old_max_i = max_i;
			}
		}
	}
	output[out_index] = max;
	indexes[out_index] = max_i;
}

extern "C" void forward_maxpool_layer_gpu_hardened(maxpool_layer layer,
		network_state state) {
	int h = layer.out_h;
	int w = layer.out_w;
	int c = layer.c;

	size_t n = h * w * c * layer.batch;

	//for the LOOKUP
	maxpool_iterator = (maxpool_iterator + 1) % MAXPOOL_N;
	int maxp = 1;
	if (maxpool_iterator == 1) {
		maxp = 3;
	} else if (maxpool_iterator == 2) {
		maxp = 8;
	} else if (maxpool_iterator == 3) {
		maxp = 19;
	}

	if (error_detected == NULL) {
		cudaMalloc(&error_detected, sizeof(unsigned long long) * MAXPOOL_N);
	}

	forward_maxpool_layer_kernel_hardened<<<cuda_gridsize(n), BLOCK>>>(n, layer.h,
			layer.w, layer.c, layer.stride, layer.size, layer.pad, state.input,
			layer.output_gpu, layer.indexes_gpu, LOOK_UP_TABLE[maxp] * FACTOR,
			error_detected, maxpool_iterator);
	check_error(cudaPeekAtLastError());
}

__global__ void memset_error(unsigned long long *error_detected) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	error_detected[i] = 0;
}

/**
 * host_error_detected must be allocated
 */
void get_and_reset_error_detected_values(error_return host_error) {
	//copy from error_detected var
	cudaMemcpy(host_error.error_detected, error_detected,
			sizeof(unsigned long long) * host_error.err_detected_size,
			cudaMemcpyDeviceToHost);

	memset_error<<<1, MAXPOOL_N>>>(error_detected);

	check_error(cudaPeekAtLastError());
}

void free_err_detected() {
	if (error_detected)
		cudaFree(error_detected);
}


