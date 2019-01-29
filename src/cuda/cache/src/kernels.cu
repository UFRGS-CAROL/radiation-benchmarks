/*
 * kernels.cu
 *
 *  Created on: 26/01/2019
 *      Author: fernando
 */

#include "utils.h"

#include <cuda_runtime.h>

texture<int, 1, cudaReadModeElementType> tex_ref;

template<int READ_ONLY_MEM_SIZE>
__global__ void test_read_only_mem(const unsigned int * __restrict__ my_array,
		int array_length, int iterations, unsigned int *index) {

	unsigned int j = 0;

	__shared__ unsigned int s_index[READ_ONLY_MEM_SIZE];

	int k;

	for (k = 0; k < READ_ONLY_MEM_SIZE; k++) {
		s_index[k] = 0;
	}

	//ldg is a direct load to const memory
	//first round
	for (k = 0; k < 16 * iterations * READ_ONLY_MEM_SIZE; k++)
		j = __ldg(&my_array[j]);

	//second round
	for (k = 0; k < iterations * READ_ONLY_MEM_SIZE; k++) {
		j = __ldg(&my_array[j]);
		s_index[k] = j;
	}

	for (k = 0; k < READ_ONLY_MEM_SIZE; k++) {
		index[k] = s_index[k];
	}
}

__global__ void test_texture(int * my_array, int size, int *index, int iter,
		unsigned int INNER_ITS) {
	const int it = 4096;

	__shared__ int s_value[it];
	int i, j;

	//initilize j
	j = 0;

	for (i = 0; i < it; i++) {
		s_value[i] = -1;
	}

	for (int k = 0; k <= iter; k++) {

		for (int cnt = 0; cnt < it; cnt++) {
			j = tex1Dfetch(tex_ref, j);
			s_value[cnt] = j;
		}

	}

	for (i = 0; i < it; i++) {
		index[i] = s_value[i];
	}

	my_array[size] = i;
	my_array[size + 1] = i;
}

template<typename int_t = unsigned, int SHARED_MEM_SIZE = 512>
__global__ void test_l1_cache(int_t * my_array, std::size_t array_length,
		std::size_t iterations, int_t * duration, int_t *index) {

	unsigned int start_time, end_time;
	unsigned int j = 0;

	__shared__ int_t s_tvalue[SHARED_MEM_SIZE];
	__shared__ int_t s_index[SHARED_MEM_SIZE];

	for (int k = 0; k < SHARED_MEM_SIZE; k++) {
		s_index[k] = 0;
		s_tvalue[k] = 0;
	}

	for (int k = -iterations * SHARED_MEM_SIZE;
			k < iterations * SHARED_MEM_SIZE; k++) {

		if (k >= 0) {
			start_time = clock();
			j = my_array[j];
			s_index[k] = j;
			end_time = clock();

			s_tvalue[k] = end_time - start_time;

		} else {
			j = my_array[j];
		}
	}

	my_array[array_length] = j;
	my_array[array_length + 1] = my_array[j];

	for (int k = 0; k < SHARED_MEM_SIZE; k++) {
		index[k] = s_index[k];
		duration[k] = s_tvalue[k];
	}
}

__global__ void test_l2_cache(unsigned int * my_array, int array_length,
		int iterations, unsigned int * duration, unsigned int *index) {

	unsigned int start_time, end_time;
	unsigned int j = 0;

	__shared__ unsigned int s_tvalue[256];
	__shared__ unsigned int s_index[256];

	int k;

	for (k = 0; k < 256; k++) {
		s_index[k] = 0;
		s_tvalue[k] = 0;
	}

	//first round, warm the TLB
	for (k = 0; k < iterations * 256; k++)
		j = my_array[j];

	//second round, begin timestamp
	for (k = 0; k < iterations * 256; k++) {

		start_time = clock();

		j = my_array[j];
		s_index[k] = j;
		end_time = clock();

		s_tvalue[k] = end_time - start_time;

	}

	my_array[array_length] = j;
	my_array[array_length + 1] = my_array[j];

	for (k = 0; k < 256; k++) {
		index[k] = s_index[k];
		duration[k] = s_tvalue[k];
	}
}

template<int CACHE_SIZE, int CACHE_LINE_SIZE>
void test_l1_cache(size_t number_of_sms) {
	cudaError_t ret = cudaFuncSetCacheConfig(test_l1_cache<int>,
			cudaFuncCachePreferL1);
	cuda_check(ret);

}

template<int CACHE_SIZE, int CACHE_LINE_SIZE>
void test_l2_cache() {

}
