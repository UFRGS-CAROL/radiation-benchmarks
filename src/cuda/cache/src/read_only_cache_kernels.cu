/*
 * read_only_cache_kernels.cu
 *
 *  Created on: Jan 31, 2019
 *      Author: carol
 */

#include "kernels.h"

template<int READ_ONLY_MEM_SIZE>
__global__ void test_read_only_kernel(const unsigned int * __restrict__ my_array,
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



Tuple test_read_only_cache(const Parameters& ){

	return Tuple();
}

