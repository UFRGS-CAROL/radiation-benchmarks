/*
 * l2_cache_kernels.cu
 *
 *  Created on: Jan 31, 2019
 *      Author: carol
 */

#include "kernels.h"


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

void test_l2_cache(std::uint32_t number_of_sms, Board device) {

}

