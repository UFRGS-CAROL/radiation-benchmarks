/*
 * read_only_cache_kernels.cu
 *
 *  Created on: Jan 31, 2019
 *      Author: carol
 */

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

