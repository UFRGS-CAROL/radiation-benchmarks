/*
 * kernels.cu
 *
 *  Created on: 26/01/2019
 *      Author: fernando
 */

#include "utils.h"
#include "kernels.h"

#include <cuda_runtime.h>

//alignas(LINE_NUMBER * SIZE_OF_T)

template<typename T, unsigned LINE_SIZE>
struct CacheLine {
	T t[LINE_SIZE];

	CacheLine() {
	}

	CacheLine(const CacheLine& a) {
		for (int i = 0; i < LINE_SIZE; i++)
			t[i] = a.t[i];
	}

	inline CacheLine& operator=(const CacheLine& a) {
		t = a.t;
		return *this;
	}

	inline bool operator==(const CacheLine& a) {
		for (int i = 0; i < LINE_SIZE; i++) {
			if (a.t[i] != t[i])
				return false;
		}
		return true;
	}

	inline bool operator!=(const CacheLine& a) {
		for (int i = 0; i < LINE_SIZE; i++) {
			if (a.t[i] != t[i])
				return true;
		}
		return false;
	}

	inline bool operator!=(const T a) {
		for (int i = 0; i < LINE_SIZE; i++) {
			if (a != t[i])
				return true;
		}
		return false;
	}


	inline CacheLine operator^(const CacheLine& rhs) {
		CacheLine ret;
		for (int i = 0; i < LINE_SIZE; i++) {
			ret.t[i] = t[i] ^ rhs.t[i];
		}
		return ret;
	}
};

texture<int, 1, cudaReadModeElementType> tex_ref;

__device__ std::uint32_t l1_cache_err;

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

/*
 * l1_size size of the L1 cache
 * V_size = l1_size / sizeof(CacheLine)
 */
template<typename cache_line, typename int_t, std::uint32_t V_SIZE>
__global__ void test_l1_cache_kernel(cache_line *v_array,
		cache_line *l1_hit_array, cache_line *l1_miss_array,
		std::uint32_t l1_size, cache_line t) {
	register std::uint32_t tx = blockIdx.x * blockDim.x + threadIdx.x;

//	__shared__ int_t l1_t_hit[V_SIZE];
//	__shared__ int_t l1_t_miss[V_SIZE];
//
//	for (std::uint32_t i = 0; i < V_SIZE; i++) {
//		int_t t1 = clock();
//		register cache_line r = v_array[i];
//		int_t t2 = clock();
//		l1_t_miss[i] = t2 - t1;
//	}
//
//	//wait for exposition to neutrons
//
//	for (std::uint32_t i = 0; i < V_SIZE; i++) {
//		//last checking
//		int_t t1 = clock();
//		register cache_line r = v_array[i];
//		int_t t2 = clock();
//		l1_t_hit[i] = t2 - t1;
//
//		//bitwise operation
//		if ((r ^ t) != 0)
//			atomicAdd(&l1_cache_err, 1);
//
//		//saving the result
//		l1_hit_array[tx + i] = l1_t_hit[i];
//		l1_miss_array[tx + i] = l1_t_miss[i];
//	}
}

void test_l1_cache(size_t number_of_sms) {
	const std::uint32_t line_size = 128; //128 bytes
	const std::uint32_t l1_size = 112 * 1024 * 1024;
	const std::uint32_t v_size = l1_size / line_size;
	const std::uint32_t siz_int = sizeof(int);
	CacheLine<int, line_size / siz_int> *a, *b, *c, t;
	cudaError_t ret = cudaFuncSetCacheConfig(
			test_l1_cache_kernel<CacheLine<int, line_size / siz_int>, int, v_size>,
			cudaFuncCachePreferShared);
	cuda_check(ret);

	cudaMalloc(&a, sizeof(CacheLine<int, line_size / siz_int>) * v_size);
	cudaMalloc(&b, sizeof(CacheLine<int, line_size / siz_int>) * v_size);
	cudaMalloc(&c, sizeof(CacheLine<int, line_size / siz_int>) * v_size);


	//	template<typename cache_line, typename int_t, std::uint32_t V_SIZE>
	test_l1_cache_kernel< CacheLine<int, line_size / siz_int>, int, v_size> <<<1, 1>>>(a, b, c, 0, t);
	cuda_check(cudaDeviceSynchronize());

	cuda_check(cudaFree(a));
	cuda_check(cudaFree(b));
	cuda_check(cudaFree(c));

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

void test_l2_cache() {

}
