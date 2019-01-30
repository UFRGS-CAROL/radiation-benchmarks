/*
 * kernels.cu
 *
 *  Created on: 26/01/2019
 *      Author: fernando
 */

#include "utils.h"
#include "kernels.h"
#include <iostream>
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

__device__ inline void sleep(std::int64_t sleep_cycles) {

	std::int64_t start = clock64();
	std::int64_t cycles_elapsed;
	do {
		cycles_elapsed = clock64() - start;
	} while (cycles_elapsed < sleep_cycles);
}

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
template<typename int_t, std::uint32_t V_SIZE>
__global__ void test_l1_cache_kernel(int_t *l1_hit_array, int_t *l1_miss_array,
		int_t *v_array, int_t *v_output_array, std::int64_t sleep_cycles) {
	register std::uint32_t tx = blockIdx.x * blockDim.x + threadIdx.x;

	__shared__ int_t l1_t_hit[V_SIZE];
	__shared__ int_t l1_t_miss[V_SIZE];

	for (std::uint32_t i = 0; i < V_SIZE; i++) {
		int_t t1 = clock();
		register int_t r = v_array[tx + i];
		int_t t2 = clock();
		l1_t_miss[i] = t2 - t1;

	}

	//wait for exposition to neutrons
	sleep(sleep_cycles);

	for (std::uint32_t i = 0; i < V_SIZE; i++) {
		//last checking
		int_t t1 = clock();
		register int_t r = v_array[tx + i];
		int_t t2 = clock();
		l1_t_hit[i] = t2 - t1;

		//bitwise operation
//		if ((r ^ t) != 0)
//			atomicAdd(&l1_cache_err, 1);

//		//saving the result
		l1_hit_array[tx + i] = l1_t_hit[i];
		l1_miss_array[tx + i] = l1_t_miss[i];
		v_output_array[tx + i] = r;
	}
}

void test_l1_cache_kepler(size_t number_of_sms) {
	const std::uint32_t l1_size = 64 * 1024; // cache l1 has 65536 bytes
	const std::uint32_t cache_line_size = 128; // size in bytes
	const std::uint32_t v_size = l1_size / cache_line_size; // 512 lines

	std::int32_t *l1_hit_array_device, *l1_miss_array_device, *v_array_device,
			*v_output_array_device;
	std::int32_t *l1_hit_array_host = new std::int32_t[v_size];
	std::int32_t *l1_miss_array_host = new std::int32_t[v_size];
	std::int32_t *v_array_host = new std::int32_t[v_size];

	cudaMalloc(&l1_hit_array_device, sizeof(std::int32_t) * v_size);
	cudaMalloc(&l1_miss_array_device, sizeof(std::int32_t) * v_size);
	cudaMalloc(&v_array_device, sizeof(std::int32_t) * v_size);
	cudaMalloc(&v_output_array_device, sizeof(std::int32_t) * v_size);

	cudaMemset(v_array_device, 3939, sizeof(std::int32_t) * v_size);
	test_l1_cache_kernel<std::int32_t, v_size> <<<1, 1>>>(l1_hit_array_device,
			l1_miss_array_device, v_array_device, v_output_array_device,
			100000);
	cuda_check(cudaDeviceSynchronize());

	cudaMemcpy(l1_hit_array_host, l1_hit_array_device,
			sizeof(std::int32_t) * v_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(l1_miss_array_host, l1_miss_array_device,
			sizeof(std::int32_t) * v_size, cudaMemcpyDeviceToHost);

	for (int i = 0; i < v_size; i++) {
		std::cout << " L1 hit " << l1_hit_array_host[i] << " L1 MISS "
				<< l1_miss_array_host[i] << std::endl;
	}

	cudaFree(l1_hit_array_device);
	cudaFree(l1_miss_array_device);
	cudaFree(v_output_array_device);
	cudaFree(v_array_device);
	delete[] v_array_host;
	delete[] l1_hit_array_host;
	delete[] l1_miss_array_host;
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
