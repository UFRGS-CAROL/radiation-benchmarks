/*
 * register_file_kernels.cu
 *
 *  Created on: Feb 2, 2019
 *      Author: carol
 */

#include "kernels.h"
#include "utils.h"
#include <cstring>

//#ifndef RFSIZE
//#define RFSIZE 256
//#endif

template<uint32 RFSIZE>
struct RF {
	uint32 rf[RFSIZE];

	__host__ __device__ RF() {
	}

	__host__ __device__ explicit RF(const RF& T) {
		for (int i = 0; i < RFSIZE; i++) {
			rf[i] = T.rf[i];
		}
	}

	__host__ __device__ RF(RF& T) {
		for (int i = 0; i < RFSIZE; i++) {
			rf[i] = T.rf[i];
		}
	}

	__host__ __device__ RF(const uint32& T) {
		for (int i = 0; i < RFSIZE; i++) {
			rf[i] = T;
		}
	}

	__host__  __device__  inline RF& operator=(const uint32& T) {
		for (int i = 0; i < RFSIZE; i++) {
			rf[i] = T;
		}
		return *this;
	}

	__host__  __device__  inline RF& operator=(volatile RF& T) {
		for (int i = 0; i < RFSIZE; i++) {
			rf[i] = T.rf[i];
		}
		return *this;
	}

	__host__  __device__  inline RF& operator=(RF& T) {
		for (int i = 0; i < RFSIZE; i++) {
			rf[i] = T.rf[i];
		}
		return *this;
	}

	__host__  __device__  inline uint32 operator [](int idx) volatile{
		return rf[idx];
	}
};

__device__ uint64 register_file_errors;

template<uint32 RFSIZE>
__global__ void test_register_file_kernel(RF<RFSIZE> *output_rf,
		const uint64 sleep_cycles, const uint32 reg_data) {
	const uint32 i = blockIdx.x * RFSIZE + threadIdx.x;
//
//	register volatile RF<RFSIZE> register_file = output_rf[i];
//
//	sleep_cuda(sleep_cycles);
//
//	__syncthreads();
//
//	for (uint32 i = 0; i < RFSIZE; i++) {
//		if (register_file[i] != reg_data)
//			atomicAdd(&register_file_errors, 1);
//	}
//
//	output_rf[i] = register_file;
	volatile register uint32 r0 = output_rf[], r1, r3;
	volatile register uint32 r0, r1, r3;
	volatile register uint32 r0, r1, r3;


}

/*
 *
 register volatile RF register_file;
 for(uint32 i = 0; i < RFSIZE; i++)
 register_file[i] = reg_data;

 sleep_cuda(sleep_cycles);

 __syncthreads();

 for (uint32 i = 0; i < RFSIZE; i++) {
 if (register_file[i] != reg_data)
 atomicAdd(&register_file_errors, 1);

 const uint32 tx = blockIdx.x * RFSIZE + threadIdx.x;
 output_rf[tx + i] = register_file;
 }
 */
Tuple test_register_file(const uint32 reg_data, const int64 cycles,
		dim3& block_size, dim3& threads_per_block) {

	const uint32 rf_size = 256;
	//Allocate an array of the size of all register bank
	uint32 out_size = block_size.x * block_size.y * threads_per_block.x;
	RF<rf_size> *output_dev;

	//error variable
	uint64 register_file_errors_host = 0;
	cuda_check(
			cudaMemcpyToSymbol(register_file_errors, &register_file_errors_host,
					sizeof(uint64), 0));

	//malloc on device
	cuda_check(cudaMalloc(&output_dev, sizeof(RF<rf_size> ) * out_size));

	//malloc on host
	std::vector < RF < rf_size >> output_host(out_size);

	test_register_file_kernel<rf_size> <<<block_size, threads_per_block>>>(
			output_dev, cycles, reg_data);
	cuda_check(cudaDeviceSynchronize());

	//Copy data back
	cuda_check(
			cudaMemcpy(output_host.data(), output_dev,
					sizeof(RF<rf_size> ) * out_size, cudaMemcpyDeviceToHost));

	//Copy error var

	cuda_check(
			cudaMemcpyFromSymbol(&register_file_errors_host,
					register_file_errors, sizeof(uint64), 0));

	cuda_check(cudaFree(output_dev));

	Tuple t;
//
//	t.register_file.assign((uint32*) output_host.data(),
//			(uint32*) output_host.data() + (sizeof(RF) * output_host.size()));

	t.errors = register_file_errors_host;

	return t;

}

Tuple test_register_file(const Parameters& parameters) {
	//Kepler and Volta have
	//fucking 256KB registers per SM
	// so I have to allocate 4 blocks of
	// 256 threads
	dim3 block_size(parameters.number_of_sms, 4);
	dim3 threads_per_block(parameters.registers_per_block / 256);
	std::cout << "SIZE OF RF "
			<< block_size.x * block_size.y * threads_per_block.x * 256
			<< std::endl;

	uint32 reg_data;
	std::memset(&reg_data, parameters.t_byte, sizeof(uint32));

	return test_register_file(reg_data, parameters.one_second_cycles,
			block_size, threads_per_block);
}
