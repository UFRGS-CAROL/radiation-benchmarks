/*
 * register_file_kernels.cu
 *
 *  Created on: Feb 2, 2019
 *      Author: carol
 */

#include "kernels.h"
#include "utils.h"
#include <cstring>

#include "register_kernel.h"

template<const uint32 RFSIZE>
Tuple test_register_file(const uint32 reg_data, const int64 cycles,
		dim3& block_size, dim3& threads_per_block) {

	//Allocate an array of the size of all register bank
	uint32 out_size = block_size.x * block_size.y * threads_per_block.x * RFSIZE;
	uint32 *output_dev;

	//error variable
	uint64 register_file_errors_host = 0;
	cuda_check(
			cudaMemcpyToSymbol(register_file_errors, &register_file_errors_host,
					sizeof(uint64), 0));

	//byte size
	const uint32 byte_size = sizeof(uint32) * out_size;

	//malloc on device
	cuda_check(cudaMalloc(&output_dev, byte_size));

	//malloc on host
	std::vector<uint32> output_host(out_size, reg_data);

	cuda_check(
			cudaMemcpy(output_dev, output_host.data(), byte_size,
					cudaMemcpyHostToDevice));
	double start = Log::mysecond();
	test_register_file_kernel<<<block_size, threads_per_block>>>(output_dev, reg_data, cycles);
	cuda_check(cudaDeviceSynchronize());

	//Copy data back
	cuda_check(
			cudaMemcpy(output_host.data(), output_dev, byte_size,
					cudaMemcpyDeviceToHost));

	//Copy error var
	cuda_check(
			cudaMemcpyFromSymbol(&register_file_errors_host,
					register_file_errors, sizeof(uint64), 0));

	cuda_check(cudaFree(output_dev));

	Tuple t;
	t.register_file = std::move(output_host);
	t.errors = register_file_errors_host;
	return t;

}

Tuple test_register_file(const Parameters& parameters) {
	const uint32 rf_size = 256;
	//Kepler and Volta have
	//fucking 256KB registers per SM
	// so I have to allocate 4 blocks of
	// 256 threads
	dim3 block_size(parameters.number_of_sms, 4);
	dim3 threads_per_block(parameters.registers_per_block / rf_size);

	uint32 reg_data;
	std::memset(&reg_data, parameters.t_byte, sizeof(uint32));

	return test_register_file<rf_size>(reg_data, parameters.one_second_cycles,
			block_size, threads_per_block);
}
