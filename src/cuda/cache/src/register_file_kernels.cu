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

template<uint32 RFSIZE>
Tuple test_register_file(uint32 reg_data, const int64 cycles,
		dim3& block_size, dim3& threads_per_block) {
        //copy before start execution
        std::vector<uint32> rf_vector_to_gpu(RFSIZE, reg_data);
        
        //copy the check vector
        cuda_check(cudaMemcpyToSymbol(trip_mem1, rf_vector_to_gpu.data(), sizeof(uint32) * RFSIZE, 0));
        cuda_check(cudaMemcpyToSymbol(trip_mem2, rf_vector_to_gpu.data(), sizeof(uint32) * RFSIZE, 0));
        cuda_check(cudaMemcpyToSymbol(trip_mem3, rf_vector_to_gpu.data(), sizeof(uint32) * RFSIZE, 0));


	//Allocate an array of the size of all register bank
	uint32 out_size = block_size.x * block_size.y * threads_per_block.x * RFSIZE;
	uint32 *output_dev1;
        uint32 *output_dev2;
        uint32 *output_dev3;

	//byte size
	const uint32 byte_size = sizeof(uint32) * out_size;

	//malloc on device
	cuda_check(cudaMalloc(&output_dev1, byte_size));
	cuda_check(cudaMalloc(&output_dev2, byte_size));
	cuda_check(cudaMalloc(&output_dev3, byte_size));

	//malloc on host
        uint32 reg_data_tmp = reg_data;
	std::vector<uint32> output_host1(out_size, reg_data_tmp);
        std::vector<uint32> output_host2(out_size, reg_data_tmp);
        std::vector<uint32> output_host3(out_size, reg_data_tmp);

	cuda_check(cudaMemcpy(output_dev1, output_host1.data(), byte_size, cudaMemcpyHostToDevice));
	cuda_check(cudaMemcpy(output_dev2, output_host2.data(), byte_size, cudaMemcpyHostToDevice));
	cuda_check(cudaMemcpy(output_dev3, output_host3.data(), byte_size, cudaMemcpyHostToDevice));


	double start = Log::mysecond();
	test_register_file_kernel<<<block_size, threads_per_block>>>(output_dev1, output_dev2, output_dev3, reg_data, cycles);
        cuda_check(cudaDeviceSynchronize());
        double end = Log::mysecond();
    
        std::cout << "KERNEL TIME " << end - start << std::endl;

	//Copy data back
	cuda_check(cudaMemcpy(output_host1.data(), output_dev1, byte_size, cudaMemcpyDeviceToHost));
        cuda_check(cudaMemcpy(output_host2.data(), output_dev2, byte_size, cudaMemcpyDeviceToHost));
        cuda_check(cudaMemcpy(output_host3.data(), output_dev3, byte_size, cudaMemcpyDeviceToHost));

	cuda_check(cudaFree(output_dev1));
        cuda_check(cudaFree(output_dev2));
        cuda_check(cudaFree(output_dev3));

	Tuple t;
        t.move_register_file(output_host1, output_host2, output_host3);
	return t;

}

Tuple test_register_file(const Parameters& parameters) {
	const uint32 rf_size = 256;
	//Kepler and Volta have
	//fucking 256KB registers per SM
	// so I have to allocate 4 blocks of
	// 256 threads
        
	dim3 block_size(parameters.number_of_sms, 1);
	dim3 threads_per_block(parameters.registers_per_block / rf_size);
    

	uint32 reg_data;
        std::memset((uint32*)&reg_data, parameters.t_byte, sizeof(uint32));

	return test_register_file<rf_size>(reg_data, parameters.one_second_cycles,
			block_size, threads_per_block);
}
