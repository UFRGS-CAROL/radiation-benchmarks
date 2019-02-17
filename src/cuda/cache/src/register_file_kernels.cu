/*
 * register_file_kernels.cu
 *
 *  Created on: Feb 2, 2019
 *      Author: carol
 */

#include "kernels.h"
#include "utils.h"
#include <cstring>
#include <ctime>

#include "register_kernel.h"
#include <bitset>


#define MAX_REG_OCCUPATION 186

template<uint32 RFSIZE>
Tuple test_register_file(uint32 reg_data, const int64 cycles, uint32  number_of_sms, uint32 number_of_threads) {
        //copy before start execution
        std::vector<uint32> rf_vector_to_gpu(RFSIZE, reg_data);
        std::vector<uint32> rf_vector_random(RFSIZE, reg_data);
        std::srand(std::time(nullptr));
        for(uint32 i = 0; i < MAX_REG_OCCUPATION; i++){
                rf_vector_random[i] = std::rand();
        }
        
        //Trapping the NVCC
        uint32 *trip_mem1, *trip_mem2, *trip_mem3;
        cuda_check(cudaMalloc(&trip_mem1, RFSIZE * sizeof(uint32)));
        cuda_check(cudaMalloc(&trip_mem2, RFSIZE * sizeof(uint32)));
        cuda_check(cudaMalloc(&trip_mem3, RFSIZE * sizeof(uint32)));
        
        cuda_check(cudaMemcpy(trip_mem1, rf_vector_to_gpu.data(), sizeof(uint32) * RFSIZE, cudaMemcpyHostToDevice));
        cuda_check(cudaMemcpy(trip_mem2, rf_vector_to_gpu.data(), sizeof(uint32) * RFSIZE, cudaMemcpyHostToDevice));
        cuda_check(cudaMemcpy(trip_mem3, rf_vector_random.data(), sizeof(uint32) * RFSIZE, cudaMemcpyHostToDevice));


	//Allocate an array of the size of all register bank
	uint32 out_size = number_of_sms * number_of_threads * RFSIZE;
	uint32 *output_dev1;
        uint32 *output_dev2;
        uint32 *output_dev3;
        
        std::cout << out_size << "\n";

	//byte size
	const uint32 byte_size = sizeof(uint32) * out_size;

	//malloc on device
	cuda_check(cudaMalloc(&output_dev1, byte_size));
	cuda_check(cudaMalloc(&output_dev2, byte_size));
	cuda_check(cudaMalloc(&output_dev3, byte_size));

	//malloc on host
	std::vector<uint32> output_host1(out_size);
        std::vector<uint32> output_host2(out_size);
        std::vector<uint32> output_host3(out_size);

	cuda_check(cudaMemcpy(output_dev1, output_host1.data(), byte_size, cudaMemcpyHostToDevice));
	cuda_check(cudaMemcpy(output_dev2, output_host2.data(), byte_size, cudaMemcpyHostToDevice));
	cuda_check(cudaMemcpy(output_dev3, output_host3.data(), byte_size, cudaMemcpyHostToDevice));


	double start = Log::mysecond();
        if(reg_data == 4294967295){
                std::cout << number_of_sms << " " << number_of_threads << "\n";
                test_register_file_kernel_or<<<number_of_sms, number_of_threads>>>(output_dev1, output_dev2, output_dev3, trip_mem1, trip_mem2, trip_mem3, reg_data, cycles);
        }else if (reg_data == 0){
                test_register_file_kernel_and<<<number_of_sms, number_of_threads>>>(output_dev1, output_dev2, output_dev3, trip_mem1, trip_mem2, trip_mem3, reg_data, cycles);
        }
        cuda_check(cudaDeviceSynchronize());
        double end = Log::mysecond();
    
        //std::cout << "KERNEL TIME " << end - start << std::endl;

	//Copy data back
	cuda_check(cudaMemcpy(output_host1.data(), output_dev1, byte_size, cudaMemcpyDeviceToHost));
        cuda_check(cudaMemcpy(output_host2.data(), output_dev2, byte_size, cudaMemcpyDeviceToHost));
        cuda_check(cudaMemcpy(output_host3.data(), output_dev3, byte_size, cudaMemcpyDeviceToHost));

	cuda_check(cudaFree(output_dev1));
        cuda_check(cudaFree(output_dev2));
        cuda_check(cudaFree(output_dev3));
        
        //Trapping the NVCCc
        cuda_check(cudaFree(trip_mem1));
        cuda_check(cudaFree(trip_mem2));
        cuda_check(cudaFree(trip_mem3));

	Tuple t;
        t.move_register_file(output_host1, output_host2, output_host3);
	return t;

}

Tuple test_register_file(const Parameters& parameters) {
	const uint32 rf_size = 256;
	//Kepler and Volta have
	//fucking 256KB registers per SM
        // 256 threads
        uint32  number_of_threads = parameters.registers_per_block / rf_size;
        std::cout << parameters.registers_per_block << std::endl;

	uint32 reg_data;
        std::memset((uint32*)&reg_data, parameters.t_byte, sizeof(uint32));

	return test_register_file<rf_size>(reg_data, parameters.one_second_cycles, parameters.number_of_sms, number_of_threads);
}
