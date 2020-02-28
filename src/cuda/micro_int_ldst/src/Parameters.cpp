/*
 * Parameters.cpp
 *
 *  Created on: Feb 1, 2020
 *      Author: fernando
 */

#include <cuda_runtime_api.h>

#include "Parameters.h"
#include "cuda_utils.h"
#include "utils.h"

Parameters::Parameters(int argc, char* argv[]) {
	this->iterations = rad::find_int_arg(argc, argv, "--iterations", 10);
	this->verbose = rad::find_arg(argc, argv, "--verbose");
	this->instruction_str = rad::find_char_arg(argc, argv, "--inst", "add");
	this->operation_num = rad::find_int_arg(argc, argv, "--opnum", LOOPING_UNROLL);
	this->micro = mic[this->instruction_str];
	this->debug = rad::find_arg(argc, argv, "--debug");

	auto dev_prop = rad::get_device();
	this->device = dev_prop.name;
	this->memory_size_to_use = gpu_ddr_by_gpu[this->device];

	//if it is ADD, MUL, or MAD use maximum allocation
	this->sm_count = dev_prop.multiProcessorCount;
	this->global_gpu_memory_bytes = dev_prop.totalGlobalMem;

	if (dev_prop.warpSize != WARP_SIZE) {
		throw_line(
				"Set the correct WARP_SIZE to "
						+ std::to_string(dev_prop.warpSize));
	}

}

std::ostream& operator<<(std::ostream& os, const Parameters& p) {
	os << "Micro type " << p.instruction_str << std::endl;
	os << "SM count = " << p.sm_count << std::endl;
	constexpr auto mb = 1 << 20;
	constexpr auto gb = 1 << 30;

	os << "Amount of global memory = " << float(p.global_gpu_memory_bytes) / gb
			<< "GB (" << p.global_gpu_memory_bytes << ") bytes" << std::endl;
	os << "Amount of memory that will be used = ";

	if (p.micro == LDST) {
		os << p.memory_size_to_use / mb;
	} else {
		os
				<< float(
						p.sm_count * WARP_PER_SM * MAX_THREAD_BLOCK
								* sizeof(int32_t)) / mb;
	}
	os << "MB" << std::endl;
	os << "Verbose: " << p.verbose << std::endl;
	os << "Iterations: " << p.iterations << std::endl;
	os << "Device " << p.device;

	return os;
}
