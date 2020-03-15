/*
 * Parameters.cpp
 *
 *  Created on: Feb 1, 2020
 *      Author: fernando
 */

#include "Parameters.h"
#include "cuda_utils.h"
#include "utils.h"

Parameters::Parameters(int argc, char* argv[]) {
	this->iterations = rad::find_int_arg(argc, argv, "--iterations", 10);
	this->verbose = rad::find_arg(argc, argv, "--verbose");
	this->instruction_str = rad::find_char_arg(argc, argv, "--inst", "add");
	this->operation_num = rad::find_int_arg(argc, argv, "--opnum",
	LOOPING_UNROLL);
	this->precision_str = rad::find_char_arg(argc, argv, "--precision",
			"float");
	this->gold = rad::find_char_arg(argc, argv, "--gold", "./gold.data");

	this->generate = rad::find_arg(argc, argv, "--generate");

	this->micro = this->mic[this->instruction_str];
	this->precision = this->pre[this->precision_str];

	this->fast_math = rad::find_arg(argc, argv, "--fast_math");

	auto dev_prop = rad::get_device();
	this->device = dev_prop.name;

	if (this->generate) {
		this->iterations = 1;
	}

	this->memory_size_to_use = this->gpu_ddr_by_gpu[this->device];

	//if it is ADD, MUL, or MAD use maximum allocation
	this->sm_count = dev_prop.multiProcessorCount;
	this->global_gpu_memory_bytes = dev_prop.totalGlobalMem;

	if (dev_prop.warpSize != WARP_SIZE) {
		throw_line(
				"Set the correct WARP_SIZE to "
						+ std::to_string(dev_prop.warpSize));
	}

	//both benchmarks will use MAX_THREAD_BLOCK size
	this->block_size = MAX_THREAD_BLOCK;

	//multiplies the grid size by the maximum number of warps per SM
	this->grid_size = this->sm_count * WARP_PER_SM;
	this->array_size = this->block_size * this->grid_size;

	if (argc < 2) {
		std::string help = "You could the following configurations: ";
		help += "\nPrecisions - ";
		for (auto p : this->pre) {
			help += p.first + " ";
		}
		help += "\nMicro benchmarks - ";
		for (auto m : this->mic) {
			help += m.first + " ";
		}

		for (auto s : this->gpu_ddr_by_gpu)
			help += "\nFor " + s.first + " GPU LD/ST micro uses "
					+ std::to_string(s.second) + "\n";

		throw_line(help);
	}
}

std::ostream& operator<<(std::ostream& os, const Parameters& p) {
	os << std::boolalpha;
	os << "Micro type: " << p.instruction_str << std::endl;
	os << "Precision: " << p.precision_str << std::endl;
	os << "SM count; " << p.sm_count << std::endl;
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
	os << "Generate: " << p.generate << std::endl;
	os << "Grids x Blocks: " << p.grid_size << " x " << p.block_size
			<< std::endl;
	os << "Verbose: " << p.verbose << std::endl;
	os << "Iterations: " << p.iterations << std::endl;
	os << "Fast math: " << p.fast_math << std::endl;
	os << "Operations per thread " << p.operation_num << " x " << LOOPING_UNROLL
			<< std::endl;
	os << "Device " << p.device << std::endl;
	os << "Gold file: " << p.gold << std::endl;
	os << "Generate: " << p.generate << std::endl;

	return os;
}
