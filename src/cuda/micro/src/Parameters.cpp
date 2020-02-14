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
	this->generate_output = rad::find_char_arg(argc, argv, "--gold",
			"./gold.data");
	this->generate = rad::find_arg(argc, argv, "--generate");

	this->micro = mic[this->instruction_str];
	this->precision = pre[this->precision_str];

	this->fast_math = rad::find_arg(argc, argv, "--fast_math");

	auto dev_prop = rad::get_device();
	this->device = dev_prop.name;

	//if it is ADD, MUL, or MAD use maximum allocation
	this->sm_count = dev_prop.multiProcessorCount;

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
}

std::ostream& operator<<(std::ostream& os, const Parameters& p) {
	os << std::boolalpha;
	os << "Micro type: " << p.instruction_str << std::endl;
	os << "Preicison: " << p.precision_str << std::endl;
	os << "SM count; " << p.sm_count << std::endl;
	os << "Grids x Blocks: " << p.grid_size << " x " << p.block_size
			<< std::endl;
	os << "Verbose: " << p.verbose << std::endl;
	os << "Iterations: " << p.iterations << std::endl;
	os << "Fast math: " << p.fast_math << std::endl;
	os << "Device " << p.device;

	return os;
}
