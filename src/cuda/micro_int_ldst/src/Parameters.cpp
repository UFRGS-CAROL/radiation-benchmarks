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
	this->iterations = find_int_arg(argc, argv, "--iterations", 10);
	this->operation_num = find_int_arg(argc, argv, "--opnum", OPS);
	this->gold_file = find_char_arg(argc, argv, "--gold", "./gold.data");
	this->input_file = find_char_arg(argc, argv, "--input", "./input.data");
	this->verbose = find_arg(argc, argv, "--verbose");
	this->instruction_str = find_char_arg(argc, argv, "--inst", "add");
	this->generate = find_arg(argc, argv, "--generate");

	this->micro = mic[this->instruction_str];

	if (this->generate) {
		this->iterations = 1;
	}

	auto dev_prop = this->get_device();
	this->device = dev_prop.name;
	//deviceProp.totalGlobalMem
	//deviceProp.warpSize
	//if it is ADD, MUL, or MAD use maximum allocation
	this->block_size = MAX_THREAD_BLOCK;
	this->grid_size = WAPR_PER_SM * dev_prop.multiProcessorCount;
	this->global_gpu_memory_bytes = dev_prop.totalGlobalMem;
	this->array_size = WAPR_PER_SM * dev_prop.multiProcessorCount * MAX_THREAD_BLOCK;

	if (this->micro == LDST){
		this->grid_size = dev_prop.multiProcessorCount;
		//Input and output arrays
		this->array_size = (this->global_gpu_memory_bytes / sizeof(int32_t)) / 2;
	}

	if (dev_prop.warpSize != WARP_SIZE) {
		throw_line(
				"Set the correct WARP_SIZE to "
						+ std::to_string(dev_prop.warpSize));
	}

}

cudaDeviceProp Parameters::get_device() {
	//================== Retrieve and set the default CUDA device
	cudaDeviceProp prop;
	int count = 0;

	rad::checkFrameworkErrors(cudaGetDeviceCount(&count));
	for (int i = 0; i < count; i++) {
		rad::checkFrameworkErrors(cudaGetDeviceProperties(&prop, i));
	}
	int *ndevice;
	int dev = 0;
	ndevice = &dev;
	rad::checkFrameworkErrors(cudaGetDevice(ndevice));

	rad::checkFrameworkErrors(cudaSetDevice(0));
	rad::checkFrameworkErrors(cudaGetDeviceProperties(&prop, 0));

	return prop;
}

std::ostream& operator<<(std::ostream& os, const Parameters& p) {
	os << "Precision " << p.instruction_str << std::endl;
	os << "Grid size = " << p.grid_size << std::endl;
	os << "Block size = " << p.block_size << std::endl;
	os << "Verbose: " << p.verbose << std::endl;
	os << "Iterations: " << p.iterations << std::endl;
	os << "Gold file: " << p.gold_file << std::endl;
	os << "Input file: " << p.input_file << std::endl;
	os << "Generate: " << p.generate << std::endl;
	os << "Number of operations per thread: " << p.operation_num;

	return os;
}

void Parameters::del_arg(int argc, char **argv, int index) {
	int i;
	for (i = index; i < argc - 1; ++i)
		argv[i] = argv[i + 1];
	argv[i] = 0;
}

int Parameters::find_int_arg(int argc, char **argv, std::string arg, int def) {
	int i;
	for (i = 0; i < argc - 1; ++i) {
		if (!argv[i])
			continue;
		if (std::string(argv[i]) == arg) {
			def = atoi(argv[i + 1]);
			del_arg(argc, argv, i);
			del_arg(argc, argv, i);
			break;
		}
	}
	return def;
}

float Parameters::find_float_arg(int argc, char **argv, std::string arg,
		float def) {
	for (int i = 0; i < argc - 1; ++i) {
		if (!argv[i])
			continue;
		if (std::string(argv[i]) == arg) {
			std::string to_convert(argv[i + 1]);

			def = std::stof(to_convert);
			del_arg(argc, argv, i);
			del_arg(argc, argv, i);
			break;
		}
	}
	return def;
}

std::string Parameters::find_char_arg(int argc, char **argv, std::string arg,
		std::string def) {
	int i;
	for (i = 0; i < argc - 1; ++i) {
		if (!argv[i])
			continue;
		if (std::string(argv[i]) == arg) {
			def = std::string(argv[i + 1]);
			del_arg(argc, argv, i);
			del_arg(argc, argv, i);
			break;
		}
	}
	return def;
}

bool Parameters::find_arg(int argc, char* argv[], std::string arg) {
	int i;
	for (i = 0; i < argc; ++i) {
		if (!argv[i])
			continue;
		if (std::string(argv[i]) == arg) {
			del_arg(argc, argv, i);
			return true;
		}
	}
	return false;
}

