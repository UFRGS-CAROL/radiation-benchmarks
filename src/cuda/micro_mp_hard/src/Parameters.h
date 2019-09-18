/*
 * Parameters.h
 *
 *  Created on: 29/04/2019
 *      Author: fernando
 */

#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include <iostream>
#include <string>
#include <unordered_map>
#include <iostream>

#include "cuda_utils.h"
#include "common.h"


struct Parameters {

	MICROINSTRUCTION micro;
	PRECISION precision;
	REDUNDANCY redundancy;

	int iterations;
	bool verbose;
	std::string instruction_str;
	std::string precision_str;
	std::string hardening_str;

	bool generate;
	bool nonconstant;

	int grid_size;
	int block_size;
	int r_size;

	int operation_num;

	std::string gold_file;
	std::string input_file;
	std::string device;
	double min_random, max_random;

	Parameters(int argc, char* argv[]) {
		auto dev_prop = this->get_device();
		this->device = dev_prop.name;

		this->grid_size = NUMBER_OF_THREAD_BLOCK;
		this->block_size = MAX_THREAD_BLOCK;
		this->r_size = grid_size * block_size;

		this->iterations = find_int_arg(argc, argv, "--iterations", 10);
		this->operation_num = find_int_arg(argc, argv, "--opnum", OPS);
		this->nonconstant = find_arg(argc, argv, "--nonconstant");
		this->min_random = find_float_arg(argc, argv, "--minrand", 0);
		this->max_random = find_float_arg(argc, argv, "--maxrand", 1000);
		this->gold_file = find_char_arg(argc, argv, "--gold", "./gold.data");
		this->input_file = find_char_arg(argc, argv, "--input", "./input.data");
		this->verbose = find_arg(argc, argv, "--verbose");
		this->hardening_str = find_char_arg(argc, argv, "--redundancy", "none");
		this->instruction_str = find_char_arg(argc, argv, "--inst", "add");
		this->precision_str = find_char_arg(argc, argv, "--precision", "single");
		this->generate = find_arg(argc, argv, "--generate");

		this->redundancy = red[this->hardening_str];
		this->precision = pre[this->precision_str];
		this->micro = mic[this->instruction_str];

		if (this->generate) {
			this->iterations = 1;
		}
	}

	cudaDeviceProp get_device() {
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

	friend std::ostream& operator<<(std::ostream& os, const Parameters& p) {
		os << "cuda micro type - " << p.precision_str << std::endl;
		os << "Precision " << p.instruction_str << std::endl;
		os << "Grid size = " << p.grid_size << std::endl;
		os << "Block size = " << p.block_size << std::endl;
		os << "Verbose: " << p.verbose << std::endl;
		os << "Iterations: " << p.iterations << std::endl;
		os << "Hardening: " << p.hardening_str << std::endl;
		os << "Gold file: " << p.gold_file << std::endl;
		os << "Input file: " << p.input_file << std::endl;
		os << "Min and max random: " << p.min_random << "-" << p.max_random
				<< std::endl;
		os << "Generate: " << p.generate << std::endl;
		os << "Number of operations per thread: " << p.operation_num;

		return os;
	}

private:
	void del_arg(int argc, char **argv, int index) {
		int i;
		for (i = index; i < argc - 1; ++i)
			argv[i] = argv[i + 1];
		argv[i] = 0;
	}

	int find_int_arg(int argc, char **argv, std::string arg, int def) {
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

	float find_float_arg(int argc, char **argv, std::string arg, float def) {
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

	std::string find_char_arg(int argc, char **argv, std::string arg,
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

	bool find_arg(int argc, char* argv[], std::string arg) {
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

};

#endif /* PARAMETERS_H_ */
