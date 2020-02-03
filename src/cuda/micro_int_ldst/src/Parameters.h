/*
 * Parameters.h
 *
 *  Created on: 29/04/2019
 *      Author: fernando
 */

#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include <string>
#include <iostream>
//#include <cuda.h>
#include <cuda_runtime.h>

#include "utils.h"

#ifndef OPS
#define OPS 10000000
#endif

struct Parameters {

	MICROINSTRUCTION micro;

	size_t iterations;
	bool verbose;
	std::string instruction_str;

	bool generate;
	uint32_t sm_count;
	uint32_t global_gpu_memory_bytes;
	std::string gold_file;
	std::string input_file;
	std::string device;

	Parameters(int argc, char* argv[]);

	friend std::ostream& operator<<(std::ostream& os, const Parameters& p);
private:
	void del_arg(int argc, char **argv, int index);

	int find_int_arg(int argc, char **argv, std::string arg, int def);
	float find_float_arg(int argc, char **argv, std::string arg, float def);
	std::string find_char_arg(int argc, char **argv, std::string arg,
			std::string def);

	bool find_arg(int argc, char* argv[], std::string arg);

	cudaDeviceProp get_device();
};

#endif /* PARAMETERS_H_ */
