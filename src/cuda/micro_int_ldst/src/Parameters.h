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

struct Parameters {

	MICROINSTRUCTION micro;
	std::string instruction_str;
	std::string device;

	size_t sm_count;
	size_t iterations;
	size_t global_gpu_memory_bytes;
	size_t operation_num;

	bool verbose;

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
