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
	size_t memory_size_to_use;

	bool verbose;

	Parameters(int argc, char* argv[]);

	friend std::ostream& operator<<(std::ostream& os, const Parameters& p);
};

#endif /* PARAMETERS_H_ */
