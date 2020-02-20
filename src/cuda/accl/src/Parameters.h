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

//#include "utils.h"

struct Parameters {
	std::string device;
	std::string input;
	std::string gold;

	size_t sm_count;
	size_t iterations;
	size_t nFrames;
	size_t nFramesPerStream;

	bool verbose;
	bool debug;
	bool generate;

	Parameters(int argc, char* argv[]);
	friend std::ostream& operator<<(std::ostream& os, const Parameters& p);
};

#endif /* PARAMETERS_H_ */
