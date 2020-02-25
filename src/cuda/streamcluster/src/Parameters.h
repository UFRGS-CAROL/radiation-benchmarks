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

#define WARPS_PER_SM 2

//it is the decimal places for
//logging errors, 20 is from old benchmarks
#define ERROR_LOG_PRECISION 20

static inline void __throw_line(std::string err, std::string line, std::string file) {
	throw std::runtime_error(err + " at " + file + ":" + line);
}

#define throw_line(err) __throw_line(std::string(err), std::to_string(__LINE__), std::string(__FILE__));

struct Parameters {
	std::string device;
	std::string input;
	std::string gold;

	size_t sm_count;
	size_t iterations;

	bool verbose;
	bool debug;
	bool generate;

	Parameters(int argc, char* argv[]);
	friend std::ostream& operator<<(std::ostream& os, const Parameters& p);
};

#endif /* PARAMETERS_H_ */
