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

#include "common.h"

struct Parameters {

	MICROINSTRUCTION micro;
	PRECISION precision;
	std::string instruction_str;
	std::string device;
	std::string precision_str;
	std::string gold;
	std::string input;

	size_t sm_count;
	size_t iterations;
	size_t operation_num;
	size_t memory_size_to_use;
	size_t global_gpu_memory_bytes;

	size_t grid_size;
	size_t block_size;
	size_t array_size;

	bool verbose;
	bool generate;
	bool fast_math;

	Parameters(int argc, char* argv[]);

	friend std::ostream& operator<<(std::ostream& os, const Parameters& p);

private:
	std::unordered_map<std::string, MICROINSTRUCTION> mic = {
	//ADD
			{ "add", ADD },
			//MUL
			{ "mul", MUL },
			//FMA
			{ "fma", FMA },
			//MAD
			{ "mad", FMA },
			//DIV
			{ "div", DIV },
			//Pythagorean
			{ "pythagorean", PYTHAGOREAN },
			//EULER
			{ "euler", EULER },
			//Branch
			{ "branch", BRANCH },
			//ldst
			{ "ldst", LDST }, };

	std::unordered_map<std::string, PRECISION> pre = {
	//half
			{ "half", HALF },
			//float
			{ "single", SINGLE },
			//another name for float
			{ "float", SINGLE },
			//double
			{ "double", DOUBLE },
			//INT32
			{ "int32", INT32 },
			//INT64
			{ "int64", INT64 }, };

	//The amount of memory that will be used
	//in the LDST test
	std::unordered_map<std::string, size_t> gpu_ddr_by_gpu = {
			{ "TITAN V", 1024ull * 1024ull * 1024ull },
			//MUL
			{ "V100", 1024ull * 1024ull * 1024ull },
			//FMA
			{ "Tesla K20c", 1024ull * 1024ull * 256ull },

			{ "Tesla K40c", 1024ull * 1024ull * 256ull }, };
};

#endif /* PARAMETERS_H_ */
