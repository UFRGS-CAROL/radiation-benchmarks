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

#include "utils.h"

#ifndef OPS
#define OPS 1000
#endif

struct Parameters {

	MICROINSTRUCTION micro;

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

	Parameters(int argc, char* argv[]);

	cudaDeviceProp get_device() ;
	friend std::ostream& operator<<(std::ostream& os, const Parameters& p);
private:
	void del_arg(int argc, char **argv, int index);

	int find_int_arg(int argc, char **argv, std::string arg, int def);
	float find_float_arg(int argc, char **argv, std::string arg, float def);
	std::string find_char_arg(int argc, char **argv, std::string arg,
			std::string def) ;

	bool find_arg(int argc, char* argv[], std::string arg);

};

#endif /* PARAMETERS_H_ */
