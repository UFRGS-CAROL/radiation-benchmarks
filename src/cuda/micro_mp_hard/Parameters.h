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

#include "helper_string.h"

enum MICROINSTRUCTION {
	ADD, MUL, FMA
};

enum PRECISION {
	HALF, SINGLE, DOUBLE
};

enum REDUNDANCY {
	NONE, DMR, TMR, DMRMIXED, TMRMIXED
};

class Parameters {
public:

	MICROINSTRUCTION micro;
	PRECISION precision;
	REDUNDANCY redundancy;

	bool is_dmr;
	int iterations;
	bool verbose;
	std::string test_type_description;
	std::string test_precision_description;
	std::string hardening;

	int gridsize;
	int blocksize;
	int r_size;

	Parameters(int argc, char* argv[]) {
		this->micro = ADD;
		this->precision = SINGLE;
		this->redundancy = NONE;

		if (checkCmdLineFlag(argc, (const char**) (argv), "iterations")) {
			iterations = getCmdLineArgumentInt(argc, (const char**) (argv),
					"iterations");
		}
		if (checkCmdLineFlag(argc, (const char**) (argv), "verbose")) {
			verbose = 1;
		}
		is_dmr = getCmdLineArgumentInt(argc, (const char**) (argv), "dmr");
	}

	void print_details() {
		std::cout << "cuda micro type - " << this->test_precision_description
				<< " precision " << this->test_type_description << std::endl;
		std::cout << "grid size = " << this->gridsize << " block size = "
				<< this->blocksize << std::endl;
	}

	virtual ~Parameters();
};

#endif /* PARAMETERS_H_ */
