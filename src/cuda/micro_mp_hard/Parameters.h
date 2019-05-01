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

//===================================== DEFINE TESTED PRECISION
//FOR DMR APPROACH I NEED to use the smallest precision
//as a limit, since it is not possible to store the bigger precisions
//on smaller precisions

//If double it means that DMR will be double and float
//so the limits are the float ones

#define INPUT_A_DOUBLE 1.1945305291614955E+103 // 0x5555555555555555
#define INPUT_B_DOUBLE 3.7206620809969885E-103 // 0x2AAAAAAAAAAAAAAA
#define OUTPUT_R_DOUBLE 4.444444444444444 //0x4011C71C71C71C71

#define INPUT_A_SINGLE 1.4660155E+13 // 0x55555555
#define INPUT_B_SINGLE 3.0316488E-13 // 0x2AAAAAAA
#define OUTPUT_R_SINGLE 4.444444 //0x408E38E3

#define INPUT_A_HALF 1.066E+2 // 0x56AA
#define INPUT_B_HALF 4.166E-2 // 0x2955
#define OUTPUT_R_HALF 4.44 // 0x4471

enum MICROINSTRUCTION {
	ADD, MUL, FMA
};

enum PRECISION {
	HALF, SINGLE, DOUBLE
};

enum REDUNDANCY {
	NONE, DMR, TMR, DMRMIXED, TMRMIXED
};

std::string redundancy_char[] = { "NONE", "DMR", "DMRMIXED", "TMRMIXED" };
std::string precision_char[] = { "HALF", "SINGLE", "DOUBLE" };
std::string microinstruction_char[] = { "ADD", "MUL", "FMA" };

struct Parameters {

	MICROINSTRUCTION micro;
	PRECISION precision;
	REDUNDANCY redundancy;

	int iterations;
	bool verbose;
	std::string test_type_description;
	std::string test_precision_description;
	std::string hardening;

	int grid_size;
	int block_size;
	int r_size;

	Parameters(int argc, char* argv[], int grid_size, int block_size) {
		this->grid_size = grid_size;
		this->block_size = block_size;
		this->r_size = grid_size * block_size;

		this->micro = ADD;
		this->precision = SINGLE;
		this->redundancy = NONE;
		this->iterations = 10;
		this->verbose = 0;

		if (checkCmdLineFlag(argc, (const char**) (argv), "iterations")) {
			this->iterations = getCmdLineArgumentInt(argc,
					(const char**) (argv), "iterations");
		}
		if (checkCmdLineFlag(argc, (const char**) (argv), "verbose")) {
			this->verbose = 1;
		}

		if (checkCmdLineFlag(argc, (const char**) (argv), "redundancy")) {
			this->redundancy = REDUNDANCY(
					getCmdLineArgumentInt(argc, (const char**) (argv),
							"redundancy"));
		}

		if (checkCmdLineFlag(argc, (const char**) (argv), "inst")) {
			this->micro = MICROINSTRUCTION(
					getCmdLineArgumentInt(argc, (const char**) (argv), "inst"));
		}

		if (checkCmdLineFlag(argc, (const char**) (argv), "precision")) {
			this->precision = PRECISION(
					getCmdLineArgumentInt(argc, (const char**) (argv),
							"precision"));
		}

		this->test_type_description = precision_char[this->precision];
		this->test_precision_description = microinstruction_char[this->micro];
		this->hardening = redundancy_char[this->redundancy];

	}

	void print_details() {
		std::cout << "cuda micro type - " << this->test_precision_description
				<< " precision " << this->test_type_description << std::endl;
		std::cout << "grid size = " << this->grid_size << " block size = "
				<< this->block_size << std::endl;
	}

	virtual ~Parameters() {
	}
};

#endif /* PARAMETERS_H_ */
