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

//#include "helper_string.h"

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

typedef enum {
	ADD, MUL, FMA
} MICROINSTRUCTION;

typedef enum {
	HALF, SINGLE, DOUBLE
} PRECISION;

typedef enum {
	NONE, DMR, TMR, DMRMIXED, TMRMIXED
} REDUNDANCY;

std::unordered_map<std::string, REDUNDANCY> red = {
//NONE
		{ "none", NONE },
		//DMR
		{ "dmr", DMR },
		// DMRMIXED
		{ "dmrmixed", DMRMIXED },
//TMRMIXED
//         {"TMRMIXED",  XAVIER}
		};

std::unordered_map<std::string, PRECISION> pre = {
//HALF
		{ "half", HALF },
		//SINGLE
		{ "single", SINGLE },
		// DOUBLE
		{ "double", DOUBLE }, };

std::unordered_map<std::string, MICROINSTRUCTION> mic = {
//ADD
		{ "add", ADD },
		//MUL
		{ "mul", MUL },
		//FMA
		{ "fma", FMA }, };

struct Parameters {

	MICROINSTRUCTION micro;
	PRECISION precision;
	REDUNDANCY redundancy;

	int iterations;
	bool verbose;
	std::string instruction_str;
	std::string precision_str;
	std::string hardening_str;

	int grid_size;
	int block_size;
	int r_size;

	Parameters(int argc, char* argv[], int grid_size, int block_size) {
		this->grid_size = grid_size;
		this->block_size = block_size;
		this->r_size = grid_size * block_size;
		this->iterations = find_int_arg(argc, argv, "iterations", 10);

		this->verbose = find_arg(argc, argv, "verbose");

		this->hardening_str = find_char_arg(argc, argv, "redundancy", "none");
		this->instruction_str = find_char_arg(argc, argv, "inst", "add");
		this->precision_str = find_char_arg(argc, argv, "precision", "single");

		this->redundancy = red[this->hardening_str];
		this->precision = pre[this->precision_str];
		this->micro = mic[this->instruction_str];
	}

	void print_details() {
		std::cout << "cuda micro type - " << this->precision_str
				<< " precision " << this->instruction_str << std::endl;
		std::cout << "grid size = " << this->grid_size << " block size = "
				<< this->block_size << std::endl;
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
