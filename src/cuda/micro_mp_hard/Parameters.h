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
#include <iostream>

//===================================== DEFINE TESTED PRECISION
//FOR DMR APPROACH I NEED to use the smallest precision
//as a limit, since it is not possible to store the bigger precisions
//on smaller precisions

//If double it means that DMR will be double and float
//so the limits are the float ones

//#define INPUT_A_DOUBLE 1.1945305291614955E+103 // 0x5555555555555555
//#define INPUT_B_DOUBLE 3.7206620809969885E-103 // 0x2AAAAAAAAAAAAAAA
//#define OUTPUT_R_DOUBLE 4.444444444444444 //0x4011C71C71C71C71

//CHANGING FOR DMR
//I'm going to use 15 digits for double and 7 digits for float
#define INPUT_A_DOUBLE 1.194530529161495E+02 // 0x5555555555555555
#define INPUT_B_DOUBLE 3.720662080996988E-10 // 0x2AAAAAAAAAAAAAAA
#define OUTPUT_R_DOUBLE 4.194530529161495 //0x4011C71C71C71C71

#define INPUT_A_SINGLE 1.4660155E+13 // 0x55555555
#define INPUT_B_SINGLE 3.0316488E-13 // 0x2AAAAAAA
#define OUTPUT_R_SINGLE 4.444444 //0x408E38E3

#define INPUT_A_HALF 1.066E+2 // 0x56AA
#define INPUT_B_HALF 4.166E-2 // 0x2955
#define OUTPUT_R_HALF 4.44 // 0x4471

typedef enum {
	ADD, MUL, FMA, ADDNOTBIASED, MULNOTBIASED, FMANOTBIASED
} MICROINSTRUCTION;

typedef enum {
	HALF, SINGLE, DOUBLE
} PRECISION;

typedef enum {
	NONE, DMR, TMR, DMRMIXED, TMRMIXED
} REDUNDANCY;



#define __DEVICE_HOST__ __device__ __host__ __forceinline__
#define __HOST__ __host__ __forceinline__
#define __DEVICE__ __device__ __forceinline__

typedef uint64_t uint64;
typedef uint32_t uint32;
typedef unsigned char byte;

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
		{ "fma", FMA },
		// NUMCOMPOSE (add not biased)
		{ "addnotbiased", ADDNOTBIASED },
		// MUL not biased
		{ "mulnotbiased", MULNOTBIASED },
		// MUL not biased
		{ "fmanotbiased", FMANOTBIASED } };

template<typename ...TypeArgs> struct Type;

template<>
struct Type<half> {
	half output_r;
	half input_a;
	half input_b;
	Type() {
		this->output_r = OUTPUT_R_HALF;
		this->input_a = INPUT_A_HALF;
		this->input_b = INPUT_B_HALF;
	}
};

template<>
struct Type<float> {
	float output_r;
	float input_a;
	float input_b;
	Type() {
		this->output_r = OUTPUT_R_SINGLE;
		this->input_a = INPUT_A_SINGLE;
		this->input_b = INPUT_B_SINGLE;
	}
};

template<>
struct Type<double> {
	double output_r;
	double input_a;
	double input_b;
	Type() {
		this->output_r = OUTPUT_R_DOUBLE;
		this->input_a = INPUT_A_DOUBLE;
		this->input_b = INPUT_B_DOUBLE;
	}
};

template<>
struct Type<half, float> {
	float output_r;
	float input_a;
	float input_b;
	Type() {
		Type<float> temp;
		this->output_r = temp.output_r;
		this->input_a = temp.input_a;
		this->input_b = temp.input_b;
	}

};

template<>
struct Type<float, double> {
	double output_r;
	double input_a;
	double input_b;
	Type() {
		Type<double> temp;
		this->output_r = temp.output_r;
		this->input_a = temp.input_a;
		this->input_b = temp.input_b;
	}
};

std::ostream& operator<<(std::ostream& os, const half& t) {
	float tmp = float(t);
	os << tmp;
	return os;
}

template<typename ...TypeArgs>
std::ostream& operator<<(std::ostream& os, const Type<TypeArgs...>& t) {
	os << std::scientific;
	os << t.output_r << " " << t.input_a << " " << t.input_b;
	return os;
}

struct Parameters {

	MICROINSTRUCTION micro;
	PRECISION precision;
	REDUNDANCY redundancy;

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
	double min_random, max_random;

	Parameters(int argc, char* argv[], int grid_size, int block_size) {
		this->grid_size = grid_size;
		this->block_size = block_size;
		this->r_size = grid_size * block_size;
		this->iterations = find_int_arg(argc, argv, "--iterations", 10);
		this->operation_num = find_int_arg(argc, argv, "--opnum",
				NUM_COMPOSE_DIVISOR);

		this->nonconstant = find_arg(argc, argv, "--nonconstant");

		this->min_random = find_float_arg(argc, argv, "--minrand", 0);
		this->max_random = find_float_arg(argc, argv, "--maxrand", 1000);

		this->gold_file = find_char_arg(argc, argv, "--gold", "./gold.data");
		this->input_file = find_char_arg(argc, argv, "--input", "./input.data");

		this->verbose = find_arg(argc, argv, "--verbose");

		this->hardening_str = find_char_arg(argc, argv, "--redundancy", "none");
		this->instruction_str = find_char_arg(argc, argv, "--inst", "add");
		this->precision_str = find_char_arg(argc, argv, "--precision",
				"single");

		this->redundancy = red[this->hardening_str];
		this->precision = pre[this->precision_str];
		this->micro = mic[this->instruction_str];

		this->generate = find_arg(argc, argv, "--generate");

//		if (this->micro == MULNOTBIASED || this->micro == FMANOTBIASED
//				|| this->micro == ADDNOTBIASED || this->nonconstant == true) {
//			this->grid_size *= 32;
//			this->r_size *= 32;
//		}

		if (this->generate) {
			this->iterations = 1;
		}
	}

	friend std::ostream& operator<<(std::ostream& os, const Parameters& p) {
		os << "cuda micro type - " << p.precision_str << std::endl;
		os << "Precision " << p.instruction_str << std::endl;
		os << "Grid size = " << p.grid_size << std::endl;
		os << "Block size = " << p.block_size << std::endl;
		os << "Verbose: " << p.verbose << std::endl;
		os << "Iterations: " << p.iterations << std::endl;
		os << "Hardening: " << p.hardening_str << std::endl;
		os << "Gold file: " << p.gold_file << std::endl;
		os << "Input file: " << p.input_file << std::endl;
		os << "Min and max random: " << p.min_random << "-" << p.max_random
				<< std::endl;
		os << "Generate: " << p.generate << std::endl;
		os << "Number of operations per thread: " << p.operation_num;

		return os;
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

	float find_float_arg(int argc, char **argv, std::string arg, float def) {
		for (int i = 0; i < argc - 1; ++i) {
			if (!argv[i])
				continue;
			if (std::string(argv[i]) == arg) {
				std::string to_convert(argv[i + 1]);

				def = std::stof(to_convert);
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
