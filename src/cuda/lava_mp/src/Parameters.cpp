/*
 * Parameters.cpp
 *
 *  Created on: 29/09/2019
 *      Author: fernando
 */

#include <cstdlib>
#include <iostream>

#include "Parameters.h"
#include "common.h"

// Helper functions
//#include "helper_cuda.h"
//#include "helper_string.h"

Parameters::Parameters(int argc, char** argv) {
	if (argc < 2) {
		usage(argc, argv);
		error("");
	}

	//=============================================================
	this->test_precision_description = this->find_char_arg(argc, argv,
			"-precision", "single");
	this->precision = PRE[this->test_precision_description];

	this->test_redundancy_description = this->find_char_arg(argc, argv,
			"-redundancy", "none");
	this->redundancy = RED[this->test_redundancy_description];

	//if not defined set the max number
	this->block_check = this->find_int_arg(argc, argv, "-opnum",
	NUMBER_PAR_PER_BOX);
	//=============================================================

	this->boxes = this->find_int_arg(argc, argv, "-boxes", 10);

	if (this->boxes <= 0) {
		error(
				"Invalid input size given on the command-line: "
						+ std::to_string(this->boxes));
	}

	this->input_distances = "lava_" + this->test_precision_description
			+ "_distances_" + std::to_string(this->boxes);
	this->input_distances = this->find_char_arg(argc, argv, "-input_distances",
			this->input_distances);

	this->input_charges = "lava_" + this->test_precision_description
			+ "_charges_" + std::to_string(boxes);

	this->input_charges = this->find_char_arg(argc, argv, "-input_charges",
			this->input_charges);

	this->output_gold = "lava_" + this->test_precision_description + "_gold_"
			+ std::to_string(boxes);

	this->output_gold = this->find_char_arg(argc, argv, "-output_gold",
			this->output_gold);

	this->iterations = this->find_int_arg(argc, argv, "-iterations", 1);

	this->nstreams = this->find_int_arg(argc, argv, "-streams", 1);

	this->verbose = this->find_arg(argc, argv, "-verbose");

	this->fault_injection = this->find_arg(argc, argv, "-debug");

	if (this->fault_injection) {
		fault_injection = 1;
		std::cout << ("!! Will be injected an input error\n");
	}

	this->generate = this->find_arg(argc, argv, "-generate");

	if (this->generate) {
		this->iterations = 1;
		std::cout
				<< ">> Output will be written to file. Only stream #0 output will be considered.\n";
	}

	this->gpu_check = this->find_arg(argc, argv, "-gpu_check");
	if (this->gpu_check) {
		error("Function not implemented");
	}

}

Parameters::~Parameters() {
	// TODO Auto-generated destructor stub
}

void Parameters::usage(int argc, char** argv) {
	printf("Usage: %s -boxes=N [-generate] [-input_distances=<path>] "
			"[-input_charges=<path>] [-output_gold=<path>] [-iterations=N] "
			"[-streams=N] [-debug] [-verbose]\n", argv[0]);
}

void Parameters::del_arg(int argc, char **argv, int index) {
	int i;
	for (i = index; i < argc - 1; ++i)
		argv[i] = argv[i + 1];
	argv[i] = 0;
}

int Parameters::find_int_arg(int argc, char **argv, std::string arg, int def) {
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

std::string Parameters::find_char_arg(int argc, char **argv, std::string arg,
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

bool Parameters::find_arg(int argc, char** argv, std::string arg) {
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

std::ostream& operator<<(std::ostream& os, const Parameters& p) {
	os << "Boxes " << p.boxes << std::endl;
	os << "Streams " << p.nstreams << std::endl;
	os << "Generate " << p.generate << std::endl;
	os << "Precision " << p.test_precision_description << std::endl;
	os << "Redundancy " << p.test_redundancy_description << std::endl;
	os << "Block check " << p.block_check << std::endl;
	os << "Fault injection " << p.fault_injection << std::endl;
	os << "Verbose " << p.verbose << std::endl;
	os << "Iterations " << p.iterations << std::endl;
	os << "Input distances " << p.input_distances << std::endl;
	os << "Input charges " << p.input_charges << std::endl;
	os << "Output gold " << p.output_gold;
	return os;
}
