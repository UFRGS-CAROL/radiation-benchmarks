/*
 * Parameters.cpp
 *
 *  Created on: 22/05/2019
 *      Author: fernando
 */

#include <unordered_map>
#include <iostream>

#include "Parameters.h"
#include "helper_cuda.h"
#include "helper_string.h"

std::unordered_map<std::string, REDUNDANCY> red = {
//NONE
		{ "none", NONE },
		//DMR
		{ "dmr", DMR },
		// DMRMIXED
		{ "dmrmixed", DMRMIXED }, };

std::unordered_map<std::string, PRECISION> pre = {
//HALF
		{ "half", HALF },
		//SINGLE
		{ "single", SINGLE },
		//SINGLE
		{ "float", SINGLE },
		// DOUBLE
		{ "double", DOUBLE }, };

Parameters::Parameters(int argc, char** argv) {
	this->iterations = 10000000;
	this->verbose = 0;
	this->fault_injection = 0;
	this->generate = 0;
	this->precision = SINGLE;
	this->redundancy = NONE;
	this->test_precision_description = "single";
	this->test_redundancy_description = "none";

	if (argc < 2) {
		this->usage(argc, argv);
		exit(EXIT_FAILURE);
	}

	if (checkCmdLineFlag(argc, (const char **) argv, "precision")) {
		char* precision = nullptr;
		getCmdLineArgumentString(argc, (const char **) argv, "precision",
				&precision);
		if (precision) {
			this->test_precision_description = std::string(precision);
			this->precision = pre[this->test_precision_description];
		}
	}

	if (checkCmdLineFlag(argc, (const char **) argv, "redundancy")) {
		char* redundancy = nullptr;
		getCmdLineArgumentString(argc, (const char **) argv, "redundancy",
				&redundancy);
		if (redundancy) {
			this->test_redundancy_description = std::string(redundancy);
			this->redundancy = red[this->test_redundancy_description];
		}
	}

	if (checkCmdLineFlag(argc, (const char **) argv, "boxes")) {
		this->boxes = getCmdLineArgumentInt(argc, (const char **) argv,
				"boxes");

		if (this->boxes <= 0) {
			std::cerr << "Invalid input size given on the command-line: "
					<< this->boxes << std::endl;
			exit(EXIT_FAILURE);
		}
	} else {
		this->usage(argc, argv);
		exit(EXIT_FAILURE);
	}

	if (checkCmdLineFlag(argc, (const char **) argv, "input_distances")) {
		char* tfile_ = nullptr;
		getCmdLineArgumentString(argc, (const char **) argv, "input_distances",
				&tfile_);
		if (tfile_) {
			this->input_distances = std::string(tfile_);
		}
	} else {
		this->input_distances = "lava_" + test_precision_description + "_"
				+ std::to_string(this->boxes);
		std::cout << "Using default input_distances path: "
				<< this->input_distances << std::endl;

	}

	if (checkCmdLineFlag(argc, (const char **) argv, "input_charges")) {
		char* tfile_ = nullptr;
		getCmdLineArgumentString(argc, (const char **) argv, "input_charges",
				&tfile_);
		if (tfile_) {
			this->input_charges = std::string(tfile_);
		}
	} else {
		this->input_charges = "lava_" + test_precision_description + "_"
				+ std::to_string(this->boxes);
		std::cout << "Using default input_charges path: " << this->input_charges
				<< std::endl;

	}

	if (checkCmdLineFlag(argc, (const char **) argv, "output_gold")) {
		char* tfile_ = nullptr;
		getCmdLineArgumentString(argc, (const char **) argv, "output_gold",
				&tfile_);
		if (tfile_) {
			this->output_gold = std::string(tfile_);
		}
	} else {
		this->output_gold = "lava_" + test_precision_description + "_"
				+ std::to_string(this->boxes);
		std::cout << "Using default output_gold path: " << this->output_gold
				<< std::endl;

	}

	if (checkCmdLineFlag(argc, (const char **) argv, "iterations")) {
		this->iterations = getCmdLineArgumentInt(argc, (const char **) argv,
				"iterations");
	}

	if (checkCmdLineFlag(argc, (const char **) argv, "streams")) {
		this->nstreams = getCmdLineArgumentInt(argc, (const char **) argv,
				"streams");
	}

	if (checkCmdLineFlag(argc, (const char **) argv, "verbose")) {
		this->verbose = 1;
	}

	if (checkCmdLineFlag(argc, (const char **) argv, "debug")) {
		this->fault_injection = true;
		std::cout << "!! Will be injected an input error\n";
	}

	if (checkCmdLineFlag(argc, (const char **) argv, "generate")) {
		this->generate = true;
		std::cout
				<< "Output will be written to file. Only stream #0 output will be considered."
				<< std::endl;
	}

	if (checkCmdLineFlag(argc, (const char **) argv, "gpu_check")) {
		this->gpu_check = true;
		printf(">> Output will be checked in gpu.\n");
	}

	if (this->generate)
		this->iterations = 1;

}

Parameters::Parameters() :
		iterations(0), verbose(false), fault_injection(false), generate(false), test_name(
				""), test_precision_description(""), test_redundancy_description(
				""), precision(SINGLE), redundancy(NONE), boxes(0), nstreams(0), gpu_check(
				false) {
}

Parameters::~Parameters() {
	// TODO Auto-generated destructor stub
}

void Parameters::usage(int argc, char** argv) {
	std::cout
			<< "Usage: " + std::string(argv[0])
					+ " -boxes=N [-generate] [-input_distances=<path>] [-input_charges=<path>] "
							"[-output_gold=<path>] [-iterations=N] [-streams=N] [-debug] [-verbose]"
			<< std::endl;
}
