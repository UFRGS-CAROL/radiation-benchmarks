/*
 * Parameters.cpp
 *
 *  Created on: 29/09/2019
 *      Author: fernando
 */

#include <cstdlib>
#include <iostream>

#include "Parameters.h"

// Helper functions
#include "helper_cuda.h"
#include "helper_string.h"

Parameters::Parameters(int argc, char** argv) {
	if (argc < 2) {
		usage(argc, argv);
		exit(EXIT_FAILURE);
	}

	generate = 0;
	iterations = 1000000;
	nstreams = 1;
	fault_injection = 0;
	verbose = 0;
	gpu_check = 0;

	if (checkCmdLineFlag(argc, (const char **) argv, "boxes")) {
		boxes = getCmdLineArgumentInt(argc, (const char **) argv, "boxes");

		if (boxes <= 0) {
			printf("Invalid input size given on the command-line: %d\n", boxes);
			exit(EXIT_FAILURE);
		}
	} else {
		usage(argc, argv);
		exit(EXIT_FAILURE);
	}

	if (checkCmdLineFlag(argc, (const char **) argv, "input_distances")) {
		char **str_ptr;
		getCmdLineArgumentString(argc, (const char **) argv, "input_distances",
				str_ptr);
		this->input_distances = std::string(*str_ptr);
	} else {
		this->input_distances = "lava_" + this->test_precision_description
				+ "_distances_" + std::to_string(this->boxes);
		std::cout << "Using default input_distances path: "
				<< this->input_distances << std::endl;

	}

	if (checkCmdLineFlag(argc, (const char **) argv, "input_charges")) {
		char **str_ptr;

		getCmdLineArgumentString(argc, (const char **) argv, "input_charges",
				str_ptr);
		this->input_charges = std::string(*str_ptr);
	} else {
		this->input_charges = "lava_" + this->test_precision_description
				+ "_charges_" + std::to_string(boxes);
		std::cout << "Using default input_charges path: " << this->input_charges
				<< std::endl;
	}

	if (checkCmdLineFlag(argc, (const char **) argv, "output_gold")) {
		char **str_ptr;

		getCmdLineArgumentString(argc, (const char **) argv, "output_gold",
				str_ptr);
		this->output_gold = std::string(*str_ptr);
	} else {
		this->output_gold = "lava_" + this->test_precision_description
				+ "_gold_" + std::to_string(boxes);
		std::cout << "Using default output_gold path: " << output_gold
				<< std::endl;
	}

	if (checkCmdLineFlag(argc, (const char **) argv, "iterations")) {
		iterations = getCmdLineArgumentInt(argc, (const char **) argv,
				"iterations");
	}

	if (checkCmdLineFlag(argc, (const char **) argv, "streams")) {
		nstreams = getCmdLineArgumentInt(argc, (const char **) argv, "streams");
	}

	if (checkCmdLineFlag(argc, (const char **) argv, "verbose")) {
		verbose = 1;
	}

	if (checkCmdLineFlag(argc, (const char **) argv, "debug")) {
		fault_injection = 1;
		std::cout << ("!! Will be injected an input error\n");
	}

	if (checkCmdLineFlag(argc, (const char **) argv, "generate")) {
		generate = 1;
		iterations = 1;
		std::cout
				<< ">> Output will be written to file. Only stream #0 output will be considered.\n";
	}

	if (checkCmdLineFlag(argc, (const char **) argv, "gpu_check")) {
		gpu_check = 1;
		std::cout << ">>Output will be checked in gpu.\n";
	}

}

Parameters::~Parameters() {
	// TODO Auto-generated destructor stub
}

void Parameters::usage(int argc, char** argv) {
	printf(
			"Usage: %s -boxes=N [-generate] [-input_distances=<path>] [-input_charges=<path>] [-output_gold=<path>] [-iterations=N] [-streams=N] [-debug] [-verbose]\n",
			argv[0]);
}
