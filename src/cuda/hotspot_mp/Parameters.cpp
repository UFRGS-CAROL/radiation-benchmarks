/*
 * Parameters.cpp
 *
 *  Created on: 18/05/2019
 *      Author: fernando
 */

#include "Parameters.h"
// Helper functions
#include "helper_cuda.h"
#include "helper_string.h"

#ifndef DEFAULT_SIM_TIME
#define DEFAULT_SIM_TIME 10000
#endif

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
	this->nstreams = 1;
	this->sim_time = DEFAULT_SIM_TIME;
	this->pyramid_height = 1;
	this->setup_loops = 10000000;
	this->verbose = 0;
	this->fault_injection = 0;
	this->generate = 0;
	this->size = 0;

	this->precision = SINGLE;
	this->redundancy = NONE;
	this->test_precision_description = "single";
	this->test_redundancy_description = "none";

	if (argc < 2) {
		usage(argc, argv);
		exit(EXIT_FAILURE);
	}

	if (checkCmdLineFlag(argc, (const char **) argv, "pyramid_height")) {
		this->pyramid_height = getCmdLineArgumentInt(argc, (const char **) argv,
				"pyramid_height");
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

	if (checkCmdLineFlag(argc, (const char **) argv, "size")) {
		this->grid_cols = getCmdLineArgumentInt(argc, (const char **) argv,
				"size");
		this->grid_rows = this->grid_cols;

		if ((this->grid_cols <= 0) || (this->grid_cols % 16 != 0)) {
			std::cerr << "Invalid input size given on the command-line: "
					<< this->grid_cols << std::endl;
			exit(EXIT_FAILURE);
		}
	} else {
		usage(argc, argv);
		exit(EXIT_FAILURE);
	}

	if (checkCmdLineFlag(argc, (const char **) argv, "generate")) {
		this->generate = true;
		std::cout
				<< "Output will be written to file. Only stream #0 output will be considered."
				<< std::endl;
	}

	if (checkCmdLineFlag(argc, (const char **) argv, "sim_time")) {
		this->sim_time = getCmdLineArgumentInt(argc, (const char **) argv,
				"sim_time");

		if (this->sim_time < 1) {
			std::cerr << "Invalid sim_time given on the command-line: "
					<< this->sim_time << std::endl;
			exit(EXIT_FAILURE);
		}
	}

	if (checkCmdLineFlag(argc, (const char **) argv, "input_temp")) {
		char* tfile_ = nullptr;
		getCmdLineArgumentString(argc, (const char **) argv, "input_temp",
				&tfile_);
		if (tfile_) {
			this->tfile = std::string(tfile_);
		}
	} else {
		this->tfile = "temp_" + std::to_string(this->grid_rows);
		std::cout << "Using default input_temp path: " << this->tfile
				<< std::endl;

	}

	if (checkCmdLineFlag(argc, (const char **) argv, "input_power")) {
		char *pfile_ = nullptr;
		getCmdLineArgumentString(argc, (const char **) argv, "input_power",
				&pfile_);
		if (pfile_) {
			this->pfile = std::string(pfile_);
		}
	} else {
		this->pfile = "power_" + std::to_string(this->grid_rows);
		std::cout << "Using default input_power path: " << this->pfile
				<< std::endl;

	}

	if (checkCmdLineFlag(argc, (const char **) argv, "gold_temp")) {
		char *ofile_ = nullptr;
		getCmdLineArgumentString(argc, (const char **) argv, "gold_temp",
				&ofile_);
		if (ofile_) {
			this->ofile = std::string(ofile_);
		}
	} else {
		this->ofile = "gold_temp_" + this->test_precision_description + "_"
				+ std::to_string(this->grid_rows) + "_"
				+ std::to_string(this->sim_time);
		std::cout << "Using default gold path: " << this->ofile << std::endl;
	}

	if (checkCmdLineFlag(argc, (const char **) argv, "iterations")) {
		this->setup_loops = getCmdLineArgumentInt(argc, (const char **) argv,
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

	if (this->generate)
		this->setup_loops = 1;

}

Parameters::Parameters() {
	this->nstreams = 1;
	this->sim_time = DEFAULT_SIM_TIME;
	this->pyramid_height = 1;
	this->setup_loops = 10000000;
	this->verbose = 0;
	this->fault_injection = 0;
	this->generate = 0;
	this->size = 0;

	this->precision = SINGLE;
	this->redundancy = NONE;
	this->grid_cols = this->grid_rows = 0;
}

void Parameters::usage(int argc, char** argv) {
	std::cout << "Usage: " << argv[0]
			<< " [-size=N] [-generate] [-sim_time=N] [-input_temp=<path>] [-input_power=<path>] "
					"[-gold_temp=<path>] [-iterations=N] [-streams=N] "
					"[-debug] [-verbose] [-redundancy=<redundancy>] [-precision=<precision>] \n";

}

Parameters::~Parameters() {
}

std::ostream& operator<<(std::ostream& os, const Parameters& p) {
	os << std::boolalpha << "";
	os << "Parameters" << std::endl;
	os << "N streams: " << p.nstreams << std::endl;
	os << "sim time: " << p.sim_time << std::endl;
	os << "pyramid height: " << p.pyramid_height << std::endl;
	os << "setup loops: " << p.setup_loops << std::endl;
	os << "verbose: " << p.verbose << std::endl;
	os << "fault injection: " << p.fault_injection << std::endl;
	os << "generate: " << p.generate << std::endl;
	os << "size: " << p.size << std::endl;
	os << "precision: " << p.test_precision_description << std::endl;
	os << "redundancy: " << p.test_redundancy_description << std::endl;
	os << "cols x rows: " << p.grid_cols << "x" << p.grid_rows;

	return os;
}
