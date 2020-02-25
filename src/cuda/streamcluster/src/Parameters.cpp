/*
 * Parameters.cpp
 *
 *  Created on: Feb 1, 2020
 *      Author: fernando
 */

#include <cuda_runtime_api.h>

#include "Parameters.h"
#include "cuda_utils.h"

Parameters::Parameters(int argc, char* argv[]) {
	this->iterations = rad::find_int_arg(argc, argv, "--iterations", 10);
	this->verbose = rad::find_arg(argc, argv, "--verbose");
	this->debug = rad::find_arg(argc, argv, "--debug");
	this->generate = rad::find_arg(argc, argv, "--generate");
	this->input = rad::find_char_arg(argc, argv, "--input", "../../../data/bfs/graph1MW_6.txt");
	this->gold = rad::find_char_arg(argc, argv, "--gold", "./gold.data");
	this->kmin = rad::find_int_arg(argc, argv, "--k1", 10);
	this->kmax = rad::find_int_arg(argc, argv, "--k2", 20);
	this->dim = rad::find_int_arg(argc, argv, "--d", 256);
	this->n = rad::find_int_arg(argc, argv, "--d", 65536);
	this->chunksize = rad::find_int_arg(argc, argv, "--chunksize", 65536);
	this->clustersize = rad::find_int_arg(argc, argv, "--clustersize", 1000);


	auto dev_prop = rad::get_device();
	this->device = dev_prop.name;
	if (this->generate) {
		this->iterations = 1;
	}

	//if it is ADD, MUL, or MAD use maximum allocation
	this->sm_count = dev_prop.multiProcessorCount;

	if (argc < 10) {
		std::string error = "usage: ./" + std::string(argv[0]);
		error +=
				" --k1 <k1> --k2 <k2> --d <d> --n <n> --chunksize <chunksize> "
				"--clustersize <clsize> --input <input file> --gold <gold>\n";
		error += "  k1:          Min. number of centers allowed\n";
		error += "  k2:          Max. number of centers allowed\n";
		error += "  d:           Dimension of each data point\n";
		error += "  n:           Number of data points\n";
		error += "  chunksize:   Number of data points to handle per step\n";
		error += "  clustersize: Maximum number of intermediate centers\n";
		error += "  input:      Input file (if n<=0)\n";
		error += "  gold:       Output file\n";
		error += "  iterations: are radiation test iterations\n";
		error += "--generate, --debug and --verbose are optional";

		error +=
				"if n > 0 and --generate is not given, points will be"
				" randomly generated instead of reading from infile.\n";
		throw_line(error);
	}
}

std::ostream& operator<<(std::ostream& os, const Parameters& p) {
	os << std::boolalpha;
	os << "Testing CFD on " << p.device << std::endl;
	os << "Input path: " << p.input << std::endl;
	os << "Gold path: " << p.gold << std::endl;
	os << "Iterations: " << p.iterations << std::endl;
	os << "Generate: " << p.generate << std::endl;
	os << "SM count = " << p.sm_count << std::endl;
	os << "Verbose: " << p.verbose;
	return os;
}

