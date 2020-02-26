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
	this->input = rad::find_char_arg(argc, argv, "--input", "./input");
	this->gold = rad::find_char_arg(argc, argv, "--gold", "./gold.data");
	this->rows = rad::find_int_arg(argc, argv, "--rows", 1024);
	this->cols = rad::find_int_arg(argc, argv, "--cols", 1024);
	this->pyramid_height = rad::find_int_arg(argc, argv, "--pyramid_height",
			10);

	auto dev_prop = rad::get_device();
	this->device = dev_prop.name;
	if (this->generate) {
		this->iterations = 1;
	}

	//if it is ADD, MUL, or MAD use maximum allocation
	this->sm_count = dev_prop.multiProcessorCount;

	if (argc < 8) {

		std::string error =
				"Usage: " + std::string(argv[0])
						+ " --rows <rows> --cols <cols> --pyramid_height <pyramid_height>"
								" --generate --iterations <iterations> --verbose --debug\n";
		error += "--generate, --verbose and --debug are not mandatory\n";
		throw_line(error);
	}

}

std::ostream& operator<<(std::ostream& os, const Parameters& p) {
	os << std::boolalpha;
	os << "Testing PathFinder on " << p.device << std::endl;
	os << "Input path: " << p.input << std::endl;
	os << "Gold path: " << p.gold << std::endl;
	os << "Iterations: " << p.iterations << std::endl;
	os << "Rows: " << p.rows << std::endl;
	os << "Cols: " << p.cols << std::endl;
	os << "Pyramid height: " << p.pyramid_height << std::endl;
	os << "Generate: " << p.generate << std::endl;
	os << "SM count = " << p.sm_count << std::endl;
	os << "Verbose: " << p.verbose;
	return os;
}

