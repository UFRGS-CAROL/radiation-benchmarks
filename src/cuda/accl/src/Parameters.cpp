/*
 * Parameters.cpp
 *
 *  Created on: Feb 1, 2020
 *      Author: fernando
 */

#include <cuda_runtime_api.h>

#include "Parameters.h"
#include "cuda_utils.h"
#include "utils.h"

Parameters::Parameters(int argc, char* argv[]) {
	this->iterations = rad::find_int_arg(argc, argv, "--iterations", 10);
	this->verbose = rad::find_arg(argc, argv, "--verbose");
	this->debug = rad::find_arg(argc, argv, "--debug");
	this->generate = rad::find_arg(argc, argv, "--generate");
	this->input = rad::find_char_arg(argc, argv, "--input", "../../../data/accl/7Frames.pgm ");
	this->gold = rad::find_char_arg(argc, argv, "--gold", "./gold.data");
	this->nFrames = rad::find_int_arg(argc, argv, "--size", 7);
	this->nFramesPerStream = rad::find_int_arg(argc, argv, "--nFramesPerStream", 7);


	auto dev_prop = rad::get_device();
	this->device = dev_prop.name;
	if (this->generate) {
		this->iterations = 1;
	}

	//if it is ADD, MUL, or MAD use maximum allocation
	this->sm_count = dev_prop.multiProcessorCount;

	if (argc < 6) {
		throw_line(
				"<mandatory arguments> [optional arguments]\nUsage: ./cudaACCL"
				" --size <N frames in the image>"
				" --frames <(HyperQ) Frames per Stream>"
				" --input <Input image path>"
				" --gold <GOLD path>"
				" --iterations <#iteractions>"
				" [--verbose] [--debug]");
	}

	if (this->nFrames < this->nFramesPerStream) {
		throw_line("Num Frames per stream should be less than or equal to numFrames in image");
	}
}

std::ostream& operator<<(std::ostream& os, const Parameters& p) {
	os << std::boolalpha;
	os << "Testing ACCL on " << p.device << std::endl;
	os << "nFrames: " << p.nFrames  << std::endl;
	os << "nFramsPerStream: " << p.nFramesPerStream  << std::endl;
	os << "Input path: " << p.input << std::endl;
	os << "Gold path: " << p.gold << std::endl;
	os << "Iterations: " << p.iterations << std::endl;
	os << "Generate: " << p.generate << std::endl;
	os << "SM count = " << p.sm_count << std::endl;
	os << "Verbose: " << p.verbose;
	return os;
}

