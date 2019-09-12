/**
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */

#include <string>
#include <random>
#include <fstream>
#include <iostream>

#define V100_STREAM_MULTIPROCESSOR 84

void create_input_file(std::string& output_file, std::vector<double>& vect_data,
		double min_, double max_, std::ofstream& ofs) {
	ofs << "__device__ __constant__ double input_volta_";
	ofs << min_ << "_" << max_ << "[V100_STREAM_MULTIPROCESSOR] = {"
			<< std::endl;

	for (auto t : vect_data) {
		ofs << "\t\t" << t << "," << std::endl;
	}

	ofs << "};" << std::endl << std::endl;
}

std::vector<double> generate_input_random(double& min_random,
		double& max_random, size_t r_size) {
	std::vector<double> input(r_size);
	std::random_device rd; //Will be used to obtain a seed for the random number engine
	std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
	std::uniform_real_distribution<double> dis(min_random, max_random);

	for (auto& in : input) {
		in = double(dis(gen));
	}
	return input;
}

int main(int argc, char **argv) {
	std::string input_constant_h_file_path = "input_constant.h";

	std::vector<std::pair<double, double> > possible_inputs = { { 0, 10 }, // 0 to 10
			{ 10, 100 }, // 10 to 100
			{ 100, 1000 }, // 100 to 1000
			};

	std::ofstream ofs(input_constant_h_file_path, std::ofstream::out);
	if (ofs.good()) {

		ofs << "#ifndef INPUT_CONSTANT_H_" << std::endl;
		ofs << "#define INPUT_CONSTANT_H_" << std::endl << std::endl;
		ofs << "#define V100_STREAM_MULTIPROCESSOR "
				<< V100_STREAM_MULTIPROCESSOR << std::endl << std::endl;

		for (auto in : possible_inputs) {
			auto generated_input = generate_input_random(in.first, in.second,
			V100_STREAM_MULTIPROCESSOR);
			create_input_file(input_constant_h_file_path, generated_input,
					in.first, in.second, ofs);
		}

	}

	ofs << std::endl << "#endif /* INPUT_CONSTANT_H_ */" << std::endl;
	ofs.close();

	return 0;
}
