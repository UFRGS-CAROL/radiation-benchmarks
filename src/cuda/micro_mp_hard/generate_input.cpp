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

void create_input_file(std::string& output_file){

}

std::vector<double> generate_input_file(double& min_random, double& max_random, size_t r_size) {
	std::vector<double> input(r_size);
	std::random_device rd; //Will be used to obtain a seed for the random number engine
	std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
	std::uniform_real_distribution<double> dis(min_random, max_random);

	for (auto& in : input) {
		in = real_t(dis(gen));
	}
	return input;
}


int main(int argc, char **argv) {

}
