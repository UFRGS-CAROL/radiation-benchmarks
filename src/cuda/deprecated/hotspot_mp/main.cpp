/*
 * main.cpp
 *
 *  Created on: 17/05/2019
 *      Author: fernando
 */

#include "Parameters.h"
#include "HotspotExecute.h"

#include <cuda_runtime.h>
#include <cuda.h>
#include "cuda_utils.h"

cudaDeviceProp GetDevice() {
//================== Retrieve and set the default CUDA device
	cudaDeviceProp prop;
	int count = 0;

	rad::checkFrameworkErrors(cudaGetDeviceCount(&count));
	for (int i = 0; i < count; i++) {
		rad::checkFrameworkErrors(cudaGetDeviceProperties(&prop, i));
	}
	int *ndevice;
	int dev = 0;
	ndevice = &dev;
	rad::checkFrameworkErrors(cudaGetDevice(ndevice));

	rad::checkFrameworkErrors(cudaSetDevice(0));
	rad::checkFrameworkErrors(cudaGetDeviceProperties(&prop, 0));

	return prop;
}



int main(int argc, char** argv) {

	Parameters setup_parameters(argc, argv);
	if (setup_parameters.verbose) {

		std::cout << setup_parameters << std::endl;
	}
	std::string test_info = std::string("streams:")
			+ std::to_string(setup_parameters.nstreams) + " precision:"
			+ setup_parameters.test_precision_description + " size:"
			+ std::to_string(setup_parameters.grid_rows) + +" pyramidHeight:"
			+ std::to_string(setup_parameters.pyramid_height) + " simTime:"
			+ std::to_string(setup_parameters.sim_time) + " redundancy:"
			+ setup_parameters.test_redundancy_description + " checkblock:"
			+ std::to_string(CHECKBLOCK);

	std::string test_name = "cuda_hotspot_"
			+ setup_parameters.test_precision_description;

	Log log(test_name, test_info, setup_parameters.generate);

	HotspotExecute setup_execution(setup_parameters, log);

	setup_execution.run();
	return 0;
}
