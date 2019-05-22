/*
 * main.cpp
 *
 *  Created on: 17/05/2019
 *      Author: fernando
 */

#include "Parameters.h"
#include "HotspotExecute.h"


int main(int argc, char** argv) {

	Parameters setup_parameters(argc, argv);
	std::cout << "Parameters" << std::endl;
	std::cout << setup_parameters << std::endl;

	std::string test_info = std::string("streams:")
			+ std::to_string(setup_parameters.nstreams) + " precision:"
			+ setup_parameters.test_precision_description + " size:"
			+ std::to_string(setup_parameters.grid_rows) + +" pyramidHeight:"
			+ std::to_string(setup_parameters.pyramid_height) + " simTime:"
			+ std::to_string(setup_parameters.sim_time) + " redundancy:"
			+ setup_parameters.test_redundancy_description;
	std::string test_name = "cuda_hotspot_"
			+ setup_parameters.test_precision_description;

	Log log(test_name, test_info, setup_parameters.generate);


	HotspotExecute setup_execution(setup_parameters, log);

	setup_execution.run();
	return 0;
}
