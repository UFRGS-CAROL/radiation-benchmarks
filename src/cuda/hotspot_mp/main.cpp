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

	HotspotExecute setup_execution(setup_parameters);

	setup_execution.run();
	return 0;
}
