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
	HotspotExecute setup_execution(setup_parameters);

	setup_execution.run();
	return 0;
}
