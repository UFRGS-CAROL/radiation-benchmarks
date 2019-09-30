/*
 * Parameters.h
 *
 *  Created on: 29/09/2019
 *      Author: fernando
 */

#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include <string>

struct Parameters {
	int boxes, generate;
	std::string input_distances, input_charges, output_gold;
	int iterations, verbose, fault_injection, nstreams, gpu_check;
	std::string test_precision_description;

	Parameters();

	Parameters(int argc, char** argv);

	virtual ~Parameters();

	void usage(int argc, char** argv);
};

#endif /* PARAMETERS_H_ */
