/*
 * MicroInt.h
 *
 *  Created on: Feb 1, 2020
 *      Author: fernando
 */

#ifndef MICROINT_H_
#define MICROINT_H_

#include <vector>

#include "Parameters.h"
#include "device_vector.h"
#include "utils.h"

struct MicroInt {
	Parameters& parameters;

	std::vector<int32_t> input_host;
	std::vector<int32_t> output_host;

	rad::DeviceVector<int32_t> input_device;
	rad::DeviceVector<int32_t> output_device;


	MicroInt(Parameters& parameters);
	virtual ~MicroInt() = default;

	void execute_micro();
	size_t compare_output();
};

#endif /* MICROINT_H_ */
