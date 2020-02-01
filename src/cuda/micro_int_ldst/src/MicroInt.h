/*
 * MicroInt.h
 *
 *  Created on: Feb 1, 2020
 *      Author: fernando
 */

#ifndef MICROINT_H_
#define MICROINT_H_

#include <vector>


#include "device_vector.h"
#include "utils.h"

struct MicroInt {
	MICROINSTRUCTION& micro;
	dim3 grid;
	dim3 block;

	std::vector<int32_t> input_host;
	std::vector<int32_t> output_host;

	rad::DeviceVector<int32_t> input_device;
	rad::DeviceVector<int32_t> output_device;


	MicroInt(MICROINSTRUCTION& m, dim3& grid, dim3& threads);
	virtual ~MicroInt();

	void select_micro();
};

#endif /* MICROINT_H_ */
