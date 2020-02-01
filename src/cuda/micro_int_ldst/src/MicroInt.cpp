/*
 * MicroInt.cpp
 *
 *  Created on: Feb 1, 2020
 *      Author: fernando
 */

#include "MicroInt.h"

MicroInt::MicroInt(Parameters& parameters) :
		parameters(parameters) {
	//Setting input and output host and device
	if (this->parameters.micro == LDST) {
		this->input_host.resize(this->parameters.array_size);
	} else {
		this->input_host.resize(MAX_THREAD_BLOCK);
		auto val = 100;
		for(auto& i : this->input_host){
			i = val++;
		}
	}

	this->output_host.resize(this->parameters.array_size);
	this->output_device.resize(this->parameters.array_size);
	this->input_device = this->input_host;
}
