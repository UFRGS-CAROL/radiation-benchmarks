/*
 * MicroInt.h
 *
 *  Created on: Mar 14, 2020
 *      Author: fernando
 */

#ifndef MICROINT_H_
#define MICROINT_H_

#include "Micro.h"
#include "Parameters.h"
#include "generic_log.h"

template<typename int_t>
struct MicroInt: public Micro<int_t> {

	MicroInt(Parameters& parameters, std::shared_ptr<rad::Log>& log) :
			Micro<int_t>(parameters, log) {
	}

	void execute_micro();

};

template<>
void MicroInt<int32_t>::execute_micro();

template<typename int_t>
struct MicroBranch: public Micro<int_t> {

	MicroBranch(Parameters& parameters, std::shared_ptr<rad::Log>& log) :
			Micro<int_t>(parameters, log) {
	}

	void execute_micro();

	void get_setup_input() {
		for(auto i = 0; i < this->gold.size(); i++){
			this->gold[i] = i;
		}
	}

};

template<>
void MicroBranch<int32_t>::execute_micro();


#endif /* MICROINT_H_ */
