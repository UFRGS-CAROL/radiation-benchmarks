/*
 * MicroLDST.h
 *
 *  Created on: Mar 15, 2020
 *      Author: fernando
 */

#ifndef MICROLDST_H_
#define MICROLDST_H_

#include "Parameters.h"
#include "generic_log.h"
#include "Micro.h"

#include "input_device.h"
//for load and store the OPS value is not used, then we use
//max thread ld/st operation
#define MAX_THREAD_LD_ST_OPERATIONS 16

//Max number of load/stores performed
//each time
#define MEM_OPERATION_NUM 64u

template<typename int_t>
struct MicroLDST: public Micro<int_t> {

	std::vector<int_t> input_host;
	rad::DeviceVector<int_t> input_device;

	MicroLDST(Parameters& parameters, std::shared_ptr<rad::Log>& log) :
			Micro<int_t>(parameters, log) {
		auto& array_size = this->parameters.array_size;
		auto& block_size = this->block_size;
		//Array size is the amount of memory that will be evaluated
		array_size = this->parameters.memory_size_to_use / sizeof(int_t);
		this->parameters.operation_num = MEM_OPERATION_NUM;
		this->grid_size = array_size
				/ (this->parameters.operation_num * block_size);

		auto start_gen = rad::mysecond();
		//Set the output size
		this->output_device.resize(array_size);
		this->input_host.resize(array_size);
		this->input_device.resize(array_size);
		auto end_gen = rad::mysecond();

		if (this->parameters.verbose) {
			std::cout << "Output/input device allocation time: "
					<< end_gen - start_gen << std::endl;
		}
	}

	void get_setup_input() {
		rad::checkFrameworkErrors(cudaMemcpyToSymbol(this->gold.data(),
				common_int_input, sizeof(int_t) * this->block_size, 0));
		this->grow_input_host(this->gold);
	}

	void execute_micro();

private:
	void grow_input_host(const std::vector<int_t>& new_input) {
		auto chunck = new_input.size();

		for (auto begin = this->input_host.begin(), end =
				this->input_host.end(); begin < end; begin += chunck) {
			std::copy(new_input.begin(), new_input.end(), begin);
		}
	}
};

template<>
void MicroLDST<int32_t>::execute_micro();

#endif /* MICROLDST_H_ */
