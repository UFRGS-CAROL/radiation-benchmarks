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


//for load and store the OPS value is not used, then we use
//max thread ld/st operation
#define MAX_THREAD_LD_ST_OPERATIONS 16

//Max number of load/stores performed
//each time
#define MEM_OPERATION_NUM 64u

template<typename int_t>
struct MicroLDST : public Micro<int_t>{

	MicroLDST(Parameters& parameters, std::shared_ptr<rad::Log>& log) :
			Micro<int_t>(parameters, log) {
//		//both benchmarks will use MAX_THREAD_BLOCK size
//			this->block_size = MAX_THREAD_BLOCK;
//			size_t final_input_size;
//			if (this->parameters.micro == LDST) {
//				//Array size is the amount of memory that will be evaluated
//				this->array_size = this->parameters.memory_size_to_use
//						/ sizeof(int_t);
//				this->parameters.operation_num = MEM_OPERATION_NUM;
//				this->grid_size = this->array_size
//						/ (this->parameters.operation_num * this->block_size);
//				final_input_size = this->array_size;
//			} else {
//				//multiplies the grid size by the maximum number of warps per SM
//				this->grid_size = this->parameters.sm_count * WARP_PER_SM;
//				this->array_size = this->grid_size * this->block_size;
//				final_input_size = this->block_size;
//			}
//
//			if (this->parameters.generate) {
//				std::cout << "Generating or reading input\n";
//				auto start_gen = rad::mysecond();
//				this->generate_input();
//				auto end_gen = rad::mysecond();
//
//				if (this->parameters.verbose) {
//					std::cout << "Input generation time: " << end_gen - start_gen
//							<< std::endl;
//				}
//			} else {
//				//It must be the final size of basic input
//				//then the input is resized
//				std::vector<int_t> new_input(this->block_size);
//
//				this->gold_host.resize(this->block_size);
//				this->read_from_file(this->parameters.input, new_input.data(), this->block_size);
//				if (this->parameters.micro == LDST) {
//					this->gold_host = new_input;
//				}else{
//					this->read_from_file(this->parameters.gold, this->gold_host.data(), this->block_size);
//				}
//				this->input_host.resize(final_input_size);
//				this->grow_input_host(new_input, final_input_size);
//
//
//			}
//			auto out_allocation = rad::mysecond();
//			//Set the output size
//			this->output_device.resize(this->array_size);
//			this->output_host.resize(this->array_size);
//			this->input_device = this->input_host;
//
//			out_allocation = rad::mysecond() - out_allocation;
//
//			if (this->parameters.verbose) {
//				std::cout << "Output device allocation time: " << out_allocation
//						<< std::endl;
//				std::cout << "Size of input array " << this->input_host.size()
//						<< " size of output " << this->output_host.size()
//						<< std::endl;
//			}
	}

	void generate_input() {
//		// First create an instance of an engine.
//		std::random_device rnd_device;
//		// Specify the engine and distribution.
//		std::mt19937 mersenne_engine { rnd_device() }; // Generates random integers
//		std::uniform_int_distribution<int_t> dist { RANGE_INT_MIN, RANGE_INT_MAX };
//
//		std::vector<int_t> new_input;
//		for (auto i = 0; i < MAX_THREAD_BLOCK; i++) {
//			auto new_element = dist(mersenne_engine);
//			new_input.push_back(new_element);
//		}
//
//		if (this->parameters.micro == LDST) {
//			this->input_host.resize(this->array_size);
//			//must be equal to the output when executing LDST
//			this->grow_input_host(new_input, this->array_size);
//		} else {
//			this->input_host = new_input;
//			if (this->file_exists(this->parameters.input)) {
//				this->read_from_file(this->parameters.input,
//						this->input_host.data(), this->input_host.size());
//			}
//		}
//
//		if (parameters.debug) {
//			this->input_host[rand() % this->input_host.size()] = 0;
//		}
	}

	void execute_micro();
};

template<>
void MicroLDST<int32_t>::execute_micro();

#endif /* MICROLDST_H_ */
