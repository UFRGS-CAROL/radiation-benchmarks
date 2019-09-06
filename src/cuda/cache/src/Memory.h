/*
 * Memory.h
 *
 *  Created on: Sep 2, 2019
 *      Author: carol
 */

#ifndef MEMORY_H_
#define MEMORY_H_

#include <vector>

#include "utils.h"
#include "device_vector.h"
#include "Parameters.h"

template<typename data_>
struct Memory {
	uint64 cycles;
	Board device;
	dim3 block_size;

	std::vector<uint64> hit_vector_host;
	std::vector<uint64> miss_vector_host;

	//Host memory output
	std::vector<data_> output_host_1;

	//Host memory input
	std::vector<data_> input_host_1;

	virtual void test(const uint32& mem) = 0;
	virtual void call_checker(const std::vector<data_>& v1,
			const uint32& valGold, Log& log, uint64 hits, uint64 misses,
			uint64 false_hits, bool verbose) = 0;

	virtual std::string error_detail(uint32 i, uint32 e, uint32 r, uint64 hits,
			uint64 misses, uint64 false_hits) = 0;

	virtual ~Memory() = default;

	Memory(const Parameters& parameter) {
		this->cycles = parameter.one_second_cycles;
		this->device = parameter.device;
		this->block_size = dim3(parameter.number_of_sms);

		this->cycles = parameter.one_second_cycles;
		this->device = parameter.device;
	}

	Memory(const Memory& b) {
		output_host_1 = b.output_host_1;
		input_host_1 = b.input_host_1;
		cycles = b.cycles;
		device = b.device;
		block_size = b.block_size;
	}

	Memory& operator=(const Memory& b) {
		this->output_host_1 = b.output_host_1;
		this->input_host_1 = b.input_host_1;
		this->cycles = b.cycles;
		this->device = b.device;
		this->block_size = b.block_size;
		return *this;
	}

	void set_cache_config(std::string mem_type) {
		if (mem_type == "L1" || mem_type == "L2" || mem_type == "REGISTERS") {
			cuda_check(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
		} else {
			cuda_check(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));
		}
	}

	std::string info_detail(uint32 i, uint32 r1, uint32 r2, uint32 r3,
			uint32 gold) {
		std::string info_detail = "m: [" + std::to_string(i) + "], r1: "
				+ std::to_string(r1) + ", r2: " + std::to_string(r2) + ", r3: "
				+ std::to_string(r3) + ", e: " + std::to_string(gold);
		return info_detail;
	}

	// Returns true if no errors are found. False if otherwise.
	// Set votedOutput pointer to retrieve the voted matrix
	template<typename raw_data_>
	bool check_output_errors(const raw_data_* v1, const raw_data_& val_gold,
			Log& log, uint64 hits, uint64 misses, uint64 false_hits,
			bool verbose, uint32 size) {

#pragma omp parallel for shared(host_errors)
		for (uint32 i = 0; i < size; i++) {
			auto val_output = v1[i];

			if (val_gold != val_output) {
#pragma omp critical
					{

						std::string errdet = this->error_detail(i, val_gold,
								val_output, hits, misses, false_hits);
						if (verbose && (log.errors < 10))
							std::cout << errdet << std::endl;

						log.log_error(errdet);
					}
			}
		}

		if (log.errors != 0) {
			std::cout << "#" << std::endl;
		}
		if (log.infos != 0) {
			std::cout << "M" << std::endl;
		}
		return log.errors == 0 || log.infos == 0;
	}

	std::tuple<uint64, uint64, uint64> compare(Log& log, const uint32& mem) {
		//Checking the misses
		uint64 hits = 0;
		uint64 misses = 0;
		uint64 false_hits = 0;
		for (uint32 i = 0; i < this->hit_vector_host.size(); i++) {
			uint64 hit = this->hit_vector_host[i];
			uint64 miss = this->miss_vector_host[i];
			if (hit <= miss) {
				hits++;
			}
			if (miss < hit) {
				false_hits++;
			} else {
				misses++;
			}
		}

		this->call_checker(this->output_host_1, mem, log, hits, misses,
				false_hits, log.verbose);

		//returning the result
		return std::make_tuple(hits, misses, false_hits);
	}
};

#endif /* MEMORY_H_ */
