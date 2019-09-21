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

template<typename data_, typename hit_miss_data_ = int64>
struct Memory {
	hit_miss_data_ cycles;
	Board device;
	dim3 block_size;

	std::vector<hit_miss_data_> hit_vector_host;
	std::vector<hit_miss_data_> miss_vector_host;

	//Host memory output
	std::vector<data_> output_host_1;

	//Host memory input
	std::vector<data_> input_host_1;

	virtual void test(const data_& mem) = 0;

	virtual std::string error_detail(hit_miss_data_ i, data_ e, data_ r,
			hit_miss_data_ hits, hit_miss_data_ misses,
			hit_miss_data_ false_hits) {
		std::string error_detail = "";
		error_detail += " i:" + std::to_string(i);
		error_detail += " cache_line:" + std::to_string(i / CACHE_LINE_SIZE);
		error_detail += " e:" + std::to_string(e);
		error_detail += " r:" + std::to_string(r);
		error_detail += " hits: " + std::to_string(hits);
		error_detail += " misses: " + std::to_string(misses);
		error_detail += " false_hits: " + std::to_string(false_hits);
		return error_detail;
	}

	virtual ~Memory() = default;

	friend std::ostream& operator<<(std::ostream& os, Memory& mem) {
		os << "Cycles: " << mem.cycles << std::endl;
		os << "Device: " << mem.device << std::endl;
		os << "Block Size: " << mem.block_size.x;
		return os;
	}

	Memory(const Parameters& parameter) {
		this->cycles = parameter.one_second_cycles;
		this->block_size = dim3(parameter.number_of_sms);
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

	void set_cache_config(std::string& mem_type) {
		if (mem_type == "L1" || mem_type == "REGISTERS") {
			cuda_check(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
		} else {
			cuda_check(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));
		}
	}

	std::string info_detail(hit_miss_data_ i, data_ r1, data_ r2, data_ r3,
			data_ gold) {
		std::string info_detail = "m: [" + std::to_string(i) + "], r1: "
				+ std::to_string(r1) + ", r2: " + std::to_string(r2) + ", r3: "
				+ std::to_string(r3) + ", e: " + std::to_string(gold);
		return info_detail;
	}

	// Returns true if no errors are found. False if otherwise.
	// Set votedOutput pointer to retrieve the voted matrix
	bool check_output_errors(const std::vector<data_>& v1,
			const data_& val_gold, Log& log, hit_miss_data_ hits,
			hit_miss_data_ misses, hit_miss_data_ false_hits, bool verbose) {

#pragma omp parallel for shared(host_errors)
		for (auto i = 0; i < v1.size(); i++) {
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

	std::tuple<hit_miss_data_, hit_miss_data_, hit_miss_data_> compare(Log& log,
			const data_& mem) {
		//Checking the misses
		auto hits = 0;
		auto misses = 0;
		auto false_hits = 0;
		auto zero_cout = 0;
		for (uint32 i = 0; i < this->hit_vector_host.size(); i++) {
			auto hit = this->hit_vector_host[i];
			auto miss = this->miss_vector_host[i];
			if (hit <= miss) {
				hits++;
			}
			if (miss < hit) {
				false_hits++;
			} else {
				misses++;
			}

			zero_cout += (hit == 0 || miss == 0);
		}

		this->check_output_errors(this->output_host_1, mem, log, hits, misses,
				false_hits, log.verbose);

		if(zero_cout != 0){
			error("Zero count is different from 0: " + std::to_string(zero_cout));
		}
		//returning the result
		return std::make_tuple(hits, misses, false_hits);
	}

};

#endif /* MEMORY_H_ */
