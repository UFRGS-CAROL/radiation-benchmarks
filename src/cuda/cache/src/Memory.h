/*
 * Memory.h
 *
 *  Created on: Sep 2, 2019
 *      Author: carol
 */

#ifndef MEMORY_H_
#define MEMORY_H_

#include <vector>
#include <sstream>

#include "utils.h"
#include "device_vector.h"
#include "Parameters.h"

//template<typename data_t, typename counter_t>
//struct Comparator{
//
//};

template<typename data_>
struct Memory {
	int64 cycles;
	Board device;
	dim3 block_size;

	std::vector<int64> hit_vector_host;
	std::vector<int64> miss_vector_host;

	//Host memory output
	std::vector<data_> output_host_1;
	std::vector<data_> output_host_2;
	std::vector<data_> output_host_3;

	//Host memory input
	std::vector<data_> input_host_1;

	virtual void test(const uint64& mem) = 0;
	virtual bool call_checker(uint64& gold, Log& log, int64& hits,
			int64& misses, int64& false_hits) = 0;

	virtual std::string error_detail(uint64 i, uint64 e, uint64 r, int64 hits,
			int64 misses, int64 false_hits) {
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

	template<typename T>
	std::tuple<T, bool> get_correct_tmr_value(const T* v1, const T* v2,
			const T* v3, const T& val_gold, const uint64 i, Log& log) {
		//For this case we don't need to check
		if(this->device == K20 || this->device == K40){
			return {true, v1[i]};
		}

		auto check_flag = true;
		auto val_output_1 = v1[i];
		auto val_output_2 = v2[i];
		auto val_output_3 = v3[i];
		auto val_output = val_output_1;

		if ((val_output_1 != val_output_2) || (val_output_1 != val_output_3)) {
#pragma omp critical
			{
				std::stringstream info_detail;
				info_detail << "m: [" << i << "], r0: " << val_output_1;
				info_detail << ", r1: " << val_output_2;
				info_detail << ", r2: " << val_output_3;
				info_detail << ", e: " << val_gold;

				if (log.verbose && (log.errors < 10))
					std::cout << info_detail.str() << std::endl;

				auto inf_det = info_detail.str();
				log.log_info(inf_det);
			}

			if ((val_output_1 != val_output_2) && (val_output_2 != val_output_3)
					&& (val_output_1 != val_output_3)) {
				// All 3 values diverge
				if (val_output_1 == val_gold) {
					val_output = val_output_1;
				} else if (val_output_2 == val_gold) {
					val_output = val_output_2;
				} else if (val_output_3 == val_gold) {
					val_output = val_output_3;
				} else {
					// NO VALUE MATCHES THE GOLD AND ALL 3 DIVERGE!
					check_flag = false;
#pragma omp critical
					{
						std::stringstream info_detail;
						info_detail << "f: [" << i << "], r0: " << val_output_1;
						info_detail << ", r1: " << val_output_2;
						info_detail << ", r2: " << val_output_3;
						info_detail << ", e: " << val_gold;

						if (log.verbose && (log.errors < 10))
							std::cout << info_detail.str() << std::endl;

						auto inf_det = info_detail.str();
						log.log_info(inf_det);
					}
				}
			} else if (val_output_2 == val_output_3) {
				// Only value 0 diverge
				val_output = val_output_2;
			} else if (val_output_1 == val_output_3) {
				// Only value 1 diverge
				val_output = val_output_1;
			} else if (val_output_1 == val_output_2) {
				// Only value 2 diverge
				val_output = val_output_1;
			}
		}

		return {val_output, check_flag};
	}

	// Returns true if no errors are found. False if otherwise.
	// Set votedOutput pointer to retrieve the voted matrix
	template<typename T>
	bool check_output_errors(const T* v1, const T* v2, const T* v3,
			const T& val_gold, Log& log, int64 hits, int64 misses,
			int64 false_hits, size_t size) {

#pragma omp parallel for shared(log)
		for (uint64 i = 0; i < size; i++) {
			T val_output;
			bool check_flag;
			std::tie(val_output, check_flag) = this->get_correct_tmr_value(v1,
					v2, v3, val_gold, i, log);

			if (val_gold != val_output && check_flag) {
#pragma omp critical
				{

					std::string errdet = this->error_detail(i, val_gold,
							val_output, hits, misses, false_hits);
					if (log.verbose && (log.errors < 10))
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

	std::tuple<int64, int64, int64> compare(Log& log, uint64& mem) {
		//Checking the misses
		int64 hits = 0;
		int64 misses = 0;
		int64 false_hits = 0;
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

		this->call_checker(mem, log, hits, misses, false_hits);

		//returning the result
		return std::make_tuple(hits, misses, false_hits);
	}
};

#endif /* MEMORY_H_ */
