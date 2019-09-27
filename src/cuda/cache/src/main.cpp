/*
 ============================================================================
 Name        : main.cpp
 Author      : Fernando
 Version     :
 Copyright   : Your copyright notice
 Description : CUDA compute reciprocals
 ============================================================================
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <cstring>
#include <iomanip>

#include "L1Cache.h"
#include "L2Cache.h"
#include "SharedMemory.h"
#include "RegisterFile.h"
#include "ReadOnly.h"

#include "utils.h"
#include "Log.h"
#include "Memory.h"

//GET ECC DATA
#ifdef BUILDPROFILER
#include "NVMLWrapper.h"
#endif

template<typename data_>
void setup_execute(Log& log, Parameters& test_parameter,
		Memory<data_>& memory_obj) {

	std::cout << std::fixed << std::setprecision(6);
	std::vector<uint64> vet = { 0xffffffffffffffff, 0x0000000000000000 };
	// SETUP THE NVWL THREAD
#ifdef BUILDPROFILER
	rad::NVMLWrapper counter_thread(DEVICE_INDEX);
#endif
	for (uint64 iteration = 0; iteration < log.iterations;) {
		for (auto mem : vet) {
#ifdef BUILDPROFILER
			//Start collecting data
			counter_thread.start_collecting_data();
#endif

			//set CUDA configuration each iteration
			memory_obj.set_cache_config(log.test_mode);

			double start_it = rad::mysecond();
			//Start iteration
			log.start_iteration_app();

			memory_obj.test(mem);

			//end iteration
			log.end_iteration_app();
			double end_it = rad::mysecond();

			//End collecting the data
			//This thing must be done before device reset
#ifdef BUILDPROFILER
			counter_thread.end_collecting_data();

			auto iteration_data = counter_thread.get_data_from_iteration();
			for (auto info_line : iteration_data) {
				log.log_info(info_line);
				//std::cout << info_line << std::endl;
			}
#endif

			double start_dev_reset = rad::mysecond();
			//reset the device
			cuda_check(cudaDeviceReset());
			double end_dev_reset = rad::mysecond();

			//Comparing the output
			double start_cmp = rad::mysecond();
			uint32 hits, misses, false_hits;
			std::tie(hits, misses, false_hits) = memory_obj.compare(log, mem);
			double end_cmp = rad::mysecond();

			//update errors
			if (log.errors) {
				log.update_error_count();
				log.update_info_count();
			}

			std::cout << "Iteration: " << iteration;
			std::cout << " Time: " << end_it - start_it;
			std::cout << " Errors: " << log.errors;
			std::cout << " Hits: " << hits;
			std::cout << " Misses: " << misses;
			std::cout << " False hits: " << false_hits;
			std::cout << std::hex;

			std::cout << " Byte: " << std::setw(sizeof(uint64) * 2)
					<< std::setfill('0') << mem;
			std::cout << std::dec;
			std::cout << " Device Reset Time: "
					<< end_dev_reset - start_dev_reset;
			std::cout << " Comparing Time: " << end_cmp - start_cmp;
			std::cout << std::endl;

			iteration++;
		}
	}
}

int main(int argc, char **argv) {
	/**
	 * CHECKING THE SIZES BEFORE START
	 */
	if (sizeof(uint64) != 8) {
		error("UINT64 is not 8 bytes it is " + std::to_string(sizeof(uint64)));
	}

	if (sizeof(uint32) != 4) {
		error("UINT4 is not 4  bytes it is " + std::to_string(sizeof(uint32)));
	}

	bool l2_checked = false;
#ifdef L2TEST
	l2_checked = true;
#endif

	//Parameter to the functions
	Parameters test_parameter;

	//Log obj
	Log log(argc, argv, test_parameter.board_name,
			test_parameter.shared_memory_size, test_parameter.l2_size,
			test_parameter.number_of_sms, test_parameter.one_second_cycles);
	log.set_info_max(2000);
	test_parameter.set_setup_sleep_time(log.seconds_sleep);

	std::cout << test_parameter << " Memory test: " << log.test_mode
			<< std::endl;

	//Test Registers
	if (log.test_mode == "REGISTERS") {
		RegisterFile rf(test_parameter);
		setup_execute(log, test_parameter, rf);
	}

	//test L1
	if (log.test_mode == "L1") {
		//L1Cache l1(test_parameter);
		//setup_execute(log, test_parameter, l1);
		error("NOT WORKING");
	}

	//Test l2
	if (log.test_mode == "L2") {
		if (l2_checked == false) {
			error("YOU MUST BUILD CUDA CACHE TEST WITH: make DISABLEL1CACHE=1");
		}
		L2Cache l2(test_parameter);
		setup_execute(log, test_parameter, l2);
	}

	//Test Shared
	if (log.test_mode == "SHARED") {
		SharedMemory shared(test_parameter);
		setup_execute(log, test_parameter, shared);
	}

	//Test ReadOnly
	if (log.test_mode == "READONLY") {
		ReadOny constant(test_parameter);
		setup_execute(log, test_parameter, constant);
	}

	return 0;

}

