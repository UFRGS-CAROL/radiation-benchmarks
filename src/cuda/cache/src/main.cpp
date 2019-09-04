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

#ifdef BUILDPROFILER
#include "NVMLWrapper.h"
#endif

#include "L1Cache.h"
#include "RegisterFile.h"
#include "utils.h"
#include "Log.h"
#include "Memory.h"

template<typename data_>
void setup_execute(Log& log, Parameters& test_parameter, Memory<data_>& mem,
		bool l2_checked) {

	std::cout << std::fixed << std::setprecision(6);
	for (uint64 iteration = 0; iteration < log.iterations;) {
		for (byte t_byte : { 0xff, 0x00 }) {
#ifdef BUILDPROFILER
			//Start collecting data
			counter_thread.start_collecting_data();
#endif
			double start_it = rad::mysecond();
			//Start iteration
			log.start_iteration_app();

			mem.test(t_byte);

			//end iteration
			log.end_iteration_app();
			double end_it = rad::mysecond();

			//End collecting the data
#ifdef BUILDPROFILER
			counter_thread.end_collecting_data();
#endif
			double start_dev_reset = rad::mysecond();
			//reset the device
			cuda_check(cudaDeviceReset());
			double end_dev_reset = rad::mysecond();

			//Comparing the output
			double start_cmp = rad::mysecond();
			uint32 hits, misses, false_hits;
			std::tie(hits, misses, false_hits) = mem.compare(log, t_byte);
			double end_cmp = rad::mysecond();

			//update errors
			if (log.errors) {
#ifdef BUILDPROFILER
				auto iteration_data = counter_thread.get_data_from_iteration();
				for (auto info_line : iteration_data) {
					log.log_info(info_line);
				}
#endif
				log.update_error_count();
				log.update_info_count();
			}

			std::cout << "Iteration: " << iteration;
			std::cout << " Time: " << end_it - start_it;
			std::cout << " Errors: " << log.errors;
			std::cout << " Hits: " << hits;
			std::cout << " Misses: " << misses;
			std::cout << " False hits: " << false_hits;
			std::cout << " Byte: " << uint32(t_byte);
			std::cout << " Device Reset Time: "	<< end_dev_reset - start_dev_reset;
			std::cout << " Comparing Time: " << end_cmp - start_cmp;
			std::cout << std::endl;

			iteration++;
		}
	}
}

int main(int argc, char **argv) {
	bool l2_checked = false;
#ifdef L2TEST
	l2_checked = true;
#endif

	//Parameter to the functions
	Parameters test_parameter;

	//Log obj
	Log log(argc, argv, test_parameter.board_name, test_parameter.shared_memory_size,
			test_parameter.l2_size, test_parameter.number_of_sms,
			test_parameter.one_second_cycles);
	log.set_info_max(2000);
	test_parameter.set_setup_sleep_time(log.seconds_sleep);

	 // SETUP THE NVWL THREAD
#ifdef BUILDPROFILER
	NVMLWrapper counter_thread(DEVICE_INDEX);
#endif
	std::cout << test_parameter << " Memory test: " << log.test_mode << std::endl;

	//Test Registers
	if (log.test_mode == "REGISTERS") {
		RegisterFile rf(test_parameter);
		setup_execute<uint32>(log, test_parameter, rf, l2_checked);
	}

	//test L1
	if (log.test_mode == "L1") {
		L1Cache l1(test_parameter);
		setup_execute<CacheLine<CACHE_LINE_SIZE>>(log, test_parameter, l1,
				l2_checked);
	}

	//Test l2
	if (log.test_mode == "L2") {
		if (l2_checked == false) {
			error("YOU MUST BUILD CUDA CACHE TEST WITH: make DISABLEL1CACHE=1");
		}
		error("NOT IMPLEMENTED");
	}

	//Test Shared
	if (log.test_mode == "SHARED") {
		error("NOT IMPLEMENTED");
	}

	return 0;

}

