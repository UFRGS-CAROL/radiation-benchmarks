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

#include "utils.h"
//#include "Log.h"

//GET ECC DATA
#ifdef BUILDPROFILER
#include "NVMLWrapper.h"
#endif

//template<typename data_>
//void setup_execute(Log& log, Parameters& test_parameter,
//		Memory<data_>& memory_obj) {
//
//	std::cout << std::fixed << std::setprecision(6);
//	std::vector<uint64> vet = { 0xffffffffffffffff, 0x0000000000000000 };
//	// SETUP THE NVWL THREAD
//#ifdef BUILDPROFILER
//	rad::NVMLWrapper counter_thread(DEVICE_INDEX);
//#endif
//	for (uint64 iteration = 0; iteration < log.iterations;) {
//		for (auto mem : vet) {
//#ifdef BUILDPROFILER
//			//Start collecting data
//			counter_thread.start_collecting_data();
//#endif
//
//			//set CUDA configuration each iteration
//			memory_obj.set_cache_config(log.test_mode);
//
//			double start_it = rad::mysecond();
//			//Start iteration
//			log.start_iteration_app();
//
//			memory_obj.test(mem);
//
//			//end iteration
//			log.end_iteration_app();
//			double end_it = rad::mysecond();
//
//			//End collecting the data
//			//This thing must be done before device reset
//#ifdef BUILDPROFILER
//			counter_thread.end_collecting_data();
//
//			auto iteration_data = counter_thread.get_data_from_iteration();
//			for (auto info_line : iteration_data) {
//				log.log_info(info_line);
//				//std::cout << info_line << std::endl;
//			}
//#endif
//
//			double start_dev_reset = rad::mysecond();
//			//reset the device
//			cuda_check(cudaDeviceReset());
//			double end_dev_reset = rad::mysecond();
//
//			//Comparing the output
//			double start_cmp = rad::mysecond();
//			uint32 hits, misses, false_hits;
//			std::tie(hits, misses, false_hits) = memory_obj.compare(log, mem);
//			double end_cmp = rad::mysecond();
//
//			//update errors
//			if (log.errors) {
//				log.update_error_count();
//				log.update_info_count();
//			}
//
//			if (log.verbose) {
//				std::cout << "Iteration: " << iteration;
//				std::cout << " Time: " << end_it - start_it;
//				std::cout << " Errors: " << log.errors;
//				std::cout << " Info(memory errors): " << log.infos;
//				std::cout << " Hits: " << hits;
//				std::cout << " Misses: " << misses;
//				std::cout << " False hits: " << false_hits;
//				std::cout << std::hex;
//
//				std::cout << " Byte: " << std::setw(sizeof(uint64) * 2)
//						<< std::setfill('0') << mem;
//				std::cout << std::dec;
//				std::cout << " Device Reset Time: "
//						<< end_dev_reset - start_dev_reset;
//				std::cout << " Comparing Time: " << end_cmp - start_cmp;
//				std::cout << std::endl;
//			}
//			iteration++;
//		}
//	}
//}

int main(int argc, char **argv) {

	return 0;

}




