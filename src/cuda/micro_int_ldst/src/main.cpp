/*
 ============================================================================
 Name        : main.cpp
 Author      : Fernando
 Version     :
 Copyright   : Your copyright notice
 Description : CUDA compute reciprocals
 ============================================================================
 */

#include <iostream>
#include "utils.h"
#include "Log.h"
#include "Parameters.h"
#include "MicroInt.h"

#include "cuda_utils.h"

//GET ECC DATA
#ifdef BUILDPROFILER
#include "NVMLWrapper.h"
#endif

template<typename int_t>
void setup_execute(Log& log, Parameters& test_parameter,
		MicroInt<int_t>& micro_obj) {

	// SETUP THE NVWL THREAD
#ifdef BUILDPROFILER
	rad::NVMLWrapper counter_thread(DEVICE_INDEX);
#endif
	for (size_t iteration = 0; iteration < test_parameter.iterations;
			iteration++) {
#ifdef BUILDPROFILER
		//Start collecting data
		counter_thread.start_collecting_data();
#endif

		auto start_it = rad::mysecond();
		//Start iteration
		log.start_iteration();
		micro_obj.execute_micro();

		rad::checkFrameworkErrors(cudaPeekAtLastError());
		rad::checkFrameworkErrors(cudaDeviceSynchronize());

		//end iteration
		log.end_iteration();
		auto end_it = rad::mysecond();

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

		//Copying from GPU
		auto start_cpy = rad::mysecond();
		micro_obj.copy_back_output();
		auto end_cpy = rad::mysecond();

		//Comparing the output
		auto start_cmp = rad::mysecond();
		auto errors = micro_obj.compare_output();
		auto end_cmp = rad::mysecond();

		//update errors
		log.update_errors();
		log.update_infos();

		if (test_parameter.verbose) {
			// PERF
			auto kernel_time = end_it - start_it;
			auto cmp_time = end_cmp - start_cmp;
			auto copy_time = end_cpy - start_cpy;

			std::cout << "Iteration " << iteration;
			std::cout << " Time: " << kernel_time;
			std::cout << " Comparison time: " << cmp_time;
			std::cout << " Copy time: " << copy_time;
			std::cout << " Output errors: " << errors << std::endl;
		}
	}
}

template<typename int_t>
void setup(Parameters& parameters){

	//================== Init logs
	std::string test_info = "";
	test_info += " gridsize:" + std::to_string(parameters.sm_count);
//	test_info += " blocksize:" + std::to_string(micro_obj.block_size);
	test_info += " type:" + parameters.instruction_str;
	test_info += " kernel_type:non-persistent";
//	test_info += " checkblock:" + std::to_string(micro_obj.operation_num);
//	test_info += " numop:" + std::to_string(micro_obj.operation_num);

	std::string test_name = std::string("cuda_micro-int-")
			+ parameters.instruction_str;

	Log log(test_name, test_info);
	MicroInt<int_t> micro_obj(parameters, log);

	if (parameters.verbose) {
		std::cout << log << std::endl;
	}

	setup_execute(log, parameters, micro_obj);
}

int main(int argc, char **argv) {
	//================== Set block and grid size for MxM kernel
	Parameters parameters(argc, argv);
	if (parameters.verbose) {
		std::cout << parameters << std::endl;
	}
	setup<int32_t>(parameters);
	return 0;

}

