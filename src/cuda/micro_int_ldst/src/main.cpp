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

void setup_execute(Log& log, Parameters& test_parameter, MicroInt& micro_obj) {

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

		double start_it = rad::mysecond();
		//Start iteration
		log.start_iteration();
		micro_obj.execute_micro();
		//end iteration
		log.end_iteration();
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

		//Comparing the output
		double start_cmp = rad::mysecond();
		double end_cmp = rad::mysecond();

		//update errors
		log.update_errors();
		log.update_infos();

		if (test_parameter.verbose) {

		}
		iteration++;

	}
}

int main(int argc, char **argv) {
	//================== Set block and grid size for MxM kernel
	Parameters parameters(argc, argv);
	//================== Init logs
	std::string test_info = "";
	test_info += " gridsize:" + std::to_string(parameters.grid_size);
	test_info += " blocksize:" + std::to_string(parameters.block_size);
	test_info += " type:" + parameters.instruction_str;
	test_info += " kernel_type:non-persistent";
	test_info += " checkblock:" + std::to_string(parameters.operation_num);
	test_info += " numop:" + std::to_string(parameters.operation_num);

	std::string test_name = std::string("cuda_micro-")
			+ parameters.instruction_str;

	Log log(test_name, test_info);

	if (parameters.verbose) {
		std::cout << "Get device Name: " << parameters.device << std::endl;
		std::cout << log << std::endl;
	}

	MicroInt micro_obj(parameters);
	setup_execute(log, parameters, micro_obj);
	return 0;

}

