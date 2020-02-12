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
#include <iomanip>      // std::setprecision

#include "utils.h"
#include "Parameters.h"
#include "Micro.h"

#include "cuda_utils.h"
#include "generic_log.h"

//GET ECC DATA
#ifdef BUILDPROFILER
#include "NVMLWrapper.h"
#endif

//#ifdef LOGS
//#include "log_helper.h"
//
//#ifdef BUILDPROFILER
//
//#ifdef FORJETSON
//#include "include/JTX2Inst.h"
//#define OBJTYPE JTX2Inst
//#else
//#include "include/NVMLWrapper.h"
//#define OBJTYPE NVMLWrapper
//#endif
//#endif
//
//#endif

template<typename real_t>
void setup_execute(Parameters& test_parameter, Micro<real_t>& micro_obj) {

	// SETUP THE NVWL THREAD
#ifdef BUILDPROFILER
	rad::NVMLWrapper counter_thread(DEVICE_INDEX);
	//TODO
	//Do the same of the other benchmarks
#endif
	for (size_t iteration = 0; iteration < test_parameter.iterations;
			iteration++) {

		auto start_it = rad::mysecond();
		//Start iteration
		micro_obj.log->start_iteration();
		micro_obj.execute_micro();

		rad::checkFrameworkErrors(cudaPeekAtLastError());
		rad::checkFrameworkErrors(cudaDeviceSynchronize());

		//end iteration
		micro_obj.log->end_iteration();
		auto end_it = rad::mysecond();

		//Copying from GPU
		auto start_cpy = rad::mysecond();
		micro_obj.copy_back_output();
		auto end_cpy = rad::mysecond();

		//Comparing the output
		auto start_cmp = rad::mysecond();
		auto errors = micro_obj.compare_output();
		//update errors
		micro_obj.log->update_errors();
		micro_obj.log->update_infos();
		auto end_cmp = rad::mysecond();

		auto start_reset_output = rad::mysecond();
		micro_obj.reset_output_device();
		auto end_reset_output = rad::mysecond();

		if (test_parameter.verbose) {
			// PERF
			auto kernel_time = end_it - start_it;
			auto cmp_time = end_cmp - start_cmp;
			auto copy_time = end_cpy - start_cpy;
			auto reset_time = end_reset_output - start_reset_output;
			auto wasted_time = cmp_time + copy_time + reset_time;

			std::cout << "Iteration " << iteration;
			std::cout << " Kernel time: " << kernel_time;
			std::cout << " Output errors: " << errors;
			std::cout << " Wasted time (copy + compare + reset): "
					<< wasted_time;
			std::cout << " Wasted time percentage: "
					<< (wasted_time / kernel_time) * 100.0 << "%";
			std::cout << " Comparison time: " << cmp_time;
			std::cout << " Copy time: " << copy_time;
			std::cout << " Reset time: " << reset_time << std::endl;
		}
	}
}

template<typename real_t>
void setup(Parameters& parameters) {
	std::cout << std::setprecision(6) << std::setfill('0');
	std::shared_ptr < rad::Log > log_ptr;

	Micro<real_t> micro_obj(parameters, log_ptr);

//#ifdef LOGS
//	char test_info[250];
//	char test_name[250];
//	snprintf(test_info, 250, "ops:%d gridsize:%d blocksize:%d type:%s-%s-precision", OPS, gridsize, blocksize, test_type_description, test_precision_description);
//	snprintf(test_name, 250, "cuda_%s_micro-%s", test_precision_description, test_type_description);
//	start_log_file(test_name, test_info);
//
//#ifdef BUILDPROFILER
//	std::string log_file_name(get_log_file_name());
//	std::shared_ptr<rad::Profiler> profiler_thread = std::make_shared<rad::OBJTYPE>(0, log_file_name);
//
////START PROFILER THREAD
//	profiler_thread->start_profile();
//#endif
//#endif

	//================== Init logs
	std::string test_info = "";
	test_info += " gridsize:" + std::to_string(parameters.grid_size);
	test_info += " blocksize:" + std::to_string(parameters.block_size);
	test_info += " type:" + parameters.instruction_str + "-"
			+ parameters.precision_str;
	test_info += " opnum:" + std::to_string(parameters.operation_num);
	test_info += " fast_math:" + std::to_string(parameters.fast_math);

	std::string test_name = std::string("cuda_micro-")
			+ parameters.instruction_str + "-" + parameters.precision_str;

	log_ptr = std::make_shared < rad::Log > (test_name, test_info);

	if (parameters.verbose) {
		std::cout << *log_ptr << std::endl;
	}

	setup_execute(parameters, micro_obj);
}

int main(int argc, char **argv) {
	//================== Set block and grid size for MxM kernel
	Parameters parameters(argc, argv);
	if (parameters.verbose) {
		std::cout << parameters << std::endl;
	}
	setup<float>(parameters);

//#ifdef LOGS
//#ifdef BUILDPROFILER
//	profiler_thread->end_profile();
//#endif
//	end_log_file();
//#endif
	return 0;

}

