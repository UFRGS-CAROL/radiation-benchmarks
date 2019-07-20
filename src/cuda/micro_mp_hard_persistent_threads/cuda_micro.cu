#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <string>
#include <omp.h>
#include <random>
#include <cuda_fp16.h>
#include <vector>
#include <sstream>

#include "dmr_kernels.h"
#include "none_kernels.h"
//#include "cuda_utils.h"
#include "Parameters.h"

#ifdef FORJETSON
#include "include/JTX2Inst.h"
#define OBJTYPE JTX2Inst
#else
#include "include/NVMLWrapper.h"
#define OBJTYPE NVMLWrapper
#endif

#include "include/persistent_lib.h"
#include "include/cuda_utils.h"
#include "include/device_vector.h"

cudaDeviceProp GetDevice() {
//================== Retrieve and set the default CUDA device
	cudaDeviceProp prop;
	int count = 0;

	rad::checkFrameworkErrors(cudaGetDeviceCount(&count));
	for (int i = 0; i < count; i++) {
		rad::checkFrameworkErrors(cudaGetDeviceProperties(&prop, i));
	}
	int *ndevice;
	int dev = 0;
	ndevice = &dev;
	rad::checkFrameworkErrors(cudaGetDevice(ndevice));

	rad::checkFrameworkErrors(cudaSetDevice(0));
	rad::checkFrameworkErrors(cudaGetDeviceProperties(&prop, 0));

	return prop;
}

unsigned long long copy_errors() {
	unsigned long long errors_host = 0;
	rad::checkFrameworkErrors(
			cudaMemcpyFromSymbol(&errors_host, errors,
					sizeof(unsigned long long), 0, cudaMemcpyDeviceToHost));

	unsigned long long temp = 0;
	//Reset the errors variable
	rad::checkFrameworkErrors(
			cudaMemcpyToSymbol(errors, &temp, sizeof(unsigned long long), 0,
					cudaMemcpyHostToDevice));
	return errors_host;
}

// helper functions
//#include "helper_cuda.h"
//
//#define HALF_ROUND_STYLE 1
//#define HALF_ROUND_TIES_TO_EVEN 1
//#include "half.hpp"

std::string get_double_representation(double val) {
	std::string output = "";
	if (sizeof(double) == 8) {

		uint64_t int_val;

		memcpy(&int_val, &val, sizeof(double));
		for (uint64_t i = uint64_t(1) << 63; i > 0; i = i / 2) {
			if (int_val & i) {
				output += "1";
			} else {
				output += "0";
			}
		}
	} else {
		std::cerr << "USING more than 64 bits double" << std::endl;
	}
	return output;
}

// Returns the number of errors found
// if no errors were found it returns 0
template<typename full>
int check_output_errors(std::vector<full> &R, full OUTPUT_R, bool verbose) {
	int host_errors = 0;
	double gold = double(OUTPUT_R);
#pragma omp parallel for shared(host_errors)
	for (int i = 0; i < R.size(); i++) {
		double output = double(R[i]);
		if (gold != output) {
#pragma omp critical
			{
				std::stringstream error_detail;
				error_detail.precision(16);
				error_detail << "p: [" << i << "], r: " << std::scientific
						<< output << ", e: " << gold;

				if (verbose && (host_errors < 10))
					std::cout << error_detail.str() << std::endl;
#ifdef LOGS
				log_error_detail(const_cast<char*>(error_detail.str().c_str()));
#endif
				host_errors++;
			}
		}
	}

	if (host_errors != 0) {
		std::cout << "#";
#ifdef LOGS
		log_error_count(host_errors);
#endif
	}
	return host_errors;
}

template<typename full>
void launch_kernel(Type<full>& type_,
		rad::DeviceVector<full>& device_vector_full,
		rad::DeviceVector<Input<full> >& device_defined_input,
		Parameters& parameters, cudaStream_t stream) {
	//================== Device computation
//	MicroBenchmarkKernel_FMA<full> <<<parameters.grid_size,
//			parameters.block_size, 0, stream>>>(device_vector_full.data(),
//			device_defined_input.data());
	MicroBenchmarkKernel_FMA<full> <<<parameters.grid_size,
			parameters.block_size, 0, stream>>>(device_vector_full.data(),
			type_.output_r, type_.input_a, type_.input_b);

	rad::checkFrameworkErrors(cudaPeekAtLastError());
//	rad::checkFrameworkErrors(cudaStreamSynchronize(stream));

}

template<typename full>
void test_radiation(Type<full>& type_, Parameters& parameters,
		const std::vector<Input<full>>& defined_input) {
	std::cout << "Printing the input values " << type_ << std::endl;
	// Init test environment
	// kernel_errors=0;
	double total_kernel_time = 0;
	double min_kernel_time = UINT_MAX;
	double max_kernel_time = 0;
	//====================================

	// FULL PRECIISON
	std::vector<full> host_vector_full(parameters.r_size, 0);
	rad::DeviceVector<full> device_vector_full(parameters.r_size);

	//For defined input only
	rad::DeviceVector<Input<full>> device_defined_input;
//	if (defined_input.size() == device_vector_full.size()) {
//		device_defined_input = defined_input;
//	}
	//====================================

	cudaStream_t stream;
	rad::checkFrameworkErrors(
			cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

	// Verbose in csv format
//	if (parameters.verbose == false) {
//		std::cout << "output/s,iteration,time,output errors,relative errors"
//				<< std::endl;
//	}

	//Profiler thread
#ifdef LOGS
	std::string log_file_name(get_log_file_name());
	std::shared_ptr<rad::Profiler> profiler_thread = std::make_shared<rad::OBJTYPE>(0, log_file_name);

	//START PROFILER THREAD
	profiler_thread->start_profile();

#endif

	rad::HostPersistentControler pt_control(parameters.grid_size);

	launch_kernel(type_, device_vector_full, device_defined_input, parameters,
			stream);

	for (size_t iteration = 0; iteration < parameters.iterations; iteration++) {
		std::cout << std::flush;
		//================== Global test loop
		double kernel_time = rad::mysecond();
#ifdef LOGS
		start_iteration();
#endif
		//================== Device computation
		pt_control.process_data_on_kernel();
		kernel_time = rad::mysecond() - kernel_time;

		//====================================
#ifdef LOGS
		end_iteration();
#endif

		total_kernel_time += kernel_time;
		min_kernel_time = std::min(min_kernel_time, kernel_time);
		max_kernel_time = std::max(max_kernel_time, kernel_time);

		//check output
		host_vector_full = device_vector_full.to_vector();
		int errors = check_output_errors(host_vector_full, type_.output_r,
				parameters.verbose);

		if (errors != 0) {
#ifdef LOGS
			profiler_thread->end_profile();
#endif
			pt_control.end_kernel();
			device_defined_input = defined_input;
#ifdef LOGS
			profiler_thread->start_profile();
#endif
			pt_control.start_kernel();
			launch_kernel(type_, device_vector_full, device_defined_input,
					parameters, stream);
		}

		double outputpersec = double(parameters.r_size) / kernel_time;
		if (parameters.verbose) {
			/////////// PERF
			std::cout << "SIZE:" << parameters.r_size;
			std::cout << " OUTPUT/S:" << outputpersec;
			std::cout << " ITERATION " << iteration;
			std::cout << " time: " << kernel_time;
			std::cout << " output errors: " << errors << std::endl;

		} else {
			// CSV format
//			std::cout << outputpersec << ",";
//			std::cout << iteration << ",";
//			std::cout << kernel_time << ",";
//			std::cout << errors << "," << std::endl;
			std::cout << ".";
		}
	}
	pt_control.end_kernel();

	if (parameters.verbose) {
		double averageKernelTime = total_kernel_time / parameters.iterations;
		std::cout << std::endl << "-- END --" << std::endl;
		std::cout << "Total kernel time: " << total_kernel_time << std::endl;
		std::cout << "Iterations: " << parameters.iterations << std::endl;
		std::cout << "Average kernel time: " << averageKernelTime << std::endl;
		std::cout << "Best: " << min_kernel_time << std::endl;
		std::cout << "Worst: " << max_kernel_time << std::endl;
	}

	rad::checkFrameworkErrors(cudaStreamDestroy(stream));
}

void dmr(Parameters& parameters) {

	Type<float> type_;
	Input<float> input(type_.output_r, type_.input_a, type_.input_b);
	std::vector<Input<float>> input_vector(
			parameters.grid_size * parameters.block_size, input);

	test_radiation<float>(type_, parameters, input_vector);

}

int main(int argc, char* argv[]) {

//================== Set block and grid size for MxM kernel
	cudaDeviceProp prop = GetDevice();
	int sm_count = prop.multiProcessorCount;
	int threads_per_block = 256;
	if (BLOCK_SIZE == 16) {
		threads_per_block *= 2;
		sm_count *= 2;
	}

	Parameters parameters(argc, argv, sm_count, threads_per_block);
	if (parameters.verbose) {
		std::cout << "Get device Name: " << prop.name << std::endl;
	}
//================== Init logs
#ifdef LOGS
	std::string test_info = std::string("ops:") + std::to_string(OPS)
	+ " gridsize:" + std::to_string(parameters.grid_size)
	+ " blocksize:" + std::to_string(parameters.block_size) + " type:"
	+ parameters.instruction_str + "-" + parameters.precision_str
	+ "-precision hard:" + parameters.hardening_str;

	std::string test_name = std::string("cuda_") + parameters.precision_str
	+ "_micro-" + parameters.instruction_str + "_persistent";
	start_log_file(const_cast<char*>(test_name.c_str()),
			const_cast<char*>(test_info.c_str()));
#endif

	parameters.print_details();

	dmr(parameters);

#ifdef LOGS
	end_log_file();
#endif
	return 0;
}
