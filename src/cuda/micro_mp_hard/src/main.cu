#include "Log.h"
#include "Microbenchmark.h"
#include "DMRConstant.h"
#include "UnhardenedConstant.h"

template<const uint32 CHECK_BLOCK, typename half_t, typename real_t>
void test_radiation(Microbenchmark<CHECK_BLOCK, half_t, real_t>& micro_test) {
	// Initial test environment
	double total_kernel_time = 0.0f;
	double min_kernel_time = UINT_MAX;
	double max_kernel_time = 0.0f;
	//====================================
	// Verbose in csv format
//	if (micro_test.parameters_.verbose == false) {
//		std::cout << "output/s,iteration,time,output errors,relative errors"
//				<< std::endl;
//	}

	if (micro_test.parameters_.generate == false) {
		micro_test.load_gold();
	}

	uint64 memory_errors, errors, relative_errors_host, relative_errors_gpu;
	// Global test loop
	for (auto it = 0; it < micro_test.parameters_.iterations; it++) {
		// Execute the test
		auto kernel_time = rad::mysecond();
		micro_test.test();
		kernel_time = rad::mysecond() - kernel_time;

		// Copy data from GPU
		auto copy_time = rad::mysecond();
		relative_errors_gpu = micro_test.copy_data_back();
		copy_time = rad::mysecond() - copy_time;

		//Check the output at the end
		auto cmp_time = rad::mysecond();
		std::tie(errors, memory_errors, relative_errors_host) =
				micro_test.check_output_errors();
		cmp_time = rad::mysecond() - cmp_time;

		//====================================

		total_kernel_time += kernel_time;
		min_kernel_time = std::min(min_kernel_time, kernel_time);
		max_kernel_time = std::max(max_kernel_time, kernel_time);

//		auto outputpersec = double(micro_test.parameters_.r_size) / kernel_time;
		if (micro_test.parameters_.verbose) {
			/////////// PERF
//			std::cout << "SIZE:" << micro_test.parameters_.r_size;
//			std::cout << " OUTPUT/S:" << outputpersec;
			std::cout << "Iteration " << it;
			std::cout << " Time: " << kernel_time;
			std::cout << " Comparison time: " << cmp_time;
			std::cout << " Copy time: " << copy_time;
			std::cout << " Output errors: " << errors;
			std::cout << " Memory errors: " << memory_errors;
			std::cout << " Hardening errors host: " << relative_errors_host;
			std::cout << " Hardening errors GPU: " << relative_errors_gpu
					<< std::endl;

		} else {
//			// CSV format
//			std::cout << outputpersec << ",";
//			std::cout << it << ",";
//			std::cout << kernel_time << ",";
//			std::cout << errors << ",";
//			std::cout << memory_errors << std::endl;
			std::cout << ".";

		}
	}

	if (micro_test.parameters_.generate == true) {
		micro_test.write_gold();
		auto max_diff = micro_test.get_max_threshold();
		std::cout << "MAX DIFF " << max_diff << std::endl;
	}

	if (micro_test.parameters_.verbose) {
		auto averageKernelTime = total_kernel_time
				/ micro_test.parameters_.iterations;
		std::cout << std::endl << "-- END --" << std::endl;
		std::cout << "Total kernel time: " << total_kernel_time << std::endl;
		std::cout << "Iterations: " << micro_test.parameters_.iterations
				<< std::endl;
		std::cout << "Average kernel time: " << averageKernelTime << std::endl;
		std::cout << "Best: " << min_kernel_time << std::endl;
		std::cout << "Worst: " << max_kernel_time << std::endl;
	}
}

template<const uint32 CHECK_BLOCK>
void setup(Parameters& parameters, Log& log) {
	/* NONE REDUNDANCY ------------------------------------------------------ */
	if (parameters.redundancy == NONE) {
		if (parameters.precision == HALF) {
//			test_radiation<half, half>(type_, parameters);
			fatalerror("Not implemented yet");
		}

		if (parameters.precision == SINGLE) {
//			UnhardenedConstant<CHECK_BLOCK, float> micro_test(parameters, log);
//			test_radiation<CHECK_BLOCK>(micro_test);
			fatalerror("Not implemented yet");

		}

		if (parameters.precision == DOUBLE) {
			UnhardenedConstant<CHECK_BLOCK, double> micro_test(parameters, log);
			test_radiation<CHECK_BLOCK>(micro_test);
		}
	}

	/* DMR REDUNDANCY ------------------------------------------------------- */
	if (parameters.redundancy == DMR) {
		if (parameters.precision == HALF) {
//			test_radiation<half, half>(type_, parameters);
			fatalerror("Not implemented yet");
		}

		if (parameters.precision == SINGLE) {
//			DMRConstant<CHECK_BLOCK, float, float> micro_test(parameters, log);
//			test_radiation<CHECK_BLOCK>(micro_test);
			fatalerror("Not implemented yet");
		}

		if (parameters.precision == DOUBLE) {
			DMRDWC<CHECK_BLOCK, double> micro_test(parameters, log);
			test_radiation<CHECK_BLOCK>(micro_test);
		}
	}

	/* DMRMIXED REDUNDANCY -------------------------------------------------- */
	if (parameters.redundancy == DMRMIXED) {

		if (parameters.precision == DOUBLE) {
			DMRConstant<CHECK_BLOCK, float, double> micro_test(parameters, log);
			test_radiation<CHECK_BLOCK>(micro_test);
		}

		if (parameters.precision == SINGLE) {
//			test_radiation<half, float>(type_, parameters);
			fatalerror("Not implemented yet");

		}
	}

}

int main(int argc, char* argv[]) {

//================== Set block and grid size for MxM kernel
	Parameters parameters(argc, argv);

	if (parameters.verbose) {
		std::cout << "Get device Name: " << parameters.device << std::endl;
	}
//================== Init logs
	std::string test_info = "";
	test_info += " gridsize:" + std::to_string(parameters.grid_size);
	test_info += " blocksize:" + std::to_string(parameters.block_size);
	test_info += " type:" + parameters.instruction_str;
	test_info += "-" + parameters.precision_str;
	test_info += "-precision hard:" + parameters.hardening_str;
	test_info += " kernel_type:non-persistent";
	test_info += " checkblock:" + std::to_string(parameters.operation_num);
	test_info += " nonconst:" + std::to_string(parameters.nonconstant);
	test_info += " numop:" + std::to_string(parameters.operation_num);

	std::string test_name = std::string("cuda_") + parameters.precision_str
			+ "_micro-" + parameters.instruction_str;

	Log log(test_name, test_info);

	std::cout << log << std::endl;
	std::cout << parameters << std::endl;
#if BUILDRELATIVEERROR != 0
	std::cout << "Relative error used" << std::endl;
#endif

	switch (parameters.operation_num) {
	case 1:
		setup<1>(parameters, log);
		break;
	case 10:
		setup<10>(parameters, log);
		break;
	case 100:
		setup<100>(parameters, log);
		break;
	case 1000:
		setup<1000>(parameters, log);
		break;
	case OPS:
		setup<OPS>(parameters, log);
		break;
	default:
		std::string result = "OPERATION NUM = "
				+ std::to_string(parameters.operation_num)
				+ " OPTION NOT COVERED";
		fatalerror(result.c_str());
	}

	return 0;
}
