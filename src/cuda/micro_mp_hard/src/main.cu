#include "Log.h"
#include "Microbenchmark.h"
#include "DMRConstant.h"
#include "DMRNonConstant.h"
#include "UnhardenedConstant.h"

template<const uint32 CHECK_BLOCK, typename half_t, typename real_t>
void test_radiation(Microbenchmark<CHECK_BLOCK, half_t, real_t>& micro_test) {
	// Initial test environment
	double total_kernel_time = 0.0f;
	double min_kernel_time = UINT_MAX;
	double max_kernel_time = 0.0f;
	//====================================
	// Verbose in csv format
	if (micro_test.parameters_.verbose == false) {
		std::cout << "output/s,iteration,time,output errors,relative errors"
				<< std::endl;
	}

	if(micro_test.parameters_.generate == false){
		micro_test.load_gold();
	}

	for (auto iteration = 0; iteration < micro_test.parameters_.iterations;
			iteration++) {
		//================== Global test loop
		auto kernel_time = micro_test.test();
		uint64 relative_errors, errors;
		double cmp_time;

		std::tie(cmp_time, errors, relative_errors) = micro_test.check_output_errors();
		//====================================

		total_kernel_time += kernel_time;
		min_kernel_time = std::min(min_kernel_time, kernel_time);
		max_kernel_time = std::max(max_kernel_time, kernel_time);

		auto outputpersec = double(micro_test.parameters_.r_size) / kernel_time;
		if (micro_test.parameters_.verbose) {
			/////////// PERF
			std::cout << "SIZE:" << micro_test.parameters_.r_size;
			std::cout << " OUTPUT/S:" << outputpersec;
			std::cout << " ITERATION " << iteration;
			std::cout << " time: " << kernel_time;
			std::cout << " output errors: " << errors;
			std::cout << " relative errors: " << relative_errors << std::endl;

		} else {
			// CSV format
			std::cout << outputpersec << ",";
			std::cout << iteration << ",";
			std::cout << kernel_time << ",";
			std::cout << errors << ",";
			std::cout << relative_errors << std::endl;

		}
	}

	if(micro_test.parameters_.generate == true){
		micro_test.write_gold();
	}

	if (micro_test.parameters_.verbose) {
		auto averageKernelTime = total_kernel_time / micro_test.parameters_.iterations;
		std::cout << std::endl << "-- END --" << std::endl;
		std::cout << "Total kernel time: " << total_kernel_time << std::endl;
		std::cout << "Iterations: " << micro_test.parameters_.iterations << std::endl;
		std::cout << "Average kernel time: " << averageKernelTime << std::endl;
		std::cout << "Best: " << min_kernel_time << std::endl;
		std::cout << "Worst: " << max_kernel_time << std::endl;
	}
}

template<const uint32 CHECK_BLOCK>
void setup(Parameters& parameters) {
	/* NONE REDUNDANCY ------------------------------------------------------ */
	if (parameters.redundancy == NONE) {
		if (parameters.precision == HALF) {
//			test_radiation<half, half>(type_, parameters);
			fatalerror("Not implemented yet");
		}

		if (parameters.precision == SINGLE) {
			UnhardenedConstant<CHECK_BLOCK, float> micro_test(parameters);
			test_radiation<CHECK_BLOCK>(micro_test);
		}

		if (parameters.precision == DOUBLE) {
			UnhardenedConstant<CHECK_BLOCK, double> micro_test(parameters);
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
			DMRConstant<CHECK_BLOCK, float, float> micro_test(parameters);
			test_radiation<CHECK_BLOCK>(micro_test);
		}

		if (parameters.precision == DOUBLE) {
			DMRConstant<CHECK_BLOCK, double, double> micro_test(parameters);
			test_radiation<CHECK_BLOCK>(micro_test);
		}
	}

	/* DMRMIXED REDUNDANCY -------------------------------------------------- */
	if (parameters.redundancy == DMRMIXED) {

		if (parameters.precision == DOUBLE) {
			DMRConstant<CHECK_BLOCK, float, double> micro_test(parameters);
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
	std::string test_info = std::string("ops:") + std::to_string(OPS);
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
	start_log_file(const_cast<char*>(test_name.c_str()),
			const_cast<char*>(test_info.c_str()));

	std::cout << "LOGFILENAME:" << get_log_file_name() << std::endl;
	std::cout << parameters << std::endl;

	switch (parameters.operation_num) {
	case 1:
		setup<1>(parameters);
		break;
	case 10:
		setup<10>(parameters);
		break;
	case 100:
		setup<100>(parameters);
		break;
	case 1000:
		setup<1000>(parameters);
		break;
	default:
		setup<OPS>(parameters);
	}

	return 0;
}
