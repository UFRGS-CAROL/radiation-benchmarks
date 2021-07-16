#include "Parameters.h"
#include "include/device_vector.h"
#include "include/multi_compiler_analysis.h"
#include "common_template_functions.h"

#include "GemmCaller.h"

extern void show_iteration_status(int it, bool verbose, double copy_time, double comparing_time,
		double computation_time, std::pair<int, int> errors);

std::string get_multi_compiler_header() {
	return rad::get_multi_compiler_header();
}

template<const uint32_t COUNT, typename half_t, typename real_t>
void setup_execute(Parameters &parameters, GemmCaller<COUNT, half_t, real_t> &mult_env,
		const uint32_t threshold = 0) {
	double elapsed_time = 0;

	std::vector<real_t> a_vector_host(parameters.size_matrices * parameters.size_matrices);
	std::vector<real_t> b_vector_host(parameters.size_matrices * parameters.size_matrices);
	std::vector<real_t> c_vector_host(parameters.size_matrices * parameters.size_matrices);
	std::vector<real_t> d_vector_host(parameters.size_matrices * parameters.size_matrices);
	std::vector<real_t> gold_host(parameters.size_matrices * parameters.size_matrices);

	//Output host vectors are set after computation
	std::vector<half_t> c_vector_host_half_t(parameters.size_matrices * parameters.size_matrices);

	if (parameters.generate) {
		std::cout << "Generating input matrices\n";
		auto read_abc_files_on_generate = (parameters.check_input_existence
				&& exists(parameters.a_input_path) && exists(parameters.b_input_path)
				&& exists(parameters.c_input_path));

		get_input_matrices(parameters.size_matrices, a_vector_host, b_vector_host, c_vector_host,
				parameters.a_input_path, parameters.b_input_path, parameters.c_input_path,
				read_abc_files_on_generate);
	} else {
		std::cout << "Reading input matrices\n";
		read_abc_files(parameters.a_input_path, a_vector_host, parameters.b_input_path,
				b_vector_host, parameters.c_input_path, c_vector_host);

		read_gold(parameters.gold_inout_path, gold_host);
	}

	//Alloc only after reading the inputs
	rad::DeviceVector<real_t> a_vector_device = a_vector_host;
	rad::DeviceVector<real_t> b_vector_device = b_vector_host;
	rad::DeviceVector<real_t> c_vector_device = c_vector_host;
	rad::DeviceVector<half_t> c_vector_device_half_t = c_vector_host_half_t;

	std::cout << "Starting the setup process...\n";
	std::cout << std::setprecision(5) << std::fixed;
	for (int it = 0; it < parameters.iterations; it++) {
		c_vector_device = c_vector_host;

		auto computation_time = rad::mysecond();

		parameters.start_iteration();

		mult_env.gemm(a_vector_device, b_vector_device, c_vector_device, c_vector_device_half_t,
				real_t(parameters.alpha), real_t(parameters.beta), parameters.size_matrices,
				parameters.size_matrices, threshold);
		rad::checkFrameworkErrors(cudaDeviceSynchronize());
		rad::checkFrameworkErrors(cudaPeekAtLastError());
		parameters.end_iteration();
		computation_time = rad::mysecond() - computation_time;
		elapsed_time += computation_time;

		double copy_time = rad::mysecond();

		mult_env.memcpy_half_t_mem(c_vector_host_half_t, c_vector_device_half_t);

		c_vector_device.to_vector(d_vector_host);
		copy_time = rad::mysecond() - copy_time;

		if (!parameters.generate) {

			auto comparing_time = rad::mysecond();
			auto errors = check_output_errors_dmr(gold_host, d_vector_host, c_vector_host_half_t,
					parameters, threshold, mult_env.duplicated);

			comparing_time = rad::mysecond() - comparing_time;

			show_iteration_status(it, parameters.verbose, copy_time, comparing_time,
					computation_time, errors);

			//If errors != 0 reload matrices to gpu
			if (errors.first != 0 || errors.second != 0) {
				read_abc_files(parameters.a_input_path, a_vector_host, parameters.b_input_path,
						b_vector_host, parameters.c_input_path, c_vector_host);
				read_gold(parameters.gold_inout_path, gold_host);

				a_vector_device.resize(0);
				b_vector_device.resize(0);
				c_vector_device.resize(0);
				c_vector_device_half_t.resize(0);

				a_vector_device = a_vector_host;
				b_vector_device = b_vector_host;
				c_vector_device = c_vector_host;
				c_vector_device_half_t = c_vector_host_half_t;

			}

		}

	}
	if (parameters.verbose) {

		std::cout << "Elapsed time: " << (elapsed_time / parameters.iterations) << " s\n";
	} else {
		std::cout << "done.\n";
	}

	if (parameters.generate) {
		auto zero_count = 0ul;
		auto nans_count = 0ul;
		for (auto s : d_vector_host) {
			zero_count += (float(s) == 0.0f);
			nans_count += (std::isnan(float(s)));
		}
		std::cout << "Zero values on gold: " << zero_count << std::endl;
		std::cout << "Nans values on gold: " << nans_count << std::endl;

		write_gold(parameters.gold_inout_path, d_vector_host);
	}
}

void setup_gemm_unhardened(Parameters &parameters) {
	if (parameters.precision == "half") {
		UnhardenedGemmCaller<half> gemm_obj(parameters.size_matrices, parameters.size_matrices);
		setup_execute(parameters, gemm_obj);
	}
//
	if (parameters.precision == "float" || parameters.precision == "single") {
		UnhardenedGemmCaller<float> gemm_obj(parameters.size_matrices, parameters.size_matrices);
		setup_execute(parameters, gemm_obj);
	}

	if (parameters.precision == "double") {
		UnhardenedGemmCaller<double> gemm_obj(parameters.size_matrices, parameters.size_matrices);
		setup_execute(parameters, gemm_obj);
	}
}

void setup_gemm_cublas(Parameters &parameters) {
	if (parameters.precision == "half") {
		CUBLASGemmCaller<half> gemm_obj(parameters.size_matrices, parameters.size_matrices,
				parameters.use_tensor_cores);
		setup_execute(parameters, gemm_obj);

	}

	if (parameters.precision == "float" || parameters.precision == "single") {
		CUBLASGemmCaller<float> gemm_obj(parameters.size_matrices, parameters.size_matrices,
				parameters.use_tensor_cores);
		setup_execute(parameters, gemm_obj);
	}

	if (parameters.precision == "double") {
		CUBLASGemmCaller<double> gemm_obj(parameters.size_matrices, parameters.size_matrices,
				parameters.use_tensor_cores);
		setup_execute(parameters, gemm_obj);
	}
}

void setup_gemm_cutlass(Parameters &parameters) {
	throw_line("CUTLASS GEMM not ready yet");
}

void setup_gemm_dmr(Parameters &parameters) {
	if (parameters.precision == "float" || parameters.precision == "single") {
		throw_line("Not ready yet");
	}

	if (parameters.precision == "double") {

		if (parameters.dmr == "mixed") {
			switch (parameters.check_block) {
			case ONE_OP_CHECK: {
				DMRMixedGemmCaller<ONE_OP_CHECK, float, double> gemm_obj(parameters.size_matrices,
						parameters.size_matrices);
				setup_execute(parameters, gemm_obj, THRESHOLD_1);
				break;

			}
			default: {
				//The counter will never be 32, so it will check only at the end
				DMRMixedGemmCaller<AT_END_OP_CHECK, float, double> gemm_obj(
						parameters.size_matrices, parameters.size_matrices);
				setup_execute(parameters, gemm_obj, THRESHOLD_AT_END);
				break;
			}
			}

		} else if (parameters.dmr == "full") {
			DMRGemmCaller<double> gemm_obj(parameters.size_matrices, parameters.size_matrices);
			setup_execute(parameters, gemm_obj);
		}
	}
}

