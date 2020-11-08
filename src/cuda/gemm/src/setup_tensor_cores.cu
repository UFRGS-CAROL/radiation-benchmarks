#include <iostream>

#include "include/device_vector.h"

#include "Parameters.h"
#include "common_template_functions.h"
#include "GemmCallerMMA.h"

template<typename half_t, typename real_t>
void setup_execute(Parameters& parameters,
		TensorCoresCaller<half_t, real_t>& mult_env, const uint32_t threshold =
				0) {
	/*double elapsed_time = 0;

	 std::vector<half_t> a_vector_host(
	 parameters.size_matrices * parameters.size_matrices);
	 std::vector<half_t> b_vector_host(
	 parameters.size_matrices * parameters.size_matrices);
	 std::vector<real_t> c_vector_host(
	 parameters.size_matrices * parameters.size_matrices);
	 std::vector<real_t> gold_host(
	 parameters.size_matrices * parameters.size_matrices);

	 //Output host vectors are set after computation
	 std::vector<real_t> d_vector_host_real_t;
	 std::vector<real_t> d_vector_host_half_t;

	 if (parameters.generate) {
	 std::cout << "Generating input matrices\n";
	 auto read_abc_files_on_generate = (parameters.check_input_existence
	 && exists(parameters.a_input_path)
	 && exists(parameters.b_input_path)
	 && exists(parameters.c_input_path));

	 get_input_matrices(parameters.size_matrices, a_vector_host,
	 b_vector_host, c_vector_host, parameters.a_input_path,
	 parameters.b_input_path, parameters.c_input_path,
	 read_abc_files_on_generate);
	 } else {
	 std::cout << "Reading input matrices\n";
	 read_abc_files(parameters.a_input_path, a_vector_host,
	 parameters.b_input_path, b_vector_host, parameters.c_input_path,
	 c_vector_host);

	 read_gold(parameters.gold_inout_path, gold_host);
	 }

	 //Alloc only after reading the inputs
	 rad::DeviceVector<half_t> a_vector_device = a_vector_host;
	 rad::DeviceVector<half_t> b_vector_device = b_vector_host;
	 rad::DeviceVector<real_t> c_vector_device = c_vector_host;

	 rad::DeviceVector<real_t> d_vector_device(
	 parameters.size_matrices * parameters.size_matrices);
	 rad::DeviceVector<real_t> d_vector_half_t_device(
	 parameters.size_matrices * parameters.size_matrices);

	 std::cout << "Starting the setup process...\n";
	 std::cout << std::setprecision(5) << std::fixed;
	 for (int it = 0; it < parameters.iterations; it++) {
	 auto computation_time = rad::mysecond();

	 parameters.start_iteration();

	 mult_env.gemm(a_vector_device, b_vector_device, c_vector_device,
	 d_vector_device, d_vector_half_t_device, parameters.alpha,
	 parameters.beta, parameters.size_matrices,
	 parameters.size_matrices, threshold);
	 rad::checkFrameworkErrors(cudaDeviceSynchronize());
	 ;
	 rad::checkFrameworkErrors(cudaPeekAtLastError());

	 //end iteration
	 parameters.end_iteration();
	 computation_time = rad::mysecond() - computation_time;
	 elapsed_time += computation_time;

	 double copy_time = rad::mysecond();
	 mult_env.memcpy_half_t_mem(d_vector_host_half_t,  d_vector_half_t_device);
	 d_vector_device.to_vector(d_vector_host_real_t);
	 copy_time = rad::mysecond() - copy_time;

	 if (!parameters.generate) {

	 auto comparing_time = rad::mysecond();
	 auto errors =  check_output_errors_dmr(gold_host, d_vector_host_real_t,
	 d_vector_host_half_t, parameters, threshold,
	 mult_env.duplicated);

	 comparing_time = rad::mysecond() - comparing_time;

	 std::cout << "Iteration: " << it << " DMR errors " << errors.first
	 << ". " << "Radiation errors: " << errors.second << ". "
	 << "Time spent on computation: " << computation_time
	 << "s. " << "Time spent on comparing: " << comparing_time
	 << "s. " << "Time spent on copying: " << copy_time << "s. "
	 << std::endl;

	 //If errors != 0 reload matrices to gpu
	 if (errors.first != 0 || errors.second != 0) {
	 read_gold(a_vector_host, b_vector_host, c_vector_host,
	 gold_host, parameters.a_input_path,
	 parameters.b_input_path, parameters.c_input_path,
	 parameters.gold_inout_path);

	 a_vector_device.resize(0);
	 b_vector_device.resize(0);
	 c_vector_device.resize(0);
	 d_vector_device.resize(0);
	 d_vector_half_t_device.resize(0);

	 a_vector_device = a_vector_host;
	 b_vector_device = b_vector_host;
	 c_vector_device = c_vector_host;
	 d_vector_device = d_vector_host_real_t;
	 d_vector_half_t_device = d_vector_host_half_t;

	 }

	 }

	 }

	 std::cout << "Elapsed time: " << (elapsed_time / parameters.iterations)
	 << " s\n";
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
	 */
}

void setup_gemm_tensor_cores_unhardened(Parameters& parameters) {
#if __CUDA_ARCH__ >= 600
	if (parameters.precision == "half") {
		UnhardenedTensorCoresCaller<half, half> gemm_obj(parameters.size_matrices,
				parameters.size_matrices);
		setup_execute(parameters, gemm_obj);

	}

#endif
	if (parameters.precision == "float" || parameters.precision == "single"
			|| parameters.precision == "double") {
		throw_line(
				parameters.precision + " using tensorcores not ready yet!!!");
	}

}
void setup_gemm_tensor_cores_dmr(Parameters& parameters) {
#if __CUDA_ARCH__ >= 600
	if (parameters.precision == "half") {
		DMRTensorCoresCaller<half> gemm_obj(parameters.size_matrices,
				parameters.size_matrices);
		setup_execute(parameters, gemm_obj);

	}
#endif

	if (parameters.precision == "float" || parameters.precision == "single"
			|| parameters.precision == "double") {
		throw_line(
				parameters.precision + " using tensorcores not ready yet!!!");
	}
}

