#include "setup.h"
#include "Log.h"
#include "include/device_vector.h"
#include "common_template_functions.h"
#include "no_tensor_kernels.h"

template<const uint32_t COUNT, typename half_t, typename real_t>
struct GemmCaller {
	dim3 dim_grid, dim_block;

	virtual ~GemmCaller() = default;
	virtual void gemm(rad::DeviceVector<half_t>& a_dev,
			rad::DeviceVector<half_t>& b_dev, rad::DeviceVector<real_t>& c_dev,
			rad::DeviceVector<real_t>& d_dev, real_t alpha, real_t beta, int wA,
			int wB, const uint32_t threshold);

	GemmCaller(uint32_t m, uint32_t n) {
		uint32_t grid_rows = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
		uint32_t grid_cols = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
		this->dim_grid = dim3(grid_cols, grid_rows);
		this->dim_block = dim3(BLOCK_SIZE, BLOCK_SIZE);
	}
};

template<typename real_t>
struct UnhardenedGemmCaller: public GemmCaller<0, real_t, real_t> {
	void gemm(rad::DeviceVector<real_t>& a_dev,
			rad::DeviceVector<real_t>& b_dev, rad::DeviceVector<real_t>& c_dev,
			rad::DeviceVector<real_t>& d_dev, real_t alpha, real_t beta, int wA,
			int wB, const uint32_t threshold) {
		matrix_mult_kernel<<<this->dim_grid, this->dim_block>>>(
				d_dev.data(), c_dev.data(), a_dev.data(), b_dev.data(), alpha,
				beta, wA, wB);
	}

	UnhardenedGemmCaller(uint32_t m, uint32_t n) :
			GemmCaller<0, real_t, real_t>(m, n) {
	}
};

template<const uint32_t COUNT, typename half_t, typename real_t>
struct DMRGemmCaller: public GemmCaller<COUNT, half_t, real_t> {
	void gemm(rad::DeviceVector<half_t>& a_dev,
			rad::DeviceVector<half_t>& b_dev, rad::DeviceVector<real_t>& c_dev,
			rad::DeviceVector<real_t>& d_dev, real_t alpha, real_t beta, int wA,
			int wB, const uint32_t threshold) {

	}
};

template<const uint32_t COUNT, typename half_t, typename real_t>
void setup_execute(Log& log_obj, GemmCaller<COUNT, half_t, real_t>& mult_env) {
	double elapsedTime = 0;

	std::vector<real_t> a_vector_host(
			log_obj.size_matrices * log_obj.size_matrices);
	std::vector<real_t> b_vector_host(
			log_obj.size_matrices * log_obj.size_matrices);
	std::vector<real_t> c_vector_host(
			log_obj.size_matrices * log_obj.size_matrices);
	std::vector<real_t> d_vector_host(
			log_obj.size_matrices * log_obj.size_matrices);
	std::vector<real_t> gold_host(
			log_obj.size_matrices * log_obj.size_matrices);

	//smaller precision
	std::vector<half_t> d_vector_half_t_host(
			log_obj.size_matrices * log_obj.size_matrices);

	if (log_obj.generate) {
		std::cout << "Generating input matrices\n";
		generate_input_matrices(log_obj.size_matrices, a_vector_host,
				b_vector_host, c_vector_host);
	} else {
		std::cout << "Reading input matrices\n";
		read_gold(a_vector_host, b_vector_host, c_vector_host, gold_host,
				log_obj.a_input_path, log_obj.b_input_path,
				log_obj.c_input_path, log_obj.gold_inout_path);
	}

	rad::DeviceVector < real_t > a_vector_device(a_vector_host);
	rad::DeviceVector < real_t > b_vector_device(b_vector_host);
	rad::DeviceVector < real_t > c_vector_device(c_vector_host);
	rad::DeviceVector < real_t > d_vector_device(d_vector_host);
	rad::DeviceVector < half_t > d_vector_half_t_device(d_vector_half_t_host);

	std::cout << "Starting the setup process...\n";
	for (int it = 0; it < log_obj.iterations; it++) {
		double start_computation = rad::mysecond();

		log_obj.start_iteration();

		mult_env.gemm(a_vector_device, b_vector_device, c_vector_device,
				d_vector_device, log_obj.alpha, log_obj.beta, log_obj.size_matrices, log_obj.size_matrices, 0);

		log_obj.end_iteration();
		double end_computation = rad::mysecond();
		elapsedTime += end_computation - start_computation;

		std::cout << "Elapsed time : " << elapsedTime << " %f ms\n";

		double copy_time = rad::mysecond();
		d_vector_half_t_host = d_vector_half_t_device.to_vector();
		d_vector_host = d_vector_device.to_vector();
		copy_time = copy_time - rad::mysecond();

		if (!log_obj.generate) {
			double start, end;

			start = rad::mysecond();
			auto errors = check_output_errors_dmr(gold_host, d_vector_host,
					d_vector_half_t_host, log_obj);
			end = rad::mysecond();

			std::cout << "Iteration: " << it << " dmr errors " << errors.first
					<< " radiation errors " << errors.second
					<< ". Time spent on computation "
					<< end_computation - start_computation
					<< "s. Time spent on comparing " << end - start << "s."
					<< " Time spent on copying " << copy_time << "s."
					<< std::endl;

			//If errors != 0 reload matrices to gpu
			if (errors.first != 0 || errors.second != 0) {
				read_gold(a_vector_host, b_vector_host, c_vector_host,
						gold_host, log_obj.a_input_path, log_obj.b_input_path,
						log_obj.c_input_path, log_obj.gold_inout_path);

				a_vector_device.resize(0);
				b_vector_device.resize(0);
				c_vector_device.resize(0);
				d_vector_device.resize(0);
				d_vector_half_t_device.resize(0);

				a_vector_device = a_vector_host;
				b_vector_device = b_vector_host;
				c_vector_device = c_vector_host;
				d_vector_device = d_vector_host;
				d_vector_half_t_device = d_vector_half_t_host;

			}

		}

	}

	std::cout << "time : " << (elapsedTime / log_obj.iterations) << " s\n";
	if (log_obj.generate) {
		write_gold(a_vector_host, b_vector_host, c_vector_host, d_vector_host,
				log_obj.a_input_path, log_obj.b_input_path,
				log_obj.c_input_path, log_obj.gold_inout_path);
	}
}

void setup_gemm_unhardened(Log& log) {
	if (log.precision == "half") {
		UnhardenedGemmCaller<half> gemm_obj(log.size_matrices,
				log.size_matrices);
		setup_execute(log, gemm_obj);
		return;
	}

	if (log.precision == "float" || log.precision == "single") {
		UnhardenedGemmCaller<float> gemm_obj(log.size_matrices,
				log.size_matrices);
		setup_execute(log, gemm_obj);
		return;
	}

	if (log.precision == "double") {
		UnhardenedGemmCaller<double> gemm_obj(log.size_matrices,
				log.size_matrices);
		setup_execute(log, gemm_obj);
		return;
	}
}

void setup_gemm_dmr(Log& log) {
	if (log.precision == "half") {
		return;
	}

	if (log.precision == "float" || log.precision == "single") {
		return;
	}

	if (log.precision == "double") {
		return;
	}
}

