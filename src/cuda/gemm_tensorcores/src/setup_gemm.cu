#include "setup.h"
#include "Log.h"
#include "include/device_vector.h"
#include "common_template_functions.h"
#include "no_tensor_kernels.h"

template<const uint32_t COUNT, typename half_t, typename real_t>
struct GemmCaller {
	dim3 dim_grid, dim_block;

	virtual ~GemmCaller() = default;
	virtual void gemm(
			rad::DeviceVector<real_t>& a_dev, 			//A matrix
			rad::DeviceVector<real_t>& b_dev, 			//B matrix
			rad::DeviceVector<real_t>& c_dev, 			//C matrix
			rad::DeviceVector<real_t>& d_dev, 			//D matrix
			rad::DeviceVector<half_t>& d_dev_half_t,  	//D_Half matrix
			real_t alpha, real_t beta, int wA, int wB,
			const uint32_t threshold);

	GemmCaller(uint32_t m, uint32_t n) {
		uint32_t grid_rows = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
		uint32_t grid_cols = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
		this->dim_grid = dim3(grid_cols, grid_rows);
		this->dim_block = dim3(BLOCK_SIZE, BLOCK_SIZE);
	}
};

template<typename real_t>
struct UnhardenedGemmCaller: public GemmCaller<0, real_t, real_t> {
	void gemm(
			rad::DeviceVector<real_t>& a_dev, 			//A matrix
			rad::DeviceVector<real_t>& b_dev, 			//B matrix
			rad::DeviceVector<real_t>& c_dev, 			//C matrix
			rad::DeviceVector<real_t>& d_dev, 			//D matrix
			rad::DeviceVector<real_t>& d_dev_half_t,  	//D_Half matrix
			real_t alpha, real_t beta, int wA, int wB,
			const uint32_t threshold) {
		matrix_mult_kernel<<<this->dim_grid, this->dim_block>>>( //call
				a_dev.data(), //a
				b_dev.data(), //b
				c_dev.data(), //c
				d_dev.data(), //d
				alpha, beta, wA, wB);
	}

	UnhardenedGemmCaller(uint32_t m, uint32_t n) :
			GemmCaller<0, real_t, real_t>(m, n) {
	} //default constructor
};

template<const uint32_t COUNT, typename half_t, typename real_t>
struct DMRMixedGemmCaller: public GemmCaller<COUNT, half_t, real_t> {
	void gemm(
			rad::DeviceVector<real_t>& a_dev, 			//A matrix
			rad::DeviceVector<real_t>& b_dev, 			//B matrix
			rad::DeviceVector<real_t>& c_dev, 			//C matrix
			rad::DeviceVector<real_t>& d_dev, 			//D matrix
			rad::DeviceVector<half_t>& d_dev_half_t,  	//D_Half matrix
			real_t alpha, real_t beta, int wA, int wB,
			const uint32_t threshold) {
		matrix_mult_kernel<COUNT> <<<this->dim_grid, this->dim_block>>>( //call
				a_dev.data(), 				//a
				b_dev.data(), 				//b
				c_dev.data(), 				//c
				d_dev.data(), 				//d
				d_dev_half_t.data(), 		//d hardening
				alpha, beta, wA, wB, threshold);
	}

	DMRMixedGemmCaller(uint32_t m, uint32_t n) :
			GemmCaller<COUNT, half_t, real_t>(m, n) {
	}
};

template<const uint32_t COUNT, typename real_t>
struct DMRGemmCaller: public DMRMixedGemmCaller<COUNT, real_t, real_t> {
	DMRGemmCaller(uint32_t m, uint32_t n) :
			DMRMixedGemmCaller<COUNT, real_t, real_t>(m, n) {
	}
};

template<const uint32_t COUNT, typename half_t, typename real_t>
void setup_execute(Log& log_obj, GemmCaller<COUNT, half_t, real_t>& mult_env,
		const uint32_t threshold = 0) {
	double elapsed_time = 0;

	std::vector<real_t> a_vector_host(
			log_obj.size_matrices * log_obj.size_matrices);
	std::vector<real_t> b_vector_host(
			log_obj.size_matrices * log_obj.size_matrices);
	std::vector<real_t> c_vector_host(
			log_obj.size_matrices * log_obj.size_matrices);
	std::vector<real_t> gold_host(
			log_obj.size_matrices * log_obj.size_matrices);

	//Output host vectors are set after computation
	std::vector<real_t> d_vector_host_real_t;
	std::vector<half_t> d_vector_host_half_t;

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

	rad::DeviceVector<real_t> a_vector_device = a_vector_host;
	rad::DeviceVector<real_t> b_vector_device = b_vector_host;
	rad::DeviceVector<real_t> c_vector_device = c_vector_host;
	rad::DeviceVector<real_t> d_vector_device(
			log_obj.size_matrices * log_obj.size_matrices);
	rad::DeviceVector<half_t> d_vector_half_t_device(
			log_obj.size_matrices * log_obj.size_matrices);

	std::cout << "Starting the setup process...\n";
	for (int it = 0; it < log_obj.iterations; it++) {
		auto computation_time = rad::mysecond();

		log_obj.start_iteration();

		mult_env.gemm(a_vector_device, b_vector_device, c_vector_device,
				d_vector_device, d_vector_half_t_device, log_obj.alpha,
				log_obj.beta, log_obj.size_matrices, log_obj.size_matrices,
				threshold);

		log_obj.end_iteration();
		computation_time = rad::mysecond() - computation_time;
		elapsed_time += computation_time;

		double copy_time = rad::mysecond();
		d_vector_host_half_t = d_vector_half_t_device.to_vector();
		d_vector_host_real_t = d_vector_device.to_vector();
		copy_time = rad::mysecond() - copy_time;

		if (!log_obj.generate) {

			auto comparing_time = rad::mysecond();
			auto errors = check_output_errors_dmr(gold_host,
					d_vector_host_real_t, d_vector_host_half_t, log_obj,
					threshold);
			comparing_time = rad::mysecond() - comparing_time;

			std::cout << "Iteration: " << it << " dmr errors " << errors.first << ". "
					<< "Radiation errors: " << errors.second << ". "
					<< "Time spent on computation: " << computation_time << "s. "
					<< "Time spent on comparing: " << comparing_time << "s. "
					<< "Time spent on copying: " << copy_time << "s. "
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
				d_vector_device = d_vector_host_real_t;
				d_vector_half_t_device = d_vector_host_half_t;

			}

		}

	}

	std::cout << "time : " << (elapsed_time / log_obj.iterations) << " s\n";
	if (log_obj.generate) {
		write_gold(a_vector_host, b_vector_host, c_vector_host,
				d_vector_host_real_t, log_obj.a_input_path,
				log_obj.b_input_path, log_obj.c_input_path,
				log_obj.gold_inout_path);
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
		if (log.dmr == "dmrmixed") {
			DMRMixedGemmCaller<1, float, double> gemm_obj(log.size_matrices,
					log.size_matrices);
			setup_execute(log, gemm_obj);
		} else {
			DMRGemmCaller<1, double> gemm_obj(log.size_matrices,
					log.size_matrices);
			setup_execute(log, gemm_obj);
		}
		return;
	}
}

