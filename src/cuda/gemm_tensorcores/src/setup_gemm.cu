#include "setup.h"
#include "Log.h"
#include "include/device_vector.h"
#include "common_template_functions.h"
#include "no_tensor_kernels.h"

template<const uint32_t COUNT, typename half_t, typename real_t>
struct GemmCaller {
	bool duplicated;
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

	virtual std::vector<half_t> memcpy_half_t_mem(
			rad::DeviceVector<half_t>& d_dev_half_t);

	GemmCaller(uint32_t m, uint32_t n) :
			duplicated(false) {
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
		matrix_mult_kernel_unhardened<<<this->dim_grid, this->dim_block>>>( //call
				a_dev.data(), //a
				b_dev.data(), //b
				c_dev.data(), //c
				d_dev.data(), //d
				alpha, beta, wA, wB);
	}

	std::vector<real_t> memcpy_half_t_mem(
			rad::DeviceVector<real_t>& d_dev_half_t) {
		return {};
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
		matrix_mult_kernel_dmr_mixed<COUNT> <<<this->dim_grid, this->dim_block>>>( //call
				a_dev.data(), 				//a
				b_dev.data(), 				//b
				c_dev.data(), 				//c
				d_dev.data(), 				//d
				d_dev_half_t.data(), 		//d hardening
				alpha, beta, wA, wB, threshold);
	}

	DMRMixedGemmCaller(uint32_t m, uint32_t n) :
			GemmCaller<COUNT, half_t, real_t>(m, n) {
		this->duplicated = true;
	}

	std::vector<half_t> memcpy_half_t_mem(
			rad::DeviceVector<half_t>& d_dev_half_t) {
		return d_dev_half_t.to_vector();
	}
};

template<const uint32_t COUNT, typename real_t>
struct DMRGemmCaller: public GemmCaller<COUNT, real_t, real_t> {

	void gemm(
			rad::DeviceVector<real_t>& a_dev, 			//A matrix
			rad::DeviceVector<real_t>& b_dev, 			//B matrix
			rad::DeviceVector<real_t>& c_dev, 			//C matrix
			rad::DeviceVector<real_t>& d_dev, 			//D matrix
			rad::DeviceVector<real_t>& d_dev_half_t,  	//D_Half matrix
			real_t alpha, real_t beta, int wA, int wB,
			const uint32_t threshold) {
		matrix_mult_kernel_dmr<COUNT> <<<this->dim_grid, this->dim_block>>>( //call
				a_dev.data(), 				//a
				b_dev.data(), 				//b
				c_dev.data(), 				//c
				d_dev.data(), 				//d
				d_dev_half_t.data(), 		//d hardening
				alpha, beta, wA, wB);
	}

	DMRGemmCaller(uint32_t m, uint32_t n) :
			GemmCaller<COUNT, real_t, real_t>(m, n) {
		this->duplicated = true;
	}

	std::vector<real_t> memcpy_half_t_mem(
			rad::DeviceVector<real_t>& d_dev_half_t) {
		return d_dev_half_t.to_vector();
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
	std::cout << std::setprecision(5) << std::fixed;
	for (int it = 0; it < log_obj.iterations; it++) {
		auto computation_time = rad::mysecond();

		log_obj.start_iteration();

		mult_env.gemm(a_vector_device, b_vector_device, c_vector_device,
				d_vector_device, d_vector_half_t_device, log_obj.alpha,
				log_obj.beta, log_obj.size_matrices, log_obj.size_matrices,
				threshold);
		rad::checkFrameworkErrors (cudaDeviceSynchronize());;
		rad::checkFrameworkErrors (cudaPeekAtLastError());;

		log_obj.end_iteration();
		computation_time = rad::mysecond() - computation_time;
		elapsed_time += computation_time;

		double copy_time = rad::mysecond();
		d_vector_host_half_t = mult_env.memcpy_half_t_mem(
				d_vector_half_t_device);
		d_vector_host_real_t = d_vector_device.to_vector();
		copy_time = rad::mysecond() - copy_time;

		if (!log_obj.generate) {

			auto comparing_time = rad::mysecond();
			auto errors = std::pair<int, int>();
			errors = check_output_errors_dmr(gold_host, d_vector_host_real_t,
					d_vector_host_half_t, log_obj, threshold,
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

	std::cout << "Elapsed time: " << (elapsed_time / log_obj.iterations)
			<< " s\n";
	if (log_obj.generate) {
		write_gold(a_vector_host, b_vector_host, c_vector_host,
				d_vector_host_real_t, log_obj.a_input_path,
				log_obj.b_input_path, log_obj.c_input_path,
				log_obj.gold_inout_path);
	}
}

void setup_gemm_unhardened(Log& log) {
	if (log.precision == "half") {
//		UnhardenedGemmCaller<half> gemm_obj(log.size_matrices,
//				log.size_matrices);
//		setup_execute(log, gemm_obj);
	}
//
//	if (log.precision == "float" || log.precision == "single") {
//		UnhardenedGemmCaller<float> gemm_obj(log.size_matrices,
//				log.size_matrices);
//		setup_execute(log, gemm_obj);
//	}
//
	if (log.precision == "double") {
		UnhardenedGemmCaller<double> gemm_obj(log.size_matrices,
				log.size_matrices);
		setup_execute(log, gemm_obj);
	}
}

void setup_gemm_dmr(Log& log) {
	if (log.precision == "half") {
		throw_line("Not ready yet");
	}

	if (log.precision == "float" || log.precision == "single") {
		throw_line("Not ready yet");
	}

	if (log.precision == "double") {

		if (log.dmr == "mixed") {
			switch (log.check_block) {
			case 1: {
//				DMRMixedGemmCaller<1, float, double> gemm_obj(log.size_matrices,
//						log.size_matrices);
//				setup_execute(log, gemm_obj, THRESHOLD_1);
				break;

			}
			case 31: {
//				DMRMixedGemmCaller<31, float, double> gemm_obj(
//						log.size_matrices, log.size_matrices);
//				setup_execute(log, gemm_obj, THRESHOLD_1);
				break;

			}
			default: {
				//The counter will never be 32, so it will check only at the end
//				DMRMixedGemmCaller<32, float, double> gemm_obj(
//						log.size_matrices, log.size_matrices);
//				setup_execute(log, gemm_obj, THRESHOLD_32);
				break;
			}
			}

		} else if (log.dmr == "full") {
			DMRGemmCaller<32, double> gemm_obj(log.size_matrices,
					log.size_matrices);
			setup_execute(log, gemm_obj);
		}
	}
}

