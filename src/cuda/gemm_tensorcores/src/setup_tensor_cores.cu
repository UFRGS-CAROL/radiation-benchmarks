#include "setup.h"
#include "Log.h"
#include "include/device_vector.h"
#include "common_template_functions.h"
#include "tensor_kernels.h"

template<typename half_t, typename real_t>
struct TensorCoresCaller {
	bool duplicated;
	dim3 dim_grid, dim_block;

	virtual ~TensorCoresCaller() = default;
	virtual void gemm(
			rad::DeviceVector<half_t>& a_dev, 			//A matrix
			rad::DeviceVector<half_t>& b_dev, 			//B matrix
			rad::DeviceVector<real_t>& c_dev, 			//C matrix
			rad::DeviceVector<real_t>& d_dev, 			//D matrix
			rad::DeviceVector<real_t>& d_dev_half_t,  	//D_Half matrix
			real_t alpha, real_t beta, int wA, int wB,
			const uint32_t threshold) ;

	virtual std::vector<real_t> memcpy_half_t_mem(
			rad::DeviceVector<real_t>& d_dev_half_t);

	TensorCoresCaller(uint32_t m, uint32_t n) :
			duplicated(false) {

		this->dim_block.x = 128;
		this->dim_block.y = 4;

		this->dim_grid.x = (m + (WMMA_M * this->dim_block.x / BLOCK_SIZE - 1))
				/ (WMMA_M * this->dim_block.x / BLOCK_SIZE);
		this->dim_grid.y = (n + WMMA_N * this->dim_block.y - 1)
				/ (WMMA_N * this->dim_block.y);
	}
};

template<typename half_t, typename real_t>
struct UnhardenedTensorCoresCaller: public TensorCoresCaller<half_t, real_t> {

	void gemm(
			rad::DeviceVector<half_t>& a_dev, 			//A matrix
			rad::DeviceVector<half_t>& b_dev, 			//B matrix
			rad::DeviceVector<real_t>& c_dev, 			//C matrix
			rad::DeviceVector<real_t>& d_dev, 			//D matrix
			rad::DeviceVector<real_t>& d_dev_half_t,  	//D_Half matrix
			real_t alpha, real_t beta, int wA, int wB,
			const uint32_t threshold) {
		matrix_mult_kernel_wmma_unhardened<<<this->dim_grid, this->dim_block>>>(
				a_dev.data(), b_dev.data(), c_dev.data(), d_dev.data(), alpha,
				beta, wA, wB, wA);

	}

	std::vector<real_t> memcpy_half_t_mem(
			rad::DeviceVector<real_t>& d_dev_half_t) {
		return {};
	}

	UnhardenedTensorCoresCaller(uint32_t m, uint32_t n) :
			TensorCoresCaller<half_t, real_t>(m, n) {
	} //default constructor
};

template<typename half_t, typename real_t>
struct DMRTensorCoresCaller: public TensorCoresCaller<half_t, real_t> {

	void gemm(
			rad::DeviceVector<half_t>& a_dev, 			//A matrix
			rad::DeviceVector<half_t>& b_dev, 			//B matrix
			rad::DeviceVector<real_t>& c_dev, 			//C matrix
			rad::DeviceVector<real_t>& d_dev, 			//D matrix
			rad::DeviceVector<real_t>& d_dev_half_t,  	//D_Half matrix
			real_t alpha, real_t beta, int wA, int wB,
			const uint32_t threshold) {
		matrix_mult_kernel_wmma_dmr<<<this->dim_grid, this->dim_block>>>(
				a_dev.data(), b_dev.data(), c_dev.data(), d_dev.data(),
				d_dev_half_t.data(), alpha, beta, wA, wB, wA);

	}

	std::vector<real_t> memcpy_half_t_mem(
			rad::DeviceVector<real_t>& d_dev_half_t) {
		return d_dev_half_t.to_vector();
	}

	DMRTensorCoresCaller(uint32_t m, uint32_t n) :
			TensorCoresCaller<half_t, real_t>(m, n) {
	} //default constructor
};

template<typename half_t, typename real_t>
void setup_execute(Log& log_obj, TensorCoresCaller<half_t, real_t>& mult_env,
		const uint32_t threshold = 0) {
	double elapsed_time = 0;

	std::vector<half_t> a_vector_host(
			log_obj.size_matrices * log_obj.size_matrices);
	std::vector<half_t> b_vector_host(
			log_obj.size_matrices * log_obj.size_matrices);
	std::vector<real_t> c_vector_host(
			log_obj.size_matrices * log_obj.size_matrices);
	std::vector<real_t> gold_host(
			log_obj.size_matrices * log_obj.size_matrices);

	//Output host vectors are set after computation
	std::vector<real_t> d_vector_host_real_t;
	std::vector<real_t> d_vector_host_half_t;

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

	rad::DeviceVector<half_t> a_vector_device = a_vector_host;
	rad::DeviceVector<half_t> b_vector_device = b_vector_host;
	rad::DeviceVector<real_t> c_vector_device = c_vector_host;

	rad::DeviceVector<real_t> d_vector_device(
			log_obj.size_matrices * log_obj.size_matrices);
	rad::DeviceVector<real_t> d_vector_half_t_device(
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
		rad::checkFrameworkErrors(cudaDeviceSynchronize());
		;
		rad::checkFrameworkErrors(cudaPeekAtLastError());

		//end iteration
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

/**
 * Setup for Tensor (GEMM)
 */
void setup_gemm_tensor_cores_unhardened(Log& log) {
//	if (log.precision == "half") {
//		UnhardenedTensorCoresCaller<half, half> gemm_obj(log.size_matrices,
//				log.size_matrices);
//		setup_execute(log, gemm_obj);
//	}
//
//	if (log.precision == "float" || log.precision == "single") {
//		UnhardenedTensorCoresCaller<half, float> gemm_obj(log.size_matrices,
//				log.size_matrices);
//		setup_execute(log, gemm_obj);
//	}
}
void setup_gemm_tensor_cores_dmr(Log& log) {
//	if (log.precision == "half") {
//		DMRTensorCoresCaller<half, half> gemm_obj(log.size_matrices,
//				log.size_matrices);
//		setup_execute(log, gemm_obj, THRESHOLD_1);
//	}
//
//	if (log.precision == "float" || log.precision == "single") {
//		DMRTensorCoresCaller<half, float> gemm_obj(log.size_matrices,
//				log.size_matrices);
//		setup_execute(log, gemm_obj, THRESHOLD_1);
//	}
}

