#include "setup.h"
#include "Log.h"
#include "include/device_vector.h"
#include "common_template_functions.h"
#include "no_tensor_kernels.h"

template<const uint32_t COUNT, typename half_t, typename real_t>
struct GemmCaller {
	bool duplicated;
	dim3 dim_grid, dim_block;

	rad::DeviceVector<real_t> a_dev;
	rad::DeviceVector<real_t> b_dev;
	rad::DeviceVector<real_t> c_dev;
	rad::DeviceVector<half_t> c_half_t_dev;

	virtual ~GemmCaller() = default;
	virtual void gemm(real_t alpha, real_t beta, int wA, int wB,
			const uint32_t threshold) = 0;

	virtual std::vector<half_t> memcpy_half_t_mem() = 0;

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

	void gemm(real_t alpha, real_t beta, int wA, int wB,
			const uint32_t threshold) {
		matrix_mult_kernel_unhardened<<<this->dim_grid, this->dim_block>>>( //call
				this->a_dev.data(), //a
				this->b_dev.data(), //b
				this->c_dev.data(), //c
				alpha, beta, wA, wB);
	}

	std::vector<real_t> memcpy_half_t_mem() {
		return {};
	}
	UnhardenedGemmCaller(uint32_t m, uint32_t n) :
			GemmCaller<0, real_t, real_t>(m, n) {
		std::cout << this->dim_block << std::endl;
		std::cout << this->dim_grid << std::endl;
	} //default constructor
};

template<typename real_t>
struct CUBLASGemmCaller: public GemmCaller<0, real_t, real_t> {
	cublasHandle_t blas_handle;

	~CUBLASGemmCaller() {
		rad::checkCublasErrors(cublasDestroy(this->blas_handle));
	}

	void gemm(double alpha, double beta, int wA, int wB,
			const uint32_t threshold) {
		rad::checkCublasErrors(
				cublasDgemm(this->blas_handle, CUBLAS_OP_N, CUBLAS_OP_N, wA, wB,
						wA, &alpha, this->a_dev.data(), wA, this->b_dev.data(),
						wB, &beta, this->c_dev.data(), wB));
	}

	void gemm(float alpha, float beta, int wA, int wB,
			const uint32_t threshold) {
		rad::checkCublasErrors(
				cublasSgemm(this->blas_handle, CUBLAS_OP_N, CUBLAS_OP_N, wA, wB,
						wA, &alpha, this->a_dev.data(), wA, this->b_dev.data(),
						wB, &beta, this->c_dev.data(), wB));
	}

#if __CUDA_ARCH__ >= 600 // more than titan

	void gemm(half alpha, half beta, int wA, int wB,
			const uint32_t threshold) {
		rad::checkCublasErrors(
				cublasHgemm(this->blas_handle, CUBLAS_OP_N, CUBLAS_OP_N, wA, wB,
						wA, &alpha, this->a_dev.data(), wA, this->b_dev.data(),
						wB, &beta, this->c_dev.data(), wB));
	}
#endif

	std::vector<real_t> memcpy_half_t_mem() {
		return {};
	}

	CUBLASGemmCaller(uint32_t m, uint32_t n, bool use_tensor_cores = false) :
			GemmCaller<0, real_t, real_t>(m, n) {
		rad::checkCublasErrors(cublasCreate(&this->blas_handle));

		if (use_tensor_cores) {
#if __CUDA_ARCH__ >= 700 // more than titan
			rad::checkCublasErrors(
					cublasSetMathMode(this->blas_handle,
							CUBLAS_TENSOR_OP_MATH));
#else
			throw_line("Tensor Cores cannot be used on CUDA_ARCH<7.0");
#endif
		}
	} //default constructor
};

template<const uint32_t COUNT, typename half_t, typename real_t>
struct DMRMixedGemmCaller: public GemmCaller<COUNT, half_t, real_t> {

	void gemm(real_t alpha, real_t beta, int wA, int wB,
			const uint32_t threshold) {
		matrix_mult_kernel_dmr_mixed<COUNT> <<<this->dim_grid, this->dim_block>>>( //call
				this->a_dev.data(), 				//a
				this->b_dev.data(), 				//b
				this->c_dev.data(), 				//c
				this->c_half_t_dev.data(), 				//d
				alpha, beta, wA, wB, threshold);

	}

	DMRMixedGemmCaller(uint32_t m, uint32_t n) :
			GemmCaller<COUNT, half_t, real_t>(m, n) {
		this->duplicated = true;
	}

	std::vector<half_t> memcpy_half_t_mem() {
		return this->c_half_t_dev.to_vector();
	}
};

template<typename real_t>
struct DMRGemmCaller: public GemmCaller<0, real_t, real_t> {

	void gemm(real_t alpha, real_t beta, int wA, int wB,
			const uint32_t threshold) {
		matrix_mult_kernel_unhardened<<<this->dim_grid, this->dim_block>>>( //call
				this->a_dev.data(), 				//a
				this->b_dev.data(), 				//b
				this->c_dev.data(), 				//c
				alpha, beta, wA, wB);
		matrix_mult_kernel_unhardened<<<this->dim_grid, this->dim_block>>>( //call
				this->a_dev.data(), 				//a
				this->b_dev.data(), 				//b
				this->c_half_t_dev.data(), 				//c
				alpha, beta, wA, wB);

		rad::checkFrameworkErrors(cudaDeviceSynchronize());
		;
		rad::checkFrameworkErrors(cudaPeekAtLastError());
		;
		uint32_t thread_block = BLOCK_SIZE * BLOCK_SIZE;
		uint32_t grid_block = (wA * wB) / thread_block;
		compare_two_outputs<<<grid_block, thread_block>>>(this->c_dev.data(),
				this->c_half_t_dev.data());

	}

	DMRGemmCaller(uint32_t m, uint32_t n) :
			GemmCaller<0, real_t, real_t>(m, n) {
		this->duplicated = true;
	}

	std::vector<real_t> memcpy_half_t_mem() {
		return this->c_half_t_dev.to_vector();
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
	std::vector<real_t> d_vector_host(
			log_obj.size_matrices * log_obj.size_matrices);
	std::vector<real_t> gold_host(
			log_obj.size_matrices * log_obj.size_matrices);

	//Output host vectors are set after computation
	std::vector<half_t> c_vector_host_half_t(
			log_obj.size_matrices * log_obj.size_matrices);
	;

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

	mult_env.a_dev = a_vector_host;
	mult_env.b_dev = b_vector_host;

	std::cout << "Starting the setup process...\n";
	std::cout << std::setprecision(5) << std::fixed;
	for (int it = 0; it < log_obj.iterations; it++) {
		mult_env.c_dev = c_vector_host;
		auto computation_time = rad::mysecond();

		log_obj.start_iteration();

		mult_env.gemm(log_obj.alpha, log_obj.beta, log_obj.size_matrices,
				log_obj.size_matrices, threshold);
		rad::checkFrameworkErrors(cudaDeviceSynchronize());
		;
		rad::checkFrameworkErrors(cudaPeekAtLastError());
		;

		log_obj.end_iteration();
		computation_time = rad::mysecond() - computation_time;
		elapsed_time += computation_time;

		double copy_time = rad::mysecond();
		c_vector_host_half_t = mult_env.memcpy_half_t_mem();
		d_vector_host = mult_env.c_dev.to_vector();
		copy_time = rad::mysecond() - copy_time;

		if (!log_obj.generate) {

			auto comparing_time = rad::mysecond();
			auto errors = std::pair<int, int>();
			errors = check_output_errors_dmr(gold_host, d_vector_host,
					c_vector_host_half_t, log_obj, threshold,
					mult_env.duplicated);

			comparing_time = rad::mysecond() - comparing_time;
			if (log_obj.verbose) {
				std::cout << "Iteration: " << it << " DMR errors "
						<< errors.first << ". " << "Radiation errors: "
						<< errors.second << ". "
						<< "Time spent on computation: " << computation_time
						<< "s. " << "Time spent on comparing: "
						<< comparing_time << "s. " << "Time spent on copying: "
						<< copy_time << "s. " << std::endl;
			} else {
				std::cout << "Iteration: " << it << " DMR errors "
						<< errors.first << ". " << "Radiation errors: "
						<< errors.second << ". " << std::endl;
			}
			//If errors != 0 reload matrices to gpu
			if (errors.first != 0 || errors.second != 0) {
				read_gold(a_vector_host, b_vector_host, c_vector_host,
						gold_host, log_obj.a_input_path, log_obj.b_input_path,
						log_obj.c_input_path, log_obj.gold_inout_path);

				mult_env.a_dev.resize(0);
				mult_env.b_dev.resize(0);
				mult_env.c_dev.resize(0);
				mult_env.c_half_t_dev.resize(0);

				mult_env.a_dev = a_vector_host;
				mult_env.b_dev = b_vector_host;
				mult_env.c_dev = c_vector_host;
				mult_env.c_half_t_dev = c_vector_host_half_t;

			}

		}

	}
	if (log_obj.verbose) {

		std::cout << "Elapsed time: " << (elapsed_time / log_obj.iterations)
				<< " s\n";
	} else {
		std::cout << "done.\n";
	}
	if (log_obj.generate) {
		write_gold(a_vector_host, b_vector_host, c_vector_host, d_vector_host,
				log_obj.a_input_path, log_obj.b_input_path,
				log_obj.c_input_path, log_obj.gold_inout_path);
	}
}

void setup_gemm_unhardened(Log& log) {
	if (log.precision == "half") {
#if __CUDA_ARCH__ >= 600
		UnhardenedGemmCaller<half> gemm_obj(log.size_matrices,
				log.size_matrices);
		setup_execute(log, gemm_obj);
#else
		throw_line("Half MxM is not available for CUDA_ARCH<6.0");
#endif
	}
//
	if (log.precision == "float" || log.precision == "single") {
		UnhardenedGemmCaller<float> gemm_obj(log.size_matrices,
				log.size_matrices);
		setup_execute(log, gemm_obj);
	}

	if (log.precision == "double") {
		UnhardenedGemmCaller<double> gemm_obj(log.size_matrices,
				log.size_matrices);
		setup_execute(log, gemm_obj);
	}
}

void setup_gemm_cublas(Log& log) {
	if (log.precision == "half") {
#if __CUDA_ARCH__ >= 600
		CUBLASGemmCaller<half> gemm_obj(log.size_matrices, log.size_matrices,
				log.use_tensor_cores);
		setup_execute(log, gemm_obj);
#else
		throw_line("Half GEMM is not available for CUDA_ARCH<6.0");
#endif
	}
//
	if (log.precision == "float" || log.precision == "single") {
		CUBLASGemmCaller<float> gemm_obj(log.size_matrices, log.size_matrices,
				log.use_tensor_cores);
		setup_execute(log, gemm_obj);
	}

	if (log.precision == "double") {
		CUBLASGemmCaller<double> gemm_obj(log.size_matrices, log.size_matrices,
				log.use_tensor_cores);
		setup_execute(log, gemm_obj);
	}
}

void setup_gemm_dmr(Log& log) {
	if (log.precision == "float" || log.precision == "single") {
		throw_line("Not ready yet");
	}

	if (log.precision == "double") {

		if (log.dmr == "mixed") {
			switch (log.check_block) {
			case ONE_OP_CHECK: {
				DMRMixedGemmCaller<ONE_OP_CHECK, float, double> gemm_obj(
						log.size_matrices, log.size_matrices);
				setup_execute(log, gemm_obj, THRESHOLD_1);
				break;

			}
			default: {
				//The counter will never be 32, so it will check only at the end
				DMRMixedGemmCaller<AT_END_OP_CHECK, float, double> gemm_obj(
						log.size_matrices, log.size_matrices);
				setup_execute(log, gemm_obj, THRESHOLD_AT_END);
				break;
			}
			}

		} else if (log.dmr == "full") {
			DMRGemmCaller<double> gemm_obj(log.size_matrices,
					log.size_matrices);
			setup_execute(log, gemm_obj);
		}
	}
}

