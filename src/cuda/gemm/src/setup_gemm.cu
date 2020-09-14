#include "setup.h"
#include "Parameters.h"
#include "include/device_vector.h"
#include "common_template_functions.h"
#include "no_tensor_kernels.h"

std::string get_cuda_cc_version() {
	std::string ret = "MAJOR_" + std::to_string(__CUDACC_VER_MAJOR__);
	ret += "_MINOR_" + std::to_string(__CUDACC_VER_MINOR__);
	return ret;
}

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

	void gemm(half alpha, half beta, int wA, int wB, const uint32_t threshold) {
#if (__CUDACC_VER_MAJOR__ >= 10)

		rad::checkCublasErrors(
				cublasHgemm(this->blas_handle, CUBLAS_OP_N, CUBLAS_OP_N, wA, wB,
						wA, &alpha, this->a_dev.data(), wA, this->b_dev.data(),
						wB, &beta, this->c_dev.data(), wB));
#endif
	}

	std::vector<real_t> memcpy_half_t_mem() {
		return {};
	}

	CUBLASGemmCaller(uint32_t m, uint32_t n, bool use_tensor_cores = false) :
			GemmCaller<0, real_t, real_t>(m, n) {
		rad::checkCublasErrors(cublasCreate(&this->blas_handle));

		if (use_tensor_cores) {
#if (__CUDACC_VER_MAJOR__ >= 10)
			rad::checkCublasErrors(
					cublasSetMathMode(this->blas_handle,
							CUBLAS_TENSOR_OP_MATH));
#endif

		}
	} //default constructor
};

template<typename real_t>
struct CUTLASSGemmCaller: public GemmCaller<0, real_t, real_t> {

	void gemm(double alpha, double beta, int wA, int wB,
			const uint32_t threshold) {
		throw_line("CUTLASS double not implemented yet\n");
	}

	void gemm(float alpha, float beta, int wA, int wB,
			const uint32_t threshold) {
		this->cutlass_sgemm_nn(wA, wB, wB, alpha, this->a_dev.data(), wA,
				this->b_dev.data(), wB, beta, this->c_dev.data(), wB);

	}

	std::vector<real_t> memcpy_half_t_mem() {
		return {};
	}

	CUTLASSGemmCaller(uint32_t m, uint32_t n, bool use_tensor_cores = false) :
			GemmCaller<0, real_t, real_t>(m, n) {
		throw_line("Not implemented yet");

	} //default constructor

	// Define a CUTLASS GEMM template and launch a GEMM kernel.
	void cutlass_sgemm_nn(int m, int n, int k, float alpha, float const *A,
			int lda, float const *B, int ldb, float beta, float *C, int ldc) {
		throw_line("Not implemented yet");
	}

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
void setup_execute(Parameters& parameters,
		GemmCaller<COUNT, half_t, real_t>& mult_env, const uint32_t threshold =
				0) {
	double elapsed_time = 0;

	std::vector<real_t> a_vector_host(
			parameters.size_matrices * parameters.size_matrices);
	std::vector<real_t> b_vector_host(
			parameters.size_matrices * parameters.size_matrices);
	std::vector<real_t> c_vector_host(
			parameters.size_matrices * parameters.size_matrices);
	std::vector<real_t> d_vector_host(
			parameters.size_matrices * parameters.size_matrices);
	std::vector<real_t> gold_host(
			parameters.size_matrices * parameters.size_matrices);

	//Output host vectors are set after computation
	std::vector<half_t> c_vector_host_half_t(
			parameters.size_matrices * parameters.size_matrices);
	;

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

	mult_env.a_dev = a_vector_host;
	mult_env.b_dev = b_vector_host;
	mult_env.c_half_t_dev.resize(b_vector_host.size());

	std::cout << "Starting the setup process...\n";
	std::cout << std::setprecision(5) << std::fixed;
	for (int it = 0; it < parameters.iterations; it++) {
		mult_env.c_dev = c_vector_host;

		auto computation_time = rad::mysecond();

		parameters.start_iteration();

		mult_env.gemm(half_t(parameters.alpha), half_t(parameters.beta),
				parameters.size_matrices, parameters.size_matrices, threshold);
		rad::checkFrameworkErrors(cudaDeviceSynchronize());
		;
		rad::checkFrameworkErrors(cudaPeekAtLastError());
		;

		parameters.end_iteration();
		computation_time = rad::mysecond() - computation_time;
		elapsed_time += computation_time;

		double copy_time = rad::mysecond();
		c_vector_host_half_t = mult_env.memcpy_half_t_mem();
		mult_env.c_dev.to_vector(d_vector_host);
		copy_time = rad::mysecond() - copy_time;

		if (!parameters.generate) {

			auto comparing_time = rad::mysecond();
			auto errors = std::pair<int, int>();
			errors = check_output_errors_dmr(gold_host, d_vector_host,
					c_vector_host_half_t, parameters, threshold,
					mult_env.duplicated);

			comparing_time = rad::mysecond() - comparing_time;
			if (parameters.verbose) {
				auto wasted_time = copy_time + comparing_time;
				auto full_time = wasted_time + computation_time;
				std::cout << "Iteration: " << it << " DMR errors "
						<< errors.first << ". " << "Radiation errors: "
						<< errors.second << ". "
						<< "Time spent on computation: " << computation_time
						<< "s. " << "Time spent on comparing: "
						<< comparing_time << "s. " << "Time spent on copying: "
						<< copy_time << "s. " << std::endl;
				std::cout << "Wasted time " << wasted_time << " ("
						<< int((wasted_time / full_time) * 100.0f) << "%)"
						<< std::endl;
			} else {
				std::cout << "Iteration: " << it << " DMR errors "
						<< errors.first << ". " << "Radiation errors: "
						<< errors.second << ". " << std::endl;
			}
			//If errors != 0 reload matrices to gpu
			if (errors.first != 0 || errors.second != 0) {
				read_abc_files(parameters.a_input_path, a_vector_host,
						parameters.b_input_path, b_vector_host,
						parameters.c_input_path, c_vector_host);
				read_gold(parameters.gold_inout_path, gold_host);

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
	if (parameters.verbose) {

		std::cout << "Elapsed time: " << (elapsed_time / parameters.iterations)
				<< " s\n";
	} else {
		std::cout << "done.\n";
	}

	if (parameters.generate) {
		auto zero_count = 0ul;
		for (auto s : d_vector_host) {
			zero_count += (float(s) == 0.0f);
		}
		std::cout << "Zero values on gold: " << zero_count << std::endl;
		write_gold(parameters.gold_inout_path, d_vector_host);
	}
}

void setup_gemm_unhardened(Parameters& parameters) {
	if (parameters.precision == "half") {
#if __CUDA_ARCH__ >= 550
		UnhardenedGemmCaller<half> gemm_obj(parameters.size_matrices,
				parameters.size_matrices);
		setup_execute(parameters, gemm_obj);
#else
		throw_line("Half MxM is not available for CUDA_ARCH<6.0");
#endif
	}
//
	if (parameters.precision == "float" || parameters.precision == "single") {
		UnhardenedGemmCaller<float> gemm_obj(parameters.size_matrices,
				parameters.size_matrices);
		setup_execute(parameters, gemm_obj);
	}

	if (parameters.precision == "double") {
		UnhardenedGemmCaller<double> gemm_obj(parameters.size_matrices,
				parameters.size_matrices);
		setup_execute(parameters, gemm_obj);
	}
}

void setup_gemm_cublas(Parameters& parameters) {
	if (parameters.precision == "half") {
		CUBLASGemmCaller<half> gemm_obj(parameters.size_matrices,
				parameters.size_matrices, parameters.use_tensor_cores);
		setup_execute(parameters, gemm_obj);

	}

	if (parameters.precision == "float" || parameters.precision == "single") {
		CUBLASGemmCaller<float> gemm_obj(parameters.size_matrices,
				parameters.size_matrices, parameters.use_tensor_cores);
		setup_execute(parameters, gemm_obj);
	}

	if (parameters.precision == "double") {
		CUBLASGemmCaller<double> gemm_obj(parameters.size_matrices,
				parameters.size_matrices, parameters.use_tensor_cores);
		setup_execute(parameters, gemm_obj);
	}
}

void setup_gemm_cutlass(Parameters& parameters) {
	if (parameters.precision == "half") {
		CUBLASGemmCaller<half> gemm_obj(parameters.size_matrices,
				parameters.size_matrices, parameters.use_tensor_cores);
		setup_execute(parameters, gemm_obj);

	}

	if (parameters.precision == "float" || parameters.precision == "single") {
		CUBLASGemmCaller<float> gemm_obj(parameters.size_matrices,
				parameters.size_matrices, parameters.use_tensor_cores);
		setup_execute(parameters, gemm_obj);
	}

	if (parameters.precision == "double") {
		CUBLASGemmCaller<double> gemm_obj(parameters.size_matrices,
				parameters.size_matrices, parameters.use_tensor_cores);
		setup_execute(parameters, gemm_obj);
	}
}

void setup_gemm_dmr(Parameters& parameters) {
	if (parameters.precision == "float" || parameters.precision == "single") {
		throw_line("Not ready yet");
	}

	if (parameters.precision == "double") {

		if (parameters.dmr == "mixed") {
			switch (parameters.check_block) {
			case ONE_OP_CHECK: {
				DMRMixedGemmCaller<ONE_OP_CHECK, float, double> gemm_obj(
						parameters.size_matrices, parameters.size_matrices);
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
			DMRGemmCaller<double> gemm_obj(parameters.size_matrices,
					parameters.size_matrices);
			setup_execute(parameters, gemm_obj);
		}
	}
}

