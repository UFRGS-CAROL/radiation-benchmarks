/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/**
 * Matrix multiplication: C = A * B.
 * Host code.
 *
 * This sample implements matrix multiplication as described in Chapter 3
 * of the programming guide.
 * It has been written for clarity of exposition to illustrate various CUDA
 * programming principles, not with the goal of providing the most
 * performant generic kernel for matrix multiplication.
 *
 * See also:
 * V. Volkov and J. Demmel, "Benchmarking GPUs to tune dense linear algebra,"
 * in Proc. 2008 ACM/IEEE Conf. on Supercomputing (SC '08),
 * Piscataway, NJ: IEEE Press, 2008, pp. Art. 31:1-11.
 */

// Helper functions and utilities to work with CUDA
#include <helper_functions.h>

// Include the kernel
#include "matrixMul_kernel.h"
#include "device_vector.h"
#include "persistent_lib.h"

#ifdef LOGS
#include "log_helper.h"

#ifdef BUILDPROFILER
#include "NVMLWrapper.h"
#define OBJTYPE NVMLWrapper
#endif

#endif

#include <omp.h>

//STD LIBS
#include <random>	//generate A and B matrices
#include <fstream>  //Input/output files
#include <iostream> //Output messages
#include <iomanip>	// random generator
#include <algorithm>    // std::count_if

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif

#define GENERATOR_MAXABSVALUE 4.1e+2
#define GENERATOR_MINABSVALUE 0

struct Parameters {
	size_t k;
	size_t n_streams;
	std::string input_a;
	std::string input_b;
	std::string gold;

	size_t iterations;

	bool verbose;
	bool generate;
	bool fault_injection;

	KernelType execution_type;

	friend std::ostream& operator<<(std::ostream& os, const Parameters& dt) {
		os << "Matrix memory space size: " << dt.k << "x" << dt.k << std::endl;
		os << "Batched matrix size: " << dt.n_streams << std::endl;
		os << "Input a: " << dt.input_a << std::endl;
		os << "Input_b: " << dt.input_b << std::endl;
		os << "Gold: " << dt.gold << std::endl;
		os << "Iterations: " << dt.iterations << std::endl;
		os << std::boolalpha << "Verbose: " << dt.verbose << std::endl;
		os << "Generate: " << dt.generate << std::endl;
		os << "Fault injection: " << dt.fault_injection << std::endl;
		os << "Kernel type: " << dt.execution_type;
		return os;
	}

	void usage(char **argv) {
		std::cout << "Usage: " << argv[0]
				<< " -size=N [-generate] [-input_a=<path>] [-input_b=<path>] [-gold=<path>] "
						"[-iterations=N] [-verbose]" << std::endl;
	}

	Parameters(int argc, char** argv) {
		if (argc < 2) {
			usage(argv);
			exit(-1);
		}

		if (checkCmdLineFlag(argc, (const char **) argv, "size")) {
			this->k = getCmdLineArgumentInt(argc, (const char **) argv, "size");

			if ((k <= 0) || (k % 16 != 0)) {
				std::cerr << "Invalid input size given on the command-line: "
						<< k << std::endl;
				exit (EXIT_FAILURE);
			}

		} else {
			usage(argv);
			exit (EXIT_FAILURE);
		}

		if (checkCmdLineFlag(argc, (const char **) argv, "batch")) {
			this->n_streams = getCmdLineArgumentInt(argc, (const char **) argv,
					"batch");
			if (this->n_streams < 1) {
				usage(argv);
				exit (EXIT_FAILURE);
			}
		}

		if (checkCmdLineFlag(argc, (const char **) argv, "input_a")) {
			char* a_matrix_path;
			getCmdLineArgumentString(argc, (const char **) argv, "input_a",
					&a_matrix_path);
			this->input_a = std::string(a_matrix_path);
		}

		if (checkCmdLineFlag(argc, (const char **) argv, "input_b")) {
			char* b_matrix_path;

			getCmdLineArgumentString(argc, (const char **) argv, "input_b",
					&b_matrix_path);
			this->input_b = std::string(b_matrix_path);
		}

		if (checkCmdLineFlag(argc, (const char **) argv, "gold")) {
			char* gold_matrix_path;
			getCmdLineArgumentString(argc, (const char **) argv, "gold",
					&gold_matrix_path);
			this->gold = std::string(gold_matrix_path);
		}

		if (checkCmdLineFlag(argc, (const char **) argv, "iterations")) {
			this->iterations = getCmdLineArgumentInt(argc, (const char **) argv,
					"iterations");
		}

		if (checkCmdLineFlag(argc, (const char **) argv, "verbose")) {
			this->verbose = 1;
		}

		if (checkCmdLineFlag(argc, (const char **) argv, "debug")) {
			this->fault_injection = 1;
			std::cout << ("!! Will be injected an input error") << std::endl;
		}

		if (checkCmdLineFlag(argc, (const char **) argv, "generate")) {
			this->generate = 1;
			this->fault_injection = 0;
			this->iterations = 1;
			std::cout
					<< ("!! Generate !! Disabling device_warmup, fault_injection and iterations limiting.")
					<< std::endl;
		}

		if (checkCmdLineFlag(argc, (const char **) argv, "kernel_type")) {
			int ty = getCmdLineArgumentInt(argc, (const char **) argv,
					"kernel_type");
			if (ty >= 0 && ty < COUNT) {
				this->execution_type = kernel_types[ty];
			} else {
				std::cerr << "Kernel type not int the range:" << std::endl
						<< " 0 for nonpersistent threads, 1 for persistent, and 2 for batched cublas GEMM"
						<< std::endl;
			}
		}
	}

};

template<typename real_t>
void generate_input(std::vector<real_t>& a_vector,
		std::vector<real_t>& b_vector) {
	std::random_device rd; //Will be used to obtain a seed for the random number engine
	std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
	std::uniform_real_distribution<real_t> dis(-GENERATOR_MAXABSVALUE,
	GENERATOR_MAXABSVALUE);

	auto generator = [&dis, &gen]() {
		return dis(gen);
	};

	generate(begin(a_vector), end(a_vector), generator);
	generate(begin(b_vector), end(b_vector), generator);

	int numZeros = std::count(a_vector.begin(), a_vector.end(), real_t(0.0));

	int numNans = std::count_if(a_vector.begin(), a_vector.end(),
			[](real_t i) {return std::isnan(i);});
	int numInfs = std::count_if(a_vector.begin(), a_vector.end(),
			[](real_t i) {return std::isinf(i);});

	std::cout << "Number of zeros/NaNs/INFs on matrix A: " << numZeros << " "
			<< numNans << " " << numInfs << std::endl;

	numZeros = std::count(b_vector.begin(), b_vector.end(), real_t(0.0));
	numNans = std::count_if(b_vector.begin(), b_vector.end(),
			[](real_t i) {return std::isnan(i);});
	numInfs = std::count_if(b_vector.begin(), b_vector.end(),
			[](real_t i) {return std::isinf(i);});

	std::cout << "Number of zeros/NaNs/INFs on matrix B: " << numZeros << " "
			<< numNans << " " << numInfs << std::endl;
}

template<typename real_t>
void load_file_data(std::string& path, std::vector<real_t>& array) {
	std::ifstream input(path, std::ios::binary);
	if (input.good()) {
		input.read(reinterpret_cast<char*>(array.data()),
				array.size() * sizeof(real_t));
	}
	input.close();
}

template<typename real_t>
void write_to_file(std::string& path, std::vector<real_t>& array) {
	std::ofstream output(path, std::ios::binary);
	if (output.good()) {
		output.write(reinterpret_cast<char*>(array.data()),
				array.size() * sizeof(real_t));
	}
	output.close();
}

template<typename real_t>
int compare_small_matrix(const int k, const int stream, bool verbose,
		bool generate, const real_t* gold, const real_t* found) {
	int host_errors = 0;
	for (int i = 0; i < k; i++) {
		for (int j = 0; j < k; j++) {
			int index = i * k + j;
			register double valGold = gold[index];
			register double valOutput = found[index];
			if (valGold != valOutput) {
#pragma OMP critical
				{
					std::stringstream error_detail;
					error_detail << std::scientific << std::setprecision(20);
					error_detail << "stream:" << stream << ",";
					error_detail << " p: [" << i << ", " << j << "],";
					error_detail << " r: " << valOutput << ", e: " << valGold;
					if (verbose && (host_errors < 10))
						std::cout << error_detail.str() << std::endl;
					host_errors++;

#ifdef LOGS
					if (!generate)
					log_error_detail(
							const_cast<char*>(error_detail.str().c_str()));
#endif

				}
			}
		}
	}
	return host_errors;
}

template<typename real_t>
int check_output(std::vector<real_t>& gold, std::vector<real_t>& found,
		int n_k_times, int k, bool verbose, bool generate) {
	int host_errors = 0, stream;

#pragma omp parallel for shared(host_errors, stream)
	for (stream = 0; stream < n_k_times; stream++) {
		int global_index = stream * k * k;
		const real_t* g_data = gold.data() + global_index;
		const real_t* f_data = found.data() + global_index;
		host_errors += compare_small_matrix(k, stream, verbose, generate,
				g_data, f_data);
	}

#ifdef LOGS
	if (!generate) {
		log_error_count(host_errors);
	}
#endif

	if (host_errors != 0)
		printf("#");

	return host_errors;
}

/**
 * Program main
 */

int main(int argc, char **argv) {
	Parameters args(argc, argv);
	std::cout << "Benchmarks parameters" << std::endl;
	std::cout << args << std::endl;
	//================== Init logs
#ifdef LOGS
	if (args.generate == false) {
		std::string test_info = "";
		std::string test_name = "cuda_float_mxm_";

		test_info += "size:" + std::to_string(args.k) + " type:float-";
		switch (args.execution_type) {
			case STATIC:
			test_info += "static";
			test_name += "static";
			break;
			case PERSISTENT:
			test_info += "persistent";
			test_name += "persistent";
			break;
			case GEMM:
			test_info += "gemm";
			test_name += "gemm";
			break;
		}

		test_info += " block_size:" + std::to_string(BLOCK_SIZE);
		test_info += " batch_size:" + std::to_string(args.n_streams);

		start_log_file(const_cast<char*>(test_name.c_str()),
				const_cast<char*>(test_info.c_str()));
		set_iter_interval_print(10);
	}

#ifdef BUILDPROFILER
	std::string log_file_name(get_log_file_name());
	if (args.generate) {
		log_file_name = "/tmp/generate.log";
	}
	//	rad::Profiler profiler_thread = new rad::JTX2Inst(log_file_name);
	std::shared_ptr < rad::Profiler > profiler_thread = std::make_shared
	< rad::OBJTYPE > (0, log_file_name);

	//START PROFILER THREAD
	profiler_thread->start_profile();
#endif

#endif

	//Batched gemm memory size
	auto num_elements = args.k * args.k * args.n_streams;

	//Host memory allocation
	std::vector<float> a_host(num_elements);
	std::vector<float> b_host(num_elements);
	std::vector<float> c_host(num_elements);
	std::vector<float> gold(num_elements);

	//Define the grid size
	int gridsize = args.k / BLOCK_SIZE < 1 ? 1 : args.k / BLOCK_SIZE;
	int blocksize = args.k / BLOCK_SIZE < 1 ? args.k : BLOCK_SIZE;
	dim3 dim_block(blocksize, blocksize);
	dim3 dim_grid(gridsize, gridsize);

	//Load or write the values to files
	if (args.generate) {
		//generate input
		generate_input(a_host, b_host);
		write_to_file(args.input_a, a_host);
		write_to_file(args.input_b, b_host);
	} else {
		load_file_data(args.input_a, a_host);
		load_file_data(args.input_b, b_host);
		load_file_data(args.gold, gold);
	}

	//Debug fault injection
	if (args.fault_injection) {
		a_host[a_host.size() / 3] = 3939393;
	}

	//Device memory allocation
	rad::DeviceVector<float> a_device = a_host;
	rad::DeviceVector<float> b_device = b_host;
	rad::DeviceVector<float> c_device = c_host;

	//Raw arrays
	float* a_dev_ptr = a_device.data();
	float* b_dev_ptr = b_device.data();
	float* c_dev_ptr = c_device.data();

	//Streams and handles cannot be copied
	//Otherwise it is necessary to destroy and realocate streams
	//so it is better to use smart pointers
	std::shared_ptr<CublasHandle> cublas_handle;

	//Streams allocation
	std::vector < std::shared_ptr
			< CudaStream >> streams(args.n_streams, nullptr);

	//Persistent case
	rad::HostPersistentControler pk(dim_grid);

	//SETUP for the type kernel
	switch (args.execution_type) {
	case PERSISTENT:
		streams[0] = std::make_shared<CudaStream>();

		matrixMulCUDA(c_dev_ptr, a_dev_ptr, b_dev_ptr, args.k, args.k, streams,
				args.execution_type, dim_grid, dim_block, cublas_handle);
		break;
	case STATIC:
		for (auto& st : streams) {
			st = std::make_shared<CudaStream>();
		}
		break;
	case GEMM:
		cublas_handle = std::make_shared<CublasHandle>();
		break;

	case DYNAMICPARALLELISM:
		streams[0] = std::make_shared<CudaStream>();
		break;
	}

	//START the processing
	for (auto it = 0; it < args.iterations; it++) {

		c_device.clear();

		auto kernel_time = rad::mysecond();
#ifdef LOGS
		if(args.generate == false) {
			start_iteration();
		}
#endif
		if (args.execution_type == PERSISTENT) {
			pk.process_data_on_kernel();
		} else {
			matrixMulCUDA(c_dev_ptr, a_dev_ptr, b_dev_ptr, args.k, args.k,
					streams, args.execution_type, dim_grid, dim_block,
					cublas_handle);
		}

#ifdef LOGS
		if(args.generate == false) {
			end_iteration();
		}
#endif
		kernel_time = rad::mysecond() - kernel_time;

		//Copy back to host
		c_host = c_device.to_vector();

		auto comparison_time = rad::mysecond();
		auto errors = 0;
		if (args.generate == false) {
			errors = check_output(gold, c_host, args.n_streams, args.k,
					args.verbose, args.generate);

			//Reload the values in the GPU DDR
			if (errors != 0) {
#ifdef LOGS
#ifdef BUILDPROFILER
				profiler_thread->end_profile();
#endif
#endif
				if (args.execution_type == PERSISTENT) {
					pk.end_kernel();
				}
				load_file_data(args.input_a, a_host);
				load_file_data(args.input_b, b_host);
				load_file_data(args.gold, gold);

				a_device = a_host;
				b_device = b_host;
				c_device = c_host;

				if (args.execution_type == PERSISTENT) {
					pk.start_kernel();
					matrixMulCUDA(c_dev_ptr, a_dev_ptr, b_dev_ptr, args.k,
							args.k, streams, args.execution_type, dim_grid,
							dim_block, cublas_handle);
				}
#ifdef LOGS
#ifdef BUILDPROFILER
				profiler_thread->start_profile();
#endif
#endif
			}
		}

		comparison_time = rad::mysecond() - comparison_time;

		if (args.verbose) {
			std::cout << "----------------------------------------------------"
					<< std::endl;
			std::cout << "Iteration: " << it << std::endl;
			std::cout << "Kernel time: " << kernel_time << std::endl;
			std::cout << "Comparison time: " << comparison_time << std::endl;
			std::cout << "Errors: " << errors << std::endl;
			std::cout << "----------------------------------------------------"
					<< std::endl;

		}

	}


	if (args.generate) {
		std::cout << gold[10] << " " << c_host[10] << std::endl;
		write_to_file(args.gold, c_host);
	}

#ifdef LOGS
	if(!args.generate) {
		end_log_file();
	}

#ifdef BUILDPROFILER
	profiler_thread->end_profile();
#endif
#endif
	return 0;
}

