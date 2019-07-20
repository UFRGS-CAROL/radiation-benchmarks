#include <iostream>
#include <string>
#include <vector>
#include <random>
#include <fstream>      // std::ifstream
#include <sstream>      // std::stringstream

#include <iomanip>
#include <limits>

#ifdef OMP
#include <omp.h>
#endif

//#include "half.hpp"
#include "Log.h"
#include "GEMM.h"

#ifndef DEFAULT_INPUT_SIZE
#define DEFAULT_INPUT_SIZE 8192
#endif

#define GENERATOR_MAXABSVALUE 2.0
#define GENERATOR_MINABSVALUE 0

// THRESHOLDS
#define ZERO_HALF 6.5469
#define ZERO_FlOAT 0.0
#define ZERO_DOUBLE 0.0
#define ZERO_DMR 0.0

//typedef half_float::half host_half;
//typedef std::vector<host_half> half_vector;

template<class half_t, class real_t> void generate_matrices_files(
		std::vector<half_t>& a_host_vector, std::vector<half_t>& b_host_vector,
		std::vector<real_t>& c_host_vector, Log& log) {

	std::ofstream f_a(log.a_input_path, std::ios::out | std::ios::binary);
	std::ofstream f_b(log.b_input_path, std::ios::out | std::ios::binary);
	std::ofstream f_c(log.c_input_path, std::ios::out | std::ios::binary);

	if (f_a.is_open() && f_b.is_open() && f_c.is_open()) {
		std::random_device rd; //Will be used to obtain a seed for the random number engine
		std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
		std::uniform_real_distribution<double> dis(-GENERATOR_MAXABSVALUE,
		GENERATOR_MAXABSVALUE);

//		for (size_t i = 0; i < log.size_matrices; i++) {
//			for (size_t j = 0; j < log.size_matrices; j++) {
//				a_host_vector[i * log.size_matrices + j] = host_half(dis(gen));
//				b_host_vector[i * log.size_matrices + j] = host_half(dis(gen));
//				c_host_vector[i * log.size_matrices + j] = real_t(dis(gen));
//			}
//		}

		for (size_t i = 0; i < log.size_matrices; i++) {
			for (size_t j = 0; j < log.size_matrices; j++) {
				a_host_vector[i * log.size_matrices + j] = (half_t) 1.0;
				b_host_vector[i * log.size_matrices + j] = (half_t) 1.0;
				c_host_vector[i * log.size_matrices + j] = (real_t) 0.0;
			}
		}

		half_t zero(0.0);
		half_t nan_ = half_t(half_float::nan("0")); //half_t nan_ = half_t(half_float::nanh("0"));
		half_t inf_ = half_t(half_t(0x7C00));

		int numZeros = std::count(a_host_vector.begin(), a_host_vector.end(),
				zero);
		int numNans = std::count(a_host_vector.begin(), a_host_vector.end(),
				nan_);

		int numInfs = std::count(a_host_vector.begin(), a_host_vector.end(),
				inf_);
		std::cout << "Number of zeros/NaNs/INFs on matrix A: " << numZeros
				<< numNans << numInfs << std::endl;

//		std::cout << "entrou generate3" << std::endl;
		numZeros = std::count(b_host_vector.begin(), b_host_vector.end(), zero);
		numNans = std::count(b_host_vector.begin(), b_host_vector.end(), nan_);
		numInfs = std::count(b_host_vector.begin(), b_host_vector.end(), inf_);

		std::cout << "Number of zeros/NaNs/INFs on matrix B: " << numZeros
				<< numNans << numInfs << std::endl;

		numZeros = std::count(c_host_vector.begin(), c_host_vector.end(), zero);
		numNans = std::count(c_host_vector.begin(), c_host_vector.end(), nan_);
		numInfs = std::count(c_host_vector.begin(), c_host_vector.end(), inf_);

		std::cout << "Number of zeros/NaNs/INFs on matrix C: " << numZeros
				<< numNans << numInfs << std::endl;

		f_a.write(reinterpret_cast<char*>(a_host_vector.data()),
				a_host_vector.size() * sizeof(half_t));
		f_b.write(reinterpret_cast<char*>(b_host_vector.data()),
				b_host_vector.size() * sizeof(half_t));
		f_c.write(reinterpret_cast<char*>(c_host_vector.data()),
				c_host_vector.size() * sizeof(real_t));

		f_a.close();
		f_b.close();
		f_c.close();

	} else {
		throw std::runtime_error(
				"Some of the imput files could not be generated\n");
	}

}

template<class real_t>
void write_gold_to_file(std::string gold_path, std::vector<real_t>& gold) {
	std::ofstream f_gold(gold_path, std::ofstream::out | std::ofstream::binary);
	if (f_gold.is_open()) {
		f_gold.write(reinterpret_cast<char*>(gold.data()),
				sizeof(real_t) * gold.size());
		f_gold.close();
	} else {
		throw std::runtime_error("Could not write gold file\n");
	}
}

template<class real_t> int is_output_ok(std::vector<real_t>& d0,
		std::vector<real_t>& d1, std::vector<real_t>& d2,
		std::vector<real_t>& correct_vector) {

	int memory_errors = 0;
	for (size_t i = 0; i < d0.size(); i++) {
		real_t val_output0 = d0[i];
		real_t val_output1 = d1[i];
		real_t val_output2 = d2[i];
		real_t val_output = val_output0;

		if ((val_output0 != val_output1) || (val_output0 != val_output2)) {
			memory_errors++;

			if ((val_output0 != val_output1) && (val_output1 != val_output2)
					&& (val_output0 != val_output2)) {
				// All 3 values diverge
				memory_errors++;
			} else if (val_output1 == val_output2) {
				// Only value 0 diverge
				val_output = val_output1;
			} else if (val_output0 == val_output2) {
				// Only value 1 diverge
				val_output = val_output0;
			} else if (val_output0 == val_output1) {
				// Only value 2 diverge
				val_output = val_output0;
			}
		}
		correct_vector[i] = val_output;
	}
	return memory_errors;
}

template<class half_t, class real_t> void retrieve_matrices(
		std::vector<half_t>& a_host_vector, std::vector<half_t>& b_host_vector,
		std::vector<real_t>& c_host_vector,
		std::vector<real_t>& gold_host_vector, Log& log) {

	double start = log.mysecond();
	std::ifstream f_a(log.a_input_path, std::ios::in | std::ios::binary);
	std::ifstream f_b(log.b_input_path, std::ios::in | std::ios::binary);
	std::ifstream f_c(log.c_input_path, std::ios::in | std::ios::binary);
	std::ifstream f_gold(log.gold_inout_path,
			std::ifstream::in | std::ifstream::binary);

	if (f_a.is_open() && f_b.is_open() && f_c.is_open() && f_gold) {

		f_a.seekg(0, std::ios::beg);
		f_a.read(reinterpret_cast<char*>(a_host_vector.data()),
				sizeof(half_t) * a_host_vector.size());

		f_b.seekg(0, std::ios::beg);
		f_b.read(reinterpret_cast<char*>(b_host_vector.data()),
				sizeof(half_t) * b_host_vector.size());

		f_c.seekg(0, std::ios::beg);
		f_c.read(reinterpret_cast<char*>(c_host_vector.data()),
				sizeof(real_t) * c_host_vector.size());

		f_gold.seekg(0, std::ios::beg);
		f_gold.read(reinterpret_cast<char*>(gold_host_vector.data()),
				sizeof(real_t) * gold_host_vector.size());

		f_a.close();
		f_b.close();
		f_c.close();
		f_gold.close();
	} else {
		log.log_error("Could not retrieve the matrices");
		throw std::runtime_error("Could not retrieve the matrices\n");
	}

	std::cout << "Done with reading matrices " << log.mysecond() - start
			<< "s\n";
}

template<class real_t>
std::pair<int, int> compare_output_matrices(long long host_is_memory_bad,
		std::vector<real_t>& gold, std::vector<real_t>& c0,
		std::vector<real_t>& c1, std::vector<real_t>& c2, Log& log) {

	int host_errors = 0;
	int memory_errors = 0;

	// printf("r = %f \n", c0[2]);

	std::cout << "host_is_memory_bad: " << host_is_memory_bad << std::endl;

	if (host_is_memory_bad != 0) {
		std::string info_detail = "b: is_memory_bad: "
				+ std::to_string(host_is_memory_bad);
		if (log.verbose)
			std::cout << info_detail << std::endl;

		log.log_error(info_detail);
		memory_errors++;
	}

#ifdef OMP
#pragma omp parallel for shared(host_errors)
#endif
	for (size_t i = 0; i < gold.size(); i++) {
		register bool checkFlag = true;
		register real_t valGold = gold[i];
		register real_t valOutput0 = c0[i];
		register real_t valOutput1 = c1[i];
		register real_t valOutput2 = c2[i];
		register real_t valOutput = valOutput0;

		if ((valOutput0 != valOutput1) || (valOutput0 != valOutput2)) {
#ifdef OMP
#pragma omp critical
#endif
			{
				std::stringstream info_detail("");
				info_detail << "m: [" << int(floor(i / log.size_matrices))
						<< ", " << i % log.size_matrices << "], r0: "
						<< valOutput0 << ", r1: " << valOutput1 << ", r2: "
						<< valOutput2;

				if (log.verbose && (memory_errors < 10))
					std::cout << info_detail.str() << std::endl;

				log.log_info(info_detail.str());
				memory_errors++;
			}
			if ((valOutput0 != valOutput1) && (valOutput1 != valOutput2)
					&& (valOutput0 != valOutput2)) {
				// All 3 values diverge
				if (valOutput0 == valGold) {
					valOutput = valOutput0;
				} else if (valOutput1 == valGold) {
					valOutput = valOutput1;
				} else if (valOutput2 == valGold) {
					valOutput = valOutput2;
				} else {
					// NO VALUE MATCHES THE GOLD AND ALL 3 DIVERGE!
					checkFlag = false;
#ifdef OMP
#pragma omp critical
#endif
					{
						std::stringstream info_detail("");
						info_detail << "t: ["
								<< int(floor(i / log.size_matrices)) << ", "
								<< i % log.size_matrices << "], r0: "
								<< valOutput0 << ", r1: " << valOutput1
								<< ", r2: " << valOutput2 << ", e: " << valGold;

						if (log.verbose && (memory_errors < 10))
							std::cout << info_detail.str() << std::endl;

						log.log_info(std::string(info_detail.str()));

						memory_errors++;
					}
				}
			} else if (valOutput1 == valOutput2) {
				// Only value 0 diverge
				valOutput = valOutput1;
			} else if (valOutput0 == valOutput2) {
				// Only value 1 diverge
				valOutput = valOutput0;
			} else if (valOutput0 == valOutput1) {
				// Only value 2 diverge
				valOutput = valOutput0;
			}
		}
		// std::cout << "val gold: " << valGold << std::endl;
		if (valGold != valOutput) {
			if (checkFlag) {
#ifdef OMP
#pragma omp critical
#endif
				{
					// std::cout << "val out: " << valOutput << std::endl;

					std::stringstream error_detail("");
					error_detail << "p: [" << int(floor(i / log.size_matrices))
							<< ", " << i % log.size_matrices << "], r: "
							<< valOutput << ", e: " << valGold;

					if (log.verbose && (host_errors < 10))
						std::cout << error_detail.str() << std::endl;

					log.log_error(error_detail.str());
					host_errors++;
				}
			}
		}
	}

// printf("numErrors:%d", host_errors);

	log.update_info_count(memory_errors);
	log.update_error_count(host_errors);

	if (memory_errors != 0)
		std::cout << "M";
	if (host_errors != 0)
		std::cout << "#";

	std::pair<int, int> res(memory_errors, host_errors);
	return res;
}

template<class real_t>
bool cmp(const real_t lhs, const real_t rhs, Log& log) {
	const real_t diff = abs(lhs - rhs);
	const real_t zero;

	// std::cout << "d0= " << lhs << "d1 = " << rhs << std::endl;	
	// std::cout << "diff= " << diff << std::endl;

	if (log.use_tensor_cores) {
		zero = real_t(ZERO_HALF);
	} else {
		if (log.precision == "float")
			zero = real_t(ZERO_FlOAT);

		if (log.precision == "double")
			zero = real_t(ZERO_DOUBLE);

		if (log.precision == "DMR")
			zero = real_t(ZERO_DMR);
	}

	if (diff > zero) {
		return false;
	}
	return true;
}

template<class real_t>
std::pair<int, int> check_output_errors_dmr(std::vector<real_t>& gold,
		std::vector<real_t>& d0, std::vector<real_t>& d1, Log& log) {
	int host_errors = 0;

#ifdef OMP
#pragma omp parallel for shared(host_errors)
#endif
	for (size_t i = 0; i < gold.size(); i++) {
		real_t valGold = gold[i];
		real_t valOutput0 = d0[i];
		real_t valOutput1 = d1[i];

		if (valGold != valOutput1 || !cmp(valOutput0, valOutput1, log)) {

			std::stringstream error_detail("");
			error_detail << "p: [" << int(floor(i / log.size_matrices)) << ", "
					<< i % log.size_matrices << "], r: " << valOutput1
					<< ", e: " << valGold << " smaller_precision: "
					<< valOutput0;

			if (log.verbose && (host_errors < 10))
				std::cout << error_detail.str() << std::endl;

			log.log_error(error_detail.str());
			host_errors++;
		}
	}
	log.update_error_count(host_errors);
	if (host_errors != 0)
		std::cout << "#";

	std::pair<int, int> res(0, host_errors);
	return res;
}

template<class real_t>
std::pair<int, int> compare_output_matrices(std::vector<real_t>& gold,
		std::vector<real_t>& c0, Log& log) {
	int host_errors = 0;

#ifdef OMP
#pragma omp parallel for shared(host_errors)
#endif
	for (size_t i = 0; i < gold.size(); i++) {
		real_t valGold = gold[i];
		real_t valOutput = c0[i];
		if (valGold != valOutput) {
			std::stringstream error_detail("");
			error_detail << "p: [" << int(floor(i / log.size_matrices)) << ", "
					<< i % log.size_matrices << "], r: " << valOutput << ", e: "
					<< valGold;

			if (log.verbose && (host_errors < 10))
				std::cout << error_detail.str() << std::endl;

			log.log_error(error_detail.str());
			host_errors++;
		}
	}
	log.update_error_count(host_errors);
	if (host_errors != 0)
		std::cout << "#";

	std::pair<int, int> res(0, host_errors);
	return res;
}

template<class half_t, class real_t>
void call_mxm(Log& log_obj) {
	cudaEvent_t start, stop;
	float elapsedTime;
	// Matrices A and B
	std::vector<half_t> host_matrix_a(
			log_obj.size_matrices * log_obj.size_matrices);
	std::vector<half_t> host_matrix_b(
			log_obj.size_matrices * log_obj.size_matrices);

// C matrix
	std::vector<real_t> host_matrix_c(
			log_obj.size_matrices * log_obj.size_matrices);
	std::vector<real_t> host_gold(
			log_obj.size_matrices * log_obj.size_matrices);
// D Matrix
	std::vector<real_t> host_matrix_d0(
			log_obj.size_matrices * log_obj.size_matrices);
	std::vector<real_t> host_matrix_d1(
			log_obj.size_matrices * log_obj.size_matrices);
	std::vector<real_t> host_matrix_d2(
			log_obj.size_matrices * log_obj.size_matrices);

	if (!log_obj.generate) {
		retrieve_matrices<half_t, real_t>(host_matrix_a, host_matrix_b,
				host_matrix_c, host_gold, log_obj);
	} else {
		generate_matrices_files<real_t, half_t>(host_matrix_a, host_matrix_b,
				host_matrix_c, log_obj);
	}

	//TODO
	//Set the correct version
	GEMMTYPE gemm_t = DMRGEMM;

	GEMM<half_t, real_t> mult_enviroment(host_matrix_a, host_matrix_b,
			host_matrix_c, log_obj.size_matrices, real_t(1.1f), real_t(1.2f));

	int tries = 0;
	cudaEventCreate(&start);
	cudaEventRecord(start, 0);
	// cudaStream_t st;
	// cudaStreamCreate(&st);	
	// assert(M_O > 512 && N_O > 512 && M_O % 64 == 0 && N_O % 16 == 0 && K_O % 16 == 0);
	for (int it = 0; it < log_obj.iterations; it++) {
		double start_computation = log_obj.mysecond();
		log_obj.start_iteration_app();
		mult_enviroment.gemm();
		log_obj.end_iteration_app();
		double end_computation = log_obj.mysecond();

		mult_enviroment.pull_array(host_matrix_d0, host_matrix_d1,
				host_matrix_d2);

		cudaEventCreate(&stop);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);

		cudaEventElapsedTime(&elapsedTime, start, stop);
		printf("Elapsed time : %f ms\n", elapsedTime);

		//TODO check this
		if (log_obj.triplicated && log_obj.generate) {
			tries++;
			int has_errors = is_output_ok(host_matrix_d0, host_matrix_d1,
					host_matrix_d2, host_gold);
			if (has_errors != 0)
				it--;

			if (tries > 5)
				throw std::runtime_error(
						"More than 5 tries on matrix generate\n");
			std::cout << "Iteration: " << it << std::endl;
		}

		if (!log_obj.generate) {
			//fault test 
//			if(it == 2){
//			host_matrix_d0[2]= (real_t) 5.00;
//			}
			//
			std::pair<int, int> errors;
			double start, end;
			if (log_obj.triplicated) {
				start = log_obj.mysecond();
				errors = compare_output_matrices(
						mult_enviroment.get_memory_errors(), host_gold,
						host_matrix_d0, host_matrix_d1, host_matrix_d2,
						log_obj);
				end = log_obj.mysecond();
			} else {
				start = log_obj.mysecond();
				//errors = compare_output_matrices(host_gold, host_matrix_d0, log_obj);

				// errors = check_output_errors_dmr(host_gold, host_matrix_d0, host_matrix_d1,log_obj);
				end = log_obj.mysecond();
			}
			std::cout << "Iteration: " << it << " memory errors "
					<< errors.first << " radiation errors " << errors.second
					<< ". Time spent on computation "
					<< end_computation - start_computation
					<< "s. Time spent on comparing " << end - start << "s."
					<< std::endl;

			//If errors != 0 reload matrices to gpu
			if (errors.first != 0 || errors.second != 0) {
				mult_enviroment.push_arrays(host_matrix_a.data(),
						host_matrix_b.data(), host_matrix_c.data());
			}

		}

	}
	// cudaStreamDestroy(st);
	cudaEventCreate(&stop);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("time : %f s\n", (elapsedTime / 1000));
	if (log_obj.generate) {
		if (log_obj.triplicated)
			write_gold_to_file<real_t>(log_obj.gold_inout_path, host_gold);
		else
			write_gold_to_file<real_t>(log_obj.gold_inout_path,
					host_matrix_d1);

	}

	// for(int z = 1;z < 15 ; z++) {
	// std::cout << "d0 = " << host_matrix_d0 [z] << " ||  d1 = " << host_matrix_d1 [z] << std::endl;
	// std::cout << "diff = " << (host_matrix_d0[0] - host_matrix_d1[0])<< std::endl;
	// }	

}

void usage(char **argv) {
	std::cout << "./" << argv[0]
			<< " --generate 0/1 --gold <gold file, DEFAULT=./gold.matrix > --size <matrix size, DEFAULT=8192> "
					"--iterations <how many iterations, optional> --input_a <input A, DEFAUL=./input_a.matrix> "
					"--input_b <input B, DEFAUL=./input_b.matrix> --input_c <input C, DEFAUL=./input_c.matrix>  --precision <float/double, DEFAULT=float>"
			<< std::endl;
}

int main(int argc, char** argv) {
	Log log_obj(argc, argv, DEFAULT_INPUT_SIZE);

	std::cout << "Generate: " << log_obj.generate << std::endl;
	std::cout << "A input path: " << log_obj.a_input_path << std::endl;
	std::cout << "B input path: " << log_obj.b_input_path << std::endl;
	std::cout << "C input path: " << log_obj.c_input_path << std::endl;
	std::cout << "Gold in/out path: " << log_obj.gold_inout_path << std::endl;
	std::cout << "Iterations: " << log_obj.iterations << std::endl;
	std::cout << "Matrix size: " << log_obj.size_matrices << std::endl;
	std::cout << "Precision: " << log_obj.precision << std::endl;
	std::cout << "Verbose: " << log_obj.verbose << std::endl;

// Alloc all memories on host
	if (log_obj.use_tensor_cores) {
		call_mxm<half, half>(log_obj);
	}

	// if (log_obj.precision == "float") {
	// 	call_mxm<float, float, half, float>(host_matrix_a, host_matrix_b, log_obj);
	// }
	// if (log_obj.precision == "double") {
	// 	call_mxm<double, double, half, double>(host_matrix_a, host_matrix_b, log_obj);
	// }
	// if (log_obj.precision == "DMR") {
	// 	call_mxm<double, double, half, float>(host_matrix_a, host_matrix_b, log_obj);
	// }

	std::cout << "Finished computation\n";
	return 0;
}
