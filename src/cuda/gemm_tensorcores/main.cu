#include <iostream>
#include <string>
#include <vector>
#include <random>
#include <fstream>      // std::ifstream

#include "half.hpp"
#include "Log.h"
#include "GEMMWMMA.h"

#define DEFAULT_INPUT_SIZE 8192
#define GENERATOR_MAXABSVALUE 2.0
#define GENERATOR_MINABSVALUE 0

typedef half_float::half host_half;

typedef std::vector<host_half> half_vector;
typedef std::vector<float> float_vector;
typedef std::vector<double> double_vector;

template<class real_t> void generate_matrices_files(std::string a_path,
		std::string b_path, std::string c_path, half_vector& a_host_vector,
		half_vector& b_host_vector, std::vector<real_t>& c_host_vector,
		int matrix_size) {

	std::ofstream f_a(a_path, std::ios::out | std::ios::binary);
	std::ofstream f_b(b_path, std::ios::out | std::ios::binary);
	std::ofstream f_c(c_path, std::ios::out | std::ios::binary);
//	std::cout << "entrou generate" << std::endl;

	if (f_a.is_open() && f_b.is_open() && f_c.is_open()) {
		std::random_device rd; //Will be used to obtain a seed for the random number engine
		std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
		std::uniform_real_distribution<double> dis(-GENERATOR_MAXABSVALUE,
		GENERATOR_MAXABSVALUE);
		std::cout << "entrou if generate1" << std::endl;

		for (int i = 0; i < matrix_size; i++) {
			for (int j = 0; j < matrix_size; j++) {
				a_host_vector[i * matrix_size + j] = host_half(dis(gen));
				b_host_vector[i * matrix_size + j] = host_half(dis(gen));
				c_host_vector[i * matrix_size + j] = real_t(dis(gen));
			}
		}
		std::cout << "entrou generate1" << std::endl;
		host_half zero(0.0);
		host_half nan_ = host_half(half_float::nanh("0"));
		host_half inf_ = host_half(host_half(0x7C00));
		std::cout << "entrou generate2" << std::endl;

		int numZeros = std::count(a_host_vector.begin(), a_host_vector.end(),
				zero);
		int numNans = std::count(a_host_vector.begin(), a_host_vector.end(),
				nan_);

		int numInfs = std::count(a_host_vector.begin(), a_host_vector.end(),
				inf_);
		std::cout << "Number of zeros/NaNs/INFs on matrix A: " << numZeros
				<< numNans << numInfs << std::endl;

		std::cout << "entrou generate3" << std::endl;
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
				a_host_vector.size() * sizeof(host_half));
		f_b.write(reinterpret_cast<char*>(b_host_vector.data()),
				b_host_vector.size() * sizeof(host_half));
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
	} else {
		throw std::runtime_error("Could not write gold file\n");
	}
}

template<class real_t> void retrieve_matrices(std::string a_path,
		std::string b_path, std::string c_path, std::string gold_path,
		half_vector& a_host_vector, half_vector& b_host_vector,
		std::vector<real_t>& c_host_vector,
		std::vector<real_t>& gold_host_vector, Log *log) {

	double start = log->mysecond();
	std::ifstream f_a(a_path, std::ios::in | std::ios::binary);
	std::ifstream f_b(b_path, std::ios::in | std::ios::binary);
	std::ifstream f_c(c_path, std::ios::in | std::ios::binary);
	std::ifstream f_gold(gold_path, std::ifstream::in | std::ifstream::binary);

	if (f_a.is_open() && f_b.is_open() && f_c.is_open() && f_gold) {

		f_a.seekg(0, std::ios::beg);
		f_a.read(reinterpret_cast<char*>(a_host_vector.data()),
				sizeof(host_half) * a_host_vector.size());

		f_b.seekg(0, std::ios::beg);
		f_b.read(reinterpret_cast<char*>(b_host_vector.data()),
				sizeof(host_half) * b_host_vector.size());

		f_c.seekg(0, std::ios::beg);
		f_c.read(reinterpret_cast<char*>(c_host_vector.data()),
				sizeof(host_half) * c_host_vector.size());

		f_gold.seekg(0, std::ios::beg);
		f_gold.read(reinterpret_cast<char*>(gold_host_vector.data()),
				sizeof(host_half) * gold_host_vector.size());

		f_a.close();
		f_b.close();
		f_c.close();
		f_gold.close();
	} else {
		if (log != nullptr)
			log->log_error("Could not retrieve the matrices");
		throw std::runtime_error("Could not retrieve the matrices\n");
	}

	std::cout << "Done with reading matrices " << log->mysecond() - start
			<< "s\n";
}

template<class real_t>
bool compare_output_matrices(long long host_is_memory_bad, bool generate,
		std::vector<real_t>& gold, std::vector<real_t>& c0,
		std::vector<real_t>& c1, std::vector<real_t>& c2,
		std::vector<real_t>& voted_output, Log *log, int matrix_size,
		bool check, bool verbose) {

	int host_errors = 0;
	int memory_errors = 0;

	if (host_is_memory_bad != 0) {
		std::string info_detail = "b: is_memory_bad: "
				+ std::to_string(host_is_memory_bad);
		if (verbose)
			std::cout << info_detail << std::endl;

		if (!generate)
			log->log_error(info_detail);
		memory_errors++;
	}

#pragma omp parallel for shared(host_errors)
	for (size_t i = 0; i < gold.size(); i++) {
		register bool checkFlag = true;
		register real_t valGold = gold[i];
		register real_t valOutput0 = c0[i];
		register real_t valOutput1 = c1[i];
		register real_t valOutput2 = c2[i];
		register real_t valOutput = valOutput0;
		if ((valOutput0 != valOutput1) || (valOutput0 != valOutput2)) {
#pragma omp critical
			{
				char info_detail[200];
				snprintf(info_detail, 150,
						"m: [%d, %d], r0: %1.20e, r1: %1.20e, r2: %1.20e",
						int(floor(i / matrix_size)), int(i % matrix_size),
						(double) valOutput0, (double) valOutput1,
						(double) valOutput2);
				if (verbose && (memory_errors < 10))
					std::cout << info_detail << std::endl;

				if (!generate)
					log->log_info(info_detail);
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
#pragma omp critical
					{
						char info_detail[200];
						snprintf(info_detail, 150,
								"t: [%d, %d], r0: %1.20e, r1: %1.20e, r2: %1.20e, e: %1.20e",
								int(floor(i / matrix_size)),
								int(i % matrix_size), (double) valOutput0,
								(double) valOutput1, (double) valOutput2,
								(double) valGold);
						if (verbose && (memory_errors < 10))
							std::cout << info_detail << std::endl;

						if (!generate)
							log->log_info(std::string(info_detail));

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
		if (voted_output.size() == gold.size())
			voted_output[i] = valOutput;
		// if ((fabs((real_t)(valOutput-valGold)/valGold) > 1e-10)||(fabs((real_t)(valOutput-valGold)/valGold) > 1e-10)) {
		if (check) {
			if (valGold != valOutput) {
				if (checkFlag) {
#pragma omp critical
					{
						char error_detail[200];
						snprintf(error_detail, 150,
								"p: [%lu, %lu], r: %1.20e, e: %1.20e",
								int(floor(i / matrix_size)),
								int(i % matrix_size), (double) valOutput,
								(double) valGold);

						if (verbose && (host_errors < 10))
							std::cout << error_detail << std::endl;
						if (!generate)
							log->log_error(error_detail);
						host_errors++;
					}
				}
			}
		}
	}

	// printf("numErrors:%d", host_errors);

	if (!generate) {
		log->update_info_count(memory_errors);
		log->update_error_count(host_errors);
	}

	if (memory_errors != 0)
		std::cout << "M";
	if (host_errors != 0)
		std::cout << "#";

	return (host_errors == 0) && (host_is_memory_bad == 0);
}

void usage(char **argv) {
	std::cout << "./" << argv[0]
			<< " --generate 0/1 --size <matrix size> --iterations <how many iterations> --input_a <input A> "
					"--input_b <input B> --input_c <input C> --gold <gold file> --precision <float/double>"
			<< std::endl;
}

int main(int argc, char** argv) {

//	if (argc < 7){
//		usage(argv);
//		throw "Wrong parameters\n";
//	}

	Log *log_obj = new Log();

	bool generate = log_obj->find_int_arg(argc, argv, "--generate", 0);

	size_t size_matrices = log_obj->find_int_arg(argc, argv, "--size",
	DEFAULT_INPUT_SIZE);

	int iterations = log_obj->find_int_arg(argc, argv, "--iterations", 1);

	std::string a_input_path = log_obj->find_char_arg(argc, argv, "--input_a",
			"./input_a.matrix");
	std::string b_input_path = log_obj->find_char_arg(argc, argv, "--input_b",
			"./input_b.matrix");
	std::string c_input_path = log_obj->find_char_arg(argc, argv, "--input_c",
			"./input_c.matrix");
	std::string gold_inout_path = log_obj->find_char_arg(argc, argv, "--gold",
			"./gold.matrix");

	std::string precision = log_obj->find_char_arg(argc, argv, "--precision",
			"float");

	bool verbose = log_obj->find_int_arg(argc, argv, "--verbose", 0);

	std::cout << "Generate: " << generate << std::endl;
	std::cout << "A input path: " << a_input_path << std::endl;
	std::cout << "B input path: " << b_input_path << std::endl;
	std::cout << "C input path: " << c_input_path << std::endl;
	std::cout << "Gold in/out path: " << gold_inout_path << std::endl;
	std::cout << "Iterations: " << iterations << std::endl;
	std::cout << "Matrix size: " << size_matrices << std::endl;
	std::cout << "Precision: " << precision << std::endl;
	std::cout << "Verbose: " << verbose << std::endl;

	// Alloc all memories on host
	half_vector host_matrix_a(size_matrices * size_matrices);
	half_vector host_matrix_b(size_matrices * size_matrices);

	if (precision == "float") {

		// C matrix
		float_vector host_matrix_c(size_matrices * size_matrices);
		float_vector host_gold(size_matrices * size_matrices);
		// D Matrix
		float_vector host_matrix_d1(size_matrices * size_matrices);
		float_vector host_matrix_d2(size_matrices * size_matrices);
		float_vector host_matrix_d3(size_matrices * size_matrices);
//		std::cout << "passou declaracao host" << std::endl;

		if (!generate) {
			retrieve_matrices(a_input_path, b_input_path, c_input_path,
					gold_inout_path, host_matrix_a, host_matrix_b,
					host_matrix_c, host_gold, log_obj);

		} else {
//			std::cout << "entrou else" << std::endl;
			generate_matrices_files(a_input_path, b_input_path, c_input_path,
					host_matrix_a, host_matrix_b, host_matrix_c, size_matrices);
		}
		//GOLD Matrix
		float_vector host_matrix_gold(size_matrices * size_matrices);

		GEMMWMMA<host_half, half, float> mult_enviroment(host_matrix_a.data(),
				host_matrix_b.data(), host_matrix_c.data(), size_matrices,
				size_matrices, size_matrices);

		int tries = 0;
		for (int it = 0; it < iterations; it++) {

			mult_enviroment.mul();
			mult_enviroment.pull_array(host_matrix_d1.data(),
					host_matrix_d2.data(), host_matrix_d3.data());

			//TODO: FIX FOR GENERATE AND TEST
			bool has_error = compare_output_matrices(mult_enviroment.get_memory_errors(),
					generate, host_gold, host_matrix_d1, host_matrix_d2,
					host_matrix_d3, host_matrix_d3, log_obj, size_matrices,
					false, verbose);

			if (!has_error) {
				tries++;
				it--;
			}

			if (generate && tries > 5)
				throw std::runtime_error(
						"More than 5 tries on matrix generate\n");

		}

		if (generate) {
			write_gold_to_file(gold_inout_path, host_gold);
		}
	} else if (precision == "double") {

	}

	if (log_obj)
		delete log_obj;
	return 0;
}
