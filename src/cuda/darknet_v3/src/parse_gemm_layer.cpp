/*
 * parse_gemm_layer.cpp
 *
 *  Created on: Apr 11, 2020
 *      Author: fernando
 */

#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <cmath>
#include <limits>
#include <sstream>
#include <unordered_map>
#include <random>

#include "parse_gemm_layer.h"
#include "log_processing.h"

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif

#define DEBUG 1

LayerOperationType operation_type = GENERATE_GOLDEN_LAYERS;
auto reset_counters_var = false;

#define MAX_FLOAT_THRESHOLD 1.0e-5f

/**
 * Represent a vector of faults as tuple of int int and double
 */
using VectorOfFaults = std::vector<std::tuple<int, int, double>>;

extern "C" void cuda_pull_array(float *x_gpu, float *x, size_t n);
extern "C" void cuda_push_array(float *x_gpu, float *x, size_t n);

void reset_counters() {
	reset_counters_var = true;
}

void set_layer_processing_parameters(LayerOperationType current_operation) {
	operation_type = current_operation;
//	std::cout << base_path << " " << current_operation << std::endl;
//	std::cout << layers_files_base_path << " " << operation_type << std::endl;
}

template<typename type_t>
void write_file(const std::vector<type_t> &src, const std::string &path) {
	std::ofstream of(path, std::ios::binary | std::ios::trunc);
	if (of.good()) {
		of.write(reinterpret_cast<const char *>(src.data()),
				sizeof(type_t) * src.size());
		of.close();
	} else {
		throw std::runtime_error(
				"Could not open file: " + path + " At line "
						+ std::to_string(__LINE__) + " file " + __FILE__);
	}
}

template<typename type_t>
void read_file(std::vector<type_t> &src, const std::string &path, size_t n) {
	std::ifstream ifs(path, std::ios::binary | std::ios::in);
	if (ifs.good()) {
		src.resize(n);
		ifs.read(reinterpret_cast<char *>(src.data()),
				sizeof(type_t) * src.size());
		ifs.close();
	} else {
		throw std::runtime_error(
				"Could not open file: " + path + " At line "
						+ std::to_string(__LINE__) + " file " + __FILE__);
	}
}

std::vector<std::string> split(const std::string &str, char del) {
	std::stringstream ss(str);
	std::vector<std::string> cont;
	std::string token;
	while (std::getline(ss, token, del)) {
		cont.push_back(token);
	}
	return cont;
}

bool str_contains(std::string &lhs, const std::string &rhs) {
	std::size_t found = lhs.find(rhs);
	return (found != std::string::npos);
}

VectorOfFaults read_injection_file(std::string &path) {
	VectorOfFaults fault_vector;
	std::ifstream ifs(path);
	if (ifs.good()) {
		//all of this to treat nan and inf
		std::string str;
		while (std::getline(ifs, str)) {
			auto splited = split(str, ' ');
			int i = std::stoi(splited[0]);
			int j = std::stoi(splited[1]);
			auto vs = splited[2];
			double val;
			if (str_contains(vs, "nan")) {
				val = std::nan("");
			} else if (str_contains(vs, "inf")) {
				val = std::numeric_limits<double>::infinity();
			} else {
				val = std::stod(vs);
			}
//			std::cout << "KKKKKK " << i << " " << j << " " << val << std::endl;
			fault_vector.emplace_back(i, j, val);
		}
	} else {
		throw std::runtime_error("FILE PATH NOT FOUND: " + path);
	}
	return fault_vector;
}

template<typename str_t>
std::string get_enviroment_var(str_t& src) {
	auto ptr = std::getenv(src);
	std::string ret = "";
	if (ptr) {
		ret = std::string(ptr);
	}
	return ret;
}

void inject_fault(int M, int N, int layer_count_output, std::vector<float>& C) {
	std::string layer_str = get_enviroment_var("FAULT_LAYER");
	std::string fault_id_str = get_enviroment_var("FAULT_ID");
	std::string fault_dir_str = get_enviroment_var("FAULT_DIR");
	if (!layer_str.empty() && !fault_dir_str.empty() && !fault_id_str.empty()) {
		auto layer_i = std::stoi(layer_str);
		if (layer_i == layer_count_output) {
			//std::cout << layer_count_output << std::endl;

			auto file_path = fault_dir_str + "/fault_id_" + fault_id_str
					+ "_layer_" + layer_str + ".txt";
			auto faults = read_injection_file(file_path);
			for (auto fault : faults) {
				int i, j;
				double val;
				std::tie(i, j, val) = fault;
//			std::cout << i << " " << j << " " << val << std::endl;
				C[i * N + j] = float(val);
			}
		}
	}

}

std::string replace(std::string &s, const std::string &to_replace,
		const std::string &replace_with) {
	std::size_t pos = s.find(to_replace);
	if (pos == std::string::npos)
		return s;
	return s.replace(pos, to_replace.length(), replace_with);
}

void simulate_scheduler_fault(int M, int N, int layer_count_output,
		std::vector<float>& C) {
	std::string fault_parameter_file_path = get_enviroment_var(
			"FAULT_PARAMETER_FILE");
	if (!fault_parameter_file_path.empty()) {
		//Open the fault injection archive
		std::ifstream parameter_file(fault_parameter_file_path);
		/**
		 * For random selection
		 */
		//Will be used to obtain a seed for the random number engine
		std::random_device rd;
		//Standard mersenne_twister_engine seeded with rd()
		std::mt19937 gen(rd());

		if (parameter_file.good()) {

			float min_relative, max_relative;
			std::string geometry_format;
			int layer_i;

			//Read the parameters from file
			parameter_file >> min_relative;
			parameter_file >> max_relative;
			parameter_file >> geometry_format;
			parameter_file >> layer_i;
			parameter_file.close();

			std::uniform_real_distribution<float> float_generator(min_relative,
					max_relative);
			std::uniform_int_distribution<int> bool_generator(0, 1);

			if (layer_i == layer_count_output) {
				if (DEBUG == 1) {
					std::cout << "DEBUG MIN RELATIVE " << min_relative
							<< std::endl;
					std::cout << "DEBUG MAX RELATIVE " << max_relative
							<< std::endl;
					std::cout << "DEBUG GEOMETRY FMT " << geometry_format
							<< std::endl;
					std::cout << "DEBUG LAYER I " << layer_i << std::endl;
					std::cout << "DEBUG CURRENT LAYER " << layer_count_output
							<< std::endl;
				}

				if (geometry_format == "RANDOM"
						|| geometry_format == "SQUARE") {
					//size selection
					std::uniform_int_distribution<int> int_m_generator(0,
							M - BLOCK_SIZE);
					std::uniform_int_distribution<int> int_p_generator(0,
							N - BLOCK_SIZE);
					auto start_i = int_m_generator(gen);
					auto start_j = int_p_generator(gen);
					auto end_i = start_i + BLOCK_SIZE;
					auto end_j = start_j + BLOCK_SIZE;

					if (geometry_format == "RANDOM") {

						for (auto i = start_i; i < end_i; i++) {
							for (auto j = start_j; j < end_j; j++) {
								auto is_necessary_to_inject = bool(
										bool_generator(gen));
								if (is_necessary_to_inject) {
									C[i * N + j] *= float_generator(gen);
								}
							}
						}
					} else {
						for (auto i = start_i; i < end_i; i++) {
							for (auto j = start_j; j < end_j; j++) {
								C[i * N + j] *= float_generator(gen);
							}
						}
					}

				} else if (geometry_format == "LINE") {
					auto col_or_line = bool(bool_generator(gen));

					if (col_or_line) {
						//select a line
						std::uniform_int_distribution<int> int_m_generator(0,
								M - 1);
						auto i = int_m_generator(gen);

						for (auto j = 0; j < N; j++) {
							C[i * N + j] *= float_generator(gen);
						}
					} else {
						//select a line
						std::uniform_int_distribution<int> int_n_generator(0,
								N - 1);
						auto j = int_n_generator(gen);
						for (auto i = 0; i < M; i++) {
							C[i * N + j] *= float_generator(gen);
						}
					}

				}
			}

		} else {
			throw std::runtime_error(
					"COULD NOT OPEN FILE: " + fault_parameter_file_path);
		}
	}
}

void compare_gemm_layers(int layer_count_output, int size_c,
		std::string& base_gold_file, std::vector<float>& c_cpu) {
	static std::unordered_map<int, std::vector<float>> layers_gold_hash;
	// Key is not present
	if (layers_gold_hash.find(layer_count_output) == layers_gold_hash.end()) {
		std::vector<float> c_gold_i_vector;
		read_file(c_gold_i_vector, base_gold_file, size_c);
		layers_gold_hash[layer_count_output] = std::move(c_gold_i_vector);
	}

	//compare and save the corrupted ones
	auto &gold = layers_gold_hash[layer_count_output];
	auto comparator = [](float &lhs, float &rhs) -> bool {
		return std::fabs(lhs - rhs) > MAX_FLOAT_THRESHOLD;
	};
	auto cmp_result = std::equal(gold.begin(), gold.end(), c_cpu.begin(),
			comparator);
	if (!cmp_result) {
		auto log_file_name = Log::get_log_path();
		auto layer_out_path = replace(log_file_name, ".log",
				"layer_" + std::to_string(layer_count_output) + ".yololayer");
		write_file(c_cpu, layer_out_path);

	}
}

#ifdef __cplusplus
extern "C" {
#endif
void parse_output_conv_layer_gpu(int TA, int TB, int M, int N, int K,
		float *C_gpu) {
	if (FLEX_GRIP_ANALYSIS != 1) {
		return;
	}

	static int layer_count_output = 0;
	if (reset_counters_var) {
		layer_count_output = 0;
		reset_counters_var = false;
	}
	layer_count_output++;
	/**
	 * If A is an m × n matrix and B is an n × p matrix,
	 * C is the m × p matrix
	 */
	auto size_c = M * N;

	std::vector<float> C_cpu(size_c);
	cuda_pull_array(C_gpu, C_cpu.data(), size_c);

	// Base path for C matrix is the same for all
	// base path for gold and injection
	std::string layers_files_base_path = "/var/radiation-benchmarks";

	auto base_gold_file = layers_files_base_path;
	base_gold_file += "/layer_gold_" + std::to_string(layer_count_output)
			+ ".yololayer";
	auto base_inj_file = layers_files_base_path;
	base_inj_file += "/layer_inj_" + std::to_string(layer_count_output)
			+ ".yololayer";

	switch (operation_type) {
	case GENERATE_GOLDEN_LAYERS:
		write_file(C_cpu, base_gold_file);
		break;
	case COMPARING_CURRENT_TO_GOLDEN: {
		compare_gemm_layers(layer_count_output, size_c, base_gold_file, C_cpu);
		break;
	}
	case INJECT_FAULT_IN_OUTPUT: {
		inject_fault(M, N, layer_count_output, C_cpu);
		break;
	}
	case SIMULATE_SCHEDULER_FAULT: {
		simulate_scheduler_fault(M, N, layer_count_output, C_cpu);
		break;
	}
	}

	cuda_push_array(C_gpu, C_cpu.data(), size_c);
}

void parse_input_conv_layer_gpu(int TA, int TB, int M, int N, int K,
		float ALPHA, float *A, int lda, float *B, int ldb, float BETA, float *C,
		int ldc) {
	if (FLEX_GRIP_ANALYSIS != 1) {
		return;
	}
}

#ifdef __cplusplus
}
#endif
