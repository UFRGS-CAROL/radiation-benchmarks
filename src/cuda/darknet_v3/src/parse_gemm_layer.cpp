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
#include "parse_gemm_layer.h"
#include "log_processing.h"

std::string layers_files_base_path;
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

void set_layer_processing_parameters(const std::string &base_path,
		LayerOperationType current_operation) {
	layers_files_base_path = base_path;
	operation_type = current_operation;
	std::cout << base_path << " " << current_operation << std::endl;
	std::cout << layers_files_base_path << " " << operation_type << std::endl;
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

void inject_fault(int TA, int M, int K, std::vector<float>& C,
		int layer_count_output) {
	auto layer_char = std::getenv("FAULT_LAYER");
	auto fault_id = std::getenv("FAULT_ID");
	auto fault_dir = std::getenv("FAULT_DIR");

	if (layer_char && fault_id && fault_dir) {
		std::string layer_str = layer_char;
		std::string fault_id_str = fault_id;
		std::string fault_dir_str = fault_dir;
		if (!layer_str.empty() && !fault_dir_str.empty()
				&& !fault_id_str.empty()) {
			auto layer_i = std::stoi(layer_str);
			if (layer_i == layer_count_output) {
				/**
				 * If A is an m × n matrix and B is an n × p matrix,
				 * C is the m × p matrix
				 */
				auto m = TA ? K : M;
//		auto p = TB ? K : N;
				//std::cout << layer_count_output << std::endl;

				auto file_path = fault_dir_str + "/fault_id_" + fault_id_str
						+ "_layer_" + layer_str + ".txt";
				auto faults = read_injection_file(file_path);
				for (auto fault : faults) {
					int i, j;
					double val;
					std::tie(i, j, val) = fault;
//			std::cout << i << " " << j << " " << val << std::endl;
					C[i * m + j] = float(val);
				}
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

template<typename T> inline
std::vector<T> pull_array(T* ptr, int size){
#ifdef GPU
	std::vector<T> ret_vec(size);
	cuda_pull_array(ptr, ret_vec.data(), size);
#else
	std::vector<T> ret_vec(ptr, ptr + size);
#endif
	return ret_vec;
}

template<typename T> inline
void push_array(T* lhs, std::vector<T>& rhs){
#ifdef GPU
	cuda_push_array(lhs, rhs.data(), rhs.size());
#else
	std::copy(lhs, lhs + rhs.size(), rhs.data());
#endif
}

#ifdef __cplusplus
extern "C" {
#endif
void parse_output_conv_layer_gpu(int TA, int TB, int M, int N, int K,
		float *C) {
	static int layer_count_output = 0;
	static std::unordered_map<int, std::vector<float>> layers_gold_hash;
	if (reset_counters_var) {
		layer_count_output = 0;
		reset_counters_var = false;
	}
	layer_count_output++;
	auto size_c = M * N;

	std::vector<float> c_cpu = pull_array(C, size_c);

	// Base path for C matrix is the same for all
	// base path for gold and injection
	auto base_gold_file = layers_files_base_path;
	base_gold_file += "/layer_gold_" + std::to_string(layer_count_output)
			+ ".yololayer";
	auto base_inj_file = layers_files_base_path;
	base_inj_file += "/layer_inj_" + std::to_string(layer_count_output)
			+ ".yololayer";

	switch (operation_type) {
	case GENERATE_GOLDEN_LAYERS:
		write_file(c_cpu, base_gold_file);
		break;
	case COMPARING_CURRENT_TO_GOLDEN: {
		// Key is not present
		if (layers_gold_hash.find(layer_count_output)
				== layers_gold_hash.end()) {
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
					"layer_" + std::to_string(layer_count_output)
							+ ".yololayer");
			write_file(c_cpu, layer_out_path);

		}
		break;
	}
	case INJECT_FAULT_IN_OUTPUT:
		inject_fault(TA, M, K, c_cpu, layer_count_output);
		push_array(C, c_cpu);
		break;

	}
}

void parse_input_conv_layer_gpu(int TA, int TB, int M, int N, int K,
		float ALPHA, float *A, int lda, float *B, int ldb, float BETA, float *C,
		int ldc) {
}

#ifdef __cplusplus
}
#endif
