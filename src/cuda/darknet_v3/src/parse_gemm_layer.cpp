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
#include "parse_gemm_layer.h"

static int layer_count = 0;

template<typename type_t>
bool write_file(type_t *src, std::string& path, std::string& header, int m,
		int n) {
	std::ofstream of(path, std::ios::trunc);
	if (of.good()) {
		of << header << std::endl;
		of << std::scientific << std::setprecision(16);
		for (auto i = 0; i < m; i++) {
			for (auto j = 0; j < n; j++) {
				of << src[i * n + j];
				if (j < (n - 1))
					of << ",";
			}
			of << std::endl;
		}

		of.close();
		return true;
	}
	return false;
}

void parse_entry(int TA, int TB, int M, int N, int K, float ALPHA, float *A,
		int lda, float *B, int ldb, float BETA, float *C, int ldc) {

	std::cout << "layer " << layer_count++ << " cpu: " << TA << " " << TB << " "
			<< M << " " << N << " " << K << " " << ALPHA << " " << lda << " "
			<< ldb << " " << BETA << " " << ldc << std::endl;

	for (int i = 0; i < M; ++i) {
		for (int j = 0; j < N; ++j) {
			if (C[i * ldc + j] != 0.0f)
				std::cout << "Vall C " << C[i * ldc + j] << std::endl;
		}
	}

	auto path_a = "matrix_a_layer_" + std::to_string(layer_count) + ".csv";
	auto header_a = "TRANSPOSE," + std::to_string(TA) + ",M,"
			+ std::to_string(M) + ",N," + std::to_string(N) + ",K,"
			+ std::to_string(K);

	//MATRIX A is M and K
	if (TA) {
		write_file(A, path_a, header_a, K, M);
	} else {
		write_file(A, path_a, header_a, M, K);
	}

	auto path_b = "matrix_b_layer_" + std::to_string(layer_count) + ".csv";
	auto header_b = "TRANSPOSE," + std::to_string(TB) + ",M,"
			+ std::to_string(M) + ",N," + std::to_string(N) + ",K,"
			+ std::to_string(K);
	//MATRIX B is K and N
	if (TB) {
		write_file(B, path_b, header_b, N, K);
	} else {
		write_file(B, path_b, header_b, K, N);
	}

}

std::vector<std::string> split(const std::string& str, char del) {
	std::stringstream ss(str);
	std::vector<std::string> cont;
	std::string token;
	while (std::getline(ss, token, del)) {
		cont.push_back(token);
	}
	return cont;
}

bool str_contains(std::string& lhs, std::string rhs) {
	std::size_t found = lhs.find(rhs);
	return (found != std::string::npos);
}

std::vector<std::tuple<int, int, double>> read_injection_file(
		std::string& path) {
	std::vector<std::tuple<int, int, double>> fault_vector;
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
			fault_vector.push_back( { i, j, val });
		}
	} else {
		throw std::runtime_error("FILE PATH NOT FOUND: " + path);
	}
	return fault_vector;
}

void inject_fault(int TA, int TB, int M, int N, int K, float *C) {
	layer_count++;
	auto layer_char = std::getenv("FAULT_LAYER");
	auto fault_id = std::getenv("FAULT_ID");
	auto fault_dir = std::getenv("FAULT_DIR");

	if (layer_char && fault_id && fault_dir) {
		std::string layer_str = layer_char;
		std::string fault_id_str = fault_id;
		std::string fault_dir_str = fault_dir;
		if (layer_str.size() != 0 && fault_dir_str.size() != 0
				&& fault_id_str.size() != 0) {
			auto layer_i = std::stoi(layer_str);
			if (layer_i == layer_count) {
				/**
				 * If A is an m × n matrix and B is an n × p matrix,
				 * C is the m × p matrix
				 */
				auto m = TA ? K : M;
//		auto p = TB ? K : N;
				//std::cout << layer_count << std::endl;

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
