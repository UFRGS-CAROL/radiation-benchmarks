/*
 * common_template_functions.h
 *
 *  Created on: 27/10/2019
 *      Author: fernando
 */

#ifndef COMMON_TEMPLATE_FUNCTIONS_H_
#define COMMON_TEMPLATE_FUNCTIONS_H_

#include <random>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cassert>

#include "device_functions.h"

#ifdef OMP
#include <omp.h>
#endif

#define CHAR_CAST(x) (reinterpret_cast<char*>(x))
#define GENERATOR_MAXABSVALUE 1000
#define GENERATOR_MINABSVALUE -GENERATOR_MAXABSVALUE

template<typename T>
bool read_from_file(std::string& path, std::vector<T>& array) {
	std::ifstream input(path, std::ios::binary);
	if (input.good()) {
		input.read(CHAR_CAST(array.data()), array.size() * sizeof(T));
		input.close();
		return true;
	}
	return false;
}

template<typename T>
bool write_to_file(std::string& path, std::vector<T>& array) {
	std::ofstream output(path, std::ios::binary);
	if (output.good()) {
		output.write(CHAR_CAST(array.data()), array.size() * sizeof(T));
		output.close();
		return true;
	}
	return false;
}

bool exists(std::string& path) {
	std::ifstream input(path);
	auto exists = input.good();
	input.close();
	return exists;
}

template<typename half_t, typename real_t>
void write_gold(std::vector<half_t>& a_vector, std::vector<half_t>& b_vector,
		std::vector<real_t>& c_vector, std::vector<real_t>& d_vector,
		std::string& a_file_path, std::string& b_file_path,
		std::string& c_file_path, std::string& d_file_path) {
	auto result = write_to_file(a_file_path, a_vector);
	result = result && write_to_file(b_file_path, b_vector);
	result = result && write_to_file(c_file_path, c_vector);
	result = result && write_to_file(d_file_path, d_vector);
	if (result == false) {
		throw_line("The gold files could not be written\n");
	}
}

template<typename half_t, typename real_t>
void read_gold(std::vector<half_t>& a_vector, std::vector<half_t>& b_vector,
		std::vector<real_t>& c_vector, std::vector<real_t>& d_vector,
		std::string& a_file_path, std::string& b_file_path,
		std::string& c_file_path, std::string& d_file_path) {
	auto result = read_from_file(a_file_path, a_vector);
	result = result && read_from_file(b_file_path, b_vector);
	result = result && read_from_file(c_file_path, c_vector);
	result = result && read_from_file(d_file_path, d_vector);
	if (result == false) {
		throw_line("Some of the files could not be read\n");
	}
}

template<typename half_t, typename real_t>
void generate_input_matrices(size_t matrix_size, std::vector<half_t>& a_vector,
		std::vector<half_t>& b_vector, std::vector<real_t>& c_vector) {

	std::random_device rd; //Will be used to obtain a seed for the random number engine
	std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
	std::uniform_real_distribution<double> dis(GENERATOR_MINABSVALUE,
	GENERATOR_MAXABSVALUE);

//	std::cout << "MIN " << GENERATOR_MINABSVALUE << " MAX " << GENERATOR_MAXABSVALUE << std::endl;

	a_vector.resize(matrix_size * matrix_size);
	b_vector.resize(matrix_size * matrix_size);
	c_vector.resize(matrix_size * matrix_size);

#pragma omp parallel for
	for (size_t i = 0; i < matrix_size * matrix_size; i++) {
		a_vector[i] = half_t(dis(gen));
		b_vector[i] = half_t(dis(gen));
		c_vector[i] = real_t(dis(gen));
	}
}

static unsigned long long dmr_errors() {
	unsigned long long ret = 0;
	rad::checkFrameworkErrors(
			cudaMemcpyFromSymbol(&ret, errors, sizeof(unsigned long long), 0,
					cudaMemcpyDeviceToHost));

	unsigned long long tmp = 0;
	rad::checkFrameworkErrors(
			cudaMemcpyToSymbol(errors, &tmp, sizeof(unsigned long long), 0,
					cudaMemcpyHostToDevice));

	return ret;
}

#if __CUDA_ARCH__ > 600
static std::ostream& operator<<(std::ostream& os, const half& rhs) {
	float temp = float(rhs);
	os << temp;
	return os;
}
#endif

template<typename real_t>
bool equals(real_t& lhs, real_t& rhs, const uint32_t threshold = 0) {
	return (std::fabs(lhs - rhs) <= ZERO_DOUBLE);
}

bool equals(float& lhs, double& rhs, const uint32_t threshold) {
	assert(sizeof(float) == sizeof(uint32_t));

	float rhs_float = float(rhs);

	uint32_t lhs_data;
	uint32_t rhs_data;
	memcpy(&lhs_data, &lhs, sizeof(uint32_t));
	memcpy(&rhs_data, &rhs_float, sizeof(uint32_t));

	return (SUB_ABS(lhs_data, rhs_data) <= threshold);
}

bool equals(double& rhs, float& lhs, const uint32_t threshold) {
	assert(sizeof(float) == sizeof(uint32_t));

	float rhs_float = float(rhs);

	uint32_t lhs_data;
	uint32_t rhs_data;
	memcpy(&lhs_data, &lhs, sizeof(uint32_t));
	memcpy(&rhs_data, &rhs_float, sizeof(uint32_t));

	return (SUB_ABS(lhs_data, rhs_data) <= threshold);
}

bool equals(float& lhs, double& rhs) {
	float relative(lhs / float(rhs));
	return (relative >= MIN_PERCENTAGE && relative <= MAX_PERCENTAGE);
}

template<class half_t, class real_t>
std::pair<int, int> check_output_errors_dmr(std::vector<real_t>& gold,
		std::vector<real_t>& real_vector, std::vector<half_t>& half_vector,
		Log& log, const uint32_t threshold, const bool dmr) {
	uint32_t host_errors = 0;

#ifdef OMP
#pragma omp parallel for shared(host_errors)
#endif
	for (size_t i = 0; i < gold.size(); i++) {
		auto gold_value = gold[i];
		auto full_precision = real_vector[i];
		half_t half_precision;
		bool is_output_diff;
		bool dmr_not_equals = false;
		if (dmr) {
			half_precision = half_vector[i];
			is_output_diff = (gold_value != full_precision
					&& !equals(half_precision, full_precision, threshold));
			dmr_not_equals = !equals(half_precision, full_precision, threshold);
		} else {
			half_precision = full_precision;
			is_output_diff = gold_value != full_precision;
		}

		if (is_output_diff || dmr_not_equals) {
#ifdef OMP
#pragma omp critical
			{
#endif

			std::stringstream error_detail("");
			error_detail << std::setprecision(20) << std::scientific;
			error_detail << "p: [" << int(floor(i / log.size_matrices)) << ", "
					<< i % log.size_matrices << "], r: " << full_precision
					<< ", e: " << gold_value << " smaller_precision: "
					<< half_precision;

			if (log.verbose && (host_errors < 10))
				std::cout << error_detail.str() << std::endl;

			log.log_error(error_detail.str());
			host_errors++;
#ifdef OMP
		}
#endif
		}
	}

	auto dmr_err = dmr_errors();
	if (dmr_err != 0) {
		std::string error_detail;
		error_detail = "detected_dmr_errors: " + std::to_string(dmr_err);
		log.log_error(error_detail);
	}

	log.update_error_count(host_errors);
	if (host_errors != 0)
		std::cout << "#";

	return {dmr_err, host_errors};
}

#endif /* COMMON_TEMPLATE_FUNCTIONS_H_ */
