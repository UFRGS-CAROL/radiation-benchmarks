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

#if (__CUDACC_VER_MAJOR__ <= 7)
#include "../../mxm/half.hpp"

using half = half_float::half;
#endif

#define CHAR_CAST(x) (reinterpret_cast<char*>(x))
#define GENERATOR_MAXABSVALUE_GEMM 10
#define GENERATOR_MINABSVALUE_GEMM -GENERATOR_MAXABSVALUE_GEMM

#define GENERATOR_MAXABSVALUE_TENSOR 10
#define GENERATOR_MINABSVALUE_TENSOR -GENERATOR_MAXABSVALUE_TENSOR

static std::ostream& operator<<(std::ostream& os, const dim3 d) {
    os << d.x << " " << d.y << " " << d.z;
    return os;
}

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

static bool exists(std::string& path) {
    std::ifstream input(path);
    auto file_exists = input.good();
    input.close();
    return file_exists;
}

template<typename half_t, typename real_t>
void write_abc_files(
        std::string& a_file_path, std::vector<half_t>& a_vector,
        std::string& b_file_path, std::vector<half_t>& b_vector,
        std::string& c_file_path, std::vector<real_t>& c_vector
        ) {
    if (write_to_file(a_file_path, a_vector) == false) {
        throw_line(a_file_path + " could not be written\n");
    }

    if (write_to_file(b_file_path, b_vector) == false) {
        throw_line(b_file_path + " could not be written\n");
    }

    if (write_to_file(c_file_path, c_vector) == false) {
        throw_line(c_file_path + " could not be written\n");
    }
}

template<typename half_t, typename real_t>
void read_abc_files(
        std::string& a_file_path, std::vector<half_t>& a_vector,
        std::string& b_file_path, std::vector<half_t>& b_vector,
        std::string& c_file_path, std::vector<real_t>& c_vector
        ) {
    if (read_from_file(a_file_path, a_vector) == false) {
        throw_line(a_file_path + " could not be read\n");
    }
    if (read_from_file(b_file_path, b_vector) == false) {
        throw_line(b_file_path + " could not be read\n");
    }
    if (read_from_file(c_file_path, c_vector) == false) {
        throw_line(c_file_path + " could not be read\n");
    }
}


template<typename real_t>
void write_gold(std::string& d_file_path, std::vector<real_t>& d_vector) {
    if (write_to_file(d_file_path, d_vector) == false) {
        throw_line(d_file_path + " could not be written\n");
    }
}

template<typename real_t>
void read_gold(std::string& d_file_path, std::vector<real_t>& d_vector) {
    if (read_from_file(d_file_path, d_vector) == false) {
        throw_line(d_file_path + " could not be read\n");
    }
}

template<typename half_t, typename real_t, const bool TENSOR_INPUT = false>
void get_input_matrices(size_t matrix_size, std::vector<half_t>& a_vector,
        std::vector<half_t>& b_vector, std::vector<real_t>& c_vector,
        std::string& a_file_path, std::string& b_file_path,
        std::string& c_file_path, const bool read_abc = false) {

    std::random_device rd; //Will be used to obtain a seed for the random number engine
    //Standard mersenne_twister_engine seeded with rd()
    std::vector<std::mt19937> generators;
    const int n_threads = omp_get_max_threads(); 
   for (int i = 0; i < n_threads; ++i) {
        generators.emplace_back(std::mt19937(rd()));
    }
    
    double min_val = GENERATOR_MINABSVALUE_GEMM;
    double max_val = GENERATOR_MAXABSVALUE_GEMM;
    if (TENSOR_INPUT) {
        min_val = GENERATOR_MINABSVALUE_TENSOR;
        max_val = GENERATOR_MAXABSVALUE_TENSOR;
    }

    a_vector.resize(matrix_size * matrix_size);
    b_vector.resize(matrix_size * matrix_size);
    c_vector.resize(matrix_size * matrix_size);

    if (read_abc == false) {
#pragma omp parallel for
        for (size_t i = 0; i < matrix_size * matrix_size; i++) {
           auto& gen = generators[omp_get_thread_num()];
           std::uniform_real_distribution<double> dis(min_val, max_val);

            a_vector[i] = half_t(dis(gen));
            b_vector[i] = half_t(dis(gen));
            c_vector[i] = real_t(dis(gen)); 
        }

        write_abc_files(
                a_file_path, a_vector,
                b_file_path, b_vector,
                c_file_path, c_vector);

    } else {
        read_abc_files(
                a_file_path, a_vector,
                b_file_path, b_vector,
                c_file_path, c_vector
                );
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

template<typename real_t>
bool equals(real_t& lhs, real_t& rhs, const uint32_t threshold = 0) {
    return lhs == rhs;
}

static bool equals(half& lhs, half& rhs, const uint32_t threshold = 0) {
    return float(lhs) == float(rhs);
}

static std::ostream& operator<<(std::ostream& os, half &rhs) {
    os << float(rhs);
    return os;
}

static float fabs(half h) {
    return fabs(float(h));
}

static bool equals(float& lhs, double& rhs, const uint32_t threshold) {
    assert(sizeof(float) == sizeof(uint32_t));

    float rhs_float = float(rhs);

    uint32_t lhs_data;
    uint32_t rhs_data;
    memcpy(&lhs_data, &lhs, sizeof(uint32_t));
    memcpy(&rhs_data, &rhs_float, sizeof(uint32_t));
    auto diff = SUB_ABS(lhs_data, rhs_data);

    return (diff <= threshold);
}

template<class t>
void debug_mxm(std::vector<t>& a, std::vector<t>& b, std::vector<t> c,
        std::vector<t>& c_gpu, float alpha, float beta, int n) {
    for (auto i = 0; i < n; ++i) {
        for (auto j = 0; j < n; ++j) {
            t sum = 0;
            for (auto k = 0; k < n; ++k) {
                sum += a[i * n + k] * b[k * n + j];
            }
            c[i * n + j] = alpha * sum + c[i * n + j] * beta;
        }

    }

    for (unsigned i = 0; i < c_gpu.size(); i++) {
        if (fabs(c[i] - c_gpu[i]) > 1.0e-4) {
            std::cout << "i: " << i << " CPU: " << c[i] << " GPU: " << c_gpu[i]
                    << std::endl;
        }
    }
}

template<class half_t, class real_t>
std::pair<int, int> check_output_errors_dmr(std::vector<real_t>& gold,
        std::vector<real_t>& real_vector, std::vector<half_t>& half_vector,
        Parameters& parameter, const uint32_t threshold, const bool dmr) {
    uint32_t host_errors = 0;
    uint32_t memory_errors = 0;

#ifdef OMP
#pragma omp parallel for shared(host_errors, memory_errors)
#endif
    for (size_t i = 0; i < gold.size(); i++) {
        auto gold_value = gold[i];
        real_t full_precision = real_vector[i];
        half_t half_precision = (dmr == true) ? half_vector[i] : real_vector[i];

        //Check if DMR is OK
        bool dmr_equals = equals(half_precision, full_precision, threshold);

        //Is output corrupted
        bool is_output_diff = !equals(gold_value, full_precision);

        if (!equals(gold_value, full_precision)) {
#ifdef OMP
#pragma omp critical
            {
#endif

            std::stringstream error_detail("");
            error_detail << std::setprecision(20) << std::scientific;
            error_detail << "p: [" << int(floor(i / parameter.size_matrices)) << ", "
                    << i % parameter.size_matrices << "], r: ";
            error_detail << full_precision;
            error_detail << ", e: " << gold_value << " smaller_precision: "
                    << half_precision;

            if (parameter.verbose && (host_errors < 10)) {
                std::cout << error_detail.str() << std::endl;

                std::cout << is_output_diff << " " << !dmr_equals << std::endl;
            }

            parameter.log_error(error_detail.str());
            host_errors++;
            if (is_output_diff && dmr_equals && dmr) {
                memory_errors++;
            }

#ifdef OMP
        }
#endif
        }
    }

    auto dmr_err = dmr_errors();
    if (dmr_err != 0) {
        std::string error_detail;
        error_detail = "detected_dmr_errors: " + std::to_string(dmr_err);
        parameter.log_info(error_detail);
    }

    if (memory_errors != 0) {
        parameter.log_info("dmr1_equals_dmr2_detected");
    }

    if (host_errors != 0){
        std::cout << "#";
        std::string error_detail;
        error_detail = "errors: " + std::to_string(host_errors);
        //log as info to not count as error detail
        parameter.log_info(error_detail);
    }
    parameter.update_error_count(host_errors);

    return {dmr_err, host_errors};
}

#endif /* COMMON_TEMPLATE_FUNCTIONS_H_ */
