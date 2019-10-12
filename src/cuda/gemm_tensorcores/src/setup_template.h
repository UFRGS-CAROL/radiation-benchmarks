/*
 * setup_template.h
 *
 *  Created on: 11/10/2019
 *      Author: fernando
 */

#ifndef SETUP_TEMPLATE_H_
#define SETUP_TEMPLATE_H_

//#include <memory>
#include <sstream>      // std::stringstream
#include <iostream>
#include <iomanip>
#include <limits>

#ifdef OMP
#include <omp.h>
#endif

#include "Log.h"
#include "GEMM.h"
#include "GEMMWMMA.h"
#include "common.h"
#include "HostVectors.h"

//#include "device_functions.h"

unsigned long long dmr_errors() {
	unsigned long long ret = 0;
	rad::checkFrameworkErrors(
			cudaMemcpyFromSymbol(&ret, errors, sizeof(unsigned long long), 0));

	unsigned long long tmp = 0;
	rad::checkFrameworkErrors(
			cudaMemcpyToSymbol(errors, &tmp, sizeof(unsigned long long), 0));

	return ret;
}

std::ostream& operator<<(std::ostream& os, const half& rhs) {
	float temp = float(rhs);
	os << temp;
	return os;
}

template<const uint32_t THRESHOLD, class real_t>
bool equals(real_t& lhs, real_t& rhs) {
	double lhs_double = double(lhs);
	double rhs_double = double(rhs);
	return (SUB_ABS(lhs_double, rhs_double) <= ZERO_DOUBLE);
}

template<const uint32_t THRESHOLD>
bool equals(float& lhs, double& rhs) {
	float rhs_float = float(rhs);

	const uint32_t lhs_data = 0, rhs_data = 0;
	memcpy((void*) &lhs_data, &lhs, sizeof(float));
	memcpy((void*) &rhs_data, &rhs_float, sizeof(float));

	return (SUB_ABS(lhs_data, rhs_data) <= THRESHOLD);
}

template<const uint32_t THRESHOLD, class half_t, class real_t>
std::pair<int, int> check_output_errors_dmr(std::vector<real_t>& gold,
		std::vector<half_t>& d0, std::vector<real_t>& d1, Log& log) {
	int host_errors = 0;
//	double threshold = -3222;
#ifdef OMP
#pragma omp parallel for shared(host_errors)
#endif
	for (size_t i = 0; i < gold.size(); i++) {
		auto gold_value = gold[i];
		auto half_precision = d0[i];
		auto full_precision = d1[i];

//		threshold = std::fmax(threshold, fabs(half_precision - full_precision));

		if (gold_value != full_precision
				|| !equals<THRESHOLD>(half_precision, full_precision)) {
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
//	std::cout << "THRESHOLD 0 " << threshold << std::endl;

	auto dmr_err = dmr_errors();
	if (dmr_err != 0) {
		std::string error_detail;
		error_detail = "detected_dmr_errors: " + std::to_string(dmr_err);
		log.log_error(error_detail);
	}

	log.update_error_count(host_errors);
	if (host_errors != 0)
		std::cout << "#";

	std::pair<int, int> res(dmr_err, host_errors);
	return res;
}

template<const uint32_t COUNT, const uint32_t THRESHOLD, class half_t, class real_t, class mixed_t>
void setup_execute(GEMMBase<COUNT, THRESHOLD, half_t, real_t, mixed_t>& mult_enviroment,
		Log& log_obj, HostVectors<half_t, real_t, mixed_t>& hd) {
	cudaEvent_t start, stop;
	float elapsedTime;

	cudaEventCreate(&start);
	cudaEventRecord(start, 0);

	for (int it = 0; it < log_obj.iterations; it++) {
		double start_computation = log_obj.mysecond();
		log_obj.start_iteration_app();
		mult_enviroment.gemm();
		log_obj.end_iteration_app();
		double end_computation = log_obj.mysecond();

		mult_enviroment.pull_array(hd.host_matrix_d, hd.host_matrix_smaller);

		cudaEventCreate(&stop);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);

		cudaEventElapsedTime(&elapsedTime, start, stop);
		printf("Elapsed time : %f ms\n", elapsedTime);

		if (!log_obj.generate) {
			std::pair<int, int> errors;
			double start, end;

			start = log_obj.mysecond();
			errors = check_output_errors_dmr<THRESHOLD>(hd.host_gold,
					hd.host_matrix_smaller, hd.host_matrix_d, log_obj);
			end = log_obj.mysecond();

			std::cout << "Iteration: " << it << " dmr errors " << errors.first
					<< " radiation errors " << errors.second
					<< ". Time spent on computation "
					<< end_computation - start_computation
					<< "s. Time spent on comparing " << end - start << "s."
					<< std::endl;

			//If errors != 0 reload matrices to gpu
			if (errors.first != 0 || errors.second != 0) {
				mult_enviroment.push_arrays(hd.host_matrix_a, hd.host_matrix_b,
						hd.host_matrix_c);
			}

		}

	}

	cudaEventCreate(&stop);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("time : %f s\n", (elapsedTime / 1000));
	if (log_obj.generate) {
		hd.write_gold_to_file(log_obj.gold_inout_path);
	}
}

#endif /* SETUP_TEMPLATE_H_ */
