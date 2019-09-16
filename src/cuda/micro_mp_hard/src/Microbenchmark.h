/*
 * Microbenchmark.h
 *
 *  Created on: 15/09/2019
 *      Author: fernando
 */

#ifndef MICROBENCHMARK_H_
#define MICROBENCHMARK_H_

#include <tuple>

#include "device_vector.h"
#include "Parameters.h"
#include "Log.h"

template<const uint32 CHECK_BLOCK, typename half_t, typename real_t>
struct Microbenchmark {

	rad::DeviceVector<real_t> input_dev_1, input_dev_2, input_dev_3;
	std::vector<real_t> input_host_1, input_host_2, input_host_3;

	Log log_;

	const Parameters& parameters_;

	Microbenchmark(const Parameters& parameters) : parameters_(parameters) {

	}


	virtual ~Microbenchmark(){

	}

	double test() {
		auto kernel_time = rad::mysecond();


//		//================== Device computation
//		if (parameters.redundancy == NONE) {
//			switch (parameters.micro) {
//			case ADD:
//				MicroBenchmarkKernel_ADD<real_t> <<<parameters.grid_size,
//						parameters.block_size>>>(device_vector_real_t.data(),
//						type_.output_r, type_.input_a);
//				break;
//			case MUL:
//				MicroBenchmarkKernel_MUL<real_t> <<<parameters.grid_size,
//						parameters.block_size>>>(device_vector_real_t.data(),
//						type_.output_r, type_.input_a);
//				break;
//			case FMA:
//				MicroBenchmarkKernel_FMA<real_t> <<<parameters.grid_size,
//						parameters.block_size>>>(device_vector_real_t.data(),
//						type_.output_r, type_.input_a, type_.input_b);
//				break;
//			}
//
//		} else {
//			switch (parameters.micro) {
//			case ADD: {
//				MicroBenchmarkKernel_ADD<half_t, real_t> <<<
//						parameters.grid_size, parameters.block_size>>>(
//						device_vector_inc.data(), device_vector_real_t.data(),
//						type_.output_r, type_.input_a);
//				break;
//			}
//			case MUL: {
//				MicroBenchmarkKernel_MUL<half_t, real_t> <<<
//						parameters.grid_size, parameters.block_size>>>(
//						device_vector_inc.data(), device_vector_real_t.data(),
//						type_.output_r, type_.input_a);
//				break;
//			}
//			case FMA: {
//				MicroBenchmarkKernel_FMA<half_t, real_t> <<<
//						parameters.grid_size, parameters.block_size>>>(
//						device_vector_inc.data(), device_vector_real_t.data(),
//						type_.output_r, type_.input_a, type_.input_b);
//				break;
//			}
//			case ADDNOTBIASED: {
//				MicroBenchmarkKernel_ADDNOTBIASAED<half_t, real_t> <<<
//						parameters.grid_size, parameters.block_size>>>(
//						device_vector_inc.data(), device_vector_real_t.data(),
//						type_.output_r);
//				break;
//			}
//			case MULNOTBIASED: {
//				MicroBenchmarkKernel_MULNOTBIASAED<half_t, real_t> <<<
//						parameters.grid_size, parameters.block_size>>>(
//						device_vector_inc.data(), device_vector_real_t.data());
//				gold = 1.10517102313140469505;
//				break;
//			}
//			case FMANOTBIASED: {
//				MicroBenchmarkKernel_FMANOTBIASAED<half_t, real_t> <<<
//						parameters.grid_size, parameters.block_size>>>(
//						device_vector_inc.data(), device_vector_real_t.data());
//				gold = 2.50000000001979527653e-01;
//				break;
//			}
//			}
//		}

		rad::checkFrameworkErrors(cudaPeekAtLastError());
		rad::checkFrameworkErrors(cudaDeviceSynchronize());
		rad::checkFrameworkErrors(cudaPeekAtLastError());
		return rad::mysecond() - kernel_time;
	}

	// Returns the number of errors found
	// if no errors were found it returns 0
	std::tuple<double, uint64, uint64> check_output_errors() {
		uint64 host_errors = 0;
		uint64 relative_errors = 0;
		auto cmp_time = rad::mysecond();

//		double gold = double(OUTPUT_R);
//		double threshold = -3;
//	#pragma omp parallel for shared(host_errors)
//		for (int i = 0; i < R.size(); i++) {
//			double output = double(R[i]);
//			double output_inc = double(R_half_t[i]);
//			threshold = max(threshold, fabs(output - output_inc));
//			if (!cmp(gold, output, 0.000000001)
//					|| !cmp(output, output_inc, ZERO_FLOAT)) {
//	#pragma omp critical
//				{
//					std::stringstream error_detail;
//					error_detail.precision(20);
//					error_detail << "p: [" << i << "], r: " << std::scientific
//							<< output << ", e: " << gold << " smaller_precision: "
//							<< output_inc;
//
//					if (verbose && (host_errors < 10))
//						std::cout << error_detail.str() << std::endl;
//					this->log_.log_error_detail(error_detail.str());
//					host_errors++;
//				}
//			}
//		}
//
//		if (dmr_errors != 0) {
//			std::stringstream error_detail;
//			error_detail << "detected_dmr_errors: " << dmr_errors;
//			;
//			this->log_.log_error_detail(error_detail.str());
//		}
//
//		if (host_errors != 0) {
//			std::cout << "#";
//			this->log_.udpdate_errors();
//		}

		return std::make_tuple(cmp_time, host_errors, relative_errors);
	}
};

#endif /* MICROBENCHMARK_H_ */
