/*
 * kernelCaller.h
 *
 *  Created on: Oct 2, 2019
 *      Author: carol
 */

#ifndef KERNELCALLER_H_
#define KERNELCALLER_H_

#include "nondmr_kernels.h"
#include "dmr_kernels.h"
#include "types.h"
#include <iomanip>      // std::setprecision
#include <sstream>      // std::stringstream
#include <iostream>

template<const uint32_t COUNT, const uint32_t THRESHOLD, typename half_t,
		typename real_t>
struct KernelCaller {

	//DMR
	VectorOfDeviceVector<FOUR_VECTOR<half_t>> d_fv_gpu_ht;
	std::vector<std::vector<FOUR_VECTOR<half_t>>>fv_cpu_ht;

	virtual ~KernelCaller() {
	}

	KernelCaller() {}

	virtual void kernel_call(dim3& blocks, dim3& threads, CudaStream& stream,
			par_str<real_t>& par_cpu, dim_str& dim_cpu, box_str* d_box_gpu,
			FOUR_VECTOR<real_t>* d_rv_gpu, real_t* d_qv_gpu,
			FOUR_VECTOR<real_t>* d_fv_gpu) = 0;

	virtual void cmp(std::vector<FOUR_VECTOR<real_t>>& fv_cpu_rt, std::vector<FOUR_VECTOR<real_t>>& fv_cpu_GOLD,
			Log& log, uint32_t streamIdx, bool verbose, uint32_t i, uint32_t& host_errors) {
		auto val_gold = fv_cpu_GOLD[i];
		auto val_output = fv_cpu_rt[i];
		if (val_gold != val_output) {
#pragma omp critical
			{
				host_errors++;
				std::stringstream error_detail;
				error_detail << std::scientific << std::setprecision(20);
				error_detail << "stream: " << streamIdx << ", p: [" << i << "], v_r: " << val_output.v << ", v_e: " << val_gold.v;
				error_detail << ", x_r: " << val_output.x << ", x_e: " << val_gold.x << ", y_r: " << val_output.y << ", y_e: " << val_gold.y;
				error_detail << ", z_r: " << val_output.z << ", z_e: " << val_gold.z;

				if (verbose && (host_errors < 10)) {
					std::cout << error_detail.str() << std::endl;
				}
				log.log_error_detail(error_detail.str());

//					snprintf(error_detail, 500, "stream: %d, p: [%d], v_r: %1.20e, v_e: %1.20e, x_r: %1.20e"
//							", x_e: %1.20e, y_r: %1.20e, y_e: %1.20e, z_r: %1.20e, z_e: %1.20e",
//							streamIdx, i, (double) valOutput.v,
//							(double) valGold.v, (double) valOutput.x,(double) valGold.x, (double) valOutput.y,
//							(double) valGold.y, (double) valOutput.z, (double) valGold.z);
			}
		}
	}

	// Returns true if no errors are found. False if otherwise.
	// Set votedOutput pointer to retrieve the voted matrix
	bool check_output_errors(bool verbose, uint32_t streamIdx,
			std::vector<FOUR_VECTOR<real_t>>& fv_cpu_rt,
			std::vector<FOUR_VECTOR<real_t>>& fv_cpu_GOLD,
			Log& log) {
		uint32_t host_errors = 0;

#pragma omp parallel for shared(host_errors)
		for (uint32_t i = 0; i < fv_cpu_GOLD.size(); i = i + 1) {
			this->cmp(fv_cpu_rt, fv_cpu_GOLD, log, streamIdx, verbose, i, host_errors);
		}

		log.update_errors(host_errors);

		if (host_errors != 0) {
			std::cout << "#";
		}
		return (host_errors == 0);
	}

};

template<const uint32_t COUNT, const uint32_t THRESHOLD, typename half_t,
		typename real_t>
struct DMRKernelCaller: public KernelCaller<COUNT, THRESHOLD, half_t, real_t> {

	DMRKernelCaller(uint32_t nstreams, uint32_t element_per_stream) {
		this->d_fv_gpu_ht.resize(nstreams);
		this->fv_cpu_ht.resize(nstreams);

		for (auto i = 0; i < nstreams; i++) {
			this->d_fv_gpu_ht[i].resize(element_per_stream);
			this->fv_cpu_ht[i].resize(element_per_stream);
		}
	}

	void cmp(std::vector<FOUR_VECTOR<real_t>>& fv_cpu_rt,
			std::vector<FOUR_VECTOR<real_t>>& fv_cpu_GOLD, Log& log,
			uint32_t streamIdx, bool verbose, uint32_t i, uint32_t& host_errors)
					override {
		auto val_gold = fv_cpu_GOLD[i];
		auto val_output = fv_cpu_rt[i];
		auto val_output_ht = this->fv_cpu_ht[i];

		if (val_gold != val_output || check_bit_error()) {
#pragma omp critical
			{
				host_errors++;
				std::stringstream error_detail;
				error_detail << std::scientific << std::setprecision(20);
				error_detail << "stream: " << streamIdx << ", p: [" << i
						<< "], v_r: " << val_output.v << ", v_e: "
						<< val_gold.v;
				error_detail << ", x_r: " << val_output.x << ", x_e: "
						<< val_gold.x << ", y_r: " << val_output.y << ", y_e: "
						<< val_gold.y;
				error_detail << ", z_r: " << val_output.z << ", z_e: "
						<< val_gold.z;

				if (verbose && (host_errors < 10)) {
					std::cout << error_detail.str() << std::endl;
				}
				log.log_error_detail(error_detail.str());
			}
		}
	}

	void kernel_call(dim3& blocks, dim3& threads, CudaStream& stream,
			par_str<real_t>& par_cpu, dim_str& dim_cpu, box_str* d_box_gpu,
			FOUR_VECTOR<real_t>* d_rv_gpu, real_t* d_qv_gpu,
			FOUR_VECTOR<real_t>* d_fv_gpu) {
		kernel_gpu_cuda<COUNT, THRESHOLD> <<<blocks, threads, 0, stream.stream>>>(
				par_cpu, dim_cpu, d_box_gpu, d_rv_gpu, d_qv_gpu, d_fv_gpu,
				this->d_fv_gpu_ht[0].data());
	}
};

template<typename real_t>
struct UnhardenedKernelCaller: public KernelCaller<0, 0, real_t, real_t> {

	void kernel_call(dim3& blocks, dim3& threads, CudaStream& stream,
			par_str<real_t>& par_cpu, dim_str& dim_cpu, box_str* d_box_gpu,
			FOUR_VECTOR<real_t>* d_rv_gpu, real_t* d_qv_gpu,
			FOUR_VECTOR<real_t>* d_fv_gpu) {
		kernel_gpu_cuda<<<blocks, threads, 0, stream.stream>>>(par_cpu, dim_cpu,
				d_box_gpu, d_rv_gpu, d_qv_gpu, d_fv_gpu);
	}
};

#endif /* KERNELCALLER_H_ */
