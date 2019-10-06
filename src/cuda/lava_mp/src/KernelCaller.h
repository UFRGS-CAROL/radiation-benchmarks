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
#include <iostream>		// std::cout
#include <algorithm> 	// std::max_element

template<const uint32_t COUNT, const uint32_t THRESHOLD, typename half_t,
		typename real_t>
struct KernelCaller {

	//DMR
	VectorOfDeviceVector<FOUR_VECTOR<half_t>> d_fv_gpu_ht;
	std::vector<std::vector<FOUR_VECTOR<half_t>>>fv_cpu_ht;

	virtual ~KernelCaller() {}

	virtual void set_half_t_vectors(uint32_t nstreams, uint32_t element_per_stream) {

	}
	virtual void sync_half_t() {

	}
	virtual void clear_half_t() {

	}

	virtual uint32_t get_max_threshold(std::vector<std::vector<FOUR_VECTOR<real_t>>>& fv_cpu_rt) {
		return 0;
	}

	KernelCaller() {}

	virtual void kernel_call(dim3& blocks, dim3& threads, CudaStream& stream,
			par_str<real_t>& par_cpu, dim_str& dim_cpu, box_str* d_box_gpu,
			FOUR_VECTOR<real_t>* d_rv_gpu, real_t* d_qv_gpu,
			FOUR_VECTOR<real_t>* d_fv_gpu, const uint32_t stream_idx) = 0;

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
				error_detail << "stream: " << streamIdx;
				error_detail << ", p: [" << i << "]";
				error_detail << ", v_r: " << val_output.v << ", v_e: " << val_gold.v;
				error_detail << ", x_r: " << val_output.x << ", x_e: " << val_gold.x;
				error_detail << ", y_r: " << val_output.y << ", y_e: " << val_gold.y;
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

		this->sync_half_t();

#pragma omp parallel for shared(host_errors)
		for (uint32_t i = 0; i < fv_cpu_GOLD.size(); i++) {
			this->cmp(fv_cpu_rt, fv_cpu_GOLD, log, streamIdx, verbose, i, host_errors);
		}

		auto dmr_errors = get_dmr_error();
		if (dmr_errors != 0) {
			std::string error_detail = "detected_dmr_errors: "
			+ std::to_string(dmr_errors);
			if (verbose)
			std::cout << error_detail << std::endl;
			log.log_info_detail(error_detail);
			log.update_infos(1);
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
struct DMRMixedKernelCaller: public KernelCaller<COUNT, THRESHOLD, half_t,
		real_t> {

	uint32_t get_max_threshold(std::vector<std::vector<FOUR_VECTOR<real_t>>>& fv_cpu_rt) {
		uint32_t max_threshold = 0;
		real_t max_threshold_float = -4444;
		auto max_before = max_threshold_float;
		auto& fht = this->fv_cpu_ht[0][0];
		auto& frt = fv_cpu_rt[0][0];

		for (uint32_t i = 0; i < fv_cpu_rt.size(); i++) {
			auto& fv_rt_i = fv_cpu_rt[i];
			auto& fv_ht_i = this->fv_cpu_ht[i];

			for(uint32_t j = 0; j < fv_rt_i.size(); j++) {
				auto& fv_rt_ij = fv_rt_i[j];
				auto& fv_ht_ij = fv_ht_i[j];

				auto diff_vector = this->get_4vector_diffs(fv_ht_ij, fv_rt_ij);
				diff_vector.push_back(max_threshold);
				max_threshold = *std::max_element(diff_vector.begin(), diff_vector.end());

				auto fl_vet = {
					std::fabs(fv_rt_ij.v - (real_t)fv_ht_ij.v),
					std::fabs(fv_rt_ij.x - (real_t) fv_ht_ij.x),
					std::fabs(fv_rt_ij.y - (real_t)fv_ht_ij.y),
					std::fabs(fv_rt_ij.z - (real_t)fv_ht_ij.z)
				};

				max_threshold_float = *std::max_element(fl_vet.begin(), fl_vet.end());
				if(max_before < max_threshold_float){
					max_before = max_threshold_float;
					fht = fv_ht_ij;
					frt = fv_rt_ij;
				}
			}
		}

		std::cout << "\n\nMAX FLOAT DIFF " << max_threshold_float << std::endl;
		std::cout << fht << std::endl;
		std::cout << frt << std::endl;

		return max_threshold;
	}

	void sync_half_t() override {
		for (uint32_t i = 0; i < this->d_fv_gpu_ht.size(); i++) {
			this->fv_cpu_ht[i] = this->d_fv_gpu_ht[i].to_vector();
		}
	}

	void clear_half_t() override {
		for (uint32_t i = 0; i < this->d_fv_gpu_ht.size(); i++) {
			std::fill(this->fv_cpu_ht[i].begin(), this->fv_cpu_ht[i].end(), FOUR_VECTOR<half_t>());
			this->d_fv_gpu_ht[i].clear();
		}

	}

	void set_half_t_vectors(uint32_t nstreams, uint32_t element_per_stream) {
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
		auto val_output_ht = this->fv_cpu_ht[streamIdx][i];

		if (val_gold != val_output
		|| check_bit_error(val_output_ht, val_output)) {
#pragma omp critical
			{
				host_errors++;
				std::stringstream error_detail;
				error_detail << std::scientific << std::setprecision(20);
				error_detail << "stream: " << streamIdx;
				error_detail << ", p: [" << i << "]";
				error_detail << ", v_r: " << val_output.v << ", v_e: "
				<< val_gold.v;
				error_detail << ", x_r: " << val_output.x << ", x_e: "
				<< val_gold.x;
				error_detail << ", y_r: " << val_output.y << ", y_e: "
				<< val_gold.y;
				error_detail << ", z_r: " << val_output.z << ", z_e: "
				<< val_gold.z;

				error_detail << ", s_v_r: " << val_output_ht.v;
				error_detail << ", s_x_r: " << val_output_ht.x;
				error_detail << ", s_y_r: " << val_output_ht.y;
				error_detail << ", s_z_r: " << val_output_ht.z;

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
	FOUR_VECTOR<real_t>* d_fv_gpu, const uint32_t stream_idx) {
		kernel_gpu_cuda<COUNT, THRESHOLD> <<<blocks, threads, 0, stream.stream>>>(
		par_cpu, dim_cpu, d_box_gpu, d_rv_gpu, d_qv_gpu, d_fv_gpu,
		this->d_fv_gpu_ht[stream_idx].data());
	}

	inline std::vector<uint32_t>
	get_4vector_diffs(FOUR_VECTOR<float>& lhs, FOUR_VECTOR<double>& rhs) {
		float rhs_float_v = float(rhs.v);
		float rhs_float_x = float(rhs.x);
		float rhs_float_y = float(rhs.y);
		float rhs_float_z = float(rhs.z);

		//To INT
		uint32_t rhs_data_v = reinterpret_cast<uint32_t&>(rhs_float_v);
		uint32_t rhs_data_x = reinterpret_cast<uint32_t&>(rhs_float_x);
		uint32_t rhs_data_y = reinterpret_cast<uint32_t&>(rhs_float_y);
		uint32_t rhs_data_z = reinterpret_cast<uint32_t&>(rhs_float_z);

		uint32_t lhs_data_v = reinterpret_cast<uint32_t&>(lhs.v);
		uint32_t lhs_data_x = reinterpret_cast<uint32_t&>(lhs.x);
		uint32_t lhs_data_y = reinterpret_cast<uint32_t&>(lhs.y);
		uint32_t lhs_data_z = reinterpret_cast<uint32_t&>(lhs.z);

		uint32_t sub_res_v = SUB_ABS(lhs_data_v, rhs_data_v);
		uint32_t sub_res_x = SUB_ABS(lhs_data_x, rhs_data_x);
		uint32_t sub_res_y = SUB_ABS(lhs_data_y, rhs_data_y);
		uint32_t sub_res_z = SUB_ABS(lhs_data_z, rhs_data_z);

		return {sub_res_v, sub_res_x, sub_res_y, sub_res_z};
	}

	inline std::vector<uint32_t>
	get_4vector_diffs(FOUR_VECTOR<real_t>& lhs, FOUR_VECTOR<real_t>& rhs) {
		return {0, 0, 0, 0};
	}

	bool check_bit_error(FOUR_VECTOR<float>& lhs, FOUR_VECTOR<double>& rhs) {
		auto diff_vec = this->get_4vector_diffs(lhs, rhs);

		for(auto it : diff_vec) {
			if(it > THRESHOLD) {
				return true;
			}
		}
		return false;

//		if ((sub_res_v > THRESHOLD) || (sub_res_x > THRESHOLD)
//		|| (sub_res_y > THRESHOLD) || (sub_res_z > THRESHOLD)) {
//			return true;
//		}
	}

	bool check_bit_error(FOUR_VECTOR<real_t>& lhs, FOUR_VECTOR<real_t>& rhs) {
		if ((std::fabs(lhs.v - rhs.v) > THRESHOLD) ||	//V
		(std::fabs(lhs.x - rhs.x) > THRESHOLD) ||//X
		(std::fabs(lhs.y - rhs.y) > THRESHOLD) ||//Y
		(std::fabs(lhs.z - rhs.z) > THRESHOLD)) {	//Z
			return true;
		}
		return false;
	}
};

template<typename real_t>
struct UnhardenedKernelCaller: public KernelCaller<0, 0, real_t, real_t> {

	void kernel_call(dim3& blocks, dim3& threads, CudaStream& stream,
			par_str<real_t>& par_cpu, dim_str& dim_cpu, box_str* d_box_gpu,
			FOUR_VECTOR<real_t>* d_rv_gpu, real_t* d_qv_gpu,
			FOUR_VECTOR<real_t>* d_fv_gpu, const uint32_t stream_idx) {
		kernel_gpu_cuda<<<blocks, threads, 0, stream.stream>>>(par_cpu, dim_cpu,
				d_box_gpu, d_rv_gpu, d_qv_gpu, d_fv_gpu);
	}
};

template<const uint32_t COUNT, typename real_t>
struct DMRKernelCaller: public DMRMixedKernelCaller<COUNT, 0, real_t, real_t> {
};

#endif /* KERNELCALLER_H_ */
