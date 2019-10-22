/*
 * device_funtions.h
 *
 *  Created on: 29/09/2019
 *      Author: fernando
 */

#ifndef DEVICE_FUNTIONS_H_
#define DEVICE_FUNTIONS_H_

#include "cuda_fp16.h"
#include "common.h"


__device__ unsigned long long errors;
__device__ uint32_t thresholds[12167] = {0};


/**
 * EXP
 */
__DEVICE_INLINE__
half exp__(half lhs) {
#if __CUDA_ARCH__ >= 600
	return hexp(lhs);
#else
	return expf(float(lhs));
#endif
}

__DEVICE_INLINE__
float exp__(float lhs) {
	return expf(lhs);
}

__DEVICE_INLINE__
double exp__(double lhs) {
	return exp(lhs);
}

template<const uint32_t THRESHOLD, typename real_t> __DEVICE_INLINE__
void check_bit_error(FOUR_VECTOR<real_t>& lhs, FOUR_VECTOR<real_t>& rhs) {
	if (lhs != rhs) {
		atomicAdd(&errors, 1);
	}
	lhs = rhs;
}

template<const uint32_t THRESHOLD> __DEVICE_INLINE__
void check_bit_error(FOUR_VECTOR<float>& lhs, FOUR_VECTOR<double>& rhs) {
	float rhs_float_v = float(rhs.v);
	float rhs_float_x = float(rhs.x);
	float rhs_float_y = float(rhs.y);
	float rhs_float_z = float(rhs.z);

	//To int
	uint32_t rhs_data_v = *((uint32_t*) (&rhs_float_v));
	uint32_t rhs_data_x = *((uint32_t*) (&rhs_float_x));
	uint32_t rhs_data_y = *((uint32_t*) (&rhs_float_y));
	uint32_t rhs_data_z = *((uint32_t*) (&rhs_float_z));

	uint32_t lhs_data_v = *((uint32_t*) (&lhs.v));
	uint32_t lhs_data_x = *((uint32_t*) (&lhs.x));
	uint32_t lhs_data_y = *((uint32_t*) (&lhs.y));
	uint32_t lhs_data_z = *((uint32_t*) (&lhs.z));

	uint32_t sub_res_v = SUB_ABS(lhs_data_v, rhs_data_v);
	uint32_t sub_res_x = SUB_ABS(lhs_data_x, rhs_data_x);
	uint32_t sub_res_y = SUB_ABS(lhs_data_y, rhs_data_y);
	uint32_t sub_res_z = SUB_ABS(lhs_data_z, rhs_data_z);

	if ((sub_res_v > THRESHOLD) || (sub_res_x > THRESHOLD)
			|| (sub_res_y > THRESHOLD) || (sub_res_z > THRESHOLD)) {
		atomicAdd(&errors, 1);
//		printf("LHS %e %e %e %e\nRHS %e %e %e %e\n%u %u %u %u %u\n",
//				lhs.v, lhs.x, lhs.y, lhs.z, rhs_float_v, rhs_float_x, rhs_float_y, rhs_float_z,
//				sub_res_v, sub_res_x, sub_res_y, sub_res_z, THRESHOLD);
		uint32_t m = 0;
		m = max(m, sub_res_v);
		m = max(m, sub_res_x);
		m = max(m, sub_res_y);
		m = max(m, sub_res_z);
		atomicMax(thresholds + blockIdx.x, m);
	}

	lhs = rhs;
}

template<const uint32_t THRESHOLD, typename real_t> __DEVICE_INLINE__
void check_bit_error(real_t& lhs, real_t& rhs) {
	if (lhs != rhs) {
		atomicAdd(&errors, 1);
	}
	lhs = rhs;
}

template<const uint32_t THRESHOLD> __DEVICE_INLINE__
void check_bit_error(float& lhs, double& rhs) {
	float rhs_float = float(rhs);
	uint32_t rhs_data = *((float*) (&rhs_float));
	uint32_t lhs_data = *((float*) (&lhs));
	uint32_t sub_res = SUB_ABS(lhs_data, rhs_data);

	if (sub_res > THRESHOLD) {
		atomicAdd(&errors, 1);
//		atomicExch
//		printf("%e %e\n", lhs, rhs);
		atomicMax(thresholds + blockIdx.x, sub_res);
	}
	lhs = rhs_float;
}

inline uint64_t get_dmr_error() {
	uint64_t dmr_errors;
	rad::checkFrameworkErrors(
			cudaMemcpyFromSymbol(&dmr_errors, errors, sizeof(uint64_t), 0,
					cudaMemcpyDeviceToHost));

	const uint64_t zero_value = 0;
	rad::checkFrameworkErrors(
			cudaMemcpyToSymbol(errors, &zero_value, sizeof(uint64_t), 0,
					cudaMemcpyHostToDevice));
	return dmr_errors;
}

#endif /* DEVICE_FUNTIONS_H_ */
