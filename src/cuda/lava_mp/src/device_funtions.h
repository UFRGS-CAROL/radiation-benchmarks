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

#define __DEVICE_INLINE__ __device__ __forceinline__

__device__ unsigned long long errors;

/**
 * EXP
 */
__DEVICE_INLINE__
half exp__(half& lhs) {
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
double exp__(double& lhs) {
	return exp(lhs);
}

/**
 * ABS
 */
__DEVICE_INLINE__
double abs__(double a) {
	return fabs(a);
}

__DEVICE_INLINE__
float abs__(float a) {
	return fabsf(a);
}

__DEVICE_INLINE__
half abs__(half a) {
	return fabsf(a);
}

template<const uint32_t THRESHOLD, typename real_t> __DEVICE_INLINE__
void check_bit_error(const FOUR_VECTOR<real_t>& lhs,
		const FOUR_VECTOR<real_t>& rhs) {
//	if ((abs__(lhs.v - rhs.v) > ZERO_DOUBLE) ||	//V
//			(abs__(lhs.x - rhs.x) > ZERO_DOUBLE) ||	//X
//			(abs__(lhs.y - rhs.y) > ZERO_DOUBLE) ||	//Y
//			(abs__(lhs.z - rhs.z) > ZERO_DOUBLE)) {	//Z
	if (lhs != rhs) {
		atomicAdd(&errors, 1);
	}
}

template<const uint32_t THRESHOLD> __DEVICE_INLINE__
void check_bit_error(const FOUR_VECTOR<float>& lhs,
		const FOUR_VECTOR<double>& rhs) {
	float rhs_float_v = float(rhs.v);
	float rhs_float_x = float(rhs.x);
	float rhs_float_y = float(rhs.y);
	float rhs_float_z = float(rhs.z);
	//To int
	uint32_t rhs_data_v = *(uint32_t*)(&rhs_float_v);
	uint32_t rhs_data_x = *(uint32_t*)(&rhs_float_x);
	uint32_t rhs_data_y = *(uint32_t*)(&rhs_float_y);
	uint32_t rhs_data_z = *(uint32_t*)(&rhs_float_z);

	uint32_t lhs_data_v = *(uint32_t*)(&lhs.v);
	uint32_t lhs_data_x = *(uint32_t*)(&lhs.x);
	uint32_t lhs_data_y = *(uint32_t*)(&lhs.y);
	uint32_t lhs_data_z = *(uint32_t*)(&lhs.z);

	uint32_t sub_res_v = SUB_ABS(lhs_data_v, rhs_data_v);
	uint32_t sub_res_x = SUB_ABS(lhs_data_x, rhs_data_x);
	uint32_t sub_res_y = SUB_ABS(lhs_data_y, rhs_data_y);
	uint32_t sub_res_z = SUB_ABS(lhs_data_z, rhs_data_z);

	if ((sub_res_v > THRESHOLD) || (sub_res_x > THRESHOLD)
			|| (sub_res_y > THRESHOLD) || (sub_res_z > THRESHOLD)) {

		atomicAdd(&errors, 1);
//		auto max_all = sub_res_v;
//		max_all = max(sub_res_x, max_all);
//		max_all = max(sub_res_y, max_all);
//		max_all = max(sub_res_z, max_all);
//		printf("v %e x %e y %e z %e\n", lhs.v, lhs.x, lhs.y, lhs.z);
//		printf("v %e x %e y %e z %e\n", rhs.v, rhs.x, rhs.y, rhs.z);
//
//		printf("MAX %d\n", max_all);
	}
//	if ((abs__(lhs.v - rhs.v) > ZERO_HALF) ||	//V
//			(abs__(lhs.x - rhs.x) > ZERO_HALF) ||	//X
//			(abs__(lhs.y - rhs.y) > ZERO_HALF) ||	//Y
//			(abs__(lhs.z - rhs.z) > ZERO_HALF)) {	//Z
//		atomicAdd(&errors, 1);
//	}
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
