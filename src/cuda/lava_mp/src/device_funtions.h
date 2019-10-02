/*
 * device_funtions.h
 *
 *  Created on: 29/09/2019
 *      Author: fernando
 */

#ifndef DEVICE_FUNTIONS_H_
#define DEVICE_FUNTIONS_H_

#include "cuda_fp16.h"

#define __DEVICE_INLINE__ __device__ __forceinline__

#define SUB_ABS(lhs, rhs) ((lhs > rhs) ? (lhs - rhs) : (rhs - lhs))

__device__ unsigned long long errors;

__DEVICE_INLINE__ half exp__(half& lhs) {
#if __CUDA_ARCH__ >= 600
	return hexp(lhs);
#else
	return expf(float(lhs));
#endif
}

__DEVICE_INLINE__ float exp__(float lhs) {
	return expf(lhs);
}

__DEVICE_INLINE__ double exp__(double& lhs) {
	return exp(lhs);
}

template<const uint32_t THRESHOLD>
__DEVICE_INLINE__ void check_bit_error(const FOUR_VECTOR<float>& lhs,
		const FOUR_VECTOR<double>& rhs) {
	//To int
	uint32_t rhs_data_v = __float_as_uint(float(rhs.v));
	uint32_t rhs_data_x = __float_as_uint(float(rhs.x));
	uint32_t rhs_data_y = __float_as_uint(float(rhs.y));
	uint32_t rhs_data_z = __float_as_uint(float(rhs.z));

	uint32_t lhs_data_v = __float_as_uint(lhs.v);
	uint32_t lhs_data_x = __float_as_uint(lhs.x);
	uint32_t lhs_data_y = __float_as_uint(lhs.y);
	uint32_t lhs_data_z = __float_as_uint(lhs.z);

	uint32_t sub_res_v = SUB_ABS(lhs_data_v, rhs_data_v);
	uint32_t sub_res_x = SUB_ABS(lhs_data_x, rhs_data_x);
	uint32_t sub_res_y = SUB_ABS(lhs_data_y, rhs_data_y);
	uint32_t sub_res_z = SUB_ABS(lhs_data_z, rhs_data_z);

	if ((sub_res_v > THRESHOLD) || (sub_res_x > THRESHOLD)
			|| (sub_res_y > THRESHOLD) || (sub_res_z > THRESHOLD)) {
		atomicAdd(&errors, 1);
	}
}

__DEVICE_INLINE__ void cast_four_vector(FOUR_VECTOR<float>& lhs, FOUR_VECTOR<double>& rhs) {
	lhs.v = rhs.v;
	lhs.x = rhs.x;
	lhs.y = rhs.y;
	lhs.z = rhs.z;
}

#endif /* DEVICE_FUNTIONS_H_ */
