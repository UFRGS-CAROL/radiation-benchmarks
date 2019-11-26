/*
 * device_functions.h
 *
 *  Created on: Jul 27, 2019
 *      Author: fernando
 */

#ifndef DEVICE_FUNCTIONS_H_
#define DEVICE_FUNCTIONS_H_

#include "common.h"

#define __DEVICE_INLINE__  __forceinline__  __device__

__device__ unsigned long long errors = 0;

template<typename real_t> __DEVICE_INLINE__
void check_relative_error(real_t lhs, real_t rhs) {
	if (lhs != rhs) {
		atomicAdd(&errors, 1);
	}
}

__DEVICE_INLINE__
void check_relative_error(float lhs, double rhs) {
	float relative = __fdividef(lhs, float(rhs));
	if (relative < MIN_PERCENTAGE || relative > MAX_PERCENTAGE) {
		atomicAdd(&errors, 1);
	}
}

__DEVICE_INLINE__
void check_relative_error(float lhs, double rhs, uint32_t threshold) {
	float rhs_as_float = float(rhs);
	uint32_t lhs_data = *((uint32_t*) &lhs);
	uint32_t rhs_data = *((uint32_t*) &rhs_as_float);

	uint32_t diff = SUB_ABS(lhs_data, rhs_data);

	if (diff > threshold) {
		atomicAdd(&errors, 1);
	}
}

#endif /* DEVICE_FUNCTIONS_H_ */
