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


__DEVICE_INLINE__ half exp__(half& lhs){
#if __CUDA_ARCH__ >= 600
	return hexp(lhs);
#else
	return expf(float(lhs));
#endif
}


__DEVICE_INLINE__ float exp__(float lhs){
	return expf(lhs);
}

__DEVICE_INLINE__ double exp__(double& lhs){
	return exp(lhs);
}

#endif /* DEVICE_FUNTIONS_H_ */
