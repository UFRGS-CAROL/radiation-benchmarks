/*
 * cuda_utils.h
 *
 *  Created on: 27/03/2019
 *      Author: fernando
 */

#ifndef CUDA_UTILS_H_
#define CUDA_UTILS_H_

#include <sys/time.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda.h>



void __checkFrameworkErrors(cudaError_t error, int line, const char* file);
cudaDeviceProp GetDevice();
#define checkFrameworkErrors(error) __checkFrameworkErrors(error, __LINE__, __FILE__)



//__device__ unsigned long long errors = 0;
//
//unsigned long long copy_errors() {
//	unsigned long long errors_host = 0;
//	checkFrameworkErrors(
//			cudaMemcpyFromSymbol(&errors_host, errors,
//					sizeof(unsigned long long), 0, cudaMemcpyDeviceToHost));
//	return errors_host;
//}

#endif /* CUDA_UTILS_H_ */
