/*
 * cuda_utils.cpp
 *
 *  Created on: 18/05/2019
 *      Author: fernando
 */

#include <helper_cuda.h>
#include <iostream>

#include "cuda_utils.h"
#include "Log.h"

void __checkFrameworkErrors(cudaError_t error, int line, const char* file) {
	if (error == cudaSuccess) {
		return;
	}
	std::string errorDescription = std::string("CUDA Framework error: ") + cudaGetErrorString(error) + " Bailing.";
	Log::force_end(errorDescription);
	std::cerr << errorDescription << " - Line: " << line << " File: " << file << std::endl;
	exit (EXIT_FAILURE);
}

cudaDeviceProp GetDevice() {
//================== Retrieve and set the default CUDA device
	cudaDeviceProp prop;
	int count = 0;

	checkFrameworkErrors(cudaGetDeviceCount(&count));
	for (int i = 0; i < count; i++) {
		checkFrameworkErrors(cudaGetDeviceProperties(&prop, i));
	}
	int *ndevice;
	int dev = 0;
	ndevice = &dev;
	checkFrameworkErrors(cudaGetDevice(ndevice));

	checkFrameworkErrors(cudaSetDevice(0));
	checkFrameworkErrors(cudaGetDeviceProperties(&prop, 0));

	return prop;
}
