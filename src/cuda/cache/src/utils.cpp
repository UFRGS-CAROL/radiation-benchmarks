/*
 * util.cpp
 *
 *  Created on: 26/01/2019
 *      Author: fernando
 */

#include "utils.h"
#include <iostream> // std::cout and std::cerr

/**
 * Check the return value of the CUDA runtime API call and exit
 * the application if the call has failed.
 */
static void check_cuda_error_(const char *file, unsigned line,
		const char *statement, cudaError_t err) {
	if (err == cudaSuccess)
		return;
	std::cerr << statement << " returned " << cudaGetErrorString(err) << "("
			<< err << ") at " << file << ":" << line << std::endl;
	exit(1);
}
