/*
 * util.cpp
 *
 *  Created on: 26/01/2019
 *      Author: fernando
 */

#include "utils.h"
#include <iostream> // std::cout and std::cerr
#include <stdexcept>
#include <chrono>
#include <thread>

/**
 * Check the return value of the CUDA runtime API call and exit
 * the application if the call has failed.
 */
void check_cuda_error_(const char *file, unsigned line, const char *statement,
		cudaError_t err) {
	cudaError_t last_err = cudaGetLastError();
	if (err == cudaSuccess && last_err == cudaSuccess)
		return;

	std::cerr << statement << " returned " << cudaGetErrorString(err) << "("
			<< err << ") at " << file << ":" << line << std::endl;

	if(last_err != cudaSuccess)
		std::cerr << "cudaGetLastError() returned " << cudaGetErrorString(last_err) << "("
				<< last_err << ") at " << file << ":" << line << std::endl;

	exit(1);
}

void error(std::string err) {
	throw std::runtime_error("ERROR:" + err);
}

//!  sleep seconds.
/*!
 \param seconds to sleep
 */
void sleep(int seconds) {
	std::this_thread::sleep_for(std::chrono::seconds(seconds));
}

//!  get time since epoch .
/*!
 \return an int value representing the seconds since epoch
 */
size_t get_time_since_epoch() {
	std::time_t result = std::time(nullptr);
	return size_t(result);
}

