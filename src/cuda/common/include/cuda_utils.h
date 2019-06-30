/*
 * cuda_utils.h
 *
 *  Created on: 17/06/2019
 *      Author: fernando
 */

#ifndef CUDA_UTILS_H_
#define CUDA_UTILS_H_

#include <chrono>
#include <thread>
#include <sys/time.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#ifdef LOGS
#include "log_helper.h"
#endif

namespace rad {

static void __checkFrameworkErrors(cudaError_t error, int line, const char* file) {
	if (error == cudaSuccess) {
		return;
	}
	char errorDescription[250];
	snprintf(errorDescription, 250, "CUDA Framework error: %s. Bailing.",
			cudaGetErrorString(error));
#ifdef LOGS
	log_error_detail((char *)errorDescription); end_log_file();
#endif
	printf("%s - Line: %d at %s\n", errorDescription, line, file);
	exit(EXIT_FAILURE);
}


#define checkFrameworkErrors(error) __checkFrameworkErrors(error, __LINE__, __FILE__)

//!  sleep seconds.
/*!
 \param seconds to sleep
 */
static void sleep(int seconds) {
	std::this_thread::sleep_for(std::chrono::seconds(seconds));
}

static void sleep(double seconds) {
	int milli = seconds * 1000.0;
	std::this_thread::sleep_for(std::chrono::milliseconds(milli));
}

static double mysecond() {
	struct timeval tp;
	struct timezone tzp;
	int i = gettimeofday(&tp, &tzp);
	return ((double) tp.tv_sec + (double) tp.tv_usec * 1.e-6);
}

}


#endif /* CUDA_UTILS_H_ */
