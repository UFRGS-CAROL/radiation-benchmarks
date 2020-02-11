/*
 * cuda_utils.h
 *
 *  Created on: 17/06/2019
 *      Author: fernando
 */

#ifndef CUDA_UTILS_H_
#define CUDA_UTILS_H_

#include <sys/time.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>

#ifdef __cplusplus
#include <chrono>
#include <thread>
//#include <helper_cuda.h>

#endif //C++ compiler defined

#ifdef LOGS
#include "log_helper.h"
#endif

#ifdef __cplusplus
namespace rad {
#endif //C++ compiler defined

static void __checkFrameworkErrors(cudaError_t error, int line,
		const char* file) {
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
	exit (EXIT_FAILURE);
}

#define checkFrameworkErrors(error) __checkFrameworkErrors(error, __LINE__, __FILE__)

static void __checkCublasErrors(cublasStatus_t error, int line,
		const char* file) {
	if (error == CUBLAS_STATUS_SUCCESS) {
		return;
	}
	char errorDescription[250];
	snprintf(errorDescription, 250, "CUDA CUBLAS error: %d. Bailing.", (error));
#ifdef LOGS
	log_error_detail((char *)errorDescription); end_log_file();
#endif
	printf("%s - Line: %d at %s\n", errorDescription, line, file);
	exit (EXIT_FAILURE);
}

#define checkCublasErrors(error) __checkCublasErrors(error, __LINE__, __FILE__)

// This will output the proper error string when calling cudaGetLastError
#define checkLastCudaError(msg) __checkLastCudaError (msg, __FILE__, __LINE__)

static void __checkLastCudaError(const char *errorMessage, const char *file,
		const int line) {
	cudaError_t err = cudaGetLastError();

	if (cudaSuccess != err) {
		char errorDescription[800];
		sprintf(errorDescription,
				"%s(%i) : getLastCudaError() CUDA error : %s : (%d) %s.\n",
				file, line, errorMessage, (int) err, cudaGetErrorString(err));
#ifdef LOGS
		log_error_detail((char *)errorDescription); end_log_file();
#endif
		cudaDeviceReset();
		exit (EXIT_FAILURE);
	}
}

static cudaDeviceProp get_device() {
	//================== Retrieve and set the default CUDA device
	cudaDeviceProp prop;
	int count = 0, i;

	checkFrameworkErrors(cudaGetDeviceCount(&count));
	for (i = 0; i < count; i++) {
		checkFrameworkErrors(cudaGetDeviceProperties(&prop, i));
	}

	int dev = 0;
	int *ndevice = &dev;
	checkFrameworkErrors(cudaGetDevice(ndevice));
	checkFrameworkErrors(cudaSetDevice(0));
	checkFrameworkErrors(cudaGetDeviceProperties(&prop, 0));

	return prop;
}

#ifdef __cplusplus

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
	gettimeofday(&tp, &tzp);
	return ((double) tp.tv_sec + (double) tp.tv_usec * 1.e-6);
}

static void del_arg(int argc, char **argv, int index) {
	int i;
	for (i = index; i < argc - 1; ++i)
		argv[i] = argv[i + 1];
	argv[i] = 0;
}

static int find_int_arg(int argc, char **argv, std::string arg, int def) {
	int i;
	for (i = 0; i < argc - 1; ++i) {
		if (!argv[i])
			continue;
		if (std::string(argv[i]) == arg) {
			def = atoi(argv[i + 1]);
			del_arg(argc, argv, i);
			del_arg(argc, argv, i);
			break;
		}
	}
	return def;
}

static float find_float_arg(int argc, char **argv, std::string arg, float def) {
	for (int i = 0; i < argc - 1; ++i) {
		if (!argv[i])
			continue;
		if (std::string(argv[i]) == arg) {
			std::string to_convert(argv[i + 1]);

			def = std::stof(to_convert);
			del_arg(argc, argv, i);
			del_arg(argc, argv, i);
			break;
		}
	}
	return def;
}

static std::string find_char_arg(int argc, char **argv, std::string arg,
		std::string def) {
	int i;
	for (i = 0; i < argc - 1; ++i) {
		if (!argv[i])
			continue;
		if (std::string(argv[i]) == arg) {
			def = std::string(argv[i + 1]);
			del_arg(argc, argv, i);
			del_arg(argc, argv, i);
			break;
		}
	}
	return def;
}

static bool find_arg(int argc, char* argv[], std::string arg) {
	int i;
	for (i = 0; i < argc; ++i) {
		if (!argv[i])
			continue;
		if (std::string(argv[i]) == arg) {
			del_arg(argc, argv, i);
			return true;
		}
	}
	return false;
}

} //namespace radiation

#endif  //C++ compiler defined

#endif /* CUDA_UTILS_H_ */
