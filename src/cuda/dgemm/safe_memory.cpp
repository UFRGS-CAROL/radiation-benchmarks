#include "safe_memory.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdio>
#include <cstring>

#ifdef LOGS
#include "log_helper.h"
#endif

static unsigned char is_crash = 0;

#define checkFrameworkErrors(error) __checkFrameworkErrors(error, __LINE__, __FILE__)

void __checkFrameworkErrors(cudaError_t error, int line, const char* file) {
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

void* safe_malloc(size_t size) {
	void* devicePtr = NULL;
	void* goldPtr = NULL;
	void* outputPtr = NULL;

	if (is_crash == 0) {
		char errorDescription[250];
		sprintf(errorDescription, "Trying_to_alloc_memory_GPU_may_crash");
#ifdef LOGS
		log_info_detail((char *)errorDescription);
#endif
		is_crash = 1;
	}

	// First, alloc DEVICE proposed memory and HOST memory for device memory checking
	checkFrameworkErrors(cudaMalloc(&devicePtr, size));
	outputPtr = malloc(size);
	goldPtr = malloc(size);

	if ((outputPtr == NULL) || (goldPtr == NULL)) {
#ifdef LOGS
		log_error_detail((char *) "error host malloc");
		end_log_file();
#endif
		printf("error host malloc\n");
		exit (EXIT_FAILURE);
	}

	// ===> FIRST PHASE: CHECK SETTING BITS TO 10101010
	checkFrameworkErrors(cudaMemset(devicePtr, 0xAA, size));
	memset(goldPtr, 0xAA, size);

	checkFrameworkErrors(
			cudaMemcpy(outputPtr, devicePtr, size, cudaMemcpyDeviceToHost));
	if (memcmp(outputPtr, goldPtr, size)) {
		// Failed
		free(outputPtr);
		free(goldPtr);
		void* newDevicePtr = safe_malloc(size);
		checkFrameworkErrors(cudaFree(devicePtr));
		return newDevicePtr;
	}
	// ===> END FIRST PHASE

	// ===> SECOND PHASE: CHECK SETTING BITS TO 01010101
	checkFrameworkErrors(cudaMemset(devicePtr, 0x55, size));
	memset(goldPtr, 0x55, size);

	checkFrameworkErrors(
			cudaMemcpy(outputPtr, devicePtr, size, cudaMemcpyDeviceToHost));
	if (memcmp(outputPtr, goldPtr, size)) {
		// Failed
		free(outputPtr);
		free(goldPtr);
		void* newDevicePtr = safe_malloc(size);
		checkFrameworkErrors(cudaFree(devicePtr));
		return newDevicePtr;
	}

	// ===> END SECOND PHASE

	free(outputPtr);
	free(goldPtr);
	return devicePtr;
}

int safe_cuda_malloc_cover(void **ptr, size_t size) {
	*ptr = safe_malloc(size);
}
