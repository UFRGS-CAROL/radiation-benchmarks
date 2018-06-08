#include "safe_memory.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdio>
#include <cstring>

#ifdef LOGS
#include "log_helper.h"
#endif

//ALL POSSIBILITIES OF MEMSET
#define XAA 0xAA
#define X55 0x55
#define X00 0x00
#define XFF 0xFF

//Max memory rounds test
#define MAX_MEM_TEST_ROUNDS 5
typedef unsigned char _byte_;
_byte_ TEST_POSSIBILITIES[] = { XAA, X55, X00, XFF };

static unsigned char is_crash = 0;

#define checkFrameworkErrors(error) __checkFrameworkErrors(error, __LINE__, __FILE__)

void inline log_error_detail_and_exit(char *error_description) {
#ifdef LOGS
	log_error_detail(error_description);
	end_log_file();
#endif
}

void inline log_error_detail_and_continue(char *error_description) {
#ifdef LOGS
	log_error_detail(error_description);
#endif
}

void __checkFrameworkErrors(cudaError_t error, int line, const char* file) {
	if (error == cudaSuccess) {
		return;
	}
	char errorDescription[250];
	snprintf(errorDescription, 250, "CUDA Framework error: %s. Bailing.",
			cudaGetErrorString(error));
	log_error_detail_and_exit(errorDescription);

	printf("%s - Line: %d at %s\n", errorDescription, line, file);
	exit (EXIT_FAILURE);
}

void* safe_malloc(size_t size) {
	void* device_ptr = NULL;
	void* gold_ptr = NULL;
	void* outputPtr = NULL;

	if (is_crash == 0) {
		char *error_description = "Trying_to_alloc_memory_GPU_may_crash";
		log_error_detail_and_continue(error_description);
		is_crash = 1;
	}

	// First, alloc DEVICE proposed memory and HOST memory for device memory checking
	checkFrameworkErrors(cudaMalloc(&device_ptr, size));
	outputPtr = malloc(size);
	gold_ptr = malloc(size);

	if ((outputPtr == NULL) || (gold_ptr == NULL)) {
		log_error_detail_and_exit((char*) "error host malloc");
		printf("error host malloc\n");
		exit (EXIT_FAILURE);
	}

	bool is_memory_corrupted = false;
	for (int round = 0; round < MAX_MEM_TEST_ROUNDS; round++) {
		for (auto mem_const_value : TEST_POSSIBILITIES) {
			// ===> FIRST PHASE: CHECK SETTING BITS TO mem_const_value, that is  XAA, X55, X00, XFF
			checkFrameworkErrors(cudaMemset(device_ptr, mem_const_value, size));
			memset(gold_ptr, mem_const_value, size);

			checkFrameworkErrors(
					cudaMemcpy(outputPtr, device_ptr, size,
							cudaMemcpyDeviceToHost));

			//check if memory is ok
			is_memory_corrupted = (memcmp(outputPtr, gold_ptr, size) != 0);

			//if corrupted we dont need to keep going
			if (is_memory_corrupted) {
				round = MAX_MEM_TEST_ROUNDS;
				break;
			}
		}
	}

	if (is_memory_corrupted) {
		// Failed
		free(outputPtr);
		free(gold_ptr);
		void* newDevicePtr = safe_malloc(size);
		checkFrameworkErrors(cudaFree(device_ptr));
		return newDevicePtr;
	}
	// ===> END FIRST PHASE

//	// ===> SECOND PHASE: CHECK SETTING BITS TO 01010101
//	checkFrameworkErrors(cudaMemset(device_ptr, 0x55, size));
//	memset(gold_ptr, 0x55, size);
//
//	checkFrameworkErrors(
//			cudaMemcpy(outputPtr, device_ptr, size, cudaMemcpyDeviceToHost));
//	if (memcmp(outputPtr, gold_ptr, size)) {
//		// Failed
//		free(outputPtr);
//		free(gold_ptr);
//		void* newDevicePtr = safe_malloc(size);
//		checkFrameworkErrors(cudaFree(device_ptr));
//		return newDevicePtr;
//	}

// ===> END SECOND PHASE

	free(outputPtr);
	free(gold_ptr);
	return device_ptr;
}

int safe_cuda_malloc_cover(void **ptr, unsigned long size) {
	*ptr = safe_malloc(size);
	return 0;
}

static void error(char *error_message){
	fprintf(stdout, "ERROR: %s, at line %d, file %s\n", error_message, __LINE__, __FILE__);
	exit(EXIT_FAILURE);
}

/**
 * memory alloc three pointers
 */
void triple_malloc(triple_memory tmr){
	//malloc host
	tmr.host_ptr1 = malloc(tmr.size);
	tmr.host_ptr2 = malloc(tmr.size);
	tmr.host_ptr3 = malloc(tmr.size);
	if (tmr.host_ptr1 == NULL || tmr.host_ptr2 == NULL 
			|| tmr.host_ptr3 == NULL){
		error("could not allocate host memory");
	} 
	
	//malloc device
	checkFrameworkErrors(cudaMalloc(&tmr.device_ptr1, tmr.size));
	checkFrameworkErrors(cudaMalloc(&tmr.device_ptr2, tmr.size));
	checkFrameworkErrors(cudaMalloc(&tmr.device_ptr3, tmr.size));
}

/**
 * free the triple memory
 */
void triple_free(triple_memory tmr){
		//malloc host
	tmr.host_ptr1 = malloc(tmr.size);
	tmr.host_ptr2 = malloc(tmr.size);
	tmr.host_ptr3 = malloc(tmr.size);
	if (tmr.host_ptr1 == NULL || tmr.host_ptr2 == NULL 
			|| tmr.host_ptr3 == NULL){
		error("could not allocate host memory");
	} 
	
	//malloc device
	checkFrameworkErrors(cudaMalloc(&tmr.device_ptr1, tmr.size));
	checkFrameworkErrors(cudaMalloc(&tmr.device_ptr2, tmr.size));
	checkFrameworkErrors(cudaMalloc(&tmr.device_ptr3, tmr.size));
}

/**
 * copy three memory to gpu
 */
void triple_host_to_device_copy(triple_memory tmr){
}
 
/**
 * copy triple memory from gpu
 */
void triple_device_to_host_copy(triple_memory tmr){
}

