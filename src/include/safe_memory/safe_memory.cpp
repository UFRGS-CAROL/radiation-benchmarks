#include "safe_memory.h"

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

unsigned long long getAvailablePhysicalSystemMemory()
{
    long av_pages = sysconf(_SC_AVPHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    return av_pages * page_size;
}

void inline log_error_detail_and_exit(std::string error_description) {
#ifdef LOGS
	log_error_detail(const_cast<char*>(error_description.c_str()));
	end_log_file();
#endif
}

void inline log_error_detail_and_continue(std::string error_description) {
#ifdef LOGS
	log_error_detail(const_cast<char*>(error_description.c_str()));
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

bool __checkFrameworkErrorsNoFail(cudaError_t error, int line, const char* file) {
	if (error == cudaSuccess) {
		return false;
	}
	char errorDescription[250];
	snprintf(errorDescription, 250, "CUDA Framework error: %s. Not failing...",
			cudaGetErrorString(error));
	log_error_detail_and_continue(errorDescription);

	printf("%s - Line: %d at %s\n", errorDescription, line, file);
	return true;
}

void* safe_host_malloc(size_t size) {
	void* host_ptr = NULL;

	if (getAvailablePhysicalSystemMemory() < size) {
		return NULL; // Too much memory used.
	}

	host_ptr = malloc(size);
	if (host_ptr == NULL) 
		return NULL;
	
	bool is_memory_corrupted = false;
	for (int round = 0; round < MAX_MEM_TEST_ROUNDS; round++) {
		for (auto mem_const_value : TEST_POSSIBILITIES) {
			// ===> FIRST PHASE: CHECK SETTING BITS TO mem_const_value, that is  XAA, X55, X00, XFF
			memset(host_ptr, mem_const_value, size);

			//check if memory is ok
			#pragma omp parallel for
			for (int k=0; k<size; k++) {
				if (host_ptr[k] != mem_const_value) {
					is_memory_corrupted = true;
					break;
				}
			}

			//if corrupted we dont need to keep going
			if (is_memory_corrupted) {
				round = MAX_MEM_TEST_ROUNDS;
				break;
			}
		}
	}

	if (is_memory_corrupted) {
		// Failed
		void* new_host_ptr = safe_malloc(size);
		free(host_ptr)
		return new_host_ptr;
	}
	// ===> END FIRST PHASE
	
	return host_ptr;
}

void* safe_malloc(size_t size) {
	void* device_ptr = NULL;
	void* gold_ptr = NULL;
	void* outputPtr = NULL;

	if (is_crash == 0) {
		const char *error_description = "Trying_to_alloc_memory_GPU_may_crash";
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

	// ===> free host memory
	free(outputPtr);
	free(gold_ptr);
	return device_ptr;
}

int safe_cuda_malloc_cover(void **ptr, size_t size) {
	*ptr = safe_malloc(size);
	return 0;
}

static void error(std::string error_message) {
	std::cout << "ERROR: " << error_message << "at line " << __LINE__ << " "
			<< __FILE__ << std::endl;
	exit (EXIT_FAILURE);
}

/**
 * memory alloc three pointers
 */
void triple_malloc(triple_memory *tmr, size_t size) {
	tmr->size = size;
	//malloc device
	tmr->device_ptr1 = safe_malloc(tmr->size);
	tmr->device_ptr2 = safe_malloc(tmr->size);
	tmr->device_ptr3 = safe_malloc(tmr->size);

	//malloc host
	tmr->host_ptr1 = malloc(tmr->size);
	tmr->host_ptr2 = malloc(tmr->size);
	tmr->host_ptr3 = malloc(tmr->size);
	if (tmr->host_ptr1 == NULL || tmr->host_ptr2 == NULL
			|| tmr->host_ptr3 == NULL) {
		error("could not allocate host memory");
	}
}

/**
 * free the triple memory
 */
void triple_free(triple_memory *tmr) {
	//malloc host
	if (tmr->host_ptr1 == NULL) {
		error("HOST_PTR1 not allocated");
	}

	if (tmr->host_ptr2 == NULL) {
		error("HOST_PTR2 not allocated");
	}

	if (tmr->host_ptr3 == NULL) {
		error("HOST_PTR3 not allocated");
	}

	//malloc device
	checkFrameworkErrors(cudaFree(tmr->device_ptr1));
	checkFrameworkErrors(cudaFree(tmr->device_ptr2));
	checkFrameworkErrors(cudaFree(tmr->device_ptr3));
}

/**
 * copy three memory to gpu
 */
void triple_host_to_device_copy(triple_memory tmr) {
	checkFrameworkErrors(
			cudaMemcpy(tmr.device_ptr1, tmr.host_ptr1, tmr.size,
					cudaMemcpyHostToDevice));

	checkFrameworkErrors(
			cudaMemcpy(tmr.device_ptr2, tmr.host_ptr2, tmr.size,
					cudaMemcpyHostToDevice));

	checkFrameworkErrors(
			cudaMemcpy(tmr.device_ptr3, tmr.host_ptr3, tmr.size,
					cudaMemcpyHostToDevice));

}

/**
 * copy triple memory from gpu
 */
void triple_device_to_host_copy(triple_memory tmr) {
	printf("%p %p %p\n", tmr.host_ptr1, tmr.host_ptr2, tmr.host_ptr3, tmr.size);
	printf("%p %p %p\n", tmr.device_ptr1, tmr.device_ptr2, tmr.device_ptr3, tmr.size);

//	checkFrameworkErrors(
//			cudaMemcpy(tmr.host_ptr1, tmr.device_ptr1, tmr.size,
//					cudaMemcpyDeviceToHost));

//	checkFrameworkErrors(
//			cudaMemcpy(tmr.host_ptr2, tmr.device_ptr2, tmr.size,
//					cudaMemcpyDeviceToHost));
//	checkFrameworkErrors(
//			cudaMemcpy(tmr.host_ptr3, tmr.device_ptr3, tmr.size,
//					cudaMemcpyDeviceToHost));
}

/**
 * fill with an specified byte
 */
void triple_memset(triple_memory tmr, unsigned char byte) {
	//set on gpu
	checkFrameworkErrors(cudaMemset(tmr.device_ptr1, byte, tmr.size));
	checkFrameworkErrors(cudaMemset(tmr.device_ptr2, byte, tmr.size));
	checkFrameworkErrors(cudaMemset(tmr.device_ptr3, byte, tmr.size));

	//set on cpu
	memset(tmr.host_ptr1, byte, tmr.size);
	memset(tmr.host_ptr2, byte, tmr.size);
	memset(tmr.host_ptr3, byte, tmr.size);
}

/**
 * set host value at i position
 */
void triple_set_host_i(triple_memory tmr, int i, size_t size_of_mem,
		void *value) {
	memcpy(tmr.host_ptr1 + (i * size_of_mem), value, size_of_mem);
	memcpy(tmr.host_ptr2 + (i * size_of_mem), value, size_of_mem);
	memcpy(tmr.host_ptr3 + (i * size_of_mem), value, size_of_mem);
}

/**
 * get host value at i position
 * TODO: implement a safe get
 */
void triple_get_host_i(triple_memory tmr, int i, size_t size_of_mem,
		void *value) {
	memcpy(value, tmr.host_ptr1 + (i * size_of_mem), size_of_mem);
}

/**
 * Copy two triple memory device
 */
void triple_copy_device(triple_memory *dst, const triple_memory src) {
	if (dst->size != src.size) {
		error("DST and SRC do not have the same size, exiting");
	}
	checkFrameworkErrors(
			cudaMemcpy(dst->device_ptr1, src.device_ptr1, src.size,
					cudaMemcpyDeviceToDevice));
	checkFrameworkErrors(
			cudaMemcpy(dst->device_ptr2, src.device_ptr2, src.size,
					cudaMemcpyDeviceToDevice));
	checkFrameworkErrors(
			cudaMemcpy(dst->device_ptr3, src.device_ptr3, src.size,
					cudaMemcpyDeviceToDevice));
}

void triple_copy_host(triple_memory *dst, const triple_memory src) {
	if (dst->size != src.size) {
		error("DST and SRC do not have the same size, exiting");
	}
	memcpy(dst->host_ptr1, src.host_ptr1, src.size);
	memcpy(dst->host_ptr2, src.host_ptr2, src.size);
	memcpy(dst->host_ptr3, src.host_ptr3, src.size);
}
