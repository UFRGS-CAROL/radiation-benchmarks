/*
 * safe_memory.h
 *
 *  Created on: Jun 6, 2018
 *      Author: carol
 */

#ifndef SAFE_MEMORY_H_
#define SAFE_MEMORY_H_

#include <stdlib.h>

typedef enum {
	CHAR, INT, FLOAT, DOUBLE
} type;

typedef struct {
	//host pointers
	void *host_ptr1;
	void *host_ptr2;
	void *host_ptr3;

	//device pointers
	void *device_ptr1;
	void *device_ptr2;
	void *device_ptr3;

	// stores the memory byte size that will be allocated
	size_t size;
} triple_memory;

#define checkFrameworkErrors(error) __checkFrameworkErrors(error, __LINE__, __FILE__)

/**
 * generic log helper checker for errors
 */
void __checkFrameworkErrors(cudaError_t error, int line, const char* file);

/**
 * memory alloc three pointers
 */
void triple_malloc(triple_memory *tmr, size_t size);

/**
 * free the triple memory
 */
void triple_free(triple_memory *tmr);

/**
 * copy three memory to gpu
 */
void triple_host_to_device_copy(triple_memory tmr);

/**
 * copy triple memory from gpu
 */
void triple_device_to_host_copy(triple_memory tmr);

/**
 * fill with an specified byte
 */
void triple_memset(triple_memory tmr, unsigned char byte);

/**
 * Safe memory allocation grant
 * that the memory was stressed before being used
 */
void* safe_malloc(unsigned long size);

/**
 * Cover for safe memory allocation for cuda
 */
int safe_cuda_malloc_cover(void **ptr, size_t size);

/**
 * set host value at i position
 */
void triple_set_host_i(triple_memory tmr, int i, size_t size_of_mem,
		void *value);

/**
 * get host value at i position
 * TODO: implement a safe get
 */
void triple_get_host_i(triple_memory tmr, int i, size_t size_of_mem,
		void *value);

/**
 * Copy two triple memory device
 */
void triple_copy_device(triple_memory *dst, const triple_memory src);

/**
 * Copy two triple memory host
 */
void triple_copy_host(triple_memory *dst, const triple_memory src);

#endif /* SAFE_MEMORY_H_ */
