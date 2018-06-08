/*
 * safe_memory.h
 *
 *  Created on: Jun 6, 2018
 *      Author: carol
 */

#ifndef SAFE_MEMORY_H_
#define SAFE_MEMORY_H_

typedef struct{
	//host pointers
	void *host_ptr1;
	void *host_ptr2;
	void *host_ptr3;
	
	//device pointers
	void *device_ptr1;
	void *device_ptr2;	
	void *device_ptr3;

	// stores the memory byte size that will be allocated
	unsigned long size;
} triple_memory;

/**
 * memory alloc three pointers
 */
void triple_malloc(triple_memory tmr);

/**
 * free the triple memory
 */
void triple_free(triple_memory tmr);

/**
 * copy three memory to gpu
 */
void triple_host_to_device_copy(triple_memory tmr);
 
/**
 * copy triple memory from gpu
 */
void triple_device_to_host_copy(triple_memory tmr);


void* safe_malloc(unsigned long size);
int safe_cuda_malloc_cover(void **ptr, unsigned long size);

#endif /* SAFE_MEMORY_H_ */
