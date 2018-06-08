/*
 * safe_memory.h
 *
 *  Created on: Jun 6, 2018
 *      Author: carol
 */

#ifndef SAFE_MEMORY_H_
#define SAFE_MEMORY_H_

void* safe_malloc(unsigned long size);
int safe_cuda_malloc_cover(void **ptr, unsigned long size);

#endif /* SAFE_MEMORY_H_ */
