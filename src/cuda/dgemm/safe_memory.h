/*
 * safe_memory.h
 *
 *  Created on: Jun 6, 2018
 *      Author: carol
 */

#ifndef SAFE_MEMORY_H_
#define SAFE_MEMORY_H_
#include <cstdlib>

void* safe_malloc(size_t size);
int safe_cuda_malloc_cover(void **ptr, size_t size);

#endif /* SAFE_MEMORY_H_ */
