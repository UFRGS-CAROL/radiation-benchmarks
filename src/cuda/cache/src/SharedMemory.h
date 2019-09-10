/*
 * SharedMemory.h
 *
 *  Created on: Sep 6, 2019
 *      Author: carol
 */

#ifndef SHAREDMEMORY_H_
#define SHAREDMEMORY_H_

#include "Memory.h"

struct SharedMemory: public Memory<uint64> {
	dim3 threads_per_block;

	SharedMemory();
	SharedMemory(const Parameters& parameters);
	virtual void test(const uint64& mem);
};


#endif /* SHAREDMEMORY_H_ */
