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
	void test(const uint64& mem) override;
	bool call_checker(uint64& gold, rad::Log& log, int64& hits,
				int64& misses, int64& false_hits, bool verbose) override;
};


#endif /* SHAREDMEMORY_H_ */
