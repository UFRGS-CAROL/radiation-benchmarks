/*
 * L2Cache.h
 *
 *  Created on: Sep 6, 2019
 *      Author: carol
 */

#ifndef L2CACHE_H_
#define L2CACHE_H_

#include "Memory.h"

struct L2Cache: public Memory<uint64> {
	dim3 threads_per_block;
	uint32 l2_size;

	L2Cache();
	L2Cache(const Parameters& parameters);
	virtual void test(const uint64& mem);
	void clear_cache(uint32 n);
};

#endif /* L2CACHE_H_ */
