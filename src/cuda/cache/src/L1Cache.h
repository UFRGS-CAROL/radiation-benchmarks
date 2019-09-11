/*
 * L1Cache.h
 *
 *  Created on: Sep 4, 2019
 *      Author: carol
 */

#ifndef L1CACHE_H_
#define L1CACHE_H_

#include "Memory.h"

struct L1Cache: public Memory<uint64> {
	dim3 threads_per_block;

	L1Cache();
	L1Cache(const Parameters& parameters);
	virtual void test(const uint64& mem) override;
};


#endif /* L1CACHE_H_ */
