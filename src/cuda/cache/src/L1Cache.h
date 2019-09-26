/*
 * L1Cache.h
 *
 *  Created on: Sep 4, 2019
 *      Author: carol
 */

#ifndef L1CACHE_H_
#define L1CACHE_H_

#include "Memory.h"
#include "CacheLine.h"

struct L1Cache: public Memory<cacheline> {
	dim3 threads_per_block;

	L1Cache();
	L1Cache(const Parameters& parameters);
	void test(const uint64& mem);

	bool call_checker(uint64& gold, Log& log, int64& hits, int64& misses,
			int64& false_hits) override;

};

#endif /* L1CACHE_H_ */
