/*
 * L1Cache.h
 *
 *  Created on: Sep 4, 2019
 *      Author: carol
 */

#ifndef L1CACHE_H_
#define L1CACHE_H_

#include "Memory.h"

struct cacheline {
	uint64 line[CACHE_LINE_SIZE_BY_INT64];
};

struct L1Cache: public Memory<cacheline> {
	dim3 threads_per_block;

	L1Cache();
	L1Cache(const Parameters& parameters);
	void test(const uint64& mem) ;

	bool call_checker(uint64& gold, Log& log, int64& hits, int64& misses,
			int64& false_hits, bool verbose) override;

};

#endif /* L1CACHE_H_ */
