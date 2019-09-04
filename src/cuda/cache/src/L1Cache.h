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

struct L1Cache: public Memory<CacheLine<CACHE_LINE_SIZE>> {
	dim3 threads_per_block;

	L1Cache();
	L1Cache(const Parameters& parameters);
	virtual void test(const byte t_byte);
	virtual std::string error_detail(uint32 i, uint32 e, uint32 r, uint64 hits, uint64 misses, uint64 false_hits) override;

	virtual void call_checker(const std::vector<CacheLine<CACHE_LINE_SIZE>>& v1,
			const std::vector<CacheLine<CACHE_LINE_SIZE>>& v2,
			const std::vector<CacheLine<CACHE_LINE_SIZE>>& v3, byte valGold,
			Log& log, uint64 hits, uint64 misses, uint64 false_hits, bool verbose) override;
};


#endif /* L1CACHE_H_ */
