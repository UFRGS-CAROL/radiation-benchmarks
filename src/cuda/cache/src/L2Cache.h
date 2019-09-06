/*
 * L2Cache.h
 *
 *  Created on: Sep 6, 2019
 *      Author: carol
 */

#ifndef L2CACHE_H_
#define L2CACHE_H_

#include "Memory.h"
#include "CacheLine.h"

struct L2Cache: public Memory<CacheLine<CACHE_LINE_SIZE>> {
	dim3 threads_per_block;
	uint32 l2_size;

	L2Cache();
	L2Cache(const Parameters& parameters);
	virtual void test(const uint32& mem);
	virtual std::string error_detail(uint32 i, uint32 e, uint32 r, uint64 hits,
			uint64 misses, uint64 false_hits)
	override;

	virtual void call_checker(const std::vector<CacheLine<CACHE_LINE_SIZE>>& v1,
			const uint32& valGold, Log& log, uint64 hits, uint64 misses,
			uint64 false_hits, bool verbose)
	override;

	void clear_cache(uint32 n);
};

#endif /* L2CACHE_H_ */
