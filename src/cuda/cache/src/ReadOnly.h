/*
 * ReadOnly.h
 *
 *  Created on: Sep 11, 2019
 *      Author: carol
 */

#ifndef READONLY_H_
#define READONLY_H_

#include "Memory.h"

struct ReadOny: public Memory<uint64> {
	dim3 threads_per_block;

	ReadOny();
	ReadOny(const Parameters& parameters);
	virtual void test(const uint64& mem) override;

	bool call_checker(uint64& gold, Log& log, int64& hits,
				int64& misses, int64& false_hits) override;
};



#endif /* READONLY_H_ */
