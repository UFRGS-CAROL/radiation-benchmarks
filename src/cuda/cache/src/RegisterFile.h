/*
 * RegisterFile.h
 *
 *  Created on: Sep 4, 2019
 *      Author: carol
 */

#ifndef REGISTERFILE_H_
#define REGISTERFILE_H_

#include <vector>
#include <string>

#include "Memory.h"
#include "Parameters.h"

struct RegisterFile: public Memory<uint32> {
	uint32 number_of_threads;
	uint32 number_of_sms;

	RegisterFile(const Parameters& parameters);

	void test(const uint64& mem);

	std::string error_detail(uint64 i, uint64 e, uint64 r, int64 hits,
			int64 misses, int64 false_hits) override;

	bool call_checker(uint64& gold, rad::Log& log, int64& hits,
			int64& misses, int64& false_hits, bool verbose) override;


};

#endif /* REGISTERFILE_H_ */
