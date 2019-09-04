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

	virtual void test(const byte reg_val);

	virtual std::string error_detail(uint32 i, uint32 e, uint32 r, uint64 hits, uint64 misses, uint64 false_hits) override;

	virtual void call_checker(const std::vector<uint32>& v1,
			const std::vector<uint32>& v2, const std::vector<uint32>& v3,
			byte valGold, Log& log, uint64 hits, uint64 misses, uint64 false_hits,
			bool verbose) override;
};


#endif /* REGISTERFILE_H_ */
