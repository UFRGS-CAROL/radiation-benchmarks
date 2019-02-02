/*
 * kernels.h
 *
 *  Created on: 26/01/2019
 *      Author: fernando
 */

#ifndef KERNELS_H_
#define KERNELS_H_

#include <vector>
#include <string>
#include "Log.h"

typedef unsigned char byte;
typedef unsigned int uint32;
typedef unsigned long long int uint64;

typedef signed int int32;
typedef signed long long int int64;

#define BLOCK_SIZE 32

enum Board {
	K40, TEGRAX2, TITANV, BOARD_COUNT,
};


struct Parameters {
	uint32 number_of_sms;
	Board device;
	uint32 shared_memory_size;
	uint32 l2_size;
	Log *log;
};


std::vector<std::string> test_l1_cache(const Parameters&);
void test_l2_cache(const Parameters&);
void test_shared_memory(const Parameters&);
void test_read_only_cache(const Parameters&);
void test_register_file(const Parameters&);

#endif /* KERNELS_H_ */
