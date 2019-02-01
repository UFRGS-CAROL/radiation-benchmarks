/*
 * kernels.h
 *
 *  Created on: 26/01/2019
 *      Author: fernando
 */

#ifndef KERNELS_H_
#define KERNELS_H_

#include <cstdint>

typedef std::uint8_t byte;
typedef std::uint32_t uint32;
typedef std::uint64_t uint64;

typedef std::int32_t int32;
typedef std::int64_t int64;

enum Board {
	K40, TEGRAX2, TITANV, BOARD_COUNT,
};

void test_l1_cache(uint32 number_of_sms, Board device);
void test_l2_cache(uint32 number_of_sms, Board device);
void test_shared_memory(uint32 number_of_sms, Board device);

#endif /* KERNELS_H_ */
