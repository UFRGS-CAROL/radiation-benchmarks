/*
 * kernels.h
 *
 *  Created on: 26/01/2019
 *      Author: fernando
 */

#ifndef KERNELS_H_
#define KERNELS_H_

typedef unsigned char byte;
typedef unsigned int uint32;
typedef unsigned long long int uint64;

typedef signed int int32;
typedef signed long long int int64;

#define BLOCK_SIZE 32

enum Board {
	K40, TEGRAX2, TITANV, BOARD_COUNT,
};

void test_l1_cache(uint32 number_of_sms, Board device);
void test_l2_cache(uint32 number_of_sms, Board device);
void test_shared_memory(const uint32 number_of_sms, const Board device,
		const uint32 shared_memory_size, const uint32 shared_line_size);
#endif /* KERNELS_H_ */
