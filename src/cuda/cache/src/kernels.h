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
void test_shared_memory(uint32 number_of_sms, Board device);

#endif /* KERNELS_H_ */
