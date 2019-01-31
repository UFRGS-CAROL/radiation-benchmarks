/*
 * kernels.h
 *
 *  Created on: 26/01/2019
 *      Author: fernando
 */

#ifndef KERNELS_H_
#define KERNELS_H_

#include <cstdint>

enum Board{
	K40,
	TEGRAX2,
	TITANV,
	BOARD_COUNT,
};

void test_l1_cache(std::uint32_t number_of_sms, Board device);
void test_l2_cache(std::uint32_t number_of_sms, Board device);



#endif /* KERNELS_H_ */
