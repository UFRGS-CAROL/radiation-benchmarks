/*
 * kernels.h
 *
 *  Created on: 26/01/2019
 *      Author: fernando
 */

#ifndef KERNELS_H_
#define KERNELS_H_



void test_l1_cache(int l1_cache_size, int number_of_lines);
void test_l2_cache(int l1_cache_size, int number_of_lines);


template<typename T, int LINE_SIZE, int TYPE_SIZE>
struct alignas(LINE_SIZE * TYPE_SIZE)  cacheLine {
	T t[LINE_SIZE];
};


#endif /* KERNELS_H_ */
