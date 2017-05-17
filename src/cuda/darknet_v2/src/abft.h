/*
 * abft.h
 *
 *  Created on: 17/05/2017
 *      Author: fernando
 */

#ifndef ABFT_H_
#define ABFT_H_
#define MAX_ABFT_TYPES 3

int abft_type;

/**
 * 0 for no abft
 * 1 for Abraham abft
 * 2 for maxpool hardened
 */
inline void set_abft(int type) {
	abft_type = type;
}

inline int get_abft() {
	return abft_type;
}

#endif /* ABFT_H_ */
