/*
 * abft.h
 *
 *  Created on: 17/05/2017
 *      Author: fernando
 */

#ifndef ABFT_H_
#define ABFT_H_


int abft_type;

/**
 * 0 for no abft
 * 1 for Abraham abft
 * 2 for maxpool hardened
 */
void set_abft(int type) {
	abft_type = type;
}

int get_abft() {
	return abft_type;
}

#endif /* ABFT_H_ */
