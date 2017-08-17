/*
 * abft.h
 *
 *  Created on: 13/08/2017
 *      Author: fernando
 */

#ifndef ABFT_H_
#define ABFT_H_

typedef struct {
	unsigned err_detected_size;
	unsigned long long *error_detected;
} error_return;

#define MAXPOOL_N 5

#define MAX_ABFT_TYPES 6

void init_error_return(error_return *e);


void free_error_return(error_return *e);

void reset_error_return(error_return *e);


#endif /* ABFT_H_ */
