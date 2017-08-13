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
//{
//	e->err_detected_size = MAXPOOL_N;
//	e->error_detected = (unsigned long long*) malloc(e->err_detected_size * sizeof(unsigned long long));
//}

void free_error_return(error_return *e);
//{
//	if(e->error_detected)
//		free(e->error_detected);
//	e->error_detected = NULL;
//}

void reset_error_return(error_return *e);
//{
//	memset(e->error_detected, 0, sizeof(unsigned long long) * e->err_detected_size);
//}

#endif /* ABFT_H_ */
