/*
 * abft.h
 *
 *  Created on: 16/11/2016
 *      Author: fernando
 */

#ifndef ABFT_H_
#define ABFT_H_

#define MAX_THRESHOLD  0.5
#define BLOCK_SIZE 1024

#define DIV_VALUE 1e5

typedef struct {
	unsigned long long int row_detected_errors;
	unsigned long long int col_detected_errors;

	//only for smart pooling
	unsigned err_detected_size;
	unsigned long long *error_detected;
} ErrorReturn;

void abraham_sum(float *a, float *b, long rows_a, long cols_a, long rows_b,
		long cols_b);
ErrorReturn abraham_check(float *c, long rows, long cols);

//ErrorReturn shared_errors;

void set_use_abft(int n);
int get_use_abft();

#define MAXPOOL_N 5

//#define MAX_ABFT_TYPES 6

void init_error_return(ErrorReturn *e);
//{
//	e->err_detected_size = MAXPOOL_N;
//	e->error_detected = (unsigned long long*) malloc(e->err_detected_size * sizeof(unsigned long long));
//}

void free_error_return(ErrorReturn *e);
//{
//	if(e->error_detected)
//		free(e->error_detected);
//	e->error_detected = NULL;
//}

void reset_error_return(ErrorReturn *e);
//{
//	memset(e->error_detected, 0, sizeof(unsigned long long) * e->err_detected_size);
//}


#endif /* ABFT_H_ */
