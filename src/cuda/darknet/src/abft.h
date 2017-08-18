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
} error_return;

void abraham_sum(float *a, float *b, long rows_a, long cols_a, long rows_b,
		long cols_b);
error_return abraham_check(float *c, long rows, long cols);

//ErrorReturn shared_errors;

void set_abft_gemm(int n);
int get_use_abft_gemm();

#define MAXPOOL_N 4

//#define MAX_ABFT_TYPES 6

void init_error_return(error_return *e);


void free_error_return(error_return *e);


void reset_error_return(error_return *e);



#endif /* ABFT_H_ */
