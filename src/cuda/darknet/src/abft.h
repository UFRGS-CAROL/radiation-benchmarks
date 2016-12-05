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

typedef struct erro_return {
	unsigned long long int row_detected_errors;
	unsigned long long int col_detected_errors;
} ErrorReturn;

void abraham_sum(float *a, float *b, long rows_a, long cols_a, long rows_b,
		long cols_b);
ErrorReturn abraham_check(float *c, long rows, long cols);

//ErrorReturn shared_errors;

void set_use_abft(int n);
int get_use_abft();

#endif /* ABFT_H_ */
