/*
 * abft.h
 *
 *  Created on: 16/11/2016
 *      Author: fernando
 */

#ifndef ABFT_H_
#define ABFT_H_

#define MAX_THRESHOLD  0.9
#define BLOCK_SIZE 1024

typedef struct erro_return {
	unsigned long long int row_detected_errors;
	unsigned long long int col_detected_errors;
} ErrorReturn;

void abraham_sum(float *a, float *b, long rows_a, long cols_a, long rows_b,
		long cols_b);
ErrorReturn abraham_check(float *c, long rows, long cols);

#endif /* ABFT_H_ */
