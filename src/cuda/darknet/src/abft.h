/*
 * abft.h
 *
 *  Created on: 16/11/2016
 *      Author: fernando
 */

#ifndef ABFT_H_
#define ABFT_H_

#define MAX_THRESHOLD  0.0001


void abraham_sum(double *a, double *b, long rows_a, long cols_a, long rows_b, long cols_b);
void abraham_check(double *c, long rows, long cols);


#endif /* ABFT_H_ */
