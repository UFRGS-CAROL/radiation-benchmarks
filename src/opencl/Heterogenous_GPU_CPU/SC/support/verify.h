/*
 * Copyright (c) 2016 University of Cordoba and University of Illinois
 * All rights reserved.
 *
 * Developed by:    IMPACT Research Group
 *                  University of Cordoba and University of Illinois
 *                  http://impact.crhc.illinois.edu/
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * with the Software without restriction, including without limitation the 
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 *      > Redistributions of source code must retain the above copyright notice,
 *        this list of conditions and the following disclaimers.
 *      > Redistributions in binary form must reproduce the above copyright
 *        notice, this list of conditions and the following disclaimers in the
 *        documentation and/or other materials provided with the distribution.
 *      > Neither the names of IMPACT Research Group, University of Cordoba, 
 *        University of Illinois nor the names of its contributors may be used 
 *        to endorse or promote products derived from this Software without 
 *        specific prior written permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
 * CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
 * THE SOFTWARE.
 *
 */
#ifndef VERIFY_H
#define VERIFY_H
#include "common.h"
#include <math.h>

//*****************************************  LOG  ***********************************//
#ifdef LOGS
#include "log_helper.h"
#endif
//************************************************************************************//

inline int new_compare_output(T *outp, T *outpCPU, int size) {
	//printf("Estou na compare\n");
	int errors=0;
    double sum_delta2,sum_delta2_x, sum_ref2, L1norm2;
    sum_delta2 = 0;
    sum_ref2   = 0;
    L1norm2    = 0;
	int i = 0;
	//printf("\tSize:%d\n",size);


/*
//************************ Compare 2 Caso não se comporte como esperado **********************
#pragma omp parallel for reduction(+:errors) private(i)
    for(i = 0; i < size; i++) {

          sum_ref2 = std::abs(outpCPU[i]);

    	if(sum_ref2 == 0)
    	    sum_ref2 = 1; //In case percent=0
		sum_delta2_x = (double)std::abs(outp[i] - outpCPU[i]) / sum_ref2 ;

			if(sum_delta2_x >= 1e-12 ){
					errors++;
					#ifdef LOGS
					char error_detail[200];
					sprintf(error_detail,"X, p: [%d], r: %d, e: %d",i,outp[i],outpCPU[i] );

		   			log_error_detail(error_detail);
					#endif				
			}
	}
*/

// ******************* Compare Testado em Maio 2018 ***************************
#pragma omp parallel for reduction(+:errors) private(i)
    for(i=0; i < size; i++) {
		if(outp[i] != outpCPU[i]){	// Computed result is different

			errors++;
			#ifdef LOGS
			char error_detail[200];
			sprintf(error_detail,"X, p: [%d], r: %d, e: %d",i,outp[i],outpCPU[i] );

   			log_error_detail(error_detail);
			#endif			
		}
    }


    return errors;
}

inline int compare_output(T *outp, T *outpCPU, int size) {
    double sum_delta2, sum_ref2, L1norm2;
    sum_delta2 = 0;
    sum_ref2   = 0;
    L1norm2    = 0;
    for(int i = 0; i < size; i++) {
        sum_delta2 += std::abs(outp[i] - outpCPU[i]);
        sum_ref2 += std::abs(outpCPU[i]);
    }
    if(sum_ref2 == 0)
        sum_ref2 = 1; //In case percent=0
    L1norm2      = (double)(sum_delta2 / sum_ref2);
    if(L1norm2 >= 1e-6){
        printf("Test failed\n");
        exit(EXIT_FAILURE);
    }
    return 0;
}

// Sequential implementation for comparison purposes
inline double cpu_streamcompaction(T *input, int size, int value) {
    struct timeval t1, t2;
    int            pos = 0;
    // start timer
    gettimeofday(&t1, NULL);
    for(int my = 0; my < size; my++) {
        if(input[my] != value) {
            input[pos] = input[my];
            pos++;
        }
    }
    // end timer
    gettimeofday(&t2, NULL);
    double timer = (t2.tv_sec - t1.tv_sec) * 1000000.0 + (t2.tv_usec - t1.tv_usec);
    //printf("Execute time: %f us\n", timer);
    return timer;
}

inline void verify(T *input, T *input_array, int size, int value, int size_compact) {
    cpu_streamcompaction(input_array, size, value);
    compare_output(input, input_array, size_compact);
}

#endif
