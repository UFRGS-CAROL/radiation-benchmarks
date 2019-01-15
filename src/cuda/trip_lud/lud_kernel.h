/*
 * lud_kernel.h
 *
 *  Created on: 14/01/2019
 *      Author: fernando
 */

#ifndef LUD_KERNEL_H_
#define LUD_KERNEL_H_

#include <cuda.h>
void lud_cuda_float(float *m, int matrix_dim);

#define max(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a > _b ? _a : _b; })

#define min(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a < _b ? _a : _b; })


#endif /* LUD_KERNEL_H_ */
