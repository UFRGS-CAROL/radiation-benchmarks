/*
 * lud_kernel.h
 *
 *  Created on: 14/01/2019
 *      Author: fernando
 */

#ifndef LUD_KERNEL_H_
#define LUD_KERNEL_H_

#include <cuda.h>

#if PRECISION == 16
#define REAL_T half
#define PRECISION_STR "Half"
#endif

#if PRECISION == 32
#define REAL_T float
#define PRECISION_STR "Float"
#endif

#if PRECISION == 64
#define REAL_T double
#define PRECISION_STR "Double"
#endif

template<typename real_t>
void lud_cuda(real_t *m, int matrix_dim);

#define max(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a > _b ? _a : _b; })

#define min(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a < _b ? _a : _b; })


#endif /* LUD_KERNEL_H_ */
