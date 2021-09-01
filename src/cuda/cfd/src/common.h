/*
 * common.h
 *
 *  Created on: 22/02/2020
 *      Author: fernando
 */

#ifndef COMMON_H_
#define COMMON_H_

/*
 * Options
 *
 */
#define GAMMA 1.4f
// #ifndef block_length
// 	#define block_length 192
// #endif

#define NDIM 3
#define NNB 4

#define RK 3    // 3rd order RK
#define ff_mach 1.2f
#define deg_angle_of_attack 0.0f


#define DEVICE 0

/*
 * not options
 */

#ifdef RD_WG_SIZE_0_0
#define BLOCK_SIZE_0 RD_WG_SIZE_0_0
#elif defined(RD_WG_SIZE_0)
#define BLOCK_SIZE_0 RD_WG_SIZE_0
#elif defined(RD_WG_SIZE)
#define BLOCK_SIZE_0 RD_WG_SIZE
#else
#define BLOCK_SIZE_0 192
#endif

#ifdef RD_WG_SIZE_1_0
#define BLOCK_SIZE_1 RD_WG_SIZE_1_0
#elif defined(RD_WG_SIZE_1)
#define BLOCK_SIZE_1 RD_WG_SIZE_1
#elif defined(RD_WG_SIZE)
#define BLOCK_SIZE_1 RD_WG_SIZE
#else
#define BLOCK_SIZE_1 192
#endif

#ifdef RD_WG_SIZE_2_0
#define BLOCK_SIZE_2 RD_WG_SIZE_2_0
#elif defined(RD_WG_SIZE_1)
#define BLOCK_SIZE_2 RD_WG_SIZE_2
#elif defined(RD_WG_SIZE)
#define BLOCK_SIZE_2 RD_WG_SIZE
#else
#define BLOCK_SIZE_2 192
#endif

#ifdef RD_WG_SIZE_3_0
#define BLOCK_SIZE_3 RD_WG_SIZE_3_0
#elif defined(RD_WG_SIZE_3)
#define BLOCK_SIZE_3 RD_WG_SIZE_3
#elif defined(RD_WG_SIZE)
#define BLOCK_SIZE_3 RD_WG_SIZE
#else
#define BLOCK_SIZE_3 192
#endif

#ifdef RD_WG_SIZE_4_0
#define BLOCK_SIZE_4 RD_WG_SIZE_4_0
#elif defined(RD_WG_SIZE_4)
#define BLOCK_SIZE_4 RD_WG_SIZE_4
#elif defined(RD_WG_SIZE)
#define BLOCK_SIZE_4 RD_WG_SIZE
#else
#define BLOCK_SIZE_4 192
#endif

// #if block_length > 128
// #warning "the kernels may fail too launch on some systems if the block length is too large"
// #endif

#define VAR_DENSITY 0
#define VAR_MOMENTUM  1
#define VAR_DENSITY_ENERGY (VAR_MOMENTUM+NDIM)
#define NVAR (VAR_DENSITY_ENERGY+1)

#endif /* COMMON_H_ */
