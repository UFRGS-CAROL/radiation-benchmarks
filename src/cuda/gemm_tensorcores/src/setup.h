/*
 * setup_gemm.h
 *
 *  Created on: 27/10/2019
 *      Author: fernando
 */

#ifndef SETUP_H_
#define SETUP_H_

struct Log;

/**
 * Setup for common MxM (GEMM)
 */
void setup_gemm_unhardened(Log& log);
void setup_gemm_dmr(Log& log);

/**
 * Setup for Tensor (GEMM)
 */
void setup_gemm_tensor_cores_unhardened(Log& log);
void setup_gemm_tensor_cores_dmr(Log& log);



#endif /* SETUP_GEMM_H_ */
