/*
 * setup_gemm.h
 *
 *  Created on: 27/10/2019
 *      Author: fernando
 */

#ifndef SETUP_H_
#define SETUP_H_

#include <string>

struct Parameters;

/**
 * Setup for common MxM (GEMM)
 */
void setup_gemm_unhardened(Parameters&);
void setup_gemm_dmr(Parameters&);
void setup_gemm_cublas(Parameters&);

/**
 * Setup for Tensor (GEMM)
 */
void setup_gemm_tensor_cores_unhardened(Parameters&);
void setup_gemm_tensor_cores_dmr(Parameters&);

/**
 * Get the __CUDACC_VER_MAJOR__ from NVCC
 */
std::string get_cuda_cc_version();

#endif /* SETUP_GEMM_H_ */
