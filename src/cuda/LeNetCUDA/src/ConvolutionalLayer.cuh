/*
 * ConvolutionalLayer.cuh
 *
 *  Created on: May 26, 2017
 *      Author: carol
 */

#ifndef CONVOLUTIONALLAYER_CUH_
#define CONVOLUTIONALLAYER_CUH_

#define THREAD_TASKS 4


/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 */
#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }





#endif /* CONVOLUTIONALLAYER_CUH_ */
