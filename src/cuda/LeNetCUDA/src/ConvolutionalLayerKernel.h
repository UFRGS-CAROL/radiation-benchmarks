/*
 * ConvolutionalLayer.cuh
 *
 *  Created on: May 26, 2017
 *      Author: carol
 */

#ifndef CONVOLUTIONALLAYER_CUH_
#define CONVOLUTIONALLAYER_CUH_

#define BLOCK_SIZE 1024

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


void call_foward_parallel(float* input_buf, float* weight_buf, float* b_buf,
		float* output_buf, int in_width, int in_height, int in_depth,
		int out_width, int out_height, int out_depth, int kernel_size);

void call_forward_batch(float* input_batch_buf, float* weight_buf, float* b_buf,
		float* output_batch_buf, int in_width, int in_height, int in_depth,
		int out_width, int out_height, int out_depth, int kernel_size,
		int batch_size);

void call_forward_batch_more(float* input_batch_buf, float* weight_buf,
		float* b_buf, float* output_batch_buf, int in_width, int in_height,
		int in_depth, int out_width, int out_height, int out_depth,
		int kernel_size, int batch_size);
#endif /* CONVOLUTIONALLAYER_CUH_ */
