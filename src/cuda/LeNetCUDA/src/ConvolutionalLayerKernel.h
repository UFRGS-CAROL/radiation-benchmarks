/*
 * ConvolutionalLayer.cuh
 *
 *  Created on: May 26, 2017
 *      Author: carol
 */

#ifndef CONVOLUTIONALLAYER_CUH_
#define CONVOLUTIONALLAYER_CUH_

#define CONV_KERNEL_SIZE 25

void call_foward_parallel(float* input_buf, float* weight_buf, float* b_buf,
		float* output_buf, int in_width, int in_height, int in_depth,
		int out_width, int out_height, int out_depth, int kernel_size);

void call_backpropagation_parallel(float *W_, //weights
		float *g_, //err array
		float *input_, //input array
		float *g_next, //b_next from this->next->g_
		float *deltaW, //deltaW array
		float *b_,  //b_ vector
		float alpha, //alpha value
		float lambda, //lambda value
		int out_depth, //size of the first for loop
		int in_depth_, //size of the second for loop
		int out_width, //size of the third for loop
		int out_height_, // size of loop
		int kernel_size_, //size of loop
		int in_width_, //width size
		int in_height_ //in height
		);

#endif /* CONVOLUTIONALLAYER_CUH_ */
