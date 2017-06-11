/*
 * ConvolutionalLayer.cuh
 *
 *  Created on: May 26, 2017
 *      Author: carol
 */

#ifndef CONVOLUTIONALLAYER_CUH_
#define CONVOLUTIONALLAYER_CUH_




void call_foward_parallel(float* input_buf, float* weight_buf, float* b_buf,
		float* output_buf, int in_width, int in_height, int in_depth,
		int out_width, int out_height, int out_depth, int kernel_size);


#endif /* CONVOLUTIONALLAYER_CUH_ */
