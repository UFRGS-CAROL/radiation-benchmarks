/*
 * OutputLayerKernel.h
 *
 *  Created on: Jun 20, 2017
 *      Author: carol
 */

#ifndef OUTPUTLAYERKERNEL_H_
#define OUTPUTLAYERKERNEL_H_


//void call_forward_output_layer(float *err, float *exp_y_vec, float *input_,
//		float *reduce_output, float *output_, int in_depth_, int exp_y);
void call_forward_output_layer(float *exp_y_vec, float *input_,
		float *reduce_output, float *output_, int in_depth_, int exp_y);

void call_backpropagation_output_layer(float *exp_y_vec, float *input_,
		float *g_, int in_depth_);



#endif /* OUTPUTLAYERKERNEL_H_ */
