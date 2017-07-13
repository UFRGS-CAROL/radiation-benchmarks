/*
 * FullyConnectedLayerKernel.h
 *
 *  Created on: Jun 6, 2017
 *      Author: carol
 */

#ifndef FULLYCONNECTEDLAYERKERNEL_H_
#define FULLYCONNECTEDLAYERKERNEL_H_


void call_forward_fully_connected(float *output_, float *input_, float *b_,
		float *W_, int out_depth_, int in_depth_, int input_size);


void call_backpropagation_fully_connected(float *input_, float *g_, float *g_next,
		float *deltaW_, float *W_, float *b_, float *r_output,
		float alpha_, float lambda_, int in_depth_, int out_depth_, int g_next_size) ;


#endif /* FULLYCONNECTEDLAYERKERNEL_H_ */
