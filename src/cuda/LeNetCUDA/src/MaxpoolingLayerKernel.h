/*
 * MaxpoolingLayerKernel.h
 *
 *  Created on: Jun 6, 2017
 *      Author: carol
 */

#ifndef MAXPOOLINGLAYERKERNEL_H_
#define MAXPOOLINGLAYERKERNEL_H_


void forward_maxpool_layer_gpu(float_t *input, float_t *output,
		float_t *max_loc, size_t out_width, size_t out_height, size_t out_depth,
		size_t in_height, size_t in_width);

void backward_maxpool_layer_gpu();


#endif /* MAXPOOLINGLAYERKERNEL_H_ */
