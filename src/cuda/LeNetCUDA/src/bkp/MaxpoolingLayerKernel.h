/*
 * MaxpoolingLayerKernel.h
 *
 *  Created on: Jun 6, 2017
 *      Author: carol
 */

#ifndef MAXPOOLINGLAYERKERNEL_H_
#define MAXPOOLINGLAYERKERNEL_H_

#include "Util.h"

void call_forward_maxpool_layer_gpu(float_t *input, float_t *output,
		Pair *max_loc, size_t out_width, size_t out_height, size_t out_depth,
		size_t in_height, size_t in_width) ;

void call_backpropagation_maxpool(Pair *max_loc, float *g_, float *g_next, size_t max_size, size_t g_max_size);

#endif /* MAXPOOLINGLAYERKERNEL_H_ */
