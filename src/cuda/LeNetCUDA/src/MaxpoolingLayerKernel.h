/*
 * MaxpoolingLayerKernel.h
 *
 *  Created on: Jun 6, 2017
 *      Author: carol
 */

#ifndef MAXPOOLINGLAYERKERNEL_H_
#define MAXPOOLINGLAYERKERNEL_H_
#include "MaxpoolingLayer.h"

void forward_maxpool_layer_gpu(MaxpoolingLayer l);
void backward_maxpool_layer_gpu(MaxpoolingLayer l);


#endif /* MAXPOOLINGLAYERKERNEL_H_ */
