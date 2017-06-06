/*
 * OutputLayer.h
 *
 *  Created on: May 26, 2017
 *      Author: carol
 */

#ifndef OUTPUTLAYER_H_
#define OUTPUTLAYER_H_

#include "Layer.h"


//namespace convnet {

class OutputLayer: public Layer {
public:
	OutputLayer(size_t in_depth);
	void forward_cpu();

	void forward_batch(int batch_size);
	void back_prop();
	void init_weight();

};

//} /* namespace convnet */

#endif /* OUTPUTLAYER_H_ */
