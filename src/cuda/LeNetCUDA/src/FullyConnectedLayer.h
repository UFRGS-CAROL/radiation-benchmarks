/*
 * FullyConnectedLayer.h
 *
 *  Created on: May 26, 2017
 *      Author: carol
 */

#ifndef FULLYCONNECTEDLAYER_H_
#define FULLYCONNECTEDLAYER_H_

#include "Layer.h"


//namespace convnet {

class FullyConnectedLayer: public Layer {
public:
	FullyConnectedLayer(size_t in_depth, size_t out_depth);

	void forward_cpu();

	void forward_batch(int batch_size);

	void back_prop() ;

	/*
	 for the activation sigmod,
	 weight init as [-4 * (6 / sqrt(fan_in + fan_out)), +4 *(6 / sqrt(fan_in + fan_out))]:
	 see also:http://deeplearning.net/tutorial/references.html#xavier10
	 */
	void init_weight() ;
private:
	vec_t get_W(size_t index);
	vec_t get_W_step(size_t in);
};

//} /* namespace convnet */

#endif /* FULLYCONNECTEDLAYER_H_ */
