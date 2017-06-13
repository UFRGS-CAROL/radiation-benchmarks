/*
 * FullyConnectedLayer.h
 *
 *  Created on: May 26, 2017
 *      Author: carol
 */

#ifndef FULLYCONNECTEDLAYER_H_
#define FULLYCONNECTEDLAYER_H_

#include "Layer.h"

class FullyConnectedLayer: public Layer {
public:
	FullyConnectedLayer(size_t in_depth, size_t out_depth);

	void back_prop() ;

	/*
	 for the activation sigmod,
	 weight init as [-4 * (6 / sqrt(fan_in + fan_out)), +4 *(6 / sqrt(fan_in + fan_out))]:
	 see also:http://deeplearning.net/tutorial/references.html#xavier10
	 */
	void init_weight() ;

	void save_layer(FILE *of);
	void load_layer(FILE *in);


private:
	vec_host get_W(size_t index);
	vec_host get_W_step(size_t in);

	void forward_cpu();
	void forward_gpu();
};


#endif /* FULLYCONNECTEDLAYER_H_ */
