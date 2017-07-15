/*
 * OutputLayer.h
 *
 *  Created on: May 26, 2017
 *      Author: carol
 */

#ifndef OUTPUTLAYER_H_
#define OUTPUTLAYER_H_

#include "Layer.h"
#ifdef GPU
#include "DeviceVector.h"
#endif


class OutputLayer: public Layer {
public:
	OutputLayer(size_t in_depth);

	void back_prop();
	void forward();

	void init_weight();

	void save_layer(FILE *of);
	void load_layer(FILE *in);

	void back_prop_L1();
	void back_prop_L2();

	void set_sum_LeNet_weights(int sum_Lenet_weights);
	void set_sum_LeNet_squared_weights(int sum_Lenet_squared_weights);

#ifdef GPU
	DeviceVector<float> reduce_output;
#endif

private:
	int lenetWeighsSum;
	int lenetSquaredWeighsSum;

};

#endif /* OUTPUTLAYER_H_ */
