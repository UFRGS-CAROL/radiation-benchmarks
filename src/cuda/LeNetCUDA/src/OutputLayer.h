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

	virtual void back_prop_L1();
	virtual void back_prop_L2();

	virtual void set_sum_LeNet_weights(float_t sum_Lenet_weights);
	virtual void set_sum_LeNet_squared_weights(
			float_t sum_Lenet_squared_weights);

#ifdef GPU
	DeviceVector<float> reduce_output;
#endif

private:
	float_t lenetWeightsSum;
	float_t lenetSquaredWeightsSum;

#ifdef GPU
//void call_forward_output_layer(float *err, float *exp_y_vec, float *input_,
//		float *reduce_output, float *output_, int in_depth_, int exp_y);
	void call_forward_output_layer(float *exp_y_vec, float *input_,
			float *reduce_output, float *output_, int in_depth_, int exp_y);

	void call_backpropagation_output_layer(float *exp_y_vec, float *input_,
			float *g_, int in_depth_);
#endif
};

#endif /* OUTPUTLAYER_H_ */
