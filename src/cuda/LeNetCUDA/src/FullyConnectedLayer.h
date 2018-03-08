/*
 * FullyConnectedLayer.h
 *
 *  Created on: May 26, 2017
 *      Author: carol
 */

#ifndef FULLYCONNECTEDLAYER_H_
#define FULLYCONNECTEDLAYER_H_

#include "Layer.h"

#ifdef GPU
#include "DeviceVector.h"
#endif

class FullyConnectedLayer: public Layer {
public:
	FullyConnectedLayer(size_t in_depth, size_t out_depth);

	void back_prop();
	void forward();

	/*
	 for the activation sigmod,
	 weight init as [-4 * (6 / sqrt(fan_in + fan_out)), +4 *(6 / sqrt(fan_in + fan_out))]:
	 see also:http://deeplearning.net/tutorial/references.html#xavier10
	 */
	void init_weight();

	void save_layer(FILE *of);
	void load_layer(FILE *in);

#ifdef GPU
	//this vector will be used only at
	//forward gpu
	DeviceVector<float> v_output;
#else
	vec_host v_output;
#endif

private:

//	DeviceVector<float> get_W_step(size_t in);
//	DeviceVector<float>  get_W(size_t index);
	vec_host get_W_step(size_t in);
	vec_host get_W(size_t index);
#ifdef GPU
	void gradient_checker(DeviceVector<float_t>& original_b, DeviceVector<float_t>& original_w, DeviceVector<float_t>& original_input);

	void call_forward_fully_connected(float *output_, float *input_, float *b_,
			float *W_, int out_depth_, int in_depth_, int input_size);

	void call_backpropagation_fully_connected(float *input_, float *g_,
			float *g_next, float *deltaW_, float *W_, float *b_, float *r_output,
			float alpha_, float lambda_, int in_depth_, int out_depth_,
			int g_next_size);

#endif

	template<typename T> double vector_norm(T v) {
		double n = 0;
		for (int i = 0; i < v.size(); i++)
			n += v[i] * v[i];

		return std::sqrt(n);
	}

};

#endif /* FULLYCONNECTEDLAYER_H_ */
