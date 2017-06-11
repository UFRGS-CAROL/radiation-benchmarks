/*
 * Layer.h
 *
 *  Created on: May 26, 2017
 *      Author: carol
 */

#ifndef LAYER_H_
#define LAYER_H_

#include <vector>
#include "Util.h"
#include <thrust/device_vector.h>

class Layer {
public:
	Layer(size_t in_width, size_t in_height, size_t in_depth, size_t out_width,
			size_t out_height, size_t out_depth, float_t alpha, float_t lambda) ;

	virtual void init_weight() = 0;
	virtual void forward_cpu() = 0;
	virtual void forward_batch(int batch_size) = 0;
	virtual void back_prop() = 0;

	void forward_gpu();

	float_t sigmod(float_t in);

	float_t df_sigmod(float_t f_x);

	size_t fan_in();

	size_t fan_out();

	size_t in_width_;
	size_t in_height_;
	size_t in_depth_;

	size_t out_width_;
	size_t out_height_;
	size_t out_depth_;

	vec_t W_;
	vec_t b_;

	vec_t deltaW_;

	vec_t input_;
	vec_t output_;

	vec_t input_batch_;
	vec_t output_batch_;

	Layer* next;

	float_t alpha_; // learning rate
	float_t lambda_; // momentum
	vec_t g_; // err terms

	/*output*/
	float_t err;
	int exp_y;
	vec_t exp_y_vec;

	vec_t exp_y_batch;
	vec_t exp_y_vec_batch;



	thrust::device_vector<float> input_buf;
	thrust::device_vector<float> weight_buf;
	thrust::device_vector<float> output_buf;
	thrust::device_vector<float> b_buf;

	float *get_raw_vector(thrust::device_vector<float> th);

};

//} /* namespace convnet */

#endif /* LAYER_H_ */
