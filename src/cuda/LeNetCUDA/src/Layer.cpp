/*
 * Layer.cpp
 *
 *  Created on: Jun 5, 2017
 *      Author: carol
 */

#include "Layer.h"

#ifdef GPU
#include <thrust/device_vector.h>
#endif

Layer::Layer(size_t in_width, size_t in_height, size_t in_depth,
		size_t out_width, size_t out_height, size_t out_depth, float_t alpha,
		float_t lambda) :
		in_width_(in_width), in_height_(in_height), in_depth_(in_depth), out_width_(
				out_width), out_height_(out_height), out_depth_(out_depth), alpha_(
				alpha), lambda_(lambda) {

}

void Layer::forward() {
#ifdef GPU
	this->forward_gpu();
#else
	this->forward_cpu();
#endif

}

float_t Layer::sigmod(float_t in) {
	return 1.0 / (1.0 + std::exp(-in));
}

float_t Layer::df_sigmod(float_t f_x) {
	return f_x * (1.0 - f_x);
}

size_t Layer::fan_in() {
	return in_width_ * in_height_ * in_depth_;
}

size_t Layer::fan_out() {
	return out_width_ * out_height_ * out_height_;
}

/**
 * 	size_t in_width_;
 size_t in_height_;
 size_t in_depth_;
 size_t out_width_;
 size_t out_height_;
 size_t out_depth_;
 float_t alpha_; // learning rate
 float_t lambda_; // momentum
 float_t err;
 int exp_y;
 vec_t W_;
 vec_t b_;
 vec_t deltaW_;
 vec_t input_;
 vec_t output_;
 vec_t g_; // err terms
 vec_t exp_y_vec;

 //Layer* parameter I did not save
 * it will be set on weights loading
 */

void Layer::save_base_layer(FILE *of) {
	this->write_layer_var<size_t>(this->in_width_, of);
	this->write_layer_var<size_t>(this->in_height_, of);
	this->write_layer_var<size_t>(this->in_depth_, of);
	this->write_layer_var<size_t>(this->out_width_, of);
	this->write_layer_var<size_t>(this->out_height_, of);
	this->write_layer_var<size_t>(this->out_depth_, of);
	this->write_layer_var<float_t>(this->alpha_, of);
	this->write_layer_var<float_t>(this->lambda_, of);
	this->write_layer_var<float_t>(this->err, of);
	this->write_layer_var<int>(this->exp_y, of);

	//vector attributes
	this->write_layer_vec<float_t>(this->W_, of);
	this->write_layer_vec<float_t>(this->b_, of);
	this->write_layer_vec<float_t>(this->deltaW_, of);
	this->write_layer_vec<float_t>(this->input_, of);
	this->write_layer_vec<float_t>(this->output_, of);
	this->write_layer_vec<float_t>(this->g_, of);
	this->write_layer_vec<float_t>(this->exp_y_vec, of);
}

void Layer::load_base_layer(FILE *in) {
	this->in_width_ = this->load_layer_var<size_t>(in);
	this->in_height_ = this->load_layer_var<size_t>(in);
	this->in_depth_ = this->load_layer_var<size_t>(in);
	this->out_width_ = this->load_layer_var<size_t>(in);
	this->out_height_ = this->load_layer_var<size_t>(in);
	this->out_depth_ = this->load_layer_var<size_t>(in);
	this->alpha_ = this->load_layer_var<float_t>(in);
	this->lambda_ = this->load_layer_var<float_t>(in);
	this->err = this->load_layer_var<float_t>(in);
	this->exp_y = this->load_layer_var<int>(in);

	//vector attributes
	this->W_ = this->load_layer_vec<float_t>(in);
	this->b_ = this->load_layer_vec<float_t>(in);
	this->deltaW_ = this->load_layer_vec<float_t>(in);
	this->input_ = this->load_layer_vec<float_t>(in);
	this->output_ = this->load_layer_vec<float_t>(in);
	this->g_ = this->load_layer_vec<float_t>(in);
	this->exp_y_vec = this->load_layer_vec<float_t>(in);

}

#ifdef GPU
float* Layer::get_raw_vector(vec_t_gpu th) {
	return thrust::raw_pointer_cast(th.data());
}
#endif
