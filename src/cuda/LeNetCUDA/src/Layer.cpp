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
void Layer::save_base_layer(std::ofstream& of){
	of.write((char*)this->in_width_, sizeof(size_t));
	of.write((char*)this->in_height_, sizeof(size_t));
	of.write((char*)this->in_depth_, sizeof(size_t));
	of.write((char*)this->out_width_, sizeof(size_t));
	of.write((char*)this->out_height_, sizeof(size_t));
	of.write((char*)this->out_depth_, sizeof(size_t));
	of.write(reinterpret_cast<const char*>(&this->alpha_), sizeof(float_t));
	of.write(reinterpret_cast<const char*>(&this->lambda_), sizeof(float_t));
	of.write(reinterpret_cast<const char*>(&this->err), sizeof(float_t));
	of.write(reinterpret_cast<const char*>(&this->exp_y), sizeof(int));
	of.write((char*)&this->W_[0], this->W_.size() * sizeof(float_t));
	of.write((char*)&this->b_[0], this->b_.size() * sizeof(float_t));
	of.write((char*)&this->deltaW_[0], this->deltaW_.size() * sizeof(float_t));
	of.write((char*)&this->input_[0], this->input_.size() * sizeof(float_t));
	of.write((char*)&this->output_[0], this->output_.size() * sizeof(float_t));
	of.write((char*)&this->g_[0], this->g_.size() * sizeof(float_t));
	of.write((char*)&this->exp_y_vec[0], this->exp_y_vec.size() * sizeof(float_t));
}

#ifdef GPU
float* Layer::get_raw_vector(vec_t_gpu th) {
	return thrust::raw_pointer_cast(th.data());
}
#endif
