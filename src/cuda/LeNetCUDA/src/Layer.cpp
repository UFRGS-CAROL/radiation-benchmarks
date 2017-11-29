/*
 * Layer.cpp
 *
 *  Created on: Jun 5, 2017
 *      Author: carol
 */

#include "Layer.h"
#include "LayerKernel.h"

Layer::Layer(size_t in_width, size_t in_height, size_t in_depth,
		size_t out_width, size_t out_height, size_t out_depth, float_t alpha,
		float_t lambda) :
		in_width_(in_width), in_height_(in_height), in_depth_(in_depth), out_width_(
				out_width), out_height_(out_height), out_depth_(out_depth), alpha_(
				alpha), lambda_(lambda) {

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

float_t Layer::getWeightsSum() {
	//funcao para a regularizacao L1
	float_t sum = 0.0;
	float_t weightsSize = this->W_.size();
#ifdef NOTUNIFIEDMEMORY
	this->W_.pop_vector();
#endif
	for (int i = 0; i < weightsSize; i++) {
		sum += std::abs(this->W_[i]);
	}
	return sum;
}

float_t Layer::getSquaredWeightsSum() {
	//funcao para a regularizacao L2
	float_t sum = 0;
	float_t weightsSize = this->W_.size();
#ifdef NOTUNIFIEDMEMORY
	this->W_.pop_vector();
#endif
	for (int i = 0; i < weightsSize; i++) {
		sum += this->W_[i] * this->W_[i];
	}
	//debug layer weights
	//std::cout << "\n debug getSquaredWeightsSum()  " << sum;
	return sum;
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

void Layer::set_sum_LeNet_squared_weights(float_t sum_Lenet_squared_weights) {
	std::cout << "CONCEPTUAL ERROR: " << sum_Lenet_squared_weights << "\n";
}

void Layer::set_sum_LeNet_weights(float_t sum_Lenet_weights) {
	std::cout << "CONCEPTUAL ERROR: " << sum_Lenet_weights << "\n";
}

void Layer::back_prop_L1() {
}

void Layer::back_prop_L2() {
}

void Layer::print_layer_weights(int layer_num) {
	std::cout << "\n Printing Layer: " << layer_num << std::endl;
	float_t weightsSize = this->W_.size();
#ifdef NOTUNIFIEDMEMORY
	this->W_.pop_vector();
#endif
	for (int i = 0; i < weightsSize; i++) {
		std::cout << this->W_[i] << ", ";
	}
	std::cout << "---";
}


bool Layer::gradient_check(){
	bool is_backprop_ok = false;
#ifdef GPU
	//must be the len(W) + len(b)
	DeviceVector<float_t> theta(this->W_.size() + this->b_.size());

	//Initialized host memory
#ifdef NOTUNIFIEDMEMORY
	theta.pop_vector();
#endif

	for(int i = 0; i < this->W_.size(); i++){
		theta[i] = this->W_[i];
		theta[i + 1] = this->b_[i];
	}

	DeviceVector<float_t> theta_minus_vector(theta);
	DeviceVector<float_t> theta_plus_vector(theta);
	DeviceVector<float_t> gradient_diff_vector(theta.size());
	vector<float_t> host_gradient_diff_vector(theta.size());

	float *theta_plus =  theta_minus_vector.data();
	float *theta_minus = theta_plus_vector.data();
	float *d_vector = deltaW_.data();
	float *gradient_diff = gradient_diff_vector.data();
	float *host_gradient_diff = host_gradient_diff_vector.data();

	isis_backprop_ok = call_gradient_check(theta_plus, theta_minus, d_vector,
			gradient_diff, host_gradient_diff, theta.size());


#ifdef NOTUNIFIEDMEMORY
	theta.push_vector();
#endif



#else
	std::cout << "NOT IMPLEMENTED FUNCTION\n";
	is_backprop_ok = true;
#endif
	return is_backprop_ok;

}
