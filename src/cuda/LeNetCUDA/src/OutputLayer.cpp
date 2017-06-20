/*
 * OutputLayer.cpp
 *
 *  Created on: 05/06/2017
 *      Author: fernando
 */

#include "OutputLayer.h"
#include "Util.h"


OutputLayer::OutputLayer(size_t in_depth) :
		Layer(1, 1, in_depth, 0, 0, 0, 0, 0) {


}

void OutputLayer::init_weight() {
}

/**
 * there is not outputlayer parameters
 */
void OutputLayer::save_layer(FILE *of){
	this->save_base_layer(of);
}

void OutputLayer::load_layer(FILE *in){
	this->load_base_layer(in);
	std::cout << "Inside Output Layer\n";
}


#ifdef GPU

void OutputLayer::forward() {
	this->err = 0;
	exp_y_vec.clear();
	exp_y_vec.resize(in_depth_);
	exp_y_vec[this->exp_y] = 1;
	for (size_t i = 0; i < in_depth_; i++) {
		err += 0.5 * (exp_y_vec[i] - input_[i]) * (exp_y_vec[i] - input_[i]);
	}
	output_ = input_;
}

void OutputLayer::back_prop() {
	/* compute err terms of output layers */
	g_.clear();

	for (size_t i = 0; i < in_depth_; i++) {
		g_.push_back((exp_y_vec[i] - input_[i]) * df_sigmod(input_[i]));

	}
}

#else

void OutputLayer::forward() {
	this->err = 0;
	exp_y_vec.clear();
	exp_y_vec.resize(in_depth_);
	exp_y_vec[this->exp_y] = 1;
	for (size_t i = 0; i < in_depth_; i++) {
		err += 0.5 * (exp_y_vec[i] - input_[i]) * (exp_y_vec[i] - input_[i]);
	}
	output_ = input_;
}

void OutputLayer::back_prop() {
	/* compute err terms of output layers */
	g_.clear();

	for (size_t i = 0; i < in_depth_; i++) {
		g_.push_back((exp_y_vec[i] - input_[i]) * df_sigmod(input_[i]));

	}
}


#endif
