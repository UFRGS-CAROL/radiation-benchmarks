/*
 * Layer.cpp
 *
 *  Created on: Jun 5, 2017
 *      Author: carol
 */

namespace convnet {

Layer::Layer(size_t in_width, size_t in_height, size_t in_depth,
		size_t out_width, size_t out_height, size_t out_depth, float_t alpha,
		float_t lambda) :
		in_width_(in_width), in_height_(in_height), in_depth_(in_depth), out_width_(
				out_width), out_height_(out_height), out_depth_(out_depth), alpha_(
				alpha), lambda_(lambda) {

	//~ this->in_width_ = in_width;
	//~ this->in_height_ = in_height;
	//~ this->in_depth_ = in_depth;
	//~ this->out_width_ = out_width;
	//~ this->out_height_ = out_height;
	//~ this->out_depth_ = out_depth;
	//~ this->alpha_ = alpha;
	//~ this->lambda_ = lambda;
	//~ this->exp_y = 0;
	//~ this->next = NULL;
	//~ this->err = 0;

}

void Layer::forward_gpu() {
	forward_cpu();
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

}        //namespace convnet

