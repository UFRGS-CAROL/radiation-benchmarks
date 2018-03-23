/*
 * MaxpoolingLayer.cpp
 *
 *  Created on: 05/06/2017
 *      Author: fernando
 */

#include "MaxpoolingLayer.h"

MaxpoolingLayer::MaxpoolingLayer(size_t in_width, size_t in_height,
		size_t in_depth) :
		Layer(2, 0, 1, in_width, in_height, in_depth, in_width / 2, in_height / 2,
				in_depth, 0, 0) {
	this->output_.resize(
			this->out_depth_ * this->out_width_ * this->out_height_);

	//it is used instead unordered map on GPU
#ifndef GPU
	Pair t;
	t.first = MAX;
	t.second = MAX;
	//this trick guarantee that I use DeviceVector or std::vector
	this->max_loc = std::vector < Pair
			> (this->out_depth_ * this->in_height_ * this->in_width_, t);
#endif
	this->indexes = std::vector <size_t>(this->out_depth_ * this->out_height_ * this->out_width_);
	this->layer_type = "maxpool";

	this->size = MAXPOOL_SIZE;

}


void MaxpoolingLayer::init_weight() {
}



inline Pair MaxpoolingLayer::get_max_loc_pair(size_t first, size_t second) {
	Pair ret;
	ret.first = first;
	ret.second = second;
	return ret;
}

inline size_t MaxpoolingLayer::getOutIndex(size_t out, size_t h_, size_t w_) {
	return out * out_width_ * out_height_ + h_ / 2 * out_width_ + (w_ / 2);
}

/**
 * 	std::unordered_map<size_t, size_t> max_loc;
 vec_t max_loc_host;
 */
void MaxpoolingLayer::save_layer(FILE *of) {
	this->save_base_layer(of);
#ifndef GPU
	this->write_layer_vec<Pair>(this->max_loc, of);
#endif
	this->write_layer_vec<size_t>(this->indexes, of);
}

void MaxpoolingLayer::load_layer(FILE *in) {
	this->load_base_layer(in);
#ifndef GPU
	this->max_loc = this->load_layer_vec<Pair>(in);
#endif
	this->indexes = this->load_layer_vec<size_t>(in);
}


#ifndef GPU
/*
FULL CPU BACKPROPAGATION
 In forward propagation, blocks are reduced to a single value.
 Then, this single value acquires an error computed from backwards
 propagation from the previous layer.
 This error is then just forwarded to the place where it came from.
 Since it only came from one place in the  block,
 the backpropagated errors from max-pooling layers are rather sparse.
 */
void MaxpoolingLayer::back_prop() {
	g_.clear();
	g_.resize(in_width_ * in_height_ * in_depth_);
	for (size_t i = 0; i < this->max_loc.size(); i++){
		auto pair = this->max_loc[i];
		if (pair.first != MAX) {
			g_[pair.second] = this->next->g_[pair.first];
		}
	}
}


void MaxpoolingLayer::forward() {
	for (size_t out = 0; out < out_depth_; out++) {
		for (size_t h_ = 0; h_ < in_height_; h_ += 2) {
			for (size_t w_ = 0; w_ < in_width_; w_ += 2) {
				output_[getOutIndex(out, h_, w_)] = max_In_(out, h_, w_,
						getOutIndex(out, h_, w_));

			}
		}
	}

}

inline float_t MaxpoolingLayer::max_In_(size_t in_index, size_t h_, size_t w_,
		size_t out_index) {
	float_t max_pixel = 0;
	size_t tmp;
	for (size_t x = 0; x < 2; x++) {
		for (size_t y = 0; y < 2; y++) {
			tmp = (in_index * in_width_ * in_height_) + ((h_ + y) * in_width_)
					+ (w_ + x);
			if (max_pixel < input_[tmp]) {
				max_pixel = input_[tmp];
				max_loc[out_index] = this->get_max_loc_pair(out_index, tmp);
			}
		}
	}
	return max_pixel;
}
#endif

