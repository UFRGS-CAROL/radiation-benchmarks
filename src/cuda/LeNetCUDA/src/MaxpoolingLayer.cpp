/*
 * MaxpoolingLayer.cpp
 *
 *  Created on: 05/06/2017
 *      Author: fernando
 */

#include "MaxpoolingLayer.h"

MaxpoolingLayer::MaxpoolingLayer(size_t in_width, size_t in_height,
		size_t in_depth) :
		Layer(in_width, in_height, in_depth, in_width / 2, in_height / 2,
				in_depth, 0, 0) {
	output_.resize(out_depth_ * out_width_ * out_height_);
}

void MaxpoolingLayer::forward_cpu() {
	for (size_t out = 0; out < out_depth_; out++) {
		for (size_t h_ = 0; h_ < in_height_; h_ += 2) {
			for (size_t w_ = 0; w_ < in_width_; w_ += 2) {
				output_[getOutIndex(out, h_, w_)] = max_In_(out, h_, w_,
						getOutIndex(out, h_, w_));
			}
		}
	}
}

void MaxpoolingLayer::forward_batch(int batch_size) {
	output_batch_.resize(batch_size * out_depth_ * out_width_ * out_height_);
	for (size_t batch = 0; batch < batch_size; batch++) {
		for (size_t out = 0; out < out_depth_; out++) {
			for (size_t h_ = 0; h_ < in_height_; h_ += 2) {
				for (size_t w_ = 0; w_ < in_width_; w_ += 2) {
					output_batch_[getOutIndex_batch(batch, out, h_, w_)] =
							max_In_batch_(batch, out, h_, w_);
				}
			}
		}
	}
}
/*
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
	for (auto pair : max_loc)
		g_[pair.second] = this->next->g_[pair.first];
}

void MaxpoolingLayer::init_weight() {
}

//private:
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
				max_loc[out_index] = tmp;
			}
		}
	}
	return max_pixel;
}

inline float_t MaxpoolingLayer::max_In_batch_(size_t batch, size_t in_index,
		size_t h_, size_t w_) {
	float_t max_pixel = 0;
	size_t tmp;
	for (size_t x = 0; x < 2; x++) {
		for (size_t y = 0; y < 2; y++) {
			tmp = (batch * in_depth_ * in_width_ * in_height_)
					+ (in_index * in_width_ * in_height_)
					+ ((h_ + y) * in_width_) + (w_ + x);
			if (max_pixel < input_batch_[tmp]) {
				max_pixel = input_batch_[tmp];
			}
		}
	}
	return max_pixel;
}

inline size_t MaxpoolingLayer::getOutIndex(size_t out, size_t h_, size_t w_) {
	return out * out_width_ * out_height_ + h_ / 2 * out_width_ + (w_ / 2);
}

inline size_t MaxpoolingLayer::getOutIndex_batch(size_t batch, size_t out,
		size_t h_, size_t w_) {
	return batch * out_depth_ * out_width_ * out_height_
			+ out * out_width_ * out_height_ + h_ / 2 * out_width_ + (w_ / 2);
}

