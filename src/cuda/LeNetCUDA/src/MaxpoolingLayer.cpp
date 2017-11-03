/*
 * MaxpoolingLayer.cpp
 *
 *  Created on: 05/06/2017
 *      Author: fernando
 */

#include "MaxpoolingLayer.h"

#ifdef GPU
#include "MaxpoolingLayerKernel.h"
#endif



MaxpoolingLayer::MaxpoolingLayer(size_t in_width, size_t in_height,
		size_t in_depth) :
		Layer(in_width, in_height, in_depth, in_width / 2, in_height / 2,
				in_depth, 0, 0) {
	this->output_.resize(
			this->out_depth_ * this->out_width_ * this->out_height_);

	//it is used instead unordered map on GPU
	Pair t;
	t.first = MAX;
	t.second = MAX;
	//this trick guarantee that I use DeviceVector or std::vector
	this->max_loc = std::vector < Pair
			> (this->out_depth_ * this->in_height_ * this->in_width_, t);
	this->layer_type = "maxpool";

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
	this->write_layer_vec<Pair>(this->max_loc, of);
}

void MaxpoolingLayer::load_layer(FILE *in) {
	this->load_base_layer(in);
	this->max_loc = this->load_layer_vec<Pair>(in);
}


///*
// In forward propagation, blocks are reduced to a single value.
// Then, this single value acquires an error computed from backwards
// propagation from the previous layer.
// This error is then just forwarded to the place where it came from.
// Since it only came from one place in the  block,
// the backpropagated errors from max-pooling layers are rather sparse.
// */
//void MaxpoolingLayer::back_prop() {
//
//	g_.clear();
//	g_.resize(in_width_ * in_height_ * in_depth_);
//#ifdef NOTUNIFIEDMEMORY
//	this->next->g_.pop_vector();
//	this->max_loc.pop_vector();
//#endif
//
//	for (size_t i = 0; i < this->max_loc.size(); i++){
//		auto pair = this->max_loc[i];
//		if (pair.first != MAX) {
//			g_[pair.second] = this->next->g_[pair.first];
//		}
//	}
//#ifdef NOTUNIFIEDMEMORY
//	this->g_.push_vector();
//#endif
//
//}


#ifdef GPU

void MaxpoolingLayer::forward() {
// execute the code on the device
		float_t *input = this->input_.data();
		float_t *output = this->output_.data();
		Pair *max_loc_buf = this->max_loc.data();
		size_t out_width = this->out_width_;
		size_t out_height = this->out_height_;
		size_t out_depth = this->out_depth_;
		size_t in_height = this->in_height_;
		size_t in_width = this->in_width_;

		call_forward_maxpool_layer_gpu(input, output, max_loc_buf, out_width,
				out_height, out_depth, in_height, in_width);

}

//#ifdef TRAINGPU

void MaxpoolingLayer::back_prop() {
	g_.clear();
	g_.resize(this->in_width_ * this->in_height_ * this->in_depth_);

	Pair *max_loc = this->max_loc.data();
	float *g_ = this->g_.data();
	float *g_next = this->next->g_.data();
	size_t max_size = this->max_loc.size();
	size_t g_max_size = this->g_.size();

	call_backpropagation_maxpool(max_loc, g_, g_next, max_size, g_max_size);

}
//#endif //TRAINGPU

#else

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

