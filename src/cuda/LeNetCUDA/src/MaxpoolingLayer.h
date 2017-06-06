/*
 * MaxpoolingLayer.h
 *
 *  Created on: May 26, 2017
 *      Author: carol
 */

#ifndef MAXPOOLINGLAYER_H_
#define MAXPOOLINGLAYER_H_
#include <numeric>
#include <unordered_map>

#include "Util.h"

//#pragma once

#include "Layer.h"

//namespace convnet {

class MaxpoolingLayer: public Layer {
public:
	MaxpoolingLayer(size_t in_width, size_t in_height, size_t in_depth);
	void forward_cpu();

	void forward_batch(int batch_size);
	/*
	 In forward propagation, k��k blocks are reduced to a single value.
	 Then, this single value acquires an error computed from backwards
	 propagation from the previous layer.
	 This error is then just forwarded to the place where it came from.
	 Since it only came from one place in the k��k block,
	 the backpropagated errors from max-pooling layers are rather sparse.
	 */
	void back_prop();
	void init_weight();

	//private:
	inline float_t max_In_(size_t in_index, size_t h_, size_t w_,
			size_t out_index);

	inline float_t max_In_batch_(size_t batch, size_t in_index, size_t h_,
			size_t w_);

	inline size_t getOutIndex(size_t out, size_t h_, size_t w_);

	inline size_t getOutIndex_batch(size_t batch, size_t out, size_t h_,
			size_t w_);
	/*
	 for each output, I store the connection index of the input,
	 which will be used in the back propagation,
	 for err translating.
	 */
	std::unordered_map<size_t, size_t> max_loc;
};

//} /* namespace convnet */

#endif /* MAXPOOLINGLAYER_H_ */
