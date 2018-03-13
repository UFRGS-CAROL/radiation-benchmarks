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
#include "Layer.h"

class MaxpoolingLayer: public Layer {
private:
	Pair get_max_loc_pair(size_t first, size_t second);
	int deb = 0;

#ifdef GPU
	void call_forward_maxpool_layer_gpu(float_t *input, float_t *output,
			Pair *max_loc, size_t out_width, size_t out_height, size_t out_depth,
			size_t in_height, size_t in_width);

	void call_backpropagation_maxpool(Pair *max_loc, float *g_, float *g_next, size_t max_size, size_t g_max_size);
#endif

public:
	MaxpoolingLayer(size_t in_width, size_t in_height, size_t in_depth);

	void save_layer(FILE *of);
	void load_layer(FILE *in);
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
	void forward();

	//private:
	inline float_t max_In_(size_t in_index, size_t h_, size_t w_,
			size_t out_index);

	inline size_t getOutIndex(size_t out, size_t h_, size_t w_);

	/*
	 for each output, I store the connection index of the input,
	 which will be used in the back propagation,
	 for err translating.
	 */
//	std::unordered_map<size_t, size_t> max_loc;
#ifdef GPU
	DeviceVector<Pair> max_loc;
#else
	unordered_vec max_loc;
#endif
};

#endif /* MAXPOOLINGLAYER_H_ */
