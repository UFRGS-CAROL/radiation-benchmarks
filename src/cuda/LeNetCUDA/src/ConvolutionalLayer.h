/*
 * ConvolutionalLayer.h
 *
 *  Created on: May 26, 2017
 *      Author: carol
 */

#ifndef CONVOLUTIONALLAYER_H_
#define CONVOLUTIONALLAYER_H_

#include "Layer.h"
#include <vector>
#include "Util.h" //class util

class ConvolutionalLayer: public Layer {
public:
	ConvolutionalLayer(size_t in_width, size_t in_height, size_t in_depth,
			size_t kernel_size, size_t out_depth);

	void init_weight();
	void save_layer(FILE *of);
	void load_layer(FILE *in);

	void forward();
	void back_prop();

private:

	inline size_t getOutIndex(size_t out, size_t h_, size_t w_);

	inline vec_host getInforKernel(size_t in, size_t h_, size_t w_);

	inline vec_host getW_(size_t in, size_t out);

	inline int getb_(size_t out, size_t h_, size_t w_);

	/*
	 2-dimension convoluton:

	 1 2 3                    1 -1 0
	 3 4 2  conv with kernel  -1 0 1
	 2 1 3                    1  1 0

	 ---->
	 1*0 + 2*1 + 3*1 + 3*1 + 4*0 + 2*-1 + 2*0 + 1*-1 + 3*1
	 return the sum.

	 see also:
	 */
	float_t conv(vec_host a, vec_host b);
	size_t kernel_size_;

#ifdef GPU

	void call_foward_parallel(float* input_buf, float* weight_buf, float* b_buf,
			float* output_buf, int in_width, int in_height, int in_depth,
			int out_width, int out_height, int out_depth, int kernel_size);

	void call_backpropagation_parallel(float *W_, //weights
			float *g_,//err array
			float *input_,//input array
			float *g_next,//b_next from this->next->g_
			float *deltaW,//deltaW array
			float *b_,//b_ vector
			float alpha,//alpha value
			float lambda,//lambda value
			int out_depth,//size of the first for loop
			int in_depth_,//size of the second for loop
			int out_width,//size of the third for loop
			int out_height_,// size of loop
			int kernel_size_,//size of loop
			int in_width_,//width size
			int in_height_//in height
	);
#endif

};
#endif /* CONVOLUTIONALLAYER_H_ */
