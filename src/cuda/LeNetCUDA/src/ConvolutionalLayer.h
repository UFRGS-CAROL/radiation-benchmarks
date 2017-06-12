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

#ifdef GPU
#include "ConvolutionalLayerKernel.h"
#endif

class ConvolutionalLayer: public Layer {
public:
	ConvolutionalLayer(size_t in_width, size_t in_height, size_t in_depth, 
			size_t kernel_size, size_t out_depth);

	void init_weight();
	void back_prop();
	void save_layer(std::ofstream& of);

private:
	void forward_cpu();
	void forward_gpu();

	inline size_t getOutIndex(size_t out, size_t h_, size_t w_);

	inline vec_t getInforKernel(size_t in, size_t h_, size_t w_);

	inline vec_t getW_(size_t in, size_t out);

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
	float_t conv(vec_t a, vec_t b);
	size_t kernel_size_;



};


#endif /* CONVOLUTIONALLAYER_H_ */
