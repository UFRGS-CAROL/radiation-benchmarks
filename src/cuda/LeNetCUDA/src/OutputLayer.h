/*
 * OutputLayer.h
 *
 *  Created on: May 26, 2017
 *      Author: carol
 */

#ifndef OUTPUTLAYER_H_
#define OUTPUTLAYER_H_

#include "Layer.h"



class OutputLayer: public Layer {
public:
	OutputLayer(size_t in_depth);

	void back_prop();
	void forward();

	void init_weight();

	void save_layer(FILE *of);
	void load_layer(FILE *in);
private:

};

#endif /* OUTPUTLAYER_H_ */
