/*
 * ActivationLayer.h
 *
 *  Created on: Mar 22, 2018
 *      Author: carol
 */

#ifndef ACTIVATIONLAYER_H_
#define ACTIVATIONLAYER_H_

#include "Layer.h"

/**
 * This code was copied from https://github.com/pjreddie/darknet
 * For research purpose only
 */
typedef enum{
    LOGISTIC, RELU, RELIE, LINEAR, RAMP, TANH, PLSE, LEAKY, ELU, LOGGY, STAIR, HARDTAN, LHTAN
}ACTIVATION;


class ActivationLayer: public Layer {
public:
	ActivationLayer();
	virtual ~ActivationLayer();

	void back_prop();
	void forward();
};

#endif /* ACTIVATIONLAYER_H_ */
