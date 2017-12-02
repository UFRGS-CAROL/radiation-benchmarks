/*
 * LayerKernel.h
 *
 *  Created on: Nov 27, 2017
 *      Author: carol
 */

#ifndef LAYERKERNEL_H_
#define LAYERKERNEL_H_

#define EPSILON 10e-4
#define MAX_ERROR_ALLOWED 10e-5

bool call_gradient_check(float *theta_plus, float *theta_minus, float *d_vector,
		float *gradient_diff, float *host_gradient_diff, int size_n);

#endif /* LAYERKERNEL_H_ */
