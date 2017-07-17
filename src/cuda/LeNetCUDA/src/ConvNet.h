/*
 * ConvNet.h
 *
 *  Created on: May 26, 2017
 *      Author: carol
 */

#ifndef CONVNET_H_
#define CONVNET_H_

//for test radiation
#include <list>

#include "Util.h"
#include "Layer.h"
#include "ConvolutionalLayer.h"
#include "MNISTParser.h"
#include "MaxpoolingLayer.h"
#include "OutputLayer.h"
#include "FullyConnectedLayer.h"
#ifdef GPU
#include "DeviceVector.h"
#endif

#define MAX_ITER 10  // maximum training iterations
#define M 10 // training sample counts in each iteration
#define END_CONDITION 1e-3

class ConvNet {
public:
	void train(vec2d_t train_x, vec_host train_y, size_t train_size);
//	void test(vec2d_t test_x, vec_t test_y, size_t test_size, int batch_size);
	void test(vec2d_t test_x, vec_host test_y, size_t test_size);

	std::list<std::pair<size_t, bool>> get_predicted_output();

	void add_layer(Layer* layer);
	void load_weights(std::string path);
	void save_weights(std::string path);

private:

#ifdef GPU
	size_t max_iter(DeviceVector<float> v);

#else
	size_t max_iter(vec_host v);
#endif

	size_t max_iter(float v[], size_t size);
	bool test_once_random();

	bool test_once(int test_x_index);

//	int test_once_random_batch(int batch_size);
//	int test_once_batch(int test_x_index, int batch_size);
//
//	bool check_batch_result(int batch_size);

	float_t train_once();

	int getSumLeNetWeights();
	int getSquaredSumLeNetWeights();

	std::vector<Layer*> layers;

	size_t train_size_;
	vec2d_t train_x_;
	vec_host train_y_;

	size_t test_size_;
	vec2d_t test_x_;
	vec_host test_y_;
	Timer mark;

	//this will save all predicted results
	std::list<std::pair<size_t, bool>> saved_output;

};

#endif /* CONVNET_H_ */
