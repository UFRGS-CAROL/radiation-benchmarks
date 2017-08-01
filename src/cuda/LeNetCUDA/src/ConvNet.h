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

#include "LogsProcessing.h"

#define MAX_ITER 30  // maximum training iterations
#define M 5 // training sample counts in each iteration
#define END_CONDITION 1e-3


class ConvNet {
public:
	void train(vec2d_t train_x, vec_host train_y, size_t train_size);

	void train(vec2d_t train_x, vec_host train_y, char normalization);

	void test(vec2d_t test_x, vec_host test_y,
			std::vector<std::pair<size_t, bool>> gold_list, //gold for radiation test
			size_t iterations, bool save_layer, int sample_count);

	void test(vec2d_t test_x, vec_host test_y, size_t test_size, bool save_layer = false);


	std::list<std::pair<size_t, bool>> get_predicted_output();

	void add_layer(Layer* layer);
	void load_weights(std::string path);
	void load_weights(FILE *in);
	void save_weights(std::string path, std::string file_mode = "wb");


	std::vector<Layer*> get_layers();

	void print_all_layer_weight_sums();
	void print_all_layer_weights();
	void print_sum_weights();

private:

#ifdef GPU
	size_t max_iter(DeviceVector<float> v);
	LayersFound layers_output;
#else
	size_t max_iter(vec_host v);
	LayersFound layers_output;
#endif

	size_t max_iter(float v[], size_t size);
	bool test_once_random();

	bool test_once(int test_x_index);

	std::pair<size_t, bool> test_once_pair(int test_x_index);

//	int test_once_random_batch(int batch_size);
//	int test_once_batch(int test_x_index, int batch_size);
//
//	bool check_batch_result(int batch_size);

	float_t train_once();

	float_t getSumLeNetWeights();
	float_t getSquaredSumLeNetWeights();


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
