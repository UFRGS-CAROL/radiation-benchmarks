/*
 * main.cpp
 *
 *  Created on: May 26, 2017
 *      Author: carol
 */

#include <cstdlib>
#include "ConvNet.h"

using namespace std;

void classify(MNISTParser& m) {
	m.load_testing();

	vec2d_t x;
	vec_t y;
	vec2d_t test_x;
	vec_t test_y;
	for (size_t i = 0; i < 10000; i++) {
		test_x.push_back(m.test_sample[i]->image);
		test_y.push_back(m.test_sample[i]->label);
	}
	ConvNet n;
	n.add_layer(new ConvolutionalLayer(32, 32, 1, 5, 6));
	n.add_layer(new MaxpoolingLayer(28, 28, 6));
	n.add_layer(new ConvolutionalLayer(14, 14, 6, 5, 16));
	n.add_layer(new MaxpoolingLayer(10, 10, 16));
	n.add_layer(new ConvolutionalLayer(5, 5, 16, 5, 100));
	n.add_layer(new FullyConnectedLayer(100, 10));
	n.train(test_x, test_y, 10000);
	int test_sample_count = 5;
	//Sleep(1000);
	printf("Testing with %d samples:\n", test_sample_count);
	const clock_t begin_time = clock();
	n.test(test_x, test_y, test_sample_count, 5);
	cout << "Time consumed in test: "
			<< float(clock() - begin_time) / (CLOCKS_PER_SEC / 1000) << " ms"
			<< endl;
}

void train(MNISTParser& m) {
	m.load_training();
	vec2d_t x;
	vec_t y;
	vec2d_t test_x;
	vec_t test_y;
	for (size_t i = 0; i < 10000; i++) {
		test_x.push_back(m.test_sample[i]->image);
		test_y.push_back(m.test_sample[i]->label);
	}
	ConvNet n;
	n.add_layer(new ConvolutionalLayer(32, 32, 1, 5, 6));
	n.add_layer(new MaxpoolingLayer(28, 28, 6));
	n.add_layer(new ConvolutionalLayer(14, 14, 6, 5, 16));
	n.add_layer(new MaxpoolingLayer(10, 10, 16));
	n.add_layer(new ConvolutionalLayer(5, 5, 16, 5, 100));
	n.add_layer(new FullyConnectedLayer(100, 10));
	n.train(test_x, test_y, 10000);
	int test_sample_count = 5;
	//Sleep(1000);
	printf("Testing with %d samples:\n", test_sample_count);
	const clock_t begin_time = clock();
	n.test(test_x, test_y, test_sample_count, 5);
	cout << "Time consumed in test: "
			<< float(clock() - begin_time) / (CLOCKS_PER_SEC / 1000) << " ms"
			<< endl;
}

inline void usage(char **argv) {
	cout << "usage: " << argv[0] << " <train\\classify> <dataset>\n";
}

int main(int argc, char **argv) {
	if (argc < 3) {
		usage(argv);
		return EXIT_FAILURE;
	}

	string mode = string(argv[1]);
	string input_data = string(argv[2]);

	if (mode == "train") {

		MNISTParser m(input_data.c_str());
		cout << "Training for " << m.get_test_img_fname() << std::endl;
		train(m);
	} else if (mode == "classify") {
		MNISTParser m(input_data.c_str());
		cout << "Classifing for " << m.get_test_img_fname() << std::endl;
		classify(m);
	} else {
		usage(argv);
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}
