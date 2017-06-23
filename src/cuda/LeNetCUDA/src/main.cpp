/*
 * main.cpp
 *
 *  Created on: May 26, 2017
 *      Author: carol
 */

#include <cstdlib>
#include "ConvNet.h"

using namespace std;

void create_lenet(ConvNet *net){
	net->add_layer(new ConvolutionalLayer(32, 32, 1, 5, 6));
	net->add_layer(new MaxpoolingLayer(28, 28, 6));
	net->add_layer(new ConvolutionalLayer(14, 14, 6, 5, 16));
	net->add_layer(new MaxpoolingLayer(10, 10, 16));
	net->add_layer(new ConvolutionalLayer(5, 5, 16, 5, 100));
	net->add_layer(new FullyConnectedLayer(100, 10));
}

void classify(MNISTParser& m, string weigths) {
	m.load_testing();

	vec2d_t test_x;
	vec_host test_y;
	for (size_t i = 0; i < 10000; i++) {
		Sample *s = m.get_sample(i);
		test_x.push_back(s->image);
		test_y.push_back(s->label);
	}
	ConvNet n;
	create_lenet(&n);
	//need to load network configurations here
	n.load_weights(weigths);

	int test_sample_count = 5;
	printf("Testing with %d samples:\n", test_sample_count);
	n.test(test_x, test_y, test_sample_count);
}

void train(MNISTParser& m, string weigths) {
	m.load_training();

	int imgs = 10000;
	vec2d_t test_x;
	vec_host test_y;
	for (size_t i = 0; i < imgs; i++) {
		Sample *temp = m.get_sample(i);
		test_x.push_back(temp->image);
		test_y.push_back(temp->label);
	}
	ConvNet n;
	create_lenet(&n);
	n.train(test_x, test_y, 10000);
	n.save_weights(weigths);
}

inline void usage(char **argv) {
	cout << "usage: " << argv[0] << " <train\\classify> <dataset> <labels> <weights>\n";
}

int main(int argc, char **argv) {
	if (argc < 5) {
		usage(argv);
		return EXIT_FAILURE;
	}

	string mode(argv[1]);
	string input_data(argv[2]);
	string input_labels(argv[3]);
	string weigths(argv[4]);

	if (mode == "train") {
		//if train training and labels must be passed
		MNISTParser m(input_data.c_str(), input_labels.c_str(), true);
		cout << "Training for " << m.get_test_img_fname() << std::endl;
		train(m, weigths);
	} else if (mode == "classify") {
		//if train classifing and labels must be passed
		MNISTParser m(input_data.c_str(), input_labels.c_str(), false);
		cout << "Classifing for " << m.get_test_img_fname() << std::endl;
		classify(m, weigths);
	} else {
		usage(argv);
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}
