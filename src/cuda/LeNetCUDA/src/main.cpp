/*
 * main.cpp
 *
 *  Created on: May 26, 2017
 *      Author: carol
 */

#include <cstdlib>
#include "ConvNet.h"

#include "LogsProcessing.h"

using namespace std;

void create_lenet(ConvNet *net) {
	net->add_layer(new ConvolutionalLayer(32, 32, 1, 5, 6));
	net->add_layer(new MaxpoolingLayer(28, 28, 6));
	net->add_layer(new ConvolutionalLayer(14, 14, 6, 5, 16));
	net->add_layer(new MaxpoolingLayer(10, 10, 16));
	net->add_layer(new ConvolutionalLayer(5, 5, 16, 5, 100));
	net->add_layer(new FullyConnectedLayer(100, 10));
}

void classify_radiation_test(MNISTParser& m, string weigths,
		string gold_input) {
//	start_count_app(gold_input.c_str(), "cudaDarknet")
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

void classify_gold_generate(MNISTParser& m, string weigths, string gold_output,
		int test_sample_count) {
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

	printf("Generating gold with %d samples:\n", test_sample_count);
	n.test(test_x, test_y, test_sample_count);

	std::ofstream gold_output_file(gold_output);

	if (gold_output_file.is_open()) {
		std::list<std::pair<size_t, bool>> output = n.get_predicted_output();
		//write gold info

		//test input
		gold_output_file << m.get_test_img_fname() << " "
				<< m.get_test_lbl_fname() << " ";
		//test size
		gold_output_file << weigths << " " << test_sample_count << "\n";

		//write the output
		for (std::pair<size_t, bool> p : output) {
			gold_output_file << p.first << " " << p.second << "\n";
		}

		gold_output_file.close();

	} else {
		printf("ERROR: On opening %s file\n", gold_output.c_str());
		exit(-1);
	}
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
	n.train(test_x, test_y);
	n.save_weights(weigths);
}

inline void usage(char **argv) {
	cout << "usage: " << argv[0]
			<< " <train\\classify\\gold_gen\\rad_test> <dataset> <labels> <weights>	"
					"[gold input/output only for gold_gen and rad_test] "
					"[sample_count only for gold_gen and rad_test]\n";
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

	string gold_in_out;
	int sample_count;
	if (argc > 5) {
		gold_in_out = argv[5];
		sample_count = atoi(argv[6]);
	}
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
	} else if (mode == "gold_gen") {
		MNISTParser m(input_data.c_str(), input_labels.c_str(), false);
		cout << "Generating gold for " << m.get_test_img_fname() << std::endl;
		classify_gold_generate(m, weigths, gold_in_out, sample_count);
	} else if (mode == "rad_test") {
		cout << "To do\n";
	} else {
		usage(argv);
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}
