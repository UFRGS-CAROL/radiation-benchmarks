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

void classify_gold_generate(MNISTParser& m, string weigths, string gold_output,
		int test_sample_count, bool save_layers) {
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
	n.test(test_x, test_y, test_sample_count, save_layers);

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
		error("ERROR: On opening " + gold_output + " file\n");
	}
}

/**
 * classifi for rad test
 */
void classify_test_rad(MNISTParser& m, string weigths, string gold_input,
		bool save_layers, int iterations) {
	string header_line = "gold_file: " + gold_input + " weights: " + weigths
			+ " iterations: " + std::to_string(iterations) + " save_layer: "
			+ std::to_string(save_layers);
	//start log file
	start_count_app(const_cast<char*>(header_line.c_str()),
			const_cast<char*>("cudaLeNET"));
	//-------------------------------------------
	//Main network
	//-------------------------------------------
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
	//-------------------------------------------
	//load golds
	//-------------------------------------------
	ifstream gold_input_file(gold_input);
	vector<pair<size_t, bool>> gold_data;
//  vector < vector<Layer*> > gold_layers;

	string test_img_fname, test_lbl_fname, weigths_read;
	int sample_count;

	if (gold_input_file.is_open()) {
		gold_input_file >> test_img_fname >> test_lbl_fname >> weigths_read
				>> sample_count;

		for (int i = 0; i < sample_count; i++) {
			pair<size_t, bool> p;
			gold_input_file >> p.first >> p.second;
			gold_data.push_back(p);
		}

		gold_input_file.close();
	} else {
		error("ERROR: On opening " + gold_input + " file\n");
	}

	//load golds for layer comparison
	if (save_layers) {
//      FILE *gold_layers_input_file = fopen(
//              (string(SAVE_LAYER_DATA) + "/gold_layers_lenet.layer").c_str(),
//              "rb");
//
//      for (int i = 0; i < sample_count; i++) {
//          if (gold_layers_input_file == NULL) {
//              error("ERROR: On reading layer gold\n");
//          }
//          ConvNet temp_gold;
//          create_lenet(&temp_gold);
//          temp_gold.load_weights(gold_layers_input_file);
//          gold_layers.push_back(temp_gold.get_layers());
//
//      }
//      fclose(gold_layers_input_file);
	}
	cout << "COmeçou a classificação \n";
	//-------------------------------------------
	//Make radiation test
	//-------------------------------------------
	n.test(test_x, test_y, gold_data, iterations, save_layers, sample_count);

	//end log file
	finish_count_app();
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

	//debug weights
	//n.print_all_layer_weights();
	n.print_sum_weights();
	n.print_all_layer_weight_sums();

	int test_sample_count = 1000;
	printf("Testing with %d samples:\n", test_sample_count);
	n.test(test_x, test_y, test_sample_count);
}

void train(MNISTParser& m, string weigths, string norm) {
	m.load_training();

	size_t imgs = 60000;
	vec2d_t test_x;
	vec_host test_y;
	for (size_t i = 0; i < imgs; i++) {
		Sample *temp = m.get_sample(i);
		test_x.push_back(temp->image);
		test_y.push_back(temp->label);
	}
	ConvNet n;
	create_lenet(&n);
	char norm_ = 'D';
	if (norm == "L1" || norm == "l1")
		norm_ = 'A';
	if (norm == "L2" || norm == "l2")
		norm_ = 'B';

	n.train(test_x, test_y, norm_);
	n.save_weights(weigths);
}

inline void usage(char **argv) {
	cout << "For classify, gold_gen and rad_test usage: " << argv[0]
			<< " <classify\\gold_gen\\rad_test> <dataset> <labels> <weights> "
					"[gold input/output] "
					"[sample_count] "
					"[save layers] "
					"[iterations]\n";
	cout << "For training, usage: " << argv[0]
			<< " <train> <dataset> <labels> <weights> [L1 or L2 normalization, if not set no norm is applied]\n";

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
	bool save_layer = false;
	int iterations = 1;
	if (argc == 9) {
		gold_in_out = argv[5];
		sample_count = atoi(argv[6]);
		save_layer = (bool) atoi(argv[7]);
		iterations = atoi(argv[8]);
	} else if (argc > 6) {
		usage(argv);
		return EXIT_FAILURE;
	}

	if (mode == "train") {
		//if train training and  labels must be passed
		MNISTParser m(input_data.c_str(), input_labels.c_str());
		string norm = "none";
		if (argc == 6)
			norm = argv[5];
		//printf("debug main norm: ");
		//cout << argv[5];
		cout << "Training for " << m.get_test_img_fname() << std::endl
				<< "normalization: " << norm << std::endl;
		train(m, weigths, norm);
	} else if (mode == "classify") {
		//if train classifing and labels must be passed
		MNISTParser m(input_data.c_str(), input_labels.c_str());
		cout << "Classifing for " << m.get_test_img_fname() << std::endl;
		classify(m, weigths);
	} else if (mode == "gold_gen") {
		MNISTParser m(input_data.c_str(), input_labels.c_str());
		cout << "Generating gold for " << m.get_test_img_fname() << std::endl;
		classify_gold_generate(m, weigths, gold_in_out, sample_count,
				save_layer);

	} else if (mode == "rad_test") {
		MNISTParser m(input_data.c_str(), input_labels.c_str());
		cout << "Testing for " << m.get_test_img_fname() << std::endl;
		classify_test_rad(m, weigths, gold_in_out, save_layer, iterations);
	} else {
		usage(argv);
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}
