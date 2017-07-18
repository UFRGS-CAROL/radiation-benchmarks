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
	//start log file
	start_count_app(const_cast<char*>(gold_input.c_str()), const_cast<char*>("cudaDarknet"));
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

	//start log file
	end_iteration_app();
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
	n.test(test_x, test_y, test_sample_count, string(SAVE_LAYER_DATA) + "/gold_layers_lenet.layer", save_layers);

	std::ofstream gold_output_file(gold_output);

	if (gold_output_file.is_open()) {
		std::list<std::pair<size_t, bool>> output = n.get_predicted_output();
		//write gold info

		//test input
		gold_output_file << m.get_test_img_fname() << ";"
				<< m.get_test_lbl_fname() << ";";
		//test size
		gold_output_file << weigths << ";" << test_sample_count << "\n";

		//write the output
		for (std::pair<size_t, bool> p : output) {
			gold_output_file << p.first << ";" << p.second << "\n";
		}

		gold_output_file.close();

	} else {
		error("ERROR: On opening " + gold_output + " file\n");
	}
}

void classify_test_rad(MNISTParser& m, string weigths, string gold_input, bool save_layers){
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
	vector<vector<Layer*>> gold_layers;

	string test_img_fname, test_lbl_fname, weigths_read;
	int sample_count;
//
//	if (gold_input_file.is_open()){
//		gold_input_file >> test_img_fname >> ";" >> test_lbl_fname >> ";" >> weigths_read >> ";" >> sample_count;
//
//		for(int i = 0; i < sample_count; i++){
//			pair<size_t, bool> p;
//			gold_input_file >> p.first >> ";" >> p.second;
//			gold_data.push_back(p);
//		}
//
//		gold_input_file.close()
//	}else{
//		error("ERROR: On opening " + gold_input + " file\n");
//	}
//
//	if(save_layers){
//		FILE *gold_layers_input_file = fopen(string(SAVE_LAYER_DATA) + "/gold_layers_lenet.layer", );
//	}

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
					"[sample_count only for gold_gen and rad_test] "
					"[save layers only for gold_gen and rad_test]\n";
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
	if (argc == 8) {
		gold_in_out = argv[5];
		sample_count = atoi(argv[6]);
		save_layer = (bool) atoi(argv[7]);
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
		classify_gold_generate(m, weigths, gold_in_out, sample_count, save_layer);

	} else if (mode == "rad_test") {

	} else {
		usage(argv);
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}
