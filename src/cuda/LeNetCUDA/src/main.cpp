/*
 * main.cpp
 *
 *  Created on: May 26, 2017
 *      Author: carol
 */

#include <cstdlib>
//#include "Util.h"
#include "ConvNet.h"


#define DATA_PATH "./"


using namespace std;
//~ using namespace convnet;

int main(int argc, char **argv){
	cout << DATA_PATH << std::endl;
		convnet::MNISTParser m(DATA_PATH);
		cout << m.get_test_img_fname() << std::endl;
		m.load_testing();
		//m.load_training();
		convnet::vec2d_t x;
		convnet::vec_t y;
		convnet::vec2d_t test_x;
		convnet::vec_t test_y;
		/*
		 for (size_t i = 0; i < 60000; i++){
		 x.push_back(m.train_sample[i]->image);
		 y.push_back(m.train_sample[i]->label);
		 }

		 */

		for (size_t i = 0; i < 10000; i++) {
			test_x.push_back(m.test_sample[i]->image);
			test_y.push_back(m.test_sample[i]->label);
		}

		convnet::ConvNet n;

		n.add_layer(new convnet::ConvolutionalLayer(32, 32, 1, 5, 6));
		n.add_layer(new convnet::MaxpoolingLayer(28, 28, 6));
		n.add_layer(new convnet::ConvolutionalLayer(14, 14, 6, 5, 16));
		n.add_layer(new convnet::MaxpoolingLayer(10, 10, 16));
		n.add_layer(new convnet::ConvolutionalLayer(5, 5, 16, 5, 100));
		n.add_layer(new convnet::FullyConnectedLayer(100, 10));

		n.train(test_x, test_y, 10000);
		int test_sample_count = 5;
		//Sleep(1000);
		printf("Testing with %d samples:\n", test_sample_count);
		const clock_t begin_time = clock();
		n.test(test_x, test_y, test_sample_count, 5);
		cout << "Time consumed in test: "
				<< float(clock() - begin_time) / (CLOCKS_PER_SEC / 1000) << " ms"
				<< endl;

	return  EXIT_SUCCESS;
}
