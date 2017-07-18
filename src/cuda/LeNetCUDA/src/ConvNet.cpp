/*
 * ConvNet.cpp
 *
 *  Created on: 05/06/2017
 *      Author: fernando
 */

#include "ConvNet.h"

#ifdef GPU
#include "cudaUtil.h"
#endif

#include "LogsProcessing.h"

void ConvNet::train(vec2d_t train_x, vec_host train_y, size_t train_size) {

#ifdef GPU
	std::cout << "Training with GPU:" << std::endl;
#else
	std::cout << "Training with CPU:" << std::endl;
#endif
	train_x_ = train_x;
	train_y_ = train_y;
	train_size_ = train_size;
	/*
	 auto add OutputLayer as the last layer.
	 */
	this->add_layer(new OutputLayer(layers.back()->out_depth_));

	/*
	 start training...
	 */
	auto stop = false;
	int iter = 0;
	this->mark.start();
	while (iter < MAX_ITER && !stop) {
		iter++;
		auto err = train_once();
		std::cout << " training cost: " << err << std::endl;
		if (err < END_CONDITION)
			stop = true;
	}
	this->mark.stop();
	std::cout << "Time spent on training " << this->mark << std::endl;
}

void ConvNet::train(vec2d_t train_x, vec_host train_y) {
#ifdef GPU
	std::cout << "Training with GPU:" << std::endl;
#else
	std::cout << "Training with CPU:" << std::endl;
#endif
	train_x_ = train_x;
	train_y_ = train_y;
	train_size_ = this->train_x_.size();
	/*
	 auto add OutputLayer as the last layer.
	 */
	auto err = 0.0;
	this->add_layer(new OutputLayer(layers.back()->out_depth_));
	this->mark.start();

	for (int i = 0; i < this->train_size_; i++) {
		layers[0]->input_ = train_x_[i];
		layers.back()->exp_y = (int) train_y_[i];
		/*
		 Start forward feeding.
		 */
		for (auto layer : layers) {
			layer->forward();
			if (layer->next != nullptr) {
				layer->next->input_ = layer->output_;
			}

		}

		/*
		 back propgation
		 */
		for (auto i = layers.rbegin(); i != layers.rend(); i++) {
			(*i)->back_prop();
		}

		err = layers.back()->err;
//		std::cout << " training cost: " << err << std::endl;
//		if(err < END_CONDITION){ // if the error is small enough
//			std::cout << "Error small enough " << err << "\n";
//			break;
//		}
	}
	this->mark.stop();
	std::cout << "Time spent on training " << this->mark << std::endl;
}

/**
 * General test case
 * test_x is where the images are located
 * test_y is the output
 * test_size is how much images
 *
 */
void ConvNet::test(vec2d_t test_x, vec_host test_y, size_t test_size,
		std::string gold_layers_path, bool save_layer) {
//	assert(batch_size > 0);
//	assert(test_size % batch_size == 0);
	test_x_ = test_x;
	test_y_ = test_y;
	test_size_ = test_size;
	int iter = 0;
	int bang = 0;

#ifdef GPU
	std::cout << "Testing with GPU " << std::endl;
#else
	std::cout << "Testing with CPU " << std::endl;
#endif // GPU
	this->mark.start();
	while (iter < test_size_) {
		int result = 0;
		result = test_once(iter) ? 1 : 0;
		bang += result;
		if(save_layer){
			//if it is the first time I rewrite the file
			this->save_weights(gold_layers_path, (iter) ? "ab" : "wb");
		}
		iter++;
	}
	this->mark.stop();
	std::cout << "bang/test_size_: " << (float) bang / test_size_ << std::endl;
	std::cout << "Time spent testing " << this->test_size_ << " samples: "
			<< this->mark << std::endl;
}

/**
 * Test case for radiation setup/fault injection
 * test_x is where the images are located
 * test_y is the output
 * test_size is how much images
 */
void ConvNet::test(vec2d_t test_x, vec_host test_y,
		std::vector<std::pair<size_t, bool>> gold_list, //gold for radiation test
		std::vector<std::vector<Layer*>> gold_layers, //gold layers
		size_t iterations, bool save_layer) {
	test_x_ = test_x;
	test_y_ = test_y;
	test_size_ = test_x_.size();

	Timer compare_timer;

#ifdef GPU
	std::cout << "Testing with GPU " << std::endl;
#else
	std::cout << "Testing with CPU " << std::endl;
#endif // GPU

	for (auto i = 0; i < iterations; i++) {
		this->mark.start();
		for (auto iter = 0; iter < test_size_; iter++) {
			auto gold_out = gold_list[i];

			//test under radiation
			start_iteration_app();
			auto result = test_once_pair(iter);
			end_iteration_app();

			//compare output
			compare_timer.start();
			auto cmp = compare_output(result, gold_out);

			//log the result
			if (cmp) {
				//err_count++
				inc_count_app();

				//layer comparison
				if (save_layer) {
					compare_and_save_layers(gold_layers[i], this->layers);
				}
			}
			compare_timer.stop();
			//-------------
		}
		this->mark.stop();
		std::cout << "Iteration: " << i << ". Time spent testing "
				<< this->test_size_ << " samples: " << this->mark << std::endl;
	}

}

std::list<std::pair<size_t, bool>> ConvNet::get_predicted_output() {
	return this->saved_output;
}

void ConvNet::add_layer(Layer* layer) {
	if (!layers.empty())
		this->layers.back()->next = layer;
	this->layers.push_back(layer);
	layer->next = NULL;
}

#ifdef GPU
size_t ConvNet::max_iter(DeviceVector<float> v) {
	size_t i = 0;
	float_t max = v[0];
	for (size_t j = 1; j < v.size(); j++) {
		if (v[j] > max) {
			max = v[j];
			i = j;
		}
	}
	return i;
}
#else
size_t ConvNet::max_iter(vec_host v) {
	size_t i = 0;
	float_t max = v[0];
	for (size_t j = 1; j < v.size(); j++) {
		if (v[j] > max) {
			max = v[j];
			i = j;
		}
	}
	return i;
}
#endif

size_t ConvNet::max_iter(float v[], size_t size) {
	size_t i = 0;
	float_t max = v[0]; //std::cout<< " raw output: "<<v[0]<<" ";
	for (size_t j = 1; j < size; j++) {
		//std::cout<< v[j] << " ";
		if (v[j] > max) {
			max = v[j];
			i = j;
		}
	}
	//std::cout<<std::endl;
	return i;
}

bool ConvNet::test_once_random() {
	int test_x_index = uniform_rand(0, test_size_ - 1);
	return test_once(test_x_index);
}

bool ConvNet::test_once(int test_x_index) {
	layers[0]->input_ = test_x_[test_x_index];
	for (auto layer : layers) {
		layer->forward();
		if (layer->next != nullptr) {
			layer->next->input_ = layer->output_;
		}
	}

	int predicted = (int) max_iter(layers.back()->output_);
	bool is_right = test_y_[test_x_index] == predicted;

	std::pair<size_t, bool> p1(predicted, is_right);
	this->saved_output.push_back(p1);
	return (int) is_right;
}

std::pair<size_t, bool> ConvNet::test_once_pair(int test_x_index) {
	layers[0]->input_ = test_x_[test_x_index];
	for (auto layer : layers) {
		layer->forward();
		if (layer->next != nullptr) {
			layer->next->input_ = layer->output_;
		}
	}

	int predicted = (int) max_iter(layers.back()->output_);
	bool is_right = test_y_[test_x_index] == predicted;

	std::pair<size_t, bool> p1(predicted, is_right);

	return p1;
}

float_t ConvNet::train_once() {
	float_t err = 0;
	int iter = 0;
#ifdef DEBUG
	//DEBUG:
	std::ofstream debugFile;
#ifdef GPU
	debugFile.open ("GPUdebugFile.txt");
#else //CPU
	debugFile.open ("CPUdebugFile.txt");
#endif
#endif
	//
	int test = 0;
	while (iter < M) {
		//auto train_x_index = iter % train_size_;
		iter++;
		auto train_x_index = uniform_rand(0, train_size_ - 1);
		layers[0]->input_ = train_x_[train_x_index];
		layers.back()->exp_y = (int) train_y_[train_x_index];
		/*
		 Start forward feeding.
		 */
		int debugIter = 0;
		for (auto layer : layers) {
			layer->forward();
			if (layer->next != nullptr) {
				layer->next->input_ = layer->output_;
			}
#ifdef DEBUG
			//debug
			debugFile << debugIter << "_output = [ ";
			for(int i = 0; i < layer->output_.size(); i++) {
				debugFile << layer->output_[i] << ", ";
			}
			debugFile << "]\n";
			debugIter++;
#endif
		}
		err += layers.back()->err;

		/*
		 back propgation
		 */
		for (auto i = layers.rbegin(); i != layers.rend(); i++) {
			(*i)->back_prop();
		}

	}
#ifdef DEBUG
	debugFile.close();
#endif	//end DEBUG

	return err / M;
}

void ConvNet::load_weights(std::string path) {
	FILE *in = fopen(path.c_str(), "rb");
//	std::ifstream in(path, std::ios::in | std::ios::binary | std::ios::ate);
	if (in != NULL) {
		for (auto i = layers.begin(); i != layers.end(); i++) {

			(*i)->load_layer(in);
		}
		fclose(in);
	} else {
		error("FAILED TO OPEN FILE " + path);
	}
}

void ConvNet::save_weights(std::string path, std::string file_mode) {
	FILE *fout = fopen(path.c_str(), file_mode.c_str());
//	std::ofstream fout(path, std::ios::out | std::ios::binary);
	if (fout != NULL) {
		for (auto i = layers.begin(); i != layers.end(); i++) {
			(*i)->save_layer(fout);
		}
		fclose(fout);
	} else {
		error("FAILED TO OPEN FILE " + path);
	}

}


int ConvNet::getSquaredSumLeNetWeights() {
	int sum = 0;
	for (auto layer : layers) {
		sum += layer->getSquaredWeightsSum();
	}
	return sum;
}

int ConvNet::getSumLeNetWeights() {
	int sum = 0;
	for (auto layer : layers) {
		sum += layer->getWeightsSum();
	}
	return sum;
}
