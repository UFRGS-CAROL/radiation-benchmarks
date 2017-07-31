/*
 * LogsProcessing.h
 *
 *  Created on: 16/07/2017
 *      Author: fernando
 */

#ifndef LOGSPROCESSING_H_
#define LOGSPROCESSING_H_

#define LAYER_THRESHOLD_ERROR 1e-5

#include "Layer.h"
#include <vector>

#define SAVE_LAYER_DATA "/var/radiation-benchmarks/data"

void start_count_app(char *test, char *app);

void finish_count_app();

void start_iteration_app();

void end_iteration_app();

void inc_count_app();

/**
 * support function only to check if two layers have
 * the same value
 */
bool compare_layer(float *l1, float *l2, int n);

bool compare_output(std::pair<size_t, bool> p1, std::pair<size_t, bool> p2,
		int img);

void log_error_app(char *error_detail);



template<typename T>
void save_gold_layers(T layers, int img) {

	for (auto i = 0; i < layers.size(); i++) {
		auto v = layers[i];
		std::string path = std::string(SAVE_LAYER_DATA)
				+ "/gold_layer_lenet_img_" + std::to_string(img) + "_layer_"
				+ std::to_string(i) + ".layer";

		FILE *fout = fopen(path.c_str(), "wb");
		if (fout != NULL) {
			size_t v_size = v->size();
			fwrite(&v_size, sizeof(size_t), 1, fout);
			fwrite(v->data(), sizeof(float), v->size(), fout);
			fclose(fout);
		} else {
			error("FAILED TO OPEN FILE " + path);
		}

	}

}

template<typename T>
T load_gold_layers(int img, int layer_size) {
	T loaded;

	loaded.resize(layer_size);
	for (auto i = 0; i < layer_size; i++) {
		std::string path = std::string(SAVE_LAYER_DATA)
				+ "/gold_layer_lenet_img_" + std::to_string(img) + "_layer_"
				+ std::to_string(i) + ".layer";

		FILE *fout = fopen(path.c_str(), "rb");
		if (fout != NULL) {
			size_t v_size;
			fread(&v_size, sizeof(size_t), 1, fout);
#ifdef GPU
			auto v = (DeviceVector<float>(v_size));
#else
			auto v = (vec_host(v_size));
#endif
			fread(v.data(), sizeof(float), v_size, fout);
			fclose(fout);
			loaded[i] = &v;

		} else {
			error("FAILED TO OPEN FILE " + path);
		}
	}
	return loaded;
}

#ifdef GPU
typedef std::vector<DeviceVector<float>*> TypeVector;
#else
typedef std::vector<vec_host*> TypeVector;
#endif

void compare_and_save_layers(TypeVector gold, TypeVector found, int iteration, int img);


#endif /* LOGSPROCESSING_H_ */
