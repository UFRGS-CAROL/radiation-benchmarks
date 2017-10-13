/*
 * LogsProcessing.cpp
 *
 *  Created on: Jul 17, 2017
 *      Author: carol
 */

#include "LogsProcessing.h"

size_t error_count = 0;

#ifdef LOGS
#include "log_helper.h"
#endif //LOGS def

std::vector<std::string> &split(const std::string &s, char delim,
		std::vector<std::string> &elems) {
	std::stringstream ss(s);
	std::string item;
	while (getline(ss, item, delim)) {
		elems.push_back(item);
	}
	return elems;
}

std::vector<std::string> split(const std::string &s, char delim) {
	std::vector < std::string > elems;
	split(s, delim, elems);
	return elems;
}

void start_count_app(char *test, char *app) {
#ifdef LOGS
	start_log_file(app, test);
#endif
}

void finish_count_app() {
#ifdef LOGS
	end_log_file();
#endif
}

void start_iteration_app() {
#ifdef LOGS
	start_iteration();
	error_count = 0;
#endif
}

void end_iteration_app() {
#ifdef LOGS
	end_iteration();
#endif
}

void inc_count_app() {
#ifdef LOGS
	log_error_count(error_count++);
#endif
}

char *get_log_filename() {
#ifdef LOGS
	return get_log_file_name();
#endif
}

/**
 * support function only to check if two layers have
 * the same value
 */
bool compare_layer(float *l1, float *l2, int n) {
	for (int i = 0; i < n; i++) {
		float diff = fabs(l1[i] - l2[i]);
		if (diff > LAYER_THRESHOLD_ERROR) {
//			printf("passou  onde nao devia %f\n\n", diff);
			return true;
		}
	}
	return false;
}

bool compare_output(std::pair<size_t, bool> gold, std::pair<size_t, bool> found,
		int img) {
//	bool cmp = (gold.first == found.first) && (gold.second == found.second);
	char err[200];
	if ((gold.first != found.first) || (gold.second != found.second)) {
		sprintf(err, "img: [%d] expected_first: [%ld] "
				"read_first: [%ld] "
				"expected_second: [%d] "
				"read_second: [%d]", img, gold.first, found.first, gold.second,
				found.second);

#ifdef LOGS
		log_error_detail(err);
		log_error_count(1);
		printf("%s\n", err);
#else
		printf("%s\n", err);
#endif
		return false;
	}
	return true;
}

LayersGold load_gold_layers(int img, int layer_size) {
	LayersGold loaded(layer_size);
	for (auto i = 0; i < layer_size; i++) {
		std::string path = std::string(SAVE_LAYER_DATA)
				+ "/gold_layer_lenet_img_" + std::to_string(img) + "_layer_"
				+ std::to_string(i) + ".layer";

		FILE *fout = fopen(path.c_str(), "rb");
		if (fout != NULL) {
			size_t v_size;
			fread(&v_size, sizeof(size_t), 1, fout);

			loaded[i].resize(v_size);

			fread(loaded[i].data(), sizeof(float), v_size, fout);

		} else {
			error("FAILED TO OPEN FILE " + path);
		}
	}
	return loaded;
}

void save_gold_layers(LayersFound layers, int img) {

	for (size_t i = 0; i < layers.size(); i++) {
		auto v = layers[i];
		std::string path = std::string(SAVE_LAYER_DATA)
				+ "/gold_layer_lenet_img_" + std::to_string(img) + "_layer_"
				+ std::to_string(i) + ".layer";

		FILE *fout = fopen(path.c_str(), "wb");
		if (fout != NULL) {
			size_t v_size = v->size();
			fwrite(&v_size, sizeof(size_t), 1, fout);
#ifdef NOTUNIFIEDMEMORY
			v->pop_vector();
			fwrite(v->h_data(), sizeof(float), v->size(), fout);
#else
			fwrite(v->data(), sizeof(float), v->size(), fout);
#endif
			fclose(fout);
		} else {
			error("FAILED TO OPEN FILE " + path);
		}

	}

}

void compare_and_save_layers(LayersGold gold, LayersFound found, int iteration,
		int img) {

	std::vector < std::string > last_part;

#ifdef LOGS
	char *temp_log_filename = get_log_filename();

	last_part = split(std::string(temp_log_filename), '/');
	const char *log_filename = last_part[last_part.size() - 1].c_str();
#else
	const char *log_filename = "test";
#endif
//	std::cout << "gold size " << gold.size() <<" found size " << found.size() << "\n";
	assert(gold.size() == found.size());

	std::string layer_file_name = std::string(SAVE_LAYER_DATA) + "/"
			+ std::string(log_filename) + "_it_" + std::to_string(iteration)
			+ "_img_" + std::to_string(img);
	std::cout << "gold size " << gold.size() << "\n";
	for (size_t i = 0; i < gold.size(); i++) {
		auto g = gold[i];
		auto f = (*found[i]);
#ifdef NOTUNIFIEDMEMORY
		f.pop_vector();
#endif
		bool error_found = true;

		assert(g.size() == f.size());
		for (size_t j = 0; j < g.size(); j++) {

			float g_val = g[j];
			float f_val = f[j];
			float diff = fabs(g_val - f_val);
			if (diff > LAYER_THRESHOLD_ERROR) {
				error_found = true;
				break;
			}
		}
		if (error_found) {
			std::string temp_layer_filename = layer_file_name + "_layer_"
					+ std::to_string(i) + ".layer";
			FILE *output_layer = fopen(temp_layer_filename.c_str(), "wb");
			if (output_layer != NULL) {
				size_t v_size = f.size();

				fwrite(&v_size, sizeof(size_t), 1, output_layer);
#ifdef NOTUNIFIEDMEMORY
				fwrite(f.h_data(), sizeof(float),f.size(),
						output_layer);
#else
				fwrite(f.data(), sizeof(float),f.size(),
						output_layer);
#endif
				fclose(output_layer);
			} else {
				error(
						("ERROR: On opening layer file " + temp_layer_filename).c_str());
			}
		}

	}
}
