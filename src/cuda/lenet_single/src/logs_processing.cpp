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
	return NULL;
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
	//TODO
	// load caffe layers
	return loaded;
}

void save_gold_layers(LayersFound layers, int img) {
	//TODO
	// Save caffe layers
}

void compare_and_save_layers(LayersGold gold, LayersFound found, int iteration,
		int img) {
	//TODO
	// compare caffe layers
}
