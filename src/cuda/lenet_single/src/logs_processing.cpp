/*
 * LogsProcessing.cpp
 *
 *  Created on: Jul 17, 2017
 *      Author: carol
 */

#include "logs_processing.h"
#include <vector>
#include <sstream>
#include <string>

#ifdef LOGS
#include "log_helper.h"
#endif //LOGS def

int error_count = 0;


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

	return false;
}

bool compare_output() {
	//TODO
	// compare caffe output
	return true;
}

void* load_gold_layers(int img, int layer_size) {
	//TODO
	// load caffe layers
	return (NULL);
}

void save_gold_layers() {
	//TODO
	// Save caffe layers
}

void compare_and_save_layers() {
	//TODO
	// compare caffe layers
}
