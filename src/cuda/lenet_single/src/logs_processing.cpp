/*
 * LogsProcessing.cpp
 *
 *  Created on: Jul 17, 2017
 *      Author: carol
 */

#include "logs_processing.h"
#include <vector>
#include <sstream>

#ifdef LOGS
#include "log_helper.h"
#endif //LOGS def

void LogsProcessing::start_iteration_app() {
#ifdef LOGS
	if (!this->generate) {
		start_iteration();
		this->error_count = 0;
	}
#endif
}

void LogsProcessing::end_iteration_app() {
#ifdef LOGS
	if (!this->generate) {
		end_iteration();
	}
#endif
}

void LogsProcessing::log_error_app(std::string error_detail) {
#ifdef LOGS
	if (!this->generate) {
		log_error_detail(error_detail.c_str());
	}
#endif
}

LogsProcessing::LogsProcessing(std::string app, bool generate, std::string gold_path,
		std::string weights, std::string prototxt, int iterations,
		int log_interval) {
	this->generate = generate;
	if (!this->generate) {
		this->app = app;
		this->error_count = 0;
		this->iterations = iterations;
		this->gold_path = gold_path;
		this->weights = weights;
		std::string
	header_line = "gold_file: " + gold_path  + " weights: " + weights
	+ " iterations: " + std::to_string(iterations) + " prototxt: " + prototxt;
#ifdef LOGS
	set_iter_interval_print(log_interval);
	start_log_file(header_line.c_str(), test.c_str());
#endif
}
}

LogsProcessing::~LogsProcessing() {
#ifdef LOGS
if (!this->generate) {
	end_log_file();
}
#endif
}

void LogsProcessing::inc_count_app() {
#ifdef LOGS
if (!this->generate) {
	log_error_count(this->error_count++);
}
#endif
}

char* LogsProcessing::get_log_filename() {
#ifdef LOGS
return get_log_file_name();
#endif
return NULL;
}

/**
 * support function only to check if two layers have
 * the same value
 */
bool LogsProcessing::compare_layer(float *l1, float *l2, int n) {

return false;
}

bool LogsProcessing::compare_output() {
//TODO
// compare caffe output
return true;
}

void* LogsProcessing::load_gold_layers(int img, int layer_size) {
//TODO
// load caffe layers
return (NULL);
}

void LogsProcessing::save_gold_layers() {
//TODO
// Save caffe layers
}

void LogsProcessing::compare_and_save_layers() {
//TODO
// compare caffe layers
}

