/*
 * LogsProcessing.cpp
 *
 *  Created on: Jul 17, 2017
 *      Author: carol
 */

#include "LogsProcessing.h"

size_t error_count = 0;

#ifdef LOGS

#if defined(__cplusplus) && !defined(GPU)
extern "C" {
#endif

#include "log_helper.h"

// Return the name of the log file generated
	char * get_log_file_name();

// Generate the log file name, log info from user about the test
// to be executed and reset log variables
	int start_log_file(char *benchmark_name, char *test_info);

// Log the string "#END" and reset global variables
	int end_log_file();

// Start time to measure kernel time, also update
// iteration number and log to file
	int start_iteration();

// Finish the measured kernel time log both
// time (total time and kernel time)
	int end_iteration();

// Update total errors variable and log both
// errors(total errors and kernel errors)
	int log_error_count(unsigned long int kernel_errors);

// Print some string with the detail of an error to log file
	int log_error_detail(char *string);

// Print some string with the detail of an error/information to log file
	int log_info_detail(char *string);

#if defined(__cplusplus) && !defined(GPU)
}
#endif //extern C

std::vector<std::string> &split(const std::string &s, char delim, std::vector<std::string> &elems) {
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

#endif //LOGS def

void start_count_app(char *test, char *app) {
#ifdef LOGS
	char test_info[500];
	snprintf(test_info, 500, "gold_file: %s", test);

	start_log_file(app, test_info);
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

bool compare_output(std::pair<size_t, bool> p1, std::pair<size_t, bool> p2,
		int img) {
	bool cmp = (p1.first == p2.first) && (p1.second == p2.second);
	char err[200];
	if (!cmp) {
		sprintf(err, "img: [%d] expected_first: [%ld] "
				"read_first: [%ld] "
				"expected_second: [%d] "
				"read_second: [%d]", img, p1.first, p2.first, p1.second,
				p2.second);

#ifdef LOGS
		log_error_detail(err);
		log_error_count(1);
#else
		printf("%s\n", err);
#endif
	}
	return cmp;
}

void compare_and_save_layers(std::vector<Layer*> gold,
		std::vector<Layer*> found, int iteration, int img) {

	std::vector < std::string > last_part;

#ifdef LOGS
	char *temp_log_filename = get_log_file_name();

	last_part = split(std::string(temp_log_filename), '/');
	const char *log_filename = last_part[last_part.size() - 1].c_str();
#else
	const char *log_filename = "test";
#endif

	assert(gold.size() == found.size());

	std::string layer_file_name = std::string(SAVE_LAYER_DATA) + "/"
			+ std::string(log_filename) + "_it_" + std::to_string(iteration)
			+ "_img_" + std::to_string(img);

	for (size_t i = 0; i < gold.size(); i++) {
		Layer *g = gold[i];
		Layer *f = found[i];
		bool error_found = false;

		for (size_t j = 0; j < g->output_.size(); j++) {
			float g_val = g->output_[i];
			float f_val = f->output_[i];
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
				f->save_layer(output_layer);
				fclose(output_layer);
			} else {
				error(
						("ERROR: On opening layer file " + temp_layer_filename).c_str());
			}
		}

	}

}

