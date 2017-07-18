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
#endif

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
		std::vector<Layer*> found) {

}

