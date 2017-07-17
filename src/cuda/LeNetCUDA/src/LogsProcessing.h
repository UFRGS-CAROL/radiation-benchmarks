/*
 * LogsProcessing.h
 *
 *  Created on: 16/07/2017
 *      Author: fernando
 */

#ifndef LOGSPROCESSING_H_
#define LOGSPROCESSING_H_

#define LAYER_THRESHOLD_ERROR 1e-5

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
#endif
}

void end_iteration_app() {
#ifdef LOGS
	end_iteration();
#endif
}

/**
 * support function only to check if two layers have
 * the same value
 */
inline bool compare_layer(float *l1, float *l2, int n) {
	for (int i = 0; i < n; i++) {
		float diff = fabs(l1[i] - l2[i]);
		if (diff > LAYER_THRESHOLD_ERROR) {
//			printf("passou  onde nao devia %f\n\n", diff);
			return true;
		}
	}
	return false;
}



#endif /* LOGSPROCESSING_H_ */
