/*
 * LogsProcessing.h
 *
 *  Created on: 16/07/2017
 *      Author: fernando
 */

#ifndef LOGSPROCESSING_H_
#define LOGSPROCESSING_H_

#define LAYER_THRESHOLD_ERROR 1e-5
#define SAVE_LAYER_DATA "/var/radiation-benchmarks/data"


void start_count_app(char *test, char *app);

void finish_count_app();

void start_iteration_app();

void end_iteration_app();

void inc_count_app();

void log_error_app(char *error_detail);

/**
 * support function only to check if two layers have
 * the same value
 */
bool compare_layer(float *l1, float *l2, int n);

bool compare_output();

void* load_gold_layers(int img, int layer_size);

void save_gold_layers();

void compare_and_save_layers();

#endif /* LOGSPROCESSING_H_ */
