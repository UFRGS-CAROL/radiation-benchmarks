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


#ifdef LOGS
#include "log_helper.h"
#endif

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

bool compare_output(std::pair<size_t, bool> p1, std::pair<size_t, bool> p2);

void compare_and_save_layers(std::vector<Layer*> gold, std::vector<Layer*> found);


#endif /* LOGSPROCESSING_H_ */
