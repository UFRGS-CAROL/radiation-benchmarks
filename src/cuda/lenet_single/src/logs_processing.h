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

#ifdef GPU

typedef std::vector<DeviceVector<float>*> LayersFound;
#else
typedef std::vector<vec_host*> LayersFound;
#endif

typedef std::vector<vec_host> LayersGold;

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

bool compare_output(std::pair<size_t, bool> gold, std::pair<size_t, bool> found,
		int img);

void log_error_app(char *error_detail);

LayersGold load_gold_layers(int img, int layer_size);

void save_gold_layers(LayersFound layers, int img);

void compare_and_save_layers(LayersGold gold, LayersFound found, int iteration,
		int img);

#endif /* LOGSPROCESSING_H_ */
