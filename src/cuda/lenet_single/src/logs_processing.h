/*
 * LogsProcessing.h
 *
 *  Created on: 16/07/2017
 *      Author: fernando
 */

#ifndef LOGSPROCESSING_H_
#define LOGSPROCESSING_H_
#include <string>

#define RADIATION_PARAMETERS 3

class LogsProcessing{
private:
	const float layer_threshold_error = 1e-5;
	const char *save_layer_data = "/var/radiation-benchmarks/data";
	bool generate;

	std::string test;
	std::string app;
	unsigned error_count;

public:
	LogsProcessing(bool generate, std::string app, std::string test,
			int log_interval = 10);
    virtual ~LogsProcessing();

	void start_iteration_app();

	void end_iteration_app();

	void inc_count_app();

	void log_error_app(std::string error_detail);

	/**
	 * support function only to check if two layers have
	 * the same value
	 */
	bool compare_layer(float *l1, float *l2, int n);

	bool compare_output();

	void* load_gold_layers(int img, int layer_size);

	void save_gold_layers();

	void compare_and_save_layers();
	char* get_log_filename();
};
#endif /* LOGSPROCESSING_H_ */
