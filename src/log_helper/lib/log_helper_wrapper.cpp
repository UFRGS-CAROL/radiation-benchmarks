/**
 * Wrapper for pure C applications
 * It is only to support legacy codes
 */
#include "include/log_helper.hpp"
#include <string>
#include "include/log_helper.h"

LogHelper *obj = nullptr;

// Set the max errors that can be found for a single iteration
// If more than max errors is found, exit the program
unsigned long int set_max_errors_iter(unsigned long int max_errors) {
	obj->set_max_errors_iter(max_errors);
	return max_errors;
}

// Set the max number of infos logged in a single iteration
unsigned long int set_max_infos_iter(unsigned long int max_infos) {
	obj->set_max_infos_iter(max_infos);
	return max_infos;
}

// Set the interval the program must print log details,
// default is 1 (each iteration)
int set_iter_interval_print(int interval) {
	obj->set_iter_interval_print(interval);
	return interval;
}

// Update with current timestamp the file where the software watchdog watchs
extern "C" void update_timestamp() {
	obj->update_timestamp();
}

// Return the name of the log file generated
char * get_log_file_name() {
	return const_cast<char*>(obj->get_log_file_name().c_str());
}

// Generate the log file name, log info from user about the test
// to be executed and reset log variables
int start_log_file(char *benchmark_name, char *test_info) {
	obj = new LogHelper(std::string(benchmark_name),
			std::string(test_info));
	if (obj != nullptr)
		return 0;
	return 1;
}

// Log the string "#END" and reset global variables
int end_log_file() {
	if (obj != nullptr) {
		delete obj;
		return 0;
	}
	return 1;
}

// Start time to measure kernel time, also update
// iteration number and log to file
int start_iteration() {
	if (obj != nullptr) {
		obj->start_iteration();
		return 0;
	}
	return 1;
}

// Finish the measured kernel time log both
// time (total time and kernel time)
int end_iteration() {
	if (obj != nullptr) {
		obj->end_iteration();
		return 0;
	}
	return 1;
}

// Update total errors variable and log both
// errors(total errors and kernel errors)
int log_error_count(unsigned long int kernel_errors) {
	if (obj != nullptr) {
		obj->log_error_count(kernel_errors);
		return 0;
	}
	return 1;
}

//Update total infos variable and log both infos(total infos and iteration infos)
int log_info_count(unsigned long int info_count) {
	if (obj != nullptr) {
		obj->log_info_count(info_count);
		return 0;
	}
	return 1;
}

// Print some string with the detail of an error to log file
int log_error_detail(char *string) {
	if (obj != nullptr) {
		obj->log_error_detail(std::string(string));
		return 0;
	}
	return 1;
}

// Print some string with the detail of an error/information to log file
int log_info_detail(char *string) {
	if (obj != nullptr) {
		obj->log_info_detail(std::string(string));
		return 0;
	}
	return 1;
}

// Get current iteration number
unsigned long int get_iteration_number() {
	return obj->get_iteration_number();
}

