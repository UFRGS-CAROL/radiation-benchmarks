/*
 * Log.h
 *
 *  Created on: Oct 4, 2018
 *      Author: carol
 */

#ifndef LOG_H_
#define LOG_H_
#include <string>
#include <sys/time.h>
#include "utils.h"

#ifdef LOGS
#include "log_helper.h"
#endif

class Log {
public:
	int32 iterations;
	bool verbose;
	std::string test_mode;
	uint64 errors;
	uint64 infos;
	uint32 seconds_sleep;

	Log(int argc, char** argv, std::string device, uint32 shared_mem_size,
			uint32 l2_size, uint32 number_of_sms, uint32 one_second_cycles) {
		bool help = this->find_arg(argc, argv, "--help");
		if (help) {
			this->usage(argv[0]);
			exit(0);
		}

		this->iterations = this->find_int_arg(argc, argv, "--iterations", 1);
		this->seconds_sleep = this->find_int_arg(argc, argv, "--sleepongpu", 1);

		this->verbose = this->find_int_arg(argc, argv, "--verbose", 0);
		this->test_mode = this->find_char_arg(argc, argv, "--memtotest", "L1");

		this->errors = 0;
		this->infos = 0;
#ifdef LOGS
		std::string test_info = std::string("iterations: ")
				+ std::to_string(this->iterations);

		test_info += " board: " + device;
		test_info += " number_sms: " + std::to_string(number_of_sms);
		test_info += " shared_mem: " + std::to_string(shared_mem_size);
		test_info += " l2_size: " + std::to_string(l2_size);
		test_info += " one_second_cycles: " + std::to_string(one_second_cycles);
		test_info += " test_mode: " + this->test_mode;

		std::string app = test_mode + "Test";
		set_iter_interval_print(10);

		start_log_file(const_cast<char*>(app.c_str()),
				const_cast<char*>(test_info.c_str()));
#endif
	}

	void usage(std::string binary) {
			std::cout << "USAGE: " << binary << " [arguments] that are:\n"
				<< "--iterations <default 1>\n"
						"--sleepongpu <default 1s>\n--verbose <default disabled>\n"
						"--memtotest <default L1 - L2,SHARED,REGISTERS,CONSTANT>"
				<< std::endl;
	}

	virtual ~Log() {
#ifdef LOGS
		end_log_file();
#endif
	}

	void end_iteration_app() {
#ifdef LOGS
		end_iteration();
#endif
	}

	void start_iteration_app() {
		this->errors = 0;
		this->infos = 0;
#ifdef LOGS
		start_iteration();
#endif
	}

	void update_timestamp_app() {
#ifdef LOGS
		update_timestamp();
#endif
	}

	void log_error(std::string& error_detail) {
		this->errors++;
#ifdef LOGS
		log_error_detail(const_cast<char*>(error_detail.c_str()));
#endif
	}

	void log_info(std::string& info_detail) {
		this->infos++;
#ifdef LOGS
		log_info_detail(const_cast<char*>(info_detail.c_str()));
#endif
	}

	static double mysecond() {
		struct timeval tp;
		struct timezone tzp;
		int i = gettimeofday(&tp, &tzp);
		return ((double) tp.tv_sec + (double) tp.tv_usec * 1.e-6);
	}

	static void del_arg(int argc, char **argv, int index) {
		int i;
		for (i = index; i < argc - 1; ++i)
			argv[i] = argv[i + 1];
		argv[i] = 0;
	}

	static int find_int_arg(int argc, char **argv, std::string arg, int def) {
		int i;
		for (i = 0; i < argc - 1; ++i) {
			if (!argv[i])
				continue;
			if (std::string(argv[i]) == arg) {
				def = atoi(argv[i + 1]);
				del_arg(argc, argv, i);
				del_arg(argc, argv, i);
				break;
			}
		}
		return def;
	}

	static std::string find_char_arg(int argc, char **argv, std::string arg,
			std::string def) {
		int i;
		for (i = 0; i < argc - 1; ++i) {
			if (!argv[i])
				continue;
			if (std::string(argv[i]) == arg) {
				def = std::string(argv[i + 1]);
				del_arg(argc, argv, i);
				del_arg(argc, argv, i);
				break;
			}
		}
		return def;
	}

	void update_error_count() {
#ifdef LOGS
		if (this->errors)
			log_error_count(this->errors);
#endif
	}

	void update_info_count() {
#ifdef LOGS
		if (this->infos)
			log_info_count(this->infos);
#endif
	}

	void set_info_max(uint64 max_infos) {
#ifdef LOGS
		set_max_infos_iter(max_infos);
#endif
	}
//
	static int find_arg(int argc, char* argv[], std::string arg) {
		int i;
		for (i = 0; i < argc; ++i) {
			if (!argv[i])
				continue;
			if (std::string(argv[i]) == arg) {
				del_arg(argc, argv, i);
				return 1;
			}
		}
		return 0;
	}
//
//	static float find_float_arg(int argc, char **argv, char *arg, float def) {
//		int i;
//		for (i = 0; i < argc - 1; ++i) {
//			if (!argv[i])
//				continue;
//			if (0 == strcmp(argv[i], arg)) {
//				def = atof(argv[i + 1]);
//				del_arg(argc, argv, i);
//				del_arg(argc, argv, i);
//				break;
//			}
//		}
//		return def;
//	}

};

#endif /* LOG_H_ */
