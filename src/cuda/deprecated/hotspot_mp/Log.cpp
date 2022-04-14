/*
 * Log.cpp
 *
 *  Created on: 18/05/2019
 *      Author: fernando
 */

#include "Log.h"
#include <iostream>

Log::Log(std::string& app, std::string& test_info, bool to_log) :
		to_log(to_log), tic(0), toc(0), info_count(0), error_count(0) {

#ifdef LOGS
	if (!this->to_log)
		start_log_file(const_cast<char*>(app.c_str()),
				const_cast<char*>(test_info.c_str()));
#endif
}

Log::Log() :
		to_log(false), tic(0), toc(0), info_count(0), error_count(0) {
}

Log::Log(const Log& a) {
	this->to_log = a.to_log;
	this->tic = a.tic;
	this->error_count = a.error_count;
	this->info_count = a.info_count;
	this->toc = a.toc;
}

Log::~Log() {
#ifdef LOGS
	if (!this->to_log)
		end_log_file();
#endif
}

void Log::end_iteration_app() {
	this->toc = this->mysecond();
#ifdef LOGS
	if (!this->to_log)
		end_iteration();
#endif
}

void Log::start_iteration_app() {
	this->tic = this->mysecond();
#ifdef LOGS
	if (!this->to_log)
		start_iteration();
#endif
}

void Log::update_timestamp_app() {
#ifdef LOGS
	if (!this->to_log)
		update_timestamp();
#endif
}

void Log::log_error(std::string error_detail) {
	this->error_count++;
#ifdef LOGS
	if (!this->to_log)
		log_error_detail(const_cast<char*>(error_detail.c_str()));
#endif
}

void Log::log_info(std::string info_detail) {
	this->info_count++;
#ifdef LOGS
	if (!this->to_log)
		log_info_detail(const_cast<char*>(info_detail.c_str()));
#endif
}

void Log::update_error_count() {
#ifdef LOGS
	if (error_count && !this->to_log)
		log_error_count(this->error_count);
#endif
	this->error_count = 0;
}

void Log::update_info_count() {
#ifdef LOGS
	if (info_count && !this->to_log)
		log_info_count(this->info_count);
#endif
	this->info_count = 0;
}

double Log::mysecond() {
	struct timeval tp;
	struct timezone tzp;
	int i = gettimeofday(&tp, &tzp);
	return ((double) tp.tv_sec + (double) tp.tv_usec * 1.e-6);
}

void Log::fatal(std::string& s) {
	std::cerr << "error: " << s << std::endl;
#ifdef LOGS
	if (!this->to_log) {
		end_log_file();
	}
#endif
	exit(1);
}

void Log::force_end(std::string& error) {
#ifdef LOGS
	log_error_detail(const_cast<char*>(error.c_str()));
	end_log_file();
#endif
}

Log& Log::operator =(const Log& other) {
	if (&other == this)
		return *this;

	this->to_log = other.to_log;
	this->tic = other.tic;

	return *this;
}

double Log::iteration_time() {
	return this->toc - this->tic;
}
