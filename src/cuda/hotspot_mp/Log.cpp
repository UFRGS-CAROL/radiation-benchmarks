/*
 * Log.cpp
 *
 *  Created on: 18/05/2019
 *      Author: fernando
 */

#include "Log.h"
#include <iostream>

Log::Log(std::string& app, std::string& test_info, bool generate) :
		generate(generate), tic_(0), info_count(0), error_count(0) {

#ifdef LOGS
	if (!this->generate)
		start_log_file(const_cast<char*>(app.c_str()),
				const_cast<char*>(test_info.c_str()));
#endif
}

Log::Log() :
		generate(false), tic_(0), info_count(0), error_count(0) {
}

Log::Log(const Log& a) {
	this->generate = a.generate;
	this->tic_ = a.tic_;
	this->error_count = a.error_count;
	this->info_count = a.info_count;
}

Log::~Log() {
	std::cout << "Passou aqui\n";
#ifdef LOGS
	if (!this->generate)
		end_log_file();
#endif
}

void Log::end_iteration_app() {
#ifdef LOGS
	if (!this->generate)
		end_iteration();
#endif
}

void Log::start_iteration_app() {
#ifdef LOGS
	if (!this->generate)
		start_iteration();
#endif
}

void Log::update_timestamp_app() {
#ifdef LOGS
	if (!this->generate)
		update_timestamp();
#endif
}

void Log::log_error(std::string error_detail) {
	this->error_count++;
#ifdef LOGS
	if (!this->generate)
		log_error_detail(const_cast<char*>(error_detail.c_str()));
#endif
}

void Log::log_info(std::string info_detail) {
	this->info_count++;
#ifdef LOGS
	if (!this->generate)
		log_info_detail(const_cast<char*>(info_detail.c_str()));
#endif
}

void Log::update_error_count() {
#ifdef LOGS
	if (error_count && !this->generate)
		log_error_count(this->error_count);
#endif
	this->error_count = 0;
}

void Log::update_info_count() {
#ifdef LOGS
	if (info_count && !this->generate)
		log_info_count(this->info_count);
#endif
	this->info_count = 0;
}

void Log::tic() {
	this->tic_ = this->mysecond();
}

double Log::toc() {
	return this->mysecond() - this->tic_;
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
	if (!this->generate) {
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

	this->generate = other.generate;
	this->tic_ = other.tic_;

	return *this;
}
