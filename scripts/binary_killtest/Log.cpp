/*
 * Log.cpp
 *
 *  Created on: 20/01/2019
 *      Author: fernando
 */

#include "Log.h"
#include <stdexcept>
#include <fstream>

namespace radiation {

Log::Log() :
		log_path("") {
}

Log::Log(std::string log_file_path) :
		log_path(log_file_path) {
}

Log::Log(const Log& b) :
		log_path(b.log_path) {
}

void Log::log_message_exception(std::string message) {
	std::fstream log_file(this->log_path,
			std::fstream::out | std::fstream::app);
	if (log_file.good()) {
		log_file << "EXCEPTION_MESSAGE:" + message << std::endl;
		log_file.close();
	} else {
		throw std::runtime_error(
				"COULD NOT OPEN THE LOG FILE " + this->log_path);
	}

	throw std::runtime_error(message);
}

void Log::log_message_info(std::string message) {
	std::fstream log_file(this->log_path,
			std::fstream::out | std::fstream::app);
	if (log_file.good()) {
		log_file << "INFO_MESSAGE:" + message << std::endl;
		log_file.close();

	} else {
		throw std::runtime_error(
				"COULD NOT OPEN THE LOG FILE " + this->log_path);

	}
}

} /* namespace radiation */

