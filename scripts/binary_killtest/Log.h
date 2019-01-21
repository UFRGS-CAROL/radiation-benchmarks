/*
 * Log.h
 *
 *  Created on: 20/01/2019
 *      Author: fernando
 */

#ifndef LOG_H_
#define LOG_H_

#include <string>

namespace radiation {

class Log {
private:
	std::string log_path;

public:
	Log(std::string log_file_path);

	Log(const Log& b);

	Log();

	void log_message_exception(std::string message);

	void log_message_info(std::string message);
};

} /* namespace radiation */

#endif /* LOG_H_ */
