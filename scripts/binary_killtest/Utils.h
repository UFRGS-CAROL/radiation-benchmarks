/*
 * Utils.h
 *
 *  Created on: 20/01/2019
 *      Author: fernando
 */
#ifndef UTILS_H_
#define UTILS_H_

#include <vector>
#include <string>
#include <sstream>
#include <ctime>
#include <sys/reboot.h>
#include <unistd.h>
#include <chrono>         // std::chrono::seconds
#include <thread>

static std::vector<std::string> &split(const std::string &s, char delim,
		std::vector<std::string> &elems) {
	std::stringstream ss(s);
	std::string item;
	while (getline(ss, item, delim)) {
		elems.push_back(item);
	}
	return elems;
}

static std::vector<std::string> split(const std::string &s, char delim) {
	std::vector < std::string > elems;
	split(s, delim, elems);
	return elems;
}

static size_t get_time_since_epoch() {
	std::time_t result = std::time(nullptr);
	return size_t(result);
}

static void system_reboot() {
	sync();
	reboot (RB_AUTOBOOT);
}

static inline void sleep(int seconds) {
	std::this_thread::sleep_for(std::chrono::seconds(seconds));
}

#endif /* UTILS_H_ */
