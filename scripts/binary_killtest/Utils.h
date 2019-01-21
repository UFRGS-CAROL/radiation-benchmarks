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

#include <stdio.h> // Remove file

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

static inline void system_reboot() {
//	system("shutdown -r now");
	sync();
	reboot (RB_AUTOBOOT);
}

static inline void sleep(int seconds) {
	std::this_thread::sleep_for(std::chrono::seconds(seconds));
}

static inline void rm(std::string file) {
//	system(("rm -f " + file).c_str());
	if (remove(file.c_str()) != 0)
		std::cout << "Error deleting file" << std::endl;
}

static inline bool is_file(std::string file) {
	std::ifstream test(file);
	bool r = test.good();
	test.close();
	return r;
}

#endif /* UTILS_H_ */
