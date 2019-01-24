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

//!  string split .
/*!
 Split a string s based on delim char
 \param string s to be split
 \param delimiter that string s will be split
 \return std vector with all elements split
 */

static std::vector<std::string> split(const std::string &s, char delim) {
	std::vector < std::string > elems;
	std::stringstream ss(s);
	std::string item;
	while (getline(ss, item, delim)) {
		elems.push_back(item);
	}
	return elems;
}

//!  get time since epoch .
/*!
 \return an int value representing the seconds since epoch
 */
static size_t get_time_since_epoch() {
	std::time_t result = std::time(nullptr);
	return size_t(result);
}

//!  reboot the system .
/*!
 */
static inline void system_reboot() {
//	system("shutdown -r now");
	sync();
	reboot (RB_AUTOBOOT);
}

//!  sleep seconds.
/*!
 \param seconds to sleep
 */
static inline void sleep(int seconds) {
	std::this_thread::sleep_for(std::chrono::seconds(seconds));
}

//! remove a file.
/*!
 \param file to be removed
 */
static inline void rm(std::string file) {
//	system(("rm -f " + file).c_str());
	if (remove(file.c_str()) != 0)
		std::cout << "Error deleting file" << std::endl;
}

//!  check if file is file.
/*!
 \param file path
 \return a boolean if file is file
 */
static inline bool is_file(std::string file) {
	std::ifstream test(file);
	bool r = test.good();
	test.close();
	return r;
}

#endif /* UTILS_H_ */
