/*
 * utils.cpp
 *
 *  Created on: Mar 15, 2020
 *      Author: fernando
 */

#include "utils.h"
#include <stdexcept>
#include <fstream>

void __throw_line(std::string err, std::string line, std::string file) {
	throw std::runtime_error(err + "\nERROR at " + file + ":" + line);
}

bool file_exists(const std::string& name) {
	std::ifstream f(name);
	bool is_good = f.good();
	f.close();
	return is_good;
}
