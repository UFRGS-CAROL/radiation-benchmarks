/*
 * File.h
 *
 *  Created on: Oct 1, 2019
 *      Author: carol
 */

#ifndef FILE_H_
#define FILE_H_

#include <fstream>
#define CHAR_CAST(x) (reinterpret_cast<char*>(x))

template<typename T>
struct File {

	static bool read_from_file(std::string& path, std::vector<T>& array) {
		std::ifstream input(path, std::ios::binary);
		if (input.good()) {
			input.read(CHAR_CAST(array.data()), array.size() * sizeof(T));
			input.close();
			return false;
		}
		return true;
	}

	static bool write_to_file(std::string& path, std::vector<T>& array) {
		std::ofstream output(path, std::ios::binary);
		if (output.good()) {
			output.write(CHAR_CAST(array.data()), array.size() * sizeof(T));
			output.close();

			return false;
		}
		return true;
	}

	static bool exists(std::string& path) {
		std::ifstream input(path);
		auto exists = input.good();
		input.close();
		return exists;
	}
};

#endif /* FILE_H_ */
