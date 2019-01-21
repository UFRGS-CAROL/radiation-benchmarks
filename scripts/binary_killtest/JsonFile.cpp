/*
 * JsonFile.cpp
 *
 *  Created on: 19/01/2019
 *      Author: fernando
 */

#include <JsonFile.h>
#include <stdexcept>
#include <regex>
#include <fstream>
#include <json.hpp>
#include <iostream>

#define MIN_LINE_SIZE 7

namespace radiation {

JsonFile::JsonFile(std::string file_path) {
	std::ifstream f(file_path);
	if (f.good()) {
		std::string file_content((std::istreambuf_iterator<char>(f)),
				std::istreambuf_iterator<char>());

		int count = 0;
		size_t pos = 0;
		while ((pos = file_content.find("}", pos)) != std::string::npos) {
			++count;
			++pos;
		}

		//count * 2 + 2
		// I am counting the [] and commas
		std::regex pattern("[({.*})]");
		std::vector<std::string> vector_of_applications(100);

		std::copy(
				std::sregex_token_iterator(file_content.begin(),
						file_content.end(), pattern, -1),
				std::sregex_token_iterator(), vector_of_applications.begin());

		//put it into the vector
		for (auto command_line : vector_of_applications) {
			if (command_line.size() > MIN_LINE_SIZE) {
				command_line.insert(0, "{");
				command_line.insert(command_line.size(), "}");

				auto json_line = nlohmann::json::parse(command_line);

				std::pair<std::string, std::string> to_execute;
				json_line.at("killcmd").get_to(to_execute.first);
				json_line.at("exec").get_to(to_execute.second);

				this->all_json_lines.push_back(to_execute);
			}
		}

		f.close();
	}
}

std::pair<std::vector<std::pair<std::string, std::string> >::iterator,
		std::vector<std::pair<std::string, std::string> >::iterator> JsonFile::get_all_command_lines() {
	return std::make_pair(this->all_json_lines.begin(),
			this->all_json_lines.end());
}

std::ostream& operator<<(std::ostream& stream, const JsonFile& jf) {
	for (auto j : jf.all_json_lines) {
		stream << "KILLCMD:" << j.first << " EXEC: " << j.second << std::endl;
	}

	return stream;
}

} /* namespace radiation */
