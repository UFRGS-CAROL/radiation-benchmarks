/*
 * main.cpp
 *
 *  Created on: 19/01/2019
 *      Author: fernando
 */

#include <iostream>
#include <unistd.h>
#include <unordered_map>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>

#include "WatchDog.h"
#include "Utils.h"
#include "JsonFile.h"

#define TIMESTAMP_MAX_DIFF_DEFAULT 30
#define CONFIG_FILE "/etc/radiation-benchmarks.conf"
#define MAX_TIME_KILL 5
#define SOCK_SERVER_IP "192.168.0.9"
#define SOCK_SERVER_PORT 1234

#define MAX_KILL_TIMES 5

std::unordered_map<std::string, std::string> parse_config_data(
		std::string config_file_path) {
	std::ifstream config_file_obj(config_file_path);

	std::unordered_map < std::string, std::string > config_file_data;

	if (config_file_obj.good()) {
		std::string str;
		std::vector < std::string > file_contents;

		// Read the next line from File untill it reaches the end.
		while (std::getline(config_file_obj, str)) {
			// Line contains string of length > 0 then save it in vector
			if (str.size() > 0)
				file_contents.push_back(str);
		}

		for (auto line : file_contents) {
			auto found = line.find('=');
			if (found != std::string::npos) {
				std::vector < std::string > split_string = split(line, '=');
				if (split_string.size() != 2) {
					throw std::runtime_error(
							"Cannot read the config file at "
									+ config_file_path);
				}

				//Make sure that spaces will not crash
				// our killtest
				split_string[0].erase(
						std::remove(split_string[0].begin(),
								split_string[0].end(), ' '),
						split_string[0].end());

				split_string[1].erase(
						std::remove(split_string[1].begin(),
								split_string[1].end(), ' '),
						split_string[1].end());

				config_file_data[split_string[0]] = split_string[1];
			}
		}
		config_file_obj.close();
	} else {
		throw std::runtime_error("Bad config file at " + config_file_path);
	}

	return config_file_data;
}

std::vector<std::string> read_json_paths(std::string text_path) {
	std::ifstream file(text_path);
	std::string str;
	std::vector < std::string > ret;
	if (file.good()) {
		while (std::getline(file, str)) {
			//Check if the json line is a file
			if (is_file(str)) {
				ret.push_back(str);
			}
		}

		file.close();
	}
	return ret;
}

int main(int argc, char **argv) {
	if (argc < 2) {
		throw std::runtime_error(
				"Usage " + std::string(argv[0])
						+ " <file with absolute paths of json files>");
	}

	std::string json_file_path = argv[1];
	if (access(argv[1], F_OK) != -1) {
		//Generate a vector with all commands
		auto list_of_jsons = read_json_paths(json_file_path);
		std::vector<radiation::Command> command_vector;

		//iterate over the json files
		for (auto jf : list_of_jsons) {
			//Json object to extract data from Json
			radiation::JsonFile json_file(jf);

			//get the iterator for the lines
			auto lines_iterator_pair = json_file.get_all_command_lines();

			for (auto it = lines_iterator_pair.first;
					it != lines_iterator_pair.second; ++it) {
				radiation::Command tmp((*it).first, (*it).second);
				command_vector.push_back(tmp);
			}
		}

		// Parse the configuration file
		auto config_hash = parse_config_data(CONFIG_FILE);

		//Setup the watchdog
		std::string log_path = config_hash["logdir"] + "/killtest.log";
		std::string install_dir = config_hash["installdir"];
		std::string var_dir = config_hash["vardir"];
		std::string tmp_dir = config_hash["tmpdir"];

		radiation::WatchDog lessie(command_vector, TIMESTAMP_MAX_DIFF_DEFAULT,
				SOCK_SERVER_IP,
				SOCK_SERVER_PORT, install_dir, var_dir, log_path, tmp_dir,
				MAX_KILL_TIMES);

		lessie.watch();
	}

	return 0;
}
