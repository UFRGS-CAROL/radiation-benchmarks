/*
 * WatchDog.cpp
 *
 *  Created on: 20/01/2019
 *      Author: fernando
 */

#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <signal.h>
#include <thread>
#include <fstream>

#include "WatchDog.h"
#include "Utils.h"

namespace radiation {

//Default time to sleep
#define TIME_TO_SLEEP 20
//Cross section is calculated based on
//1h so the default time window is
//60 * 60
#define TIME_WINDOW 60 * 60

size_t timestamp_signal;
const std::vector<Command> *kill_list;

WatchDog::WatchDog(const std::vector<Command>& command_list,
		size_t timestamp_max_diff, std::string server_address, int server_port,
		std::string install_dir, std::string var_dir, std::string log_path,
		std::string tmp_dir, int max_kill_times) :
		command_list(command_list), timestamp_max_diff(timestamp_max_diff), server_address(
				server_address), port(server_port), install_dir(install_dir), var_dir(
				var_dir), tmp_dir(tmp_dir), max_kill_times(max_kill_times) {

	// The SIGUSR1 and SIGUSR2 are necessary for killtest signal
	signal(SIGUSR1, this->signal_handler);
	signal(SIGUSR2, this->signal_handler);

	// SIGINT is necessary to crtl - c keys
	signal(SIGINT, this->signal_handler);

	//---------------------------------------------------------------------------
	// Start last kill timestamp with an old enough timestamp
	// The last benchmarks must had not been run more than 50*max_diff
	this->last_kill_timestamp = get_time_since_epoch()
			- 50 * this->timestamp_max_diff;

	//---------------------------------------------------------------------------
	//Outside scope var initialization
	timestamp_signal = get_time_since_epoch();

	this->kill_count = 0;
	this->log = Log(log_path);

	//to kill on crtl c
	kill_list = &this->command_list;
	//---------------------------------------------------------------------------
	//Debug
//	std::cout << "Server address " << this->server_address << " Server port "
//			<< this->port << " Install dir " << this->install_dir << " var dir "
//			<< this->var_dir << " tmp dir " << this->tmp_dir << std::endl;
}

void WatchDog::signal_handler(int signal) {
	switch (signal) {
	case SIGUSR1:
		timestamp_signal = get_time_since_epoch();
		break;
	case SIGUSR2:
		timestamp_signal = get_time_since_epoch();
		break;
	case SIGINT:
		for (auto cmd : *kill_list) {
			cmd.kill();
		}
		throw std::runtime_error(
				"\n\tKeyboardInterrupt detected, exiting gracefully!( at least trying :) )");
		break;
	}
}

void WatchDog::kill_all() {
	bool result = false;
	for (auto cmd : this->command_list)
		result = cmd.kill();
	if (result) {
		std::cerr
				<< "Could not issue the kill command for each entry, config file error!"
				<< std::endl;
	}
}

void WatchDog::watch() {

	//Initialize socket
	this->client_socket = ClientSocket(this->server_address, this->port);

	this->kill_count = 0;
	auto curr_command = this->select_command();
	this->exec_command(curr_command);

	while (true) {
		//Telling to the server that we are alive
		this->connect_and_disconnect();

		// Get the current timestampWatchDog
		auto now = get_time_since_epoch();

		// timestampDiff = now - timestamp
		auto timestamp_diff = now - timestamp_signal;

		std::cout << "Timestamp diff: " << timestamp_diff << " Now: " << now
				<< " Timestamp signal: " << timestamp_signal << std::endl;

		// If timestamp was not update properly
		if (timestamp_diff > this->timestamp_max_diff) {
			// Check if last kill was in the last 60 seconds and reboot
			this->kill_all();

			now = get_time_since_epoch();

			if ((now - this->last_kill_timestamp)
					< 3 * this->timestamp_max_diff) {
				this->log.log_message_info(
						"Rebooting, last kill too recent, timestampDiff: "
								+ std::to_string(timestamp_diff)
								+ ", current command:"
								+ curr_command.get_exec_command());

				//Telling to the server that we are alive
				this->connect_and_disconnect();

				system_reboot();
				sleep(TIME_TO_SLEEP);
			} else {
				this->last_kill_timestamp = now;
			}

			this->kill_count += 1;

			this->log.log_message_info(
					"timestampMaxDiff kill(#" + std::to_string(kill_count)
							+ "), timestampDiff:"
							+ std::to_string(timestamp_diff) + " command '"
							+ curr_command.get_exec_command() + "'");

			// Reboot if we reach the max number of kills allowed
			if (this->kill_count >= this->max_kill_times) {
				this->log.log_message_info(
						"Rebooting, maxKill reached, current command:"
								+ curr_command.get_exec_command());

				//Telling to the server that we are alive
				this->connect_and_disconnect();

				system_reboot();
				sleep(TIME_TO_SLEEP);
			} else {
				curr_command = this->select_command(); // select properly the current command to be executed
				this->exec_command(curr_command);
			}
		}
		sleep(1);
	}

}

void WatchDog::exec_command(const Command& cmd) {
	//start the command
	if (cmd.execute_command() == false)
		this->log.log_message_info(
				"Error launching command '" + cmd.get_exec_command() + "';'");
}

Command WatchDog::select_command() {
	if (this->check_command_list_changes()) {
		this->clean_command_exec_logs();
	}

	// Get the index of last existent file
	auto i = 0;
	while (is_file(this->var_dir + "/command_execstart_" + std::to_string(i)))
		i++;

	i--;

	// If there is no file, create the first file with current timestamp
	// and return the first command of commands list
	if (i == -1) {
		std::ofstream ofp(this->var_dir + "/command_execstart_0");
		ofp << get_time_since_epoch();
		ofp.close();
		return this->get_command(0);
	}

	//------------------------------------------------------------------------------------
	// Check if last command executed is still in the defined time window for each command
	// and return it

	// Read the timestamp file
	auto timestamp_path = this->var_dir + "/command_execstart_"
			+ std::to_string(i);
	size_t timestamp = 0;

	if (is_file(timestamp_path)) {
		std::ifstream ifp(timestamp_path);
		ifp >> timestamp;
		ifp.close();
	} else {
		this->log.log_message_info(
				"Rebooting, command execstart timestamp read error");
		this->connect_and_disconnect();
		system(("rm -f " + timestamp_path).c_str());
		system_reboot();
		sleep(TIME_TO_SLEEP);
	}

	auto now = get_time_since_epoch();
	if ((now - timestamp) < TIME_WINDOW) {
		return this->get_command(i);
	}

	//Increment the command
	i++;

	// If all commands executed their time window, start all over again
	if (i >= this->command_list.size()) {
		this->clean_command_exec_logs();
		std::ofstream ofp(this->var_dir + "/command_execstart_0");
		ofp << get_time_since_epoch();
		ofp.close();
		return this->get_command(0);
	}

	// Finally, select the next command not executed so far
	timestamp_path = this->var_dir + "/command_execstart_" + std::to_string(i);
	std::ofstream ofp(timestamp_path);
	ofp << timestamp;
	ofp.close();

	return this->get_command(i);

}

void WatchDog::connect_and_disconnect() {
	//This function works only for radiation tests
	this->client_socket.connect_host();
	this->client_socket.disconnect_host();
}

bool WatchDog::check_command_list_changes() {
	auto current_file = this->var_dir + "/currentCommandFile";
	auto last_file = this->var_dir + "/lastCommandFile";
	std::string curren_file_content = "";
	//----------------------------------------------------------------

	//Save the current command
	for (auto current_command : this->command_list)
		curren_file_content += std::to_string(
				current_command.generate_hash_number());

	//save if last file does not exist
	if (is_file(last_file) == false) {
		std::ofstream curr_of(last_file);
		if (curr_of.good()) {
			curr_of << curren_file_content;
		}
		curr_of.close();
		return true;
	}

	std::ifstream last_if(last_file);
	std::string last_file_content((std::istreambuf_iterator<char>(last_if)),
			std::istreambuf_iterator<char>());
	last_if.close();
	if (curren_file_content == last_file_content) {
		return false;
	} else {
		std::ofstream curr_of(last_file);
		if (curr_of.good()) {
			curr_of << curren_file_content;
		}
		curr_of.close();
		return true;
	}

}

void WatchDog::clean_command_exec_logs() {
	system(("rm -f " + this->var_dir + "/command_execstart_*").c_str());
}

Command WatchDog::get_command(int index) {
	return this->command_list[index];
}

} /* namespace radiation */
