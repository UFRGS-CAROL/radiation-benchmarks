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

#include "WatchDog.h"
#include "Utils.h"

namespace radiation {

#define TIME_TO_SLEEP 20

size_t timestamp_signal;
std::vector<Command> *kill_list;

WatchDog::WatchDog(size_t timestamp_max_diff, std::string server_address,
		int server_port, std::string install_dir, std::string var_dir,
		std::string log_path, std::string tmp_dir, int max_kill_times) :
		timestamp_max_diff(timestamp_max_diff), server_address(server_address), port(
				server_port), install_dir(install_dir), var_dir(var_dir), tmp_dir(
				tmp_dir), max_kill_times(max_kill_times) {

	// The SIGUSR1 and SIGUSR2 are necessary for killtest signal
	signal(SIGUSR1, this->signal_handler);
	signal(SIGUSR2, this->signal_handler);

	// SIGINT is necessary to crtl - c keys
	signal(SIGINT, this->interrupt_processing);

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

	//---------------------------------------------------------------------------
	//Debug
//	std::cout << "Server address " << this->server_address << " Server port "
//			<< this->port << " Install dir " << this->install_dir << " var dir "
//			<< this->var_dir << " tmp dir " << this->tmp_dir << std::endl;
}

void WatchDog::signal_handler(int signal) {
	if (signal == SIGUSR1 || signal == SIGUSR1) {
		timestamp_signal = get_time_since_epoch();
	}
}

void WatchDog::interrupt_processing(int signal) {
	for (auto cmd : *kill_list) {
		cmd.kill();
	}
	throw std::runtime_error(
			"\n\tKeyboardInterrupt detected, exiting gracefully!( at least trying :) )");
}

void WatchDog::kill_all(std::vector<Command>& command_list) {
	bool result = false;
	for (auto cmd : command_list)
		result = cmd.kill();
	if (result) {
		std::cerr
				<< "Could not issue the kill command for each entry, config file error!"
				<< std::endl;
	}
}

void WatchDog::watch(std::vector<Command>& command_list) {
	//to kill on crtl c
	kill_list = &command_list;

	//Initialize socket
	this->client_socket = ClientSocket(this->server_address, this->port,
			this->log);

	this->kill_count = 0;
	auto curr_command = this->select_command(command_list);
	curr_command.execute_command();

	while (true) {
		//Telling to the server that we are alive
		this->connect_and_disconnect();

		// Get the current timestamp
		auto now = get_time_since_epoch();

		// timestampDiff = now - timestamp
		auto timestamp_diff = now - timestamp_signal;

		std::cout << "Timestamp diff: " << timestamp_diff << " Now: " << now <<
				" Timestamp signal: " << timestamp_signal << std::endl;

		// If timestamp was not update properly
		if (timestamp_diff > this->timestamp_max_diff) {
			// Check if last kill was in the last 60 seconds and reboot
			this->kill_all(command_list);

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
				curr_command = this->select_command(command_list); // select properly the current command to be executed
				curr_command.execute_command(); //start the command
			}
		}
		sleep(1);
	}

}

Command WatchDog::select_command(std::vector<Command>& command_list) {
	//TODO: FINISH IT
	return Command();
}

void WatchDog::connect_and_disconnect() {
	//This function works only for radiation tests
	//TODO: Change it to work with messages
	this->client_socket.connect_host();
	this->client_socket.disconnect_host();
}

} /* namespace radiation */
