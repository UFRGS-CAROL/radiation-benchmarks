/*
 * WatchDog.h
 *
 *  Created on: 20/01/2019
 *      Author: fernando
 */

#ifndef WATCHDOG_H_
#define WATCHDOG_H_

#include <string>
#include <vector>

#include "Log.h"
#include "ClientSocket.h"
#include "Command.h"

namespace radiation {

//!  Watch dog class.
/*!
 This class will watch a list of commands
 in indefinitely
 */
class WatchDog {
private:
	//!  Max timestamp difference
	/*!
	 */
	size_t timestamp_max_diff;

	//!  last kill timestamp
	/*!
	 */
	size_t last_kill_timestamp;

	//!  Log object
	/*!
	 */
	Log log;

	//! Installation directory
	/*!
	 */
	std::string install_dir;

	//! system var directory
	/*!
	 */
	std::string var_dir;

	//! system tmp directory
	/*!
	 */
	std::string tmp_dir;

	//! Kill count
	/*!
	 Counts how many kills were executed throughout execution
	 */

	size_t kill_count;

	//! max kill times
	/*!
	 */

	int max_kill_times;

	//! server address
	/*!
	 */
	std::string server_address;

	//! server port
	/*!
	 */
	int port;

	//! Client socket object
	/*!
	 */
	ClientSocket client_socket;

	//! Command list
	/*!
	 */
	std::vector<Command> command_list;

	//! Signal handler
	/*!
	 \param signal value that will be passed by the system
	 */
	static void signal_handler(int signal);

	//! Select a command in the list
	/*!
	 \return a command to be executed
	 */
	Command select_command();

	//! Kill all commands
	/*!
	 */
	void kill_all();

	//! Connect and disconnect to the server
	/*!
	 */
	void connect_and_disconnect();

	//! Check if the list was changed in reboot
	/*!
	 \return true if the list was changes, false otherwise
	 */
	bool check_command_list_changes();

	//! Clean all commands and execution logs
	/*!
	 */
	void clean_command_exec_logs();

	//! get a command based on an index
	/*!
	 \param index of the command
	 \return a command
	 */
	Command get_command(int index);

	//! Execute a command
	/*!
	 \param a command to be executed
	 */
	void exec_command(const Command& cmd);

public:
	//! WatchDog constructor
	/*!
	 \param command_list list of commands
	 \param timestamp_max_diff timestamp max diff
	 \param server_addres address of the host
	 \param server_port port to connect
	 \param install_dir installation dir
	 \param var_dir var directory
	 \param log_path log path
	 \param tmp_dir temp dir
	 \param max_kill_times maximum kill times
	 */
	WatchDog(const std::vector<Command>& command_list,
			size_t timestamp_max_diff, std::string server_address,
			int server_port, std::string install_dir, std::string var_dir,
			std::string log_path, std::string tmp_dir, int max_kill_times);

	//! Watch method
	/*!
	 */
	void watch();

};

} /* namespace radiation */

#endif /* WATCHDOG_H_ */
