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

class WatchDog {
private:

	size_t timestamp_max_diff;

	size_t last_kill_timestamp;

	Log log;

	std::string install_dir;
	std::string var_dir;
	std::string tmp_dir;

	//Counts how many kills were executed throughout execution
	size_t kill_count;
	int max_kill_times;

	std::string server_address;
	int port;

	ClientSocket client_socket;

	std::vector<Command> command_list;

	static void signal_handler(int signal);
	static void interrupt_processing(int signal);
	Command select_command();
	void kill_all();
	void connect_and_disconnect();
	bool check_command_list_changes();
	void clean_command_exec_logs();

	Command get_command(int index);

	void exec_command(const Command& cmd);

public:
	WatchDog(const std::vector<Command>& command_list,
			size_t timestamp_max_diff, std::string server_address,
			int server_port, std::string install_dir, std::string var_dir,
			std::string log_path, std::string tmp_dir, int max_kill_times);

	void watch();

};

} /* namespace radiation */

#endif /* WATCHDOG_H_ */
