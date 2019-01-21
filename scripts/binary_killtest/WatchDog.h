/*
 * WatchDog.h
 *
 *  Created on: 20/01/2019
 *      Author: fernando
 */

#ifndef WATCHDOG_H_
#define WATCHDOG_H_

#include <string>

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


	static void signal_handler(int signal);
	static void interrupt_processing(int signal);

public:
//    installDir = config.get('DEFAULT', 'installdir') + "/"
//    varDir = config.get('DEFAULT', 'vardir') + "/"
//    logDir = config.get('DEFAULT', 'logdir') + "/"
//    tmpDir = config.get('DEFAULT', 'tmpdir') + "/"
	WatchDog(size_t timestamp_max_diff, std::string server_address,
			int server_port, std::string install_dir, std::string var_dir,
			std::string log_path, std::string tmp_dir, int max_kill_times);

	void watch(std::vector<Command>& command_list);

    Command select_command(std::vector<Command>& command_list);

    void kill_all(std::vector<Command>& command_list);

};

} /* namespace radiation */

#endif /* WATCHDOG_H_ */
