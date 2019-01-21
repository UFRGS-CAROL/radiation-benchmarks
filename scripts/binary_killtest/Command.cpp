/*
 * Command.cpp
 *
 *  Created on: 20/01/2019
 *      Author: fernando
 */

#include "Command.h"
#include <iostream>

namespace radiation {

Command::Command(std::string kill_cmd, std::string exec_cmd,
		size_t time_window) {
	this->kill_command = kill_cmd;
	this->exec_command = exec_cmd;
	this->time_acc = 0;
	this->time_window = time_window;
}

Command::Command() {
	this->kill_command = "";
	this->exec_command = "";
	this->time_acc = 0;
	this->time_window = -1;
}

Command::Command(const Command& b) {
	this->exec_command = b.exec_command;
	this->kill_command = b.kill_command;
	this->time_acc = b.time_acc;
	this->time_window = b.time_window;
}

std::ostream& operator<<(std::ostream& stream, const Command& cmd) {
	stream << std::string("KILL CMD: ") << cmd.kill_command << std::endl << "EXEC CMD: "
			<< cmd.exec_command << std::endl << "TIME ACC: " << cmd.time_acc
			<< std::endl;
	return stream;
}

bool Command::kill() {
	//TODO: FINISH
	return false;
}

void Command::execute_command() {
	//TODO:FINISH
	std::cout << this->exec_command << std::endl;
}

std::string Command::get_exec_command() {
	return this->exec_command;
}

} /* namespace radiation */

