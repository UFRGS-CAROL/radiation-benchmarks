/*
 * Command.cpp
 *
 *  Created on: 20/01/2019
 *      Author: fernando
 */

#include "Command.h"
#include <iostream>
#include <functional> //for std::hash

namespace radiation {

Command::Command(std::string kill_cmd, std::string exec_cmd, size_t time_acc) {
	this->kill_command = kill_cmd;
	this->exec_command = exec_cmd;
	this->time_acc = time_acc;
}

Command::Command() {
	this->kill_command = "";
	this->exec_command = "";
	this->time_acc = 0;
}

Command::Command(const Command& b) {
	this->exec_command = b.exec_command;
	this->kill_command = b.kill_command;
	this->time_acc = b.time_acc;
}

std::ostream& operator<<(std::ostream& stream, const Command& cmd) {
	stream << cmd.kill_command << std::endl << cmd.exec_command << std::endl
			<< cmd.time_acc;
	return stream;
}

std::istream& operator>>(std::istream& stream, Command& cmd) {
	stream >> cmd.kill_command;
	stream >> cmd.exec_command;
	stream >> cmd.time_acc;
	return stream;
}

bool Command::operator==(const Command& rhs) const {
	return (this->exec_command == rhs.exec_command
			&& this->kill_command == rhs.kill_command
			&& this->time_acc == rhs.time_acc);
}



bool Command::kill() {
	if (system(this->kill_command.c_str())) {
		return true;
	}
	return false;
}

Command& Command::operator=(const Command arg) noexcept {
	this->exec_command = arg.exec_command;
	this->kill_command = arg.kill_command;
	this->time_acc = arg.time_acc;

	return *this;
}

bool Command::execute_command() {
	std::string cmd = this->exec_command;
	//Search & after half of the string
	if (!(cmd.find('&', cmd.size() / 2) != std::string::npos))
		cmd += '&';
	return (system(cmd.c_str()) != 0);
}

std::string Command::get_exec_command() {
	return this->exec_command;
}

size_t Command::generate_hash_number() {
	std::string final_string = this->exec_command + this->kill_command;
	std::hash<std::string> hasher;
	size_t hashed = hasher(final_string);
	return hashed;
}

} /* namespace radiation */

