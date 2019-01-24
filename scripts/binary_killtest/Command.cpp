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

Command::Command(std::string kill_cmd, std::string exec_cmd) {
	this->kill_command = kill_cmd;
	this->exec_command = exec_cmd;
}

Command::Command() {
	this->kill_command = "";
	this->exec_command = "";
}

Command::Command(const Command& b) {
	this->exec_command = b.exec_command;
	this->kill_command = b.kill_command;
}

std::ostream& operator<<(std::ostream& stream, const Command& cmd) {
	stream << cmd.kill_command << cmd.exec_command;
	return stream;
}

std::istream& operator>>(std::istream& stream, Command& cmd) {
	stream >> cmd.kill_command;
	stream >> cmd.exec_command;
	return stream;
}

bool Command::operator==(const Command& rhs) const {
	return (this->exec_command == rhs.exec_command
			&& this->kill_command == rhs.kill_command);
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
	return *this;
}

bool Command::execute_command() const {
	std::string cmd = this->exec_command;
	//Search & after half of the string
	if (!(cmd.find('&', cmd.size() / 2) != std::string::npos))
		cmd += '&';
	return (system(cmd.c_str()) != 0);
}

std::string Command::get_exec_command() const {
	return this->exec_command;
}

size_t Command::generate_hash_number() {
	std::string final_string = this->exec_command;
	std::hash < std::string > hasher;
	size_t hashed = hasher(final_string);
	return hashed;
}

} /* namespace radiation */

