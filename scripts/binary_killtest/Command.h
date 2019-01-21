/*
 * Command.h
 *
 *  Created on: 20/01/2019
 *      Author: fernando
 */

#ifndef COMMAND_H_
#define COMMAND_H_

#include <string>

namespace radiation {

class Command {
private:
	std::string kill_command;
	std::string exec_command;

	size_t time_acc;
	size_t time_window;


public:
	Command(std::string kill_cmd, std::string exec_cmd, size_t time_window);
	Command();

	Command(const Command& b);

	friend std::ostream& operator<<(std::ostream& stream, const Command& cmd);

	bool kill();

	void execute_command();

	std::string get_exec_command();

};

} /* namespace radiation */

#endif /* COMMAND_H_ */
