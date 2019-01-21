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

public:
	Command(std::string kill_cmd, std::string exec_cmd, size_t time_acc = 0);
	Command();

	Command(const Command& b);

	friend std::ostream& operator<<(std::ostream& stream, const Command& cmd);
	friend std::istream& operator>>(std::istream& stream, Command& cmd);

	bool operator==(const Command& rhs) const;

	Command& operator=(const Command arg) noexcept;

	bool kill();

	bool execute_command();

	std::string get_exec_command();

	size_t generate_hash_number();

};

} /* namespace radiation */

#endif /* COMMAND_H_ */
