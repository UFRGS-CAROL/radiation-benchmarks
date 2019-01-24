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

//!  Command class.
/*!
 This class will be accountable to run commands on
 the killtest
 */
class Command {
private:
	//! Kill command string
	/*! Kill command string, this string
	 * is extracted from json file. This will run when
	 * the application must be killed
	 */
	std::string kill_command;

	//! Execute command
	/*! Execute command string, this will be executed.
	 * It is extracted from json file.
	 * */
	std::string exec_command;

public:
	//! Constructor.
	/*!
	 * Constructor that has two parameters
	 \param kill_cmd kill command string
	 \param exec_cmd command that will be executed
	 */
	Command(std::string kill_cmd, std::string exec_cmd);

	//! Default constructor.
	/*!
	 Default constructor for the Command class
	 */
	Command();

	//! Copy constructor.
	/*!
	 Copy constructor for the Command class
	 \param b the object to be copied
	 */
	Command(const Command& b);

	//! Output stream operator
	/*!
	 \return the output stream
	 */
	friend std::ostream& operator<<(std::ostream& stream, const Command& cmd);

	//! Input stream operator
	/*!
	 \return the input stream
	 */
	friend std::istream& operator>>(std::istream& stream, Command& cmd);

	//! Comparator operator
	/*!
	 \return True if the two Commands are equal
	 */
	bool operator==(const Command& rhs) const;

	//! Assignment operator
	/*!
	 */
	Command& operator=(const Command arg) noexcept;

	//! Kill method
	/*! This method will execute the kill_cmd on
	 * the system() function call
	 \return True if the kill was correctly executed
	 */
	bool kill();

	//! Execute method
	/*! This method will execute the command on
	 * the system() function call
	 \return True if the kill was correctly executed
	 */
	bool execute_command() const;

	//! Get the execute command
	/*!
	 \return the command string
	 */
	std::string get_exec_command() const;

	//! Generate a hash from command string
	/*!
	 \return a size_t number equivalent to the string hash value
	 */
	size_t generate_hash_number();

};

} /* namespace radiation */

#endif /* COMMAND_H_ */
