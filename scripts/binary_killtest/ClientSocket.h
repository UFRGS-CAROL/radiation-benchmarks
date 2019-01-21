/*
 * ClientSocket.h
 *
 *  Created on: 19/01/2019
 *      Author: fernando
 */

#ifndef CLIENTSOCKET_H_
#define CLIENTSOCKET_H_

#include<string.h>	//strlen
#include<string>	//string
#include<sys/socket.h>	//socket
#include<arpa/inet.h>	//inet_addr
#include<netdb.h>	//hostent

#include "Log.h"

namespace radiation {

//!  Client Socket class.
/*!
 This class will be accountable to make network connections
 */
class ClientSocket {
private:
	//! Socket id
	/*! Socket file descriptor, that must be opened by socket function
	 * and closed by close function
	 */
	int sock;

	//! Host.
	/*! Host address to be connected */
	std::string address;

	//! Port.
	/*! network port. */
	int port;

	//! Server.
	/*! struct for the server. */
	struct sockaddr_in server;

	//! Connected var.
	/*! boolean to save the conection state. */
	bool connected;

	Log log;

public:
	void connect_host();
	void disconnect_host();

	//! Client Socket Constructor.
	/*!
	 Default Constructor for the ClientSocket class
	 */
	ClientSocket();

	//! Client Socket Constructor.
	/*!
	 Constructor for the ClientSocket class that receives two parameters
	 \param address the ip address
	 \param log logfile obj
	 */
	ClientSocket(std::string address, int port, Log& log);

	//! ClientSocket destructor.
	/*!
	 It will close the socket connection
	 */
	virtual ~ClientSocket();
};

} /* namespace radiation */

#endif /* CLIENTSOCKET_H_ */
