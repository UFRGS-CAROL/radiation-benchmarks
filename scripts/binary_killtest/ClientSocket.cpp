/*
 * ClientSocket.cpp
 *
 *  Created on: 19/01/2019
 *      Author: fernando
 */

#include <iostream>
#include <unistd.h>

#include "ClientSocket.h"

namespace radiation {

ClientSocket::ClientSocket() :
		address(""), connected(false), port(0), sock(-1) {
}

ClientSocket::ClientSocket(std::string address, int port) :
		address(address), port(port), connected(false), sock(-1) {
}

ClientSocket::~ClientSocket() {
	this->disconnect_host();
}

void ClientSocket::disconnect_host() {
	if (this->connected) {
		shutdown(this->sock, SHUT_RDWR);
		close(this->sock);
		this->connected = false;
		this->sock = -1;
	}
}

void ClientSocket::connect_host() {
	this->connected = false;
	//create socket if it is not already created
	if (this->sock == -1) {
		//Create socket
		this->sock = socket(AF_INET, SOCK_STREAM, 0);
		if (this->sock == -1) {
			std::cerr << "CLIENT_SOCKET - Could not create socket";
			return;
		}
	}

	//plain ip address
	this->server.sin_addr.s_addr = inet_addr(this->address.c_str());
	this->server.sin_family = AF_INET;
	this->server.sin_port = htons(this->port);

	//Connect to remote server
	if (connect(this->sock, (struct sockaddr*) (&this->server),
			sizeof(this->server)) < 0) {
		std::cerr << "CLIENT_SOCKET - connect failed. Error" << std::endl;

		//Sock already open so close it first
		shutdown(this->sock, SHUT_RDWR);
		this->connected = false;
		close(this->sock);
		return;
	}
	this->connected = true;
}

} /* namespace radiation */

