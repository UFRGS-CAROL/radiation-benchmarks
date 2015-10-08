#!/usr/bin/python

import socket

#create an INET, STREAMing socket
s = socket.socket(
    socket.AF_INET, socket.SOCK_STREAM)
#now connect to the web server on port 80
# - the normal http port
#s.connect(("feliz", 8080))
s.connect(("143.54.10.104", 8080))
