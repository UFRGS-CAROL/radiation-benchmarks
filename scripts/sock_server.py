#!/usr/bin/python
import threading
import socket
import time
from datetime import datetime
import sys
#datetime.fromtimestamp(time.time())

socketPort = 8080 # PORT the socket will listen to
sleepTime = 5 # Time between checks
timeDiffReboot = 15 # Time in seconds since last connection to reboot machine
timeDiffBootProblem = 40 # Time in seconds since last connection to stop trying to reboot machine

# Add the machines IP to check
IPmachines = ["143.54.10.104", "143.54.10.105"]



serverSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
def startSocket():
	# Create an INET, STREAMing socket
	# Bind the socket to a public host, and a well-known port
	serverSocket.bind((socket.gethostbyname(socket.gethostname()), socketPort))
	print "\tServer bind to: ",socket.gethostbyname(socket.gethostname())
	# Become a server socket
	serverSocket.listen(15)
	
	while 1:
		# Accept connections from outside
		(clientSocket, address) = serverSocket.accept()
		print "connection from "+str(address[0])
		if address[0] in IPmachines:
			IPLastConn[address[0]]=time.time() # Set new timestamp
		clientSocket.close()

def checkMachines():
	for address, timestamp in IPLastConn.copy().iteritems():
		now = datetime.now()
		then = datetime.fromtimestamp(timestamp)
		seconds = (now - then).total_seconds()
		if seconds > timeDiffReboot and seconds < timeDiffBootProblem:
			reboot = datetime.fromtimestamp(rebooting[address])
			if (now - reboot).total_seconds() > timeDiffReboot:
				rebooting[address] = time.time()
				print "Rebooting IP "+address

class handleMachines(threading.Thread):
	def __init__(self):
		threading.Thread.__init__(self)
	def run(self):
		print "\tStarting thread to check machine connections"
		while 1:
			checkMachines()
			time.sleep(sleepTime)

################################################
# Main Execution
################################################

try:
	# Set the initial timestamp for all IPs
	IPLastConn = dict()
	rebooting = dict()
	for ip in IPmachines:
		rebooting[ip]=time.time()
		IPLastConn[ip]=time.time() # Current timestamp
	
	handle = handleMachines()
	handle.setDaemon(True)
	handle.start()
	startSocket()
except KeyboardInterrupt:
	print "\n\tKeyboardInterrupt detected, exiting gracefully!( at least trying :) )"
	serverSocket.close()
	sys.exit(1)


