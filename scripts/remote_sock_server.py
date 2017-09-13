#!/usr/bin/python
import threading
import socket
import time
import os
from datetime import datetime
import sys
#datetime.fromtimestamp(time.time())

socketPort = 8080 # PORT the socket will listen to
sleepTime = 5 # Time between checks
timeDiffReboot = 30 # Time in seconds since last connection to reboot machine
timeDiffBootProblem = 120 # Time in seconds since last connection to stop trying to reboot machine

serverIP = "192.168.1.5" # IP of the remote socket server (hardware watchdog)
#serverIP = socket.gethostbyname(socket.gethostname())

# Set the machines IP to check, comment the ones we are not checking
IPmachines = [
	"192.168.1.1",  #CarolK201
	"192.168.1.2",  #CarolK401
	"192.168.1.6",  #CarolXeon1
	"192.168.1.7",  #CarolXeon2
	"192.168.1.10", #CarolAPU1
	"192.168.1.11", #CarolAPU2
	"192.168.1.12", #CarolX2A
	"192.168.1.13", #CarolX2B

        # Not used:
#	"192.168.1.8",  #CarolK401
#	"192.168.1.9",  #CarolK402
#	"192.168.1.14", #CarolK1C
#	"192.168.1.15", #CarolHSA2
#	"192.168.1.20", #PI_1
#	"192.168.1.21", #PI_2
#	"192.168.1.22", #PI_3
]

# Set the machine names for each IP
IPtoNames = {
	"192.168.1.1" : "CarolK201",
	"192.168.1.2" : "CarolK401",
	"192.168.1.6" : "CarolXeon1",
	"192.168.1.7" : "CarolXeon2",
	"192.168.1.10" : "CarolAPU1",
	"192.168.1.11" : "CarolAPU2",
	"192.168.1.12" : "CarolX2A",
	"192.168.1.13" : "CarolX2B",

        # Not used:
#	"192.168.1.8" : "CarolK401",
#	"192.168.1.9" : "CarolK402",
#	"192.168.1.14" : "CarolK1C",
#	"192.168.1.15" : "CarolHSA2",
#	"192.168.1.20" : "PI_1",
#	"192.168.1.21" : "PI_2",
#	"192.168.1.22" : "PI_3",
}

# Set the switch IP that a machine IP is connected
IPtoSwitchIP = {
	"192.168.1.1" : "192.168.1.100",  #CarolK201
	"192.168.1.2" : "192.168.1.102",  #CarolK401
	"192.168.1.6" : "192.168.1.100",  #CarolXeon1
	"192.168.1.7" : "192.168.1.101",  #CarolXeon2
	"192.168.1.10" : "192.168.1.101", #CarolAPU1
	"192.168.1.11" : "192.168.1.102", #CarolAPU2
	"192.168.1.12" : "192.168.1.101", #CarolX2A
	"192.168.1.13" : "192.168.1.102", #CarolX2B

        # Not used:
#	"192.168.1.8" : "192.168.1.100",  #CarolK401
#	"192.168.1.9" : "192.168.1.102",  #CarolK402
#	"192.168.1.14" : "192.168.1.100", #CarolK1C
#	"192.168.1.15" : "192.168.1.101", #CarolHSA2
#	"192.168.1.20" : "192.168.1.101", #PI_1
#	"192.168.1.21" : "192.168.1.101", #PI_2
#	"192.168.1.22" : "192.168.1.101", #PI_3
}

# Set the switch Port that a machine IP is connected
IPtoSwitchPort = {
	"192.168.1.1" : 3,  #CarolK201
	"192.168.1.2" : 1,  #CarolK401
	"192.168.1.6" : 1,  #CarolXeon1
	"192.168.1.7" : 1,  #CarolXeon2
	"192.168.1.10" : 3, #CarolAPU1
	"192.168.1.11" : 3, #CarolAPU2
	"192.168.1.12" : 2, #CarolX2A
	"192.168.1.13" : 2, #CarolX2B

        # Not used:
#	"192.168.1.8" : 1,  #CarolK401
#	"192.168.1.9" : 1,  #CarolK402
#	"192.168.1.14" : 2, #CarolK1C
#	"192.168.1.15" : 3, #CarolHSA2
#	"192.168.1.20" : 2, #PI_1
#	"192.168.1.21" : 4, #PI_2
#	"192.168.1.22" : 2, #PI_3
}

# log in whatever path you are executing this script
logFile = "server.log"

# Log messages adding timestamp before the message
def logMsg(msg):
	now = datetime.now()
	fp = open(logFile, 'a')
	print >>fp, now.ctime()+": "+str(msg)
	fp.close()
################################################
# Routines to perform power cycle user IP SWITCH
################################################
class Switch():
    def __init__(self, ip, portCount):
        self.ip = ip
        self.portCount = portCount
        self.portList = []
        for i in range(0, self.portCount):
            self.portList.append(
                        'pw%1dName=&P6%1d=%%s&P6%1d_TS=&P6%1d_TC=&' %
                        (i+1, i, i, i)
                    )

    def cmd(self, port, c):
        assert(port <= self.portCount)

        cmd = 'curl --data \"'

        # the port list is indexed from 0, so fix it
        port = port - 1

        for i in range(0, self.portCount):
            if i == (port):
                cmd += self.portList[i] % c
            else:
                cmd += self.portList[i]

        cmd += '&Apply=Apply\" '
        cmd += 'http://%s/tgi/iocontrol.tgi ' % (self.ip)
        cmd += '-o /dev/null 2>/dev/null'
        return os.system(cmd)

    def on(self, port):
        return self.cmd(port, 'On')

    def off(self, port):
        return self.cmd(port, 'Off')

def setIPSwitch(portNumber, status, switchIP):
	s = Switch(switchIP, 4)
	if status == 'on' or status == 'On' or status == 'ON':
		cmd = 'On'
	elif status == 'off' or status == 'Off' or status == 'OFF':
		cmd = 'Off'
	else:
		return 1
	return s.cmd(int(portNumber), cmd)

class RebootMachine(threading.Thread):
	def __init__(self, address):
		threading.Thread.__init__(self)
		self.address = address
	def run(self):
		port = IPtoSwitchPort[self.address]
		switchIP = IPtoSwitchIP[self.address]
		print "\tRebooting machine: "+self.address+", switch IP: "+str(switchIP)+", switch port: "+str(port)
		setIPSwitch(port, "Off", switchIP)
		time.sleep(10)
		setIPSwitch(port, "On", switchIP)


################################################
# Socket server
################################################
# Create an INET, STREAMing socket
serverSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
def startSocket():
	# Bind the socket to a public host, and a well-known port
	serverSocket.bind((serverIP, socketPort))
	print "\tServer bind to: ",serverIP
	# Become a server socket
	serverSocket.listen(15)
	
	while 1:
		# Accept connections from outside
		(clientSocket, address) = serverSocket.accept()
		print "connection from "+str(address[0])
		if address[0] in IPmachines:
			IPLastConn[address[0]]=time.time() # Set new timestamp
			# If machine was set to not check again, now it's alive so start to check again
			IPActiveTest[address[0]]=True 
		clientSocket.close()

################################################
# Routines to check machine status
################################################
def checkMachines():
	for address, timestamp in IPLastConn.copy().iteritems():
		# If machine had a boot problem, stop rebooting it
		if not IPActiveTest[address]:
			continue

		# Check if machine is working fine
		now = datetime.now()
		then = datetime.fromtimestamp(timestamp)
		seconds = (now - then).total_seconds()
		# If machine is not working fine reboot it
		if seconds > timeDiffReboot and seconds < timeDiffBootProblem:
			reboot = datetime.fromtimestamp(rebooting[address])
			if (now - reboot).total_seconds() > timeDiffReboot:
				rebooting[address] = time.time()
				if address in IPtoNames:
					print "Rebooting IP "+address+" ("+IPtoNames[address]+")"
					logMsg("Rebooting IP "+address+" ("+IPtoNames[address]+")")
				else:
					print "Rebooting IP "+address
					logMsg("Rebooting IP "+address)
				# Reboot machine in another thread
				RebootMachine(address).start()
		# If machine did not reboot, log this and set it to not check again
		elif seconds > timeDiffBootProblem:
			if address in IPtoNames:
				print "Boot Problem IP "+address+" ("+IPtoNames[address]+")"
				logMsg("Boot Problem IP "+address+" ("+IPtoNames[address]+")")
			else:
				print "Boot Problem IP "+address
				logMsg("Boot Problem IP "+address)
			IPActiveTest[address]=False


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
	IPActiveTest = dict()
	rebooting = dict()
	for ip in IPmachines:
		rebooting[ip]=time.time()
		IPLastConn[ip]=time.time() # Current timestamp
		IPActiveTest[ip]=True
		port = IPtoSwitchPort[ip]
		switchIP = IPtoSwitchIP[ip]
		setIPSwitch(port, "On", switchIP)
	
	handle = handleMachines()
	handle.setDaemon(True)
	handle.start()
	startSocket()
except KeyboardInterrupt:
	print "\n\tKeyboardInterrupt detected, exiting gracefully!( at least trying :) )"
	serverSocket.close()
	sys.exit(1)


