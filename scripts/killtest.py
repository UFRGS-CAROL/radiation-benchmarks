#!/usr/bin/python

import threading
import socket
import time
import os.path
import ConfigParser
import sys
import filecmp
from datetime import datetime
#from subprocess import call
#from subprocess import Popen

# Commands to be executed by KillTest
# each command should be in this format:
# ["command", <hours to be executed>]
# for example: 
# commandList = [
# 	["./lavaMD 15 4 input1 input2 gold 10000", 1, "lavaMD"],
# 	["./gemm 1024 4 input1 input2 gold 10000", 2, "gemm"]
# ]
# will execute lavaMD for about one hour and then execute gemm for two hours
# When the list finish executing, it start to execute from the beginning
commandList = [
	["comando 1", 1, "nameToKill (exec name)"],
	["comando 2", 2, "nameToKill (exec name)"],
	["comando 3", 5, "nameToKill (exec name)"]
]


timestampMaxDiff = 20 # Time in seconds
maxConsecKill = 2 # Max number of consective kills allowed
maxKill = 5 # Max number of kills allowed

sockServerIP = "192.168.1.5"
sockServerPORT = 8080

# Connect to server and close connection, kind of ping
def sockConnect():
	try:
		#create an INET, STREAMing socket
		s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		# Now, connect with IP (or hostname) and PORT
		# s.connect(("feliz", 8080)) or s.connect(("143.54.10.100", 8080))
		s.connect((sockServerIP, sockServerPORT))
		s.close()
	except socket.error:
		print "could not connect to remote server, socket error"

# Log messages adding timestamp before the message
def logMsg(msg):
	now = datetime.now()
	fp = open(logFile, 'a')
	print >>fp, now.ctime()+": "+str(msg)
	fp.close()
	print now.ctime()+": "+str(msg)

# Update the timestamp file with machine current timestamp
def updateTimestamp():
	command = "echo "+str(time.time())+" > "+timestampFile
	retcode = os.system(command)


# Remove files with start timestamp of commands executing
def cleanCommandExecLogs():
	i=len(commandList)
	while i >= 0:
		if os.path.isfile(varDir+"command_execstart_"+str(i)):
			os.remove(varDir+"command_execstart_"+str(i))
		i -= 1

# Return True if the variable commandList from this file changed from the 
# last time it was executed. If the file was never executed returns False
def checkCommandListChanges():
	curList = varDir+"currentCommandList"
	lastList = varDir+"lastCommandList"
	fp = open(curList,'w')
	print >>fp, commandList
	fp.close()
	if not os.path.isfile(lastList):
		fp = open(lastList,'w')
		print >>fp, commandList
		fp.close()
		return True

	if filecmp.cmp(curList, lastList, shallow=False):
		return False
	else:
		return True

# Select the correct command to be executed from the commandList variable
def selectCommand():
	if checkCommandListChanges():
		cleanCommandExecLogs()

	# Get the index of last existent file	
	i=0
	while os.path.isfile(varDir+"command_execstart_"+str(i)):
		i += 1
	i -= 1

	# If there is no file, create the first file with current timestamp
	# and return the first command of commandList
	if i == -1:
		os.system("echo "+str(time.time())+" > "+varDir+"command_execstart_0")
		return commandList[0][0]

	# Check if last command executed is still in its execution time window
	# and return it
	timeWindow = commandList[i][1] * 60 * 60 # Time window in seconds
	fp = open(varDir+"command_execstart_"+str(i),'r')
	timestamp = int(float(fp.readline().strip()))
	fp.close()
	now = time.time()
	if (now - timestamp) < timeWindow:
		return commandList[i][0]

	# If all commands executed their time window, start all over again
	if i >= len(commandList):
		cleanCommandExecLogs()
		os.system("echo "+str(time.time())+" > "+varDir+"command_execstart_0")
		return commandList[0][0]

	# Finally, select the next command not executed so far
	i += 1
	os.system("echo "+str(time.time())+" > "+varDir+"command_execstart_"+str(i))
	return commandList[i][0]


def execCommand(command):
	try:
		updateTimestamp()
		return os.system(command)
		#procPopen = Popen(command, shell=True)
		#return procPopen
	except OSError as detail:
		logMsg("Error launching command '"+command+"'; error detail: "+str(detail))
		return None


################################################
# KillTest Main Execution
################################################
confFile = '/etc/radiation-benchmarks.conf'

if not os.path.isfile(confFile):
	print >> sys.stderr, "Configuration file not found!("+confFile+")"
	sys.exit(1)

try:
	config = ConfigParser.RawConfigParser()
	config.read(confFile)
	
	installDir = config.get('DEFAULT', 'installdir')+"/"
	varDir =  config.get('DEFAULT', 'vardir')+"/"
	logDir =  config.get('DEFAULT', 'logdir')+"/"
	tmpDir =  config.get('DEFAULT', 'tmpdir')+"/"
	
	#logDir = varDir+"log/"
	
	if not os.path.isdir(logDir):
		os.mkdir(logDir, 0777)
		os.chmod(logDir, 0777)
	
except IOError as e:
	print >> sys.stderr, "Configuration setup error: "+str(e)
	sys.exit(1)
	
logFile = logDir+"killtest.log"
timestampFile = varDir+"timestamp.txt"

# Start last kill timestamo with an old enough timestamp
lastKillTimestamp = time.time() - 120

proc = None
try:
	consecKillCount = 0 # Counts how many consecutive kills to reboot machine
	killCount = 0 # Counts how many kills were executed throughout execution
	curCommand = selectCommand()
	proc = execCommand(curCommand)
	while True:
		sockConnect()
		# Read the timestamp file
		fp = open(timestampFile, 'r')
		timestamp = int(float(fp.readline().strip()))
		fp.close()
		# Get the current timestamp
		now = time.time()
	
		# If timestamp was not update properly
		if (now - timestamp) > timestampMaxDiff:
			#if proc is not None:
				#proc.kill() # Kill current command

				# Check if last kill was in the last 60 seconds and reboot
				#now = time.time()
				#if (now - lastKillTimestamp) < 3*timestampMaxDiff:
				#	logMsg("Rebooting, current command:"+curCommand)
				#	sockConnect()
				#	os.system("shutdown -r now")
				#else:
				#	lastKillTimestamp = now
					
			# Check if last kill was in the last 60 seconds and reboot
			for cmd in commandList:
				os.system(cmd[2])
			now = time.time()
			if (now - lastKillTimestamp) < 3*timestampMaxDiff:
				logMsg("Rebooting, current command:"+curCommand)
				sockConnect()
				os.system("shutdown -r now")
			else:
				lastKillTimestamp = now

			logMsg("timestampMaxDiff kill command '"+curCommand+"'")
			curCommand = selectCommand() # select properly the current command to be executed
			proc = execCommand(curCommand) # start the command
			consecKillCount += 1
			killCount += 1
		else:
			consecKillCount = 0
	
		# Reboot if we reach the max number of kills allowed 
		if consecKillCount >= maxConsecKill or killCount >= maxKill:
			logMsg("Rebooting, current command:"+curCommand)
			sockConnect()
			os.system("shutdown -r now")
	
		time.sleep(timestampMaxDiff)	
except KeyboardInterrupt: # Ctrl+c
	print "\n\tKeyboardInterrupt detected, exiting gracefully!( at least trying :) )"
	for cmd in commandList:
		os.system(cmd[2])
	sys.exit(1)
