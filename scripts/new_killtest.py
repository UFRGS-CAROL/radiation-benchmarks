#!/usr/bin/python

import threading
import socket
import time
import os
import os.path
import ConfigParser
import sys
import filecmp
import re
import shutil
import signal
from datetime import datetime


timestampMaxDiff = 30 # Time in seconds to wait for the timestamp update

maxKill = 5 # Max number of kills allowed

timeWindowCommands = 1 * 60 * 60 # How long each command will execute, time window in seconds

sockServerIP = "192.168.1.5"
sockServerPORT = 8080

# Connect to server and close connection, kind of ping
def sockConnect():
	try:
		#create an INET, STREAMing socket
		s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		# Now, connect with IP (or hostname) and PORT
		s.connect((sockServerIP, sockServerPORT))
		s.close()
	except socket.error as eDetail:
		print "could not connect to remote server, socket error"
		#logMsg("socket connect error: "+str(eDetail))

# Log messages adding timestamp before the message
def logMsg(msg):
	now = datetime.now()
	fp = open(logFile, 'a')
	print >>fp, now.ctime()+": "+str(msg)
	fp.close()
	print now.ctime()+": "+str(msg)

# Update the timestamp file with machine current timestamp
def updateTimestamp():
	command = "echo "+str(int(time.time()))+" > "+timestampFile
	retcode = os.system(command)


# Remove files with start timestamp of commands executing
def cleanCommandExecLogs():
	os.system("rm -f "+varDir+"command_execstart_*")

# Return True if the commandFile changed from the 
# last time it was executed. If the file was never executed returns False
def checkCommandListChanges():
	curFile = varDir+"currentCommandFile"
	lastFile = varDir+"lastCommandFile"
        shutil.copyfile(commandFile, curFile)
	if not os.path.isfile(lastFile):
                shutil.copyfile(commandFile, lastFile)
		return True

	if filecmp.cmp(curFile, lastFile, shallow=False):
		return False
	else:
                shutil.copyfile(commandFile, lastFile)
		return True

# Select the correct command to be executed from the configcmd configuration file
def selectCommand():
	if checkCommandListChanges():
		cleanCommandExecLogs()

	# Get the index of last existent file	
	i=0
	while os.path.isfile(varDir+"command_execstart_"+str(i)):
		i += 1
	i -= 1

	# If there is no file, create the first file with current timestamp
	# and return the first command of configcmd configuration file
	if i == -1:
		os.system("echo "+str(int(time.time()))+" > "+varDir+"command_execstart_0")
		return getCommand(0)

	# Check if last command executed is still in the defined time window for each command
	# and return it

	# Read the timestamp file
	try:
		fp = open(varDir+"command_execstart_"+str(i),'r')
		timestamp = int(float(fp.readline().strip()))
		fp.close()
	except ValueError as eDetail:
		logMsg("Rebooting, command execstart timestamp read error: "+str(eDetail))
		sockConnect()
		os.system("rm -f "+varDir+"command_execstart_"+str(i))
		os.system("shutdown -r now")
		time.sleep(20)
	#fp = open(varDir+"command_execstart_"+str(i),'r')
	#timestamp = int(float(fp.readline().strip()))
	#fp.close()

	now = int(time.time())
	if (now - timestamp) < timeWindowCommands:
		return getCommand(i)

	i += 1
	# If all commands executed their time window, start all over again
	if i >= len(configcmd.sections()):
		cleanCommandExecLogs()
		os.system("echo "+str(int(time.time()))+" > "+varDir+"command_execstart_0")
		return getCommand(0)

	# Finally, select the next command not executed so far
	os.system("echo "+str(int(time.time()))+" > "+varDir+"command_execstart_"+str(i))
	return getCommand(i)


def execCommand(command):
	try:
		updateTimestamp()
		if re.match(".*&\s*$", command):
			#print "command should be ok"
			return os.system(command)
		else:
			#print "command not ok, inserting &"
			return os.system(command+" &")
	except OSError as detail:
		logMsg("Error launching command '"+command+"'; error detail: "+str(detail))
		return None

def getCommand(index):
	sec = configcmd.sections()[index]
        return configcmd.get(sec,"exec")

def killall():
        try:
		for sec in configcmd.sections():
			os.system(configcmd.get(sec,"killcmd"))
        except:
		print >> sys.stderr, "Could not issue the kill command for each entry, config file error!"


# When SIGUSR1 or SIGUSR2 is received update timestamp
def receive_signal(signum, stack):
	timestampSignal = int(time.time())

################################################
# KillTest Main Execution
################################################
confFile = '/etc/radiation-benchmarks.conf'

# call the routine "receive_sginal" when SIGUSR1 is received
signal.signal(signal.SIGUSR1, receive_signal) 
# call the routine "receive_sginal" when SIGUSR2 is received
signal.signal(signal.SIGUSR2, receive_signal)

if not os.path.isfile(confFile):
	print >> sys.stderr, "System configuration file not found!("+confFile+")"
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
	print >> sys.stderr, "System configuration setup error: "+str(e)
	sys.exit(1)
	

if (len(sys.argv) != 2):
    print "Usage: "+sys.argv[0]+" <command conf file>"
    sys.exit(1)

commandFile = sys.argv[1]

if not os.path.isfile(commandFile):
	print >> sys.stderr, "Command configuration file not found!("+confFile+")"
	sys.exit(1)

configcmd = ConfigParser.RawConfigParser()
try:
	configcmd.read(commandFile)
except IOError as e:
	print >> sys.stderr, "Command configuration setup error: "+str(e)
	sys.exit(1)


logFile = logDir+"killtest.log"
timestampFile = varDir+"timestamp.txt"

# Start last kill timestamp with an old enough timestamp
lastKillTimestamp = int(time.time()) - 50*timestampMaxDiff
timestampSignal = int(time.time())

contTimestampReadError=0
try:
	killCount = 0 # Counts how many kills were executed throughout execution
	curCommand = selectCommand()
	execCommand(curCommand)
	while True:
		sockConnect()
		## Read the timestamp file
		#try:
		#	#fp = open(timestampFile, 'r')
		#	#timestamp = int(float(fp.readline().strip()))
		#	#fp.close()
		#	timestamp = int(os.path.getmtime(timestampFile))
		#except (ValueError, OSError) as eDetail:
		#	fp.close()
		#	updateTimestamp()
		#	contTimestampReadError += 1
		#	logMsg("timestamp read error(#"+str(contTimestampReadError)+"): "+str(eDetail))
		#	if contTimestampReadError > 1:
		#		logMsg("Rebooting, timestamp read error: "+str(eDetail))
		#		sockConnect()
		#		os.system("shutdown -r now")
		#		time.sleep(20)
		#	timestamp = int(float(time.time()))
			
		# Get the current timestamp
		now = int(time.time())
		#timestampDiff = now - timestamp
		timestampDiff = now - timestampSignal
		# If timestamp was not update properly
		if timestampDiff > timestampMaxDiff:
			# Check if last kill was in the last 60 seconds and reboot
			killall()
			now = int(time.time())
			if (now - lastKillTimestamp) < 3*timestampMaxDiff:
				logMsg("Rebooting, last kill too recent, timestampDiff: "+str(timestampDiff)+", current command:"+curCommand)
				sockConnect()
				os.system("shutdown -r now")
				time.sleep(20)
			else:
				lastKillTimestamp = now

			killCount += 1
			logMsg("timestampMaxDiff kill(#"+str(killCount)+"), timestampDiff:"+str(timestampDiff)+" command '"+curCommand+"'")
			# Reboot if we reach the max number of kills allowed 
			if killCount >= maxKill:
				logMsg("Rebooting, maxKill reached, current command:"+curCommand)
				sockConnect()
				os.system("shutdown -r now")
				time.sleep(20)
			else:
				curCommand = selectCommand() # select properly the current command to be executed
				execCommand(curCommand) # start the command
	
	
		time.sleep(1)	
except KeyboardInterrupt: # Ctrl+c
	print "\n\tKeyboardInterrupt detected, exiting gracefully!( at least trying :) )"
	killall()
	sys.exit(1)
