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

# Log messages adding timestamp before the message
def logMsg(msg):
	now = datetime.now()
	fp = open(logFile, 'a')
	print >>fp, now.ctime()+": "+str(msg)
	fp.close()
	print now.ctime()+": "+str(msg)

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
	print >> sys.stderr, "Command configuration file not found!("+commandFile+")"
	sys.exit(1)

configcmd = ConfigParser.RawConfigParser()
try:
	configcmd.read(commandFile)
except IOError as e:
	print >> sys.stderr, "Command configuration setup error: "+str(e)
	sys.exit(1)


logFile = logDir+"test_killtest_commands.log"
timestampFile = varDir+"timestamp.txt"

# Start last kill timestamp with an old enough timestamp
lastKillTimestamp = int(time.time()) - 50*timestampMaxDiff
timestampSignal = int(time.time())

try:
    for i in range(0,len(configcmd.sections())-1):

        logMsg("Executing command: "+getCommand(i))
	execStart = int(time.time())
	execCommand(getCommand(i))
	while True:
		# Read the timestamp file
		try:
			timestamp = int(os.path.getmtime(timestampFile))
		except (ValueError, OSError) as eDetail:
			fp.close()
			contTimestampReadError += 1
			logMsg("timestamp read error(#"+str(contTimestampReadError)+"): "+str(eDetail))
			timestamp = int(float(time.time()))
			
		# Get the current timestamp
		now = int(time.time())
		timestampDiff = now - timestamp
		timestampDiffSignal = now - timestampSignal
                if timestampDiff > maxDiff:
                    maxDiff = timestampDiff
                if timestampDiffSignal > maxDiffSignal:
                    maxDiffSignal = timestampDiff
		# If timestamp was not update properly
		if timestampDiff > timestampMaxDiff:
                        logMsg("ERROR: Timestamp diff from file higher than expected!")
		if timestampDiffSignal > timestampMaxDiff:
                        logMsg("ERROR: Timestamp diff from signal higher than expected!")
	
                # Execute each command for 3 min only
                if (now - execStart) > (60 * 3):
                    killall()
                    break
	
		time.sleep(1)
except KeyboardInterrupt: # Ctrl+c
	print "\n\tKeyboardInterrupt detected, exiting gracefully!( at least trying :) )"
	killall()
	sys.exit(1)
