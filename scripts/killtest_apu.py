#!/usr/bin/python

import threading
import socket
import time
import os.path
import ConfigParser
import sys
import filecmp
import re
from datetime import datetime

# Commands to be executed by KillTest
# each command should be in this format:
# [<launch command>, <hours to be executed>, <command name to be used by 'killall -9'>]
# for example: 
# commandList = [
# 	["./lavaMD 15 4 input1 input2 gold 10000", 1, "lavaMD"],
# 	["./gemm 1024 4 input1 input2 gold 10000", 2, "gemm"]
# ]
# will execute lavaMD for about one hour and then execute gemm for two hours
# When the list finish executing, it start to execute from the beginning
commandList = [
# 	["./lavaMD 15 4 input1 input2 gold 10000", 1, "lavaMD"],
#	["./gemm 1024 4 input1 input2 gold 10000", 2, "gemm"],

#----------------------------------------- Bezier Surface ---------------------------------------
#./gen_input -n 2500  
#./gen_gold -n 2500
# CPU
 	["/home/carol/radiation-benchmarks/src/opencl/Heterogenous_GPU_CPU/BS/bs -p 0 -d 0 -i 5 -g 5 -a 1.00 -t 4 -n 2500 -r 1000000 -f /home/carol/radiation-benchmarks/src/opencl/Heterogenous_GPU_CPU/BS/input/control.txt", 1, "bs"],
#GPU
 	["/home/carol/radiation-benchmarks/src/opencl/Heterogenous_GPU_CPU/BS/bs -p 0 -d 0 -i 16 -g 8 -a 0.00 -t 1 -n 2500 -r 1000000 -f /home/carol/radiation-benchmarks/src/opencl/Heterogenous_GPU_CPU/BS/input/control.txt", 1, "bs"],
#CPU+GPU
 	["/home/carol/radiation-benchmarks/src/opencl/Heterogenous_GPU_CPU/BS/bs -p 0 -d 0 -i 16 -g 8 -a 0.10 -t 2 -n 2500 -r 1000000 -f /home/carol/radiation-benchmarks/src/opencl/Heterogenous_GPU_CPU/BS/input/control.txt", 1, "bs"],
#30% GPU
 	["/home/carol/radiation-benchmarks/src/opencl/Heterogenous_GPU_CPU/BS/bs -p 0 -d 0 -i 16 -g 8 -a 0.70 -t 2 -n 2500 -r 1000000 -f /home/carol/radiation-benchmarks/src/opencl/Heterogenous_GPU_CPU/BS/input/control.txt", 1, "bs"],
#50% GPU
 	["/home/carol/radiation-benchmarks/src/opencl/Heterogenous_GPU_CPU/BS/bs -p 0 -d 0 -i 16 -g 8 -a 0.50 -t 2 -n 2500 -r 1000000 -f /home/carol/radiation-benchmarks/src/opencl/Heterogenous_GPU_CPU/BS/input/control.txt", 1, "bs"],
#60% GPU
 	["/home/carol/radiation-benchmarks/src/opencl/Heterogenous_GPU_CPU/BS/bs -p 0 -d 0 -i 16 -g 8 -a 0.40 -t 2 -n 2500 -r 1000000 -f /home/carol/radiation-benchmarks/src/opencl/Heterogenous_GPU_CPU/BS/input/control.txt", 1, "bs"],
#------------------------------------------------------------------------------------------------
#---------------------------------------- Stream Compaction --------------------------------
#./gen_input -i 256 -n 367001600 -c 50
#./gen_gold -i 256 -n 367001600 -c 50
# CPU
 	["/home/carol/radiation-benchmarks/src/opencl/Heterogenous_GPU_CPU/SC/sc -p 0 -d 0 -i 256 -g 8 -a 1.00 -t 4 -n 367001600 -c 50 -r 1000000", 1, "sc"],
#GPU
 	["/home/carol/radiation-benchmarks/src/opencl/Heterogenous_GPU_CPU/BS/bs -p 0 -d 0 -i 256 -g 8 -a 0.00 -t 1 -n 367001600 -c 50 -r 1000000", 1, "sc"],
#CPU+GPU
  	["/home/carol/radiation-benchmarks/src/opencl/Heterogenous_GPU_CPU/BS/bs -p 0 -d 0 -i 256 -g 8 -a 0.70 -t 4 -n 367001600 -c 50 -r 1000000", 1, "sc"],
#30% GPU
#  	["/home/carol/radiation-benchmarks/src/opencl/Heterogenous_GPU_CPU/BS/bs -p 0 -d 0 -i 256 -g 8 -a 0.70 -t 4 -n 367001600 -c 50 -r 1000000", 1, "sc"],
#50% GPU
  	["/home/carol/radiation-benchmarks/src/opencl/Heterogenous_GPU_CPU/BS/bs -p 0 -d 0 -i 256 -g 8 -a 0.50 -t 4 -n 367001600 -c 50 -r 1000000", 1, "sc"],
#60% GPU
  	["/home/carol/radiation-benchmarks/src/opencl/Heterogenous_GPU_CPU/BS/bs -p 0 -d 0 -i 256 -g 8 -a 0.40 -t 4 -n 367001600 -c 50 -r 1000000", 1, "sc"],
#------------------------------------------------------------------------------------------------
#---------------------------------------- Canny Edge Detection 1 --------------------------------
# 2500 Frames de Input "Dame 5"
# CPU
 	["/home/carol/radiation-benchmarks/src/opencl/Heterogenous_GPU_CPU/CEDD/cedd -p 0 -d 0 -i 16 -a 1.00 -t 4 -r 2490 -w 10 -f /home/carol/radiation-benchmarks/src/opencl/Heterogenous_GPU_CPU/CEDD/input/dame_5/ -c /home/carol/radiation-benchmarks/src/opencl/Heterogenous_GPU_CPU/CEDD/output/dame_5/ -l 1000000", 1, "cedd"],
#GPU
 	["/home/carol/radiation-benchmarks/src/opencl/Heterogenous_GPU_CPU/CEDD/cedd -p 0 -d 0 -i 16 -a 0.00 -t 1 -r 2490 -w 10 -f /home/carol/radiation-benchmarks/src/opencl/Heterogenous_GPU_CPU/CEDD/input/dame_5/ -c /home/carol/radiation-benchmarks/src/opencl/Heterogenous_GPU_CPU/CEDD/output/dame_5/ -l 1000000", 1, "cedd"],
#CPU+GPU
 	["/home/carol/radiation-benchmarks/src/opencl/Heterogenous_GPU_CPU/CEDD/cedd -p 0 -d 0 -i 8 -a 0.10 -t 3 -r 2490 -w 10 -f /home/carol/radiation-benchmarks/src/opencl/Heterogenous_GPU_CPU/CEDD/input/dame_5/ -c /home/carol/radiation-benchmarks/src/opencl/Heterogenous_GPU_CPU/CEDD/output/dame_5/ -l 1000000", 1, "cedd"],
#30% GPU
 	["/home/carol/radiation-benchmarks/src/opencl/Heterogenous_GPU_CPU/CEDD/cedd -p 0 -d 0 -i 8 -a 0.70 -t 3 -r 2490 -w 10 -f /home/carol/radiation-benchmarks/src/opencl/Heterogenous_GPU_CPU/CEDD/input/dame_5/ -c /home/carol/radiation-benchmarks/src/opencl/Heterogenous_GPU_CPU/CEDD/output/dame_5/ -l 1000000", 1, "cedd"],
#50% GPU
 	["/home/carol/radiation-benchmarks/src/opencl/Heterogenous_GPU_CPU/CEDD/cedd -p 0 -d 0 -i 8 -a 0.50 -t 3 -r 2490 -w 10 -f /home/carol/radiation-benchmarks/src/opencl/Heterogenous_GPU_CPU/CEDD/input/dame_5/ -c /home/carol/radiation-benchmarks/src/opencl/Heterogenous_GPU_CPU/CEDD/output/dame_5/ -l 1000000", 1, "cedd"],
#60% GPU
 	["/home/carol/radiation-benchmarks/src/opencl/Heterogenous_GPU_CPU/CEDD/cedd -p 0 -d 0 -i 8 -a 0.40 -t 3 -r 2490 -w 10 -f /home/carol/radiation-benchmarks/src/opencl/Heterogenous_GPU_CPU/CEDD/input/dame_5/ -c /home/carol/radiation-benchmarks/src/opencl/Heterogenous_GPU_CPU/CEDD/output/dame_5/ -l 1000000", 1, "cedd"],
#-----------------------------------------------------------------------------------------------
#---------------------------------------- Canny Edge Detection 2 --------------------------------
# 1110 Frames de Input "Urban"
# CPU
 	["/home/carol/radiation-benchmarks/src/opencl/Heterogenous_GPU_CPU/CEDD/cedd -p 0 -d 0 -i 16 -a 1.00 -t 4 -r 1100 -w 10 -f /home/carol/radiation-benchmarks/src/opencl/Heterogenous_GPU_CPU/CEDD/input/urban_input/ -c /home/carol/radiation-benchmarks/src/opencl/Heterogenous_GPU_CPU/CEDD/output/urban_output/ -l 1000000", 1, "cedd"],
#GPU
 	["/home/carol/radiation-benchmarks/src/opencl/Heterogenous_GPU_CPU/CEDD/cedd -p 0 -d 0 -i 16 -a 0.00 -t 1 -r 1100 -w 10 -f /home/carol/radiation-benchmarks/src/opencl/Heterogenous_GPU_CPU/CEDD/input/urban_input/ -c /home/carol/radiation-benchmarks/src/opencl/Heterogenous_GPU_CPU/CEDD/output/urban_output/ -l 1000000", 1, "cedd"],
#CPU+GPU
 	["/home/carol/radiation-benchmarks/src/opencl/Heterogenous_GPU_CPU/CEDD/cedd -p 0 -d 0 -i 8 -a 0.10 -t 3 -r 1100 -w 10 -f /home/carol/radiation-benchmarks/src/opencl/Heterogenous_GPU_CPU/CEDD/input/urban_input/ -c /home/carol/radiation-benchmarks/src/opencl/Heterogenous_GPU_CPU/CEDD/output/urban_output/ -l 1000000", 1, "cedd"],
#30% GPU
 	["/home/carol/radiation-benchmarks/src/opencl/Heterogenous_GPU_CPU/CEDD/cedd -p 0 -d 0 -i 8 -a 0.70 -t 3 -r 1100 -w 10 -f /home/carol/radiation-benchmarks/src/opencl/Heterogenous_GPU_CPU/CEDD/input/urban_input/ -c /home/carol/radiation-benchmarks/src/opencl/Heterogenous_GPU_CPU/CEDD/output/urban_output/ -l 1000000", 1, "cedd"],
#50% GPU
 	["/home/carol/radiation-benchmarks/src/opencl/Heterogenous_GPU_CPU/CEDD/cedd -p 0 -d 0 -i 8 -a 0.50 -t 3 -r 1100 -w 10 -f /home/carol/radiation-benchmarks/src/opencl/Heterogenous_GPU_CPU/CEDD/input/urban_input/ -c /home/carol/radiation-benchmarks/src/opencl/Heterogenous_GPU_CPU/CEDD/output/urban_output/ -l 1000000", 1, "cedd"],
#60% GPU
 	["/home/carol/radiation-benchmarks/src/opencl/Heterogenous_GPU_CPU/CEDD/cedd -p 0 -d 0 -i 8 -a 0.60 -t 3 -r 1100 -w 10 -f /home/carol/radiation-benchmarks/src/opencl/Heterogenous_GPU_CPU/CEDD/input/urban_input/ -c /home/carol/radiation-benchmarks/src/opencl/Heterogenous_GPU_CPU/CEDD/output/urban_output/ -l 1000000", 1, "cedd"],
#-----------------------------------------------------------------------------------------------
#---------------------------------------- Canny Edge Detection 3 --------------------------------
# 1000 Frames de Input "Caltech"
# CPU
 	["/home/carol/radiation-benchmarks/src/opencl/Heterogenous_GPU_CPU/CEDD/cedd -p 0 -d 0 -i 16 -a 1.00 -t 4 -r 990 -w 10 -f /home/carol/radiation-benchmarks/src/opencl/Heterogenous_GPU_CPU/CEDD/input/caltech_input/ -c /home/carol/radiation-benchmarks/src/opencl/Heterogenous_GPU_CPU/CEDD/output/caltech_output/ -l 1000000", 1, "cedd"],
#GPU
 	["/home/carol/radiation-benchmarks/src/opencl/Heterogenous_GPU_CPU/CEDD/cedd -p 0 -d 0 -i 16 -a 0.00 -t 1 -r 990 -w 10 -f /home/carol/radiation-benchmarks/src/opencl/Heterogenous_GPU_CPU/CEDD/input/caltech_input/ -c /home/carol/radiation-benchmarks/src/opencl/Heterogenous_GPU_CPU/CEDD/output/caltech_output/ -l 1000000", 1, "cedd"],
#CPU+GPU
 	["/home/carol/radiation-benchmarks/src/opencl/Heterogenous_GPU_CPU/CEDD/cedd -p 0 -d 0 -i 8 -a 0.10 -t 3 -r 990 -w 10 -f /home/carol/radiation-benchmarks/src/opencl/Heterogenous_GPU_CPU/CEDD/input/caltech_input/ -c /home/carol/radiation-benchmarks/src/opencl/Heterogenous_GPU_CPU/CEDD/output/caltech_output/ -l 1000000", 1, "cedd"],
#30% GPU
 	["/home/carol/radiation-benchmarks/src/opencl/Heterogenous_GPU_CPU/CEDD/cedd -p 0 -d 0 -i 8 -a 0.70 -t 3 -r 990 -w 10 -f /home/carol/radiation-benchmarks/src/opencl/Heterogenous_GPU_CPU/CEDD/input/caltech_input/ -c /home/carol/radiation-benchmarks/src/opencl/Heterogenous_GPU_CPU/CEDD/output/caltech_output/ -l 1000000", 1, "cedd"],
#50% GPU
 	["/home/carol/radiation-benchmarks/src/opencl/Heterogenous_GPU_CPU/CEDD/cedd -p 0 -d 0 -i 8 -a 0.50 -t 3 -r 990 -w 10 -f /home/carol/radiation-benchmarks/src/opencl/Heterogenous_GPU_CPU/CEDD/input/caltech_input/ -c /home/carol/radiation-benchmarks/src/opencl/Heterogenous_GPU_CPU/CEDD/output/caltech_output/ -l 1000000", 1, "cedd"],
#60% GPU
 	["/home/carol/radiation-benchmarks/src/opencl/Heterogenous_GPU_CPU/CEDD/cedd -p 0 -d 0 -i 8 -a 0.40 -t 3 -r 990 -w 10 -f /home/carol/radiation-benchmarks/src/opencl/Heterogenous_GPU_CPU/CEDD/input/caltech_input/ -c /home/carol/radiation-benchmarks/src/opencl/Heterogenous_GPU_CPU/CEDD/output/caltech_output/ -l 1000000", 1, "cedd"],
#-----------------------------------------------------------------------------------------------

]

# Command used to kill application
killcmd="killall -9 "


timestampMaxDiff = 30 # Time in seconds
maxKill = 5 # Max number of kills allowed

sockServerIP = "192.168.1.5"
sockServerPORT = 8080

# Connect to server and close connection, kind of ping
def sockConnect():
	return
	try:
		#create an INET, STREAMing socket
		s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		# Now, connect with IP (or hostname) and PORT
		# s.connect(("feliz", 8080)) or s.connect(("143.54.10.100", 8080))
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
		fp = open(lastList,'w')
		print >>fp, commandList
		fp.close()
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
		os.system("echo "+str(int(time.time()))+" > "+varDir+"command_execstart_0")
		return commandList[0][0]

	# Check if last command executed is still in its execution time window
	# and return it
	timeWindow = commandList[i][1] * 60 * 60 # Time window in seconds
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
	if (now - timestamp) < timeWindow:
		return commandList[i][0]

	i += 1
	# If all commands executed their time window, start all over again
	if i >= len(commandList):
		cleanCommandExecLogs()
		os.system("echo "+str(int(time.time()))+" > "+varDir+"command_execstart_0")
		return commandList[0][0]

	# Finally, select the next command not executed so far
	os.system("echo "+str(int(time.time()))+" > "+varDir+"command_execstart_"+str(i))
	return commandList[i][0]


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
lastKillTimestamp = int(time.time()) - 50*timestampMaxDiff

contTimestampReadError=0
try:
	killCount = 0 # Counts how many kills were executed throughout execution
	curCommand = selectCommand()
	execCommand(curCommand)
	while True:
		sockConnect()
		# Read the timestamp file
		try:
			#fp = open(timestampFile, 'r')
			#timestamp = int(float(fp.readline().strip()))
			#fp.close()
			timestamp = int(os.path.getmtime(timestampFile))
		except (ValueError, OSError) as eDetail:
			fp.close()
			updateTimestamp()
			contTimestampReadError += 1
			logMsg("timestamp read error(#"+str(contTimestampReadError)+"): "+str(eDetail))
			if contTimestampReadError > 1:
				logMsg("Rebooting, timestamp read error: "+str(eDetail))
				sockConnect()
				os.system("shutdown -r now")
				time.sleep(20)
			timestamp = int(float(time.time()))
			
		# Get the current timestamp
		now = int(time.time())
		timestampDiff = now - timestamp
		# If timestamp was not update properly
		if timestampDiff > timestampMaxDiff:
			# Check if last kill was in the last 60 seconds and reboot
			for cmd in commandList:
				os.system(killcmd+" "+cmd[2])
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
	for cmd in commandList:
		os.system(killcmd+" "+cmd[2])
	sys.exit(1)
