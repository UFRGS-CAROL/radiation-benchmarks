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
# ["command", <hours to be executed>]
# for example: 
# commandList = [
# 	["./lavaMD 15 4 input1 input2 gold 10000", 1, "lavaMD"],
# 	["./gemm 1024 4 input1 input2 gold 10000", 2, "gemm"]
# ]
# will execute lavaMD for about one hour and then execute gemm for two hours
# When the list finish executing, it start to execute from the beginning
sshcmd="sshpass -p qwerty0 ssh carol@mic0 "
commandList = [
 	["cd /home/carol/run_clamr_mic;./runCLAMRMic0.sh", 1, "cd /home/carol/run_clamr_mic;./killCLAMRMic0.sh"],
 	[sshcmd+"\" /micNfs/bin/hotspot/hotspot_check 1024 1024 1000 228 /micNfs/bin/hotspot/temp_1024 /micNfs/bin/hotspot/power_1024 /micNfs/bin/hotspot/GOLD_1024grid_1000simiter_228ths 100000\"", 1, sshcmd+"\"  killall -9 hotspot_check\""],
 	[sshcmd+"\" /micNfs/bin/hotspot/hotspot_check 1024 1024 10000 228 /micNfs/bin/hotspot/temp_1024 /micNfs/bin/hotspot/power_1024 /micNfs/bin/hotspot/GOLD_1024grid_10000simiter_228ths 100000\"", 1, sshcmd+"\"  killall -9 hotspot_check\""],
 	[sshcmd+"\" /micNfs/bin/lavamd/lavamd 228 15 /micNfs/bin/lavamd/input_distance_228_15 /micNfs/bin/lavamd/input_charges_228_15 /micNfs/bin/lavamd/output_gold_228_15 1000000\"", 1, sshcmd+"\"  killall -9 lavamd\""],
 	[sshcmd+"\" /micNfs/bin/lavamd/lavamd 228 19 /micNfs/bin/lavamd/input_distance_228_19 /micNfs/bin/lavamd/input_charges_228_19 /micNfs/bin/lavamd/output_gold_228_19 1000000\"", 1, sshcmd+"\"  killall -9 lavamd\""],
 	[sshcmd+"\" /micNfs/bin/lavamd/lavamd 228 23 /micNfs/bin/lavamd/input_distance_228_23 /micNfs/bin/lavamd/input_charges_228_23 /micNfs/bin/lavamd/output_gold_228_23 1000000\"", 1, sshcmd+"\"  killall -9 lavamd\""],
 	[sshcmd+"\" /micNfs/bin/lud/lud_check -n 228 -s 2048 -i /micNfs/bin/lud/input_2048_th_228 -g /micNfs/bin/lud/gold_2048_th_228 -l 1000000\"", 1, sshcmd+"\"  killall -9 lud_check\""],
 	[sshcmd+"\" /micNfs/bin/lud/lud_check -n 228 -s 4096 -i /micNfs/bin/lud/input_4096_th_228 -g /micNfs/bin/lud/gold_4096_th_228 -l 1000000\"", 1, sshcmd+"\"  killall -9 lud_check\""],
 	[sshcmd+"\" /micNfs/bin/nw/needle_check 8192 10 228 /micNfs/bin/nw/input_8192_th_228_pen_10 /micNfs/bin/nw/gold_8192_th_228_pen_10 10000000\"", 1, sshcmd+"\"  killall -9 needle_check\""],
#
# LUD input 8192 NOT WORKING, PROBLEM WITH INPUT SIZE AND NFS???
# 	[sshcmd+"\" /micNfs/bin/lud/lud_check -n 228 -s 8192 -i /micNfs/bin/lud/input_8192_th_228 -g /micNfs/bin/lud/gold_8192_th_228 -l 1000000\"", 0.05, sshcmd+"\"  killall -9 lud_check\""],
# NW input 16384 NOT WORKING, PROBLEM WITH INPUT SIZE AND NFS???
# 	[sshcmd+"\" /micNfs/bin/nw/needle_check 16384 10 228 /micNfs/bin/nw/input_16384_th_228_pen_10 /micNfs/bin/nw/gold_16384_th_228_pen_10 10000000\"", 0.1, sshcmd+"\"  killall -9 needle_check\""],
]

# Command used to kill application
#killcmd="killall -9 "
killcmd=""


timestampMaxDiff = 30 # Time in seconds
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
	micLog =  config.get('DEFAULT', 'miclogdir')+"/"
	varDir = micLog
	
	
	if not os.path.isdir(micLog):
		os.mkdir(micLog, 0777)
		os.chmod(micLog, 0777)
	
except IOError as e:
	print >> sys.stderr, "Configuration setup error: "+str(e)
	sys.exit(1)
	
logFile = micLog+"killtest.log"
timestampFile = micLog+"timestamp.txt"
updateTimestamp()
os.chmod(timestampFile, 0777)

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
			#execCommand(killcmd)
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
