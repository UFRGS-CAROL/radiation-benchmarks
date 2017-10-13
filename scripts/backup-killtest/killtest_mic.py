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
	["sshpass -p qwerty0 ssh carol@mic0 \" /micNfs/codes/nw/needle_check 16394 10 228 /micNfs/codes/nw/input_16394_th_228_pen_10 /micNfs/codes/nw/gold_16394_th_228_pen_10 1000000\"",1,"sshpass -p qwerty0 ssh carol@mic0 \"  killall -9 needle_check\""],
	["sshpass -p qwerty0 ssh carol@mic0 \" /micNfs/codes/nw/needle_check 8192 10 228 /micNfs/codes/nw/input_8192_th_228_pen_10 /micNfs/codes/nw/gold_8192_th_228_pen_10 1000000\"",1,"sshpass -p qwerty0 ssh carol@mic0 \"  killall -9 needle_check\""],
	["sshpass -p qwerty0 ssh carol@mic0 \" /micNfs/codes/lavamd/lavamd_check 228 8 /micNfs/codes/lavamd/input_distance_228_8 /micNfs/codes/lavamd/input_charges_228_8 /micNfs/codes/lavamd/output_gold_228_8 10000000\"",1,"sshpass -p qwerty0 ssh carol@mic0 \"  killall -9 lavamd_check\""],
	["sshpass -p qwerty0 ssh carol@mic0 \" /micNfs/codes/lavamd/lavamd_check 228 9 /micNfs/codes/lavamd/input_distance_228_9 /micNfs/codes/lavamd/input_charges_228_9 /micNfs/codes/lavamd/output_gold_228_9 10000000\"",1,"sshpass -p qwerty0 ssh carol@mic0 \"  killall -9 lavamd_check\""],
	["sshpass -p qwerty0 ssh carol@mic0 \" /micNfs/codes/lavamd/lavamd_check 228 10 /micNfs/codes/lavamd/input_distance_228_10 /micNfs/codes/lavamd/input_charges_228_10 /micNfs/codes/lavamd/output_gold_228_10 10000000\"",1,"sshpass -p qwerty0 ssh carol@mic0 \"  killall -9 lavamd_check\""],
	["sshpass -p qwerty0 ssh carol@mic0 \" /micNfs/codes/hotspot/hotspot_check 1024 1024 1000 228 /micNfs/codes/hotspot/temp_1024 /micNfs/codes/hotspot/power_1024 /micNfs/codes/hotspot/gold_1024_1000 1000000\"",1,"sshpass -p qwerty0 ssh carol@mic0 \" killall -9 hotspot_check\""],
	["sshpass -p qwerty0 ssh carol@mic0 \" /micNfs/codes/kmeans/kmeans_check -i /micNfs/codes/kmeans/kdd_cup -o /micNfs/codes/kmeans/gold-kdd -n 228 -l 9999999\"",1,"sshpass -p qwerty0 ssh carol@mic0 \"  killall -9 kmeans_check\""],
	["sshpass -p qwerty0 ssh carol@mic0 \" /micNfs/codes/lud/lud_check -n 228 -s 1024 -i /micNfs/codes/lud/input_1024_th_228 -g /micNfs/codes/lud/gold_1024_th_228 -l 10000000\"",1,"sshpass -p qwerty0 ssh carol@mic0 \"  killall -9 lud_check \""],
	["sshpass -p qwerty0 ssh carol@mic0 \" /micNfs/codes/lud/lud_check -n 228 -s 2048 -i /micNfs/codes/lud/input_2048_th_228 -g /micNfs/codes/lud/gold_2048_th_228 -l 10000000\"",1,"sshpass -p qwerty0 ssh carol@mic0 \"  killall -9 lud_check \""],
	["sshpass -p qwerty0 ssh carol@mic0 \" /micNfs/codes/lud/lud_check -n 228 -s 4096 -i /micNfs/codes/lud/input_4096_th_228 -g /micNfs/codes/lud/gold_4096_th_228 -l 10000000\"",1,"sshpass -p qwerty0 ssh carol@mic0 \"  killall -9 lud_check \""],
	["sshpass -p qwerty0 ssh carol@mic0 \" /micNfs/codes/lulesh/lulesh_check -s 15 -g /micNfs/codes/lulesh/gold_15\"",1,"sshpass -p qwerty0 ssh carol@mic0 \"  killall -9 lulesh_check\""],
	["sshpass -p qwerty0 ssh carol@mic0 \" /micNfs/codes/mergesort/merge_check 67108864 228 /micNfs/codes/mergesort/inputsort_134217728 /micNfs/codes/mergesort/gold_67108864 99999999\"",1,"sshpass -p qwerty0 ssh carol@mic0 \"  killall -9 merge_check\""],
	["sshpass -p qwerty0 ssh carol@mic0 \" /micNfs/codes/nn/nn_check  /micNfs/codes/nn/list8192k.txt 10 10 80 /micNfs/codes/nn/gold-list8192k.txt 8388608 99999999\"",1,"sshpass -p qwerty0 ssh carol@mic0 \"  killall -9 nn_check\""],
	["sshpass -p qwerty0 ssh carol@mic0 \" /micNfs/codes/quicksort/quick_check 67108864 228 /micNfs/codes/mergesort/inputsort_134217728 /micNfs/codes/quicksort/gold_67108864 99999999\"",1,"sshpass -p qwerty0 ssh carol@mic0 \"  killall -9 quick_check\""],
	["sshpass -p qwerty0 ssh carol@mic0 \" /micNfs/codes/bfs/bfs_check 228 /micNfs/codes/bfs/graph2M.txt /micNfs/codes/bfs/gold-graph2M 9999999\"",1,"sshpass -p qwerty0 ssh carol@mic0 \"  killall -9 bfs_check\""],

]

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
			#fp.close()
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
