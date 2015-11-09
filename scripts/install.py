#!/usr/bin/python

import ConfigParser
import sys
import os
import re

def checkPath(path):
	print "The install directory is '"+path+"', is that correct [Y/n]: "
	yes = set(['yes','y', 'ye', ''])
	no = set(['no','n'])


	choice = raw_input().lower()
	if choice in yes:
		return True
	elif choice in no:
		return False
	else:
		sys.stdout.write("Please respond with 'yes' or 'no'")
		return False
	

def installPath():
	path=os.getcwd()
	path=re.sub(r'/scripts', '', path)
	while not checkPath(path):
		print "Please, enter the install path (radiation-benchmarks directory): "
		path = raw_input()
	return path

def check():
	datafile = file("/proc/cpuinfo");
	for line in datafile:
		if 'jetson-tk1' in line:
			return True;
	return False;

varDir = "/var/radiation-benchmarks"
confFile = "/etc/radiation-benchmarks.conf"

#check if it is K1
if check():
	#if it is save everything in sd card
	varDir = raw_input("Where is mounted the sd card??? ") + "/radiation-benchmarks";

logDir = varDir+"/log"
micLog = "/micNfs/carol/logs"

installPath=installPath()

config = ConfigParser.ConfigParser()

config.set("DEFAULT", "installdir", installPath)
config.set("DEFAULT", "vardir", varDir)
config.set("DEFAULT", "logdir", logDir)
config.set("DEFAULT", "tmpdir", "/tmp")
config.set("DEFAULT", "miclogdir", micLog)
try:
	if not os.path.isdir(varDir):
		os.mkdir(varDir, 0777)
	os.chmod(varDir, 0777)
	if not os.path.isdir(logDir):
		os.mkdir(logDir, 0777)
	os.chmod(logDir, 0777)
	with open(confFile, 'w') as configfile:
		config.write(configfile)
except IOError:
	print "I/O Error, please make sure to run as root (sudo)"
	sys.exit(1)

print "var directory created ("+varDir+")"
print "log directory created ("+logDir+")"
print "Config file written ("+confFile+")"
