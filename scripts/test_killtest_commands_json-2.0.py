#!/usr/bin/python3

import time
import os
import os.path
import configparser
import sys
import re
import signal
import json
from datetime import datetime

timestampMaxDiff = 30  # Time in seconds to wait for the timestamp update

maxKill = 5  # Max number of kills allowed

timeWindowCommand = 10  # * 60 * 60 # How long each command will execute, time window in seconds


# Log messages adding timestamp before the message
def logMsg(msg):
    now = datetime.now()
    with open(logFile, 'a') as fp:
        fp.write(str(now.ctime()) + ": " + str(msg))

    print(now.ctime() + ": " + str(msg))


def updateTimestamp():
    command = "echo " + str(int(time.time())) + " > " + timestampFile
    retcode = os.system(command)
    global timestampSignal
    timestampSignal = int(time.time())


def execCommand(command):
    try:
        updateTimestamp()
        if re.match(".*&\s*$", command):
            # print "command should be ok"
            return os.system(command)
        else:
            # print "command not ok, inserting &"
            return os.system(command + " &")
    except OSError as detail:
        logMsg("Error launching command '" + command + "'; error detail: " + str(detail))
        return None


def getCommand(index):
    return commands[index]["exec"]


def killall():
    try:
        for cmd in commands:
            os.system(cmd["killcmd"])
    except KeyError:
        print("Could not issue the kill command for each entry, config file error!", file=sys.stderr)


def readCommands(filelist):
    global commands
    if os.path.isfile(filelist):
        with open(filelist, "r") as fp:
            lines = fp.readlines()
            for f in lines:
                f = f.strip()
                if f.startswith("#") or f.startswith("%"):
                    continue
                if os.path.isfile(f):
                    fjson = open(f, "r")
                    data = json.load(fjson)
                    fjson.close()
                    commands.extend(data)
                else:
                    logMsg(f"ERROR: File with commands not found - {f} - continuing with other files")


# When SIGUSR1 or SIGUSR2 is received update timestamp
def receive_signal(signum, stack):
    global timestampSignal
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
    print(f"System configuration file not found!({confFile})", file=sys.stderr)
    sys.exit(1)

try:
    config = configparser.RawConfigParser()
    config.read(confFile)

    installDir = config.get('DEFAULT', 'installdir') + "/"
    varDir = config.get('DEFAULT', 'vardir') + "/"
    logDir = config.get('DEFAULT', 'logdir') + "/"
    tmpDir = config.get('DEFAULT', 'tmpdir') + "/"

    # logDir = varDir+"log/"

    if not os.path.isdir(logDir):
        os.mkdir(logDir, 0o777)
        os.chmod(logDir, 0o777)

except IOError as e:
    print("System configuration setup error: " + str(e), file=sys.stderr)
    sys.exit(1)

logFile = "log_test.txt"
timestampFile = varDir + "timestamp.txt"

if len(sys.argv) != 2:
    print("Usage: " + sys.argv[0] + " <file with absolute paths of json files>")
    sys.exit(1)

commands = list()

readCommands(sys.argv[1])

if len(commands) < 1:
    print("ERROR: No commands read, there is nothing to execute", file=sys.stderr)
    sys.exit(1)

# Start last kill timestamp with an old enough timestamp
timestampSignal = int(time.time())

try:
    for i in range(0, len(commands)):

        logMsg("Executing command: " + getCommand(i))
        execStart = int(time.time())
        execCommand(getCommand(i))
        maxDiff = 0
        maxDiffSignal = 0
        flagDiff = False
        flagDiffSignal = False
        while True:
            # Read the timestamp file
            try:
                timestamp = int(os.path.getmtime(timestampFile))
            except (ValueError, OSError) as eDetail:
                logMsg("timestamp read error: " + str(eDetail))
                timestamp = int(float(time.time()))

            # Get the current timestamp
            now = int(time.time())
            timestampDiff = now - timestamp
            timestampDiffSignal = now - timestampSignal
            if timestampDiff > maxDiff:
                maxDiff = timestampDiff
            if timestampDiffSignal > maxDiffSignal:
                maxDiffSignal = timestampDiffSignal
            # If timestamp was not update properly
            if timestampDiff > timestampMaxDiff:
                if not flagDiff:
                    logMsg("ERROR: Timestamp diff from file higher than expected!")
                    flagDiff = True
            if timestampDiffSignal > timestampMaxDiff:
                if not flagDiffSignal:
                    logMsg("ERROR: Timestamp diff from signal higher than expected!")
                    flagDiffSignal = True

            # Execute each command for 5 min only
            if (now - execStart) > timeWindowCommand:
                killall()
                logMsg("maxDiff from file: " + str(maxDiff))
                logMsg("maxDiff from signal: " + str(maxDiffSignal))
                logMsg("Finished executing command: " + getCommand(i) + "\n\n")
                break

            time.sleep(1)
except KeyboardInterrupt:  # Ctrl+c
    print("\n\tKeyboardInterrupt detected, exiting gracefully!( at least trying :) )")
    killall()
    sys.exit(1)
