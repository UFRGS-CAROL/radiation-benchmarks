#!/usr/bin/python3
import socket
import time
import os
import os.path
import configparser
import sys
import filecmp
import re
import shutil
import signal
import json
from datetime import datetime

from server_package.server_parameters import SERVER_IP, SOCKET_PORT

# Time in seconds to wait for the timestamp update
timestampMaxDiffDefault = 30

# Max number of kills allowed
maxKill = 5

# How long each command will execute, time window in seconds
timeWindowCommands = 3600

# Config file
confFile = '/etc/radiation-benchmarks.conf'


def sockConnect():
    """
    Connect to server and close connection, kind of ping
    :return:
    """
    try:
        # create an INET, STREAMing socket
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # python 3 requires a TIMEOUT
        s.settimeout(2)
        # Now, connect with IP (or hostname) and PORT
        s.connect((SERVER_IP, SOCKET_PORT))
        s.close()
    except socket.error:
        print("could not connect to remote server, socket error")


def logMsg(msg):
    """
    Log messages adding timestamp before the message
    :param msg: message to print
    :return:
    """
    now = datetime.now()
    # fp = open(logFile, 'a')
    date_str = str(now.ctime()) + ": " + str(msg)
    with open(logFile, 'a') as fp:
        fp.write(date_str + '\n')

    # fp.close()
    print(date_str)


def updateTimestamp():
    """
    Update the timestamp file with machine current timestamp
    :return:
    """
    command = "echo " + str(int(time.time())) + " > " + timestampFile
    retcode = os.system(command)
    global timestampSignal
    timestampSignal = int(time.time())


def cleanCommandExecLogs():
    """
    Remove files with start timestamp of commands executing
    :return:
    """
    os.system("rm -f " + varDir + "command_execstart_*")


def checkCommandListChanges():
    """
    Return True if the commandFile changed from the
    last time it was executed. If the file was never executed returns False
    :return:
    """
    curFile = varDir + "currentCommandFile"
    lastFile = varDir + "lastCommandFile"
    fp = open(curFile, "w")
    json.dump(commands, fp)
    fp.close()
    if not os.path.isfile(lastFile):
        shutil.copyfile(curFile, lastFile)
        return True

    if filecmp.cmp(curFile, lastFile, shallow=False):
        return False
    else:
        shutil.copyfile(curFile, lastFile)
        return True


def selectCommand():
    """
    Select the correct command to be executed from the commands
    :return:
    """
    if checkCommandListChanges():
        cleanCommandExecLogs()

    # Get the index of last existent file
    i = 0
    while os.path.isfile(varDir + "command_execstart_" + str(i)):
        i += 1
    i -= 1

    # If there is no file, create the first file with current timestamp
    # and return the first command of commands list
    if i == -1:
        os.system("echo " + str(int(time.time())) + " > " + varDir + "command_execstart_0")
        return getCommand(0)

    # Check if last command executed is still in the defined time window for each command
    # and return it

    # Read the timestamp file
    try:
        fp = open(varDir + "command_execstart_" + str(i), 'r')
        timestamp = int(float(fp.readline().strip()))
        fp.close()
    except ValueError as eDetail:
        logMsg("Rebooting, command execstart timestamp read error: " + str(eDetail))
        sockConnect()
        os.system("rm -f " + varDir + "command_execstart_" + str(i))
        os.system("shutdown -r now")
        time.sleep(20)
    # fp = open(varDir+"command_execstart_"+str(i),'r')
    # timestamp = int(float(fp.readline().strip()))
    # fp.close()

    now = int(time.time())
    if (now - timestamp) < timeWindowCommands:
        return getCommand(i)

    i += 1
    # If all commands executed their time window, start all over again
    if i >= len(commands):
        cleanCommandExecLogs()
        os.system("echo " + str(int(time.time())) + " > " + varDir + "command_execstart_0")
        return getCommand(0)

    # Finally, select the next command not executed so far
    os.system("echo " + str(int(time.time())) + " > " + varDir + "command_execstart_" + str(i))
    return getCommand(i)


def execCommand(command):
    """
    Execute command
    :param command: cmd to execute
    :return:
    """
    try:
        updateTimestamp()
        if re.match(r".*&\s*$", command):
            # print "command should be ok"
            return os.system(command)
        else:
            # print "command not ok, inserting &"
            return os.system(command + " &")
    except OSError as detail:
        logMsg("Error launching command '" + command + "'; error detail: " + str(detail))
        return None


def getCommand(index):
    global timestampMaxDiff
    try:
        if commands[index]["tmDiff"]:
            timestampMaxDiff = commands[index]["tmDiff"]
    except (KeyError, IndexError, ValueError, TypeError):
        timestampMaxDiff = timestampMaxDiffDefault
    return commands[index]["exec"]


def killall():
    try:
        for cmd in commands:
            os.system(cmd["killcmd"])
    except (KeyError, ValueError, TypeError):
        print("Could not issue the kill command for each entry, config file error!")


# When SIGUSR1 or SIGUSR2 is received update timestamp
def receive_signal(signum, stack):
    global timestampSignal
    timestampSignal = int(time.time())


def readCommands(filelist):
    global commands
    if os.path.isfile(filelist):
        fp = open(filelist, "r")
        for f in fp:
            f = f.strip()
            if f.startswith("#") or f.startswith("%"):
                continue
            if os.path.isfile(f):
                fjson = open(f, "r")
                data = json.load(fjson)
                fjson.close()
                commands.extend(data)
            else:
                logMsg("ERROR: File with commands not found - " + str(f) + " - continuing with other files")


################################################
# KillTest Main Execution
################################################
# call the routine "receive_sginal" when SIGUSR1 is received
signal.signal(signal.SIGUSR1, receive_signal)
# call the routine "receive_sginal" when SIGUSR2 is received
signal.signal(signal.SIGUSR2, receive_signal)

if not os.path.isfile(confFile):
    raise FileNotFoundError("System configuration file not found!(" + confFile + ")")

try:
    config = configparser.RawConfigParser()
    config.read(confFile)

    installDir = config.get('DEFAULT', 'installdir') + "/"
    varDir = config.get('DEFAULT', 'vardir') + "/"
    logDir = config.get('DEFAULT', 'logdir') + "/"
    tmpDir = config.get('DEFAULT', 'tmpdir') + "/"

    if not os.path.isdir(logDir):
        os.mkdir(logDir, 0o777)
        os.chmod(logDir, 0o777)

except IOError as e:
    raise IOError("System configuration setup error: " + str(e))
    # sys.exit(1)

logFile = logDir + "killtest.log"
timestampFile = varDir + "timestamp.txt"

if len(sys.argv) != 2:
    print("Usage: " + sys.argv[0] + " <file with absolute paths of json files>")
    sys.exit(1)

commands = list()

readCommands(sys.argv[1])

if len(commands) < 1:
    raise ValueError("ERROR: No commands read, there is nothing to execute")
    # sys.exit(1)

timestampMaxDiff = timestampMaxDiffDefault
# Start last kill timestamp with an old enough timestamp
lastKillTimestamp = int(time.time()) - 50 * timestampMaxDiff
timestampSignal = int(time.time())

try:
    killCount = 0  # Counts how many kills were executed throughout execution
    curCommand = selectCommand()
    execCommand(curCommand)
    while True:
        sockConnect()

        # Get the current timestamp
        now = int(time.time())
        # timestampDiff = now - timestamp
        timestampDiff = now - timestampSignal
        # If timestamp was not update properly
        if timestampDiff > timestampMaxDiff:
            # Check if last kill was in the last 60 seconds and reboot
            killall()
            now = int(time.time())
            if (now - lastKillTimestamp) < 3 * timestampMaxDiff:
                logMsg("Rebooting, last kill too recent, timestampDiff: " + str(
                    timestampDiff) + ", current command:" + curCommand)
                sockConnect()
                os.system("shutdown -r now")
                time.sleep(20)
            else:
                lastKillTimestamp = now

            killCount += 1
            logMsg("timestampMaxDiff kill(#" + str(killCount) + "), timestampDiff:" + str(
                timestampDiff) + " command '" + curCommand + "'")
            # Reboot if we reach the max number of kills allowed
            if killCount >= maxKill:
                logMsg("Rebooting, maxKill reached, current command:" + curCommand)
                sockConnect()
                os.system("shutdown -r now")
                time.sleep(20)
            else:
                curCommand = selectCommand()  # select properly the current command to be executed
                execCommand(curCommand)  # start the command

        time.sleep(1)
except KeyboardInterrupt:  # Ctrl+c
    print("\n\tKeyboardInterrupt detected, exiting gracefully!( at least trying :) )")
    killall()
    # 130 is the key for CTRL + c
    exit(130)
