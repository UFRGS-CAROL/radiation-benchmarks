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
# from datetime import datetime

import logging

timestamp_max_diff_default = 30  # Time in seconds to wait for the timestamp update

max_kill = 5  # Max number of kills allowed

time_window_commands = 1 * 60 * 60  # How long each command will execute, time window in seconds

sock_server_ip = "192.168.1.5"
sock_server_port = 8080


# Connect to server and close connection, kind of ping
def sockConnect():
    try:
        # create an INET, STREAMing socket
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Now, connect with IP (or hostname) and PORT
        s.connect((sock_server_ip, sock_server_port))
        s.close()
    except socket.error:
        print("could not connect to remote server, socket error")
    # logMsg("socket connect error: "+str(eDetail))


# Remove files with start timestamp of commands executing
def cleanCommandExecLogs():
    global var_dir
    os.system("rm -f " + var_dir + "command_execstart_*")


# Return True if the commandFile changed from the
# last time it was executed. If the file was never executed returns False
def check_command_list_changes():
    cur_file = var_dir + "currentCommandFile"
    last_file = var_dir + "lastCommandFile"
    with open(cur_file, "w") as fp:
        json.dump(commands, fp)
    if not os.path.isfile(last_file):
        shutil.copyfile(cur_file, last_file)
        return True

    if filecmp.cmp(cur_file, last_file, shallow=False):
        return False
    else:
        shutil.copyfile(cur_file, last_file)
        return True


# Select the correct command to be executed from the commands
def select_command():
    global timestamp
    if check_command_list_changes():
        cleanCommandExecLogs()

    # Get the index of last existent file
    i = 0
    while os.path.isfile(var_dir + "command_execstart_" + str(i)):
        i += 1
    i -= 1

    # If there is no file, create the first file with current timestamp
    # and return the first command of commands list
    if i == -1:
        os.system("echo " + str(int(time.time())) + " > " + var_dir + "command_execstart_0")
        return getCommand(0)

    # Check if last command executed is still in the defined time window for each command
    # and return it
    # Read the timestamp file
    try:
        fp = open(var_dir + "command_execstart_" + str(i), 'r')
        timestamp = int(float(fp.readline().strip()))
        fp.close()
    except ValueError as eDetail:
        logging.exception("Rebooting, command execstart timestamp read error: " + str(eDetail))
        sockConnect()
        os.system("rm -f " + var_dir + "command_execstart_" + str(i))
        os.system("shutdown -r now")
        time.sleep(20)

    now = int(time.time())
    if (now - timestamp) < time_window_commands:
        return getCommand(i)

    i += 1
    # If all commands executed their time window, start all over again
    if i >= len(commands):
        cleanCommandExecLogs()
        os.system("echo " + str(int(time.time())) + " > " + var_dir + "command_execstart_0")
        return getCommand(0)

    # Finally, select the next command not executed so far
    os.system("echo " + str(int(time.time())) + " > " + var_dir + "command_execstart_" + str(i))
    return getCommand(i)


def execCommand(command):
    try:
        # updateTimestamp()
        if re.match(r".*&\s*$", command):
            # print "command should be ok"
            return os.system(command)
        else:
            # print "command not ok, inserting &"
            return os.system(command + " &")
    except OSError as detail:
        logging.exception("Error launching command '" + command + "'; error detail: " + str(detail))
        return None


def getCommand(index):
    global timestamp_max_diff
    try:
        if commands[index]["tmDiff"]:
            timestamp_max_diff = commands[index]["tmDiff"]
    except (IndexError, KeyError):
        timestamp_max_diff = timestamp_max_diff_default
    return commands[index]["exec"]


def killall():
    try:
        for cmd in commands:
            os.system(cmd["killcmd"])
    except KeyError:
        print("Could not issue the kill command for each entry, config file error!", file=sys.stderr)


# When SIGUSR1 or SIGUSR2 is received update timestamp
def receive_signal(signum, stack):
    global timestamp_signal
    timestamp_signal = int(time.time())


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
                logging.error("ERROR: File with commands not found - " + str(f) + " - continuing with other files")


def main():
    global commands, var_dir, timestamp_max_diff, last_kill_timestamp
    global timestamp_diff, timestamp_signal
    ################################################
    # KillTest Main Execution
    ################################################
    conf_file = '/etc/radiation-benchmarks.conf'

    # call the routine "receive_sginal" when SIGUSR1 is received
    signal.signal(signal.SIGUSR1, receive_signal)
    # call the routine "receive_sginal" when SIGUSR2 is received
    signal.signal(signal.SIGUSR2, receive_signal)

    if not os.path.isfile(conf_file):
        raise FileNotFoundError("System configuration file not found!(" + conf_file + ")")
        # sys.exit(1)

    try:
        config = configparser.RawConfigParser()
        config.read(conf_file)

        # install_dir = config.get('DEFAULT', 'installdir') + "/"
        var_dir = config.get('DEFAULT', 'vardir') + "/"
        log_dir = config.get('DEFAULT', 'logdir') + "/"
        # tmp_dir = config.get('DEFAULT', 'tmpdir') + "/"

        if not os.path.isdir(log_dir):
            os.mkdir(log_dir, 0o777)
            os.chmod(log_dir, 0o777)

    except IOError as e:
        raise IOError("System configuration setup error: " + str(e))
        # sys.exit(1)

    log_file = log_dir + "killtest.log"
    # timestampFile = var_dir + "timestamp.txt"

    # Logging definition
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M',
        filename=log_file,
        filemode='w'
    )

    if len(sys.argv) != 2:
        print("Usage: " + sys.argv[0] + " <file with absolute paths of json files>")
        sys.exit(1)

    commands = list()

    readCommands(sys.argv[1])

    if len(commands) < 1:
        raise ValueError("ERROR: No commands read, there is nothing to execute")
        # sys.exit(1)

    timestamp_max_diff = timestamp_max_diff_default
    # Start last kill timestamp with an old enough timestamp
    last_kill_timestamp = int(time.time()) - 50 * timestamp_max_diff
    timestamp_signal = int(time.time())

    try:
        kill_count = 0  # Counts how many kills were executed throughout execution
        cur_command = select_command()
        execCommand(cur_command)
        while True:
            sockConnect()

            # Get the current timestamp
            now = int(time.time())
            # timestamp_diff = now - timestamp
            timestamp_diff = now - timestamp_signal
            # If timestamp was not update properly
            if timestamp_diff > timestamp_max_diff:
                # Check if last kill was in the last 60 seconds and reboot
                killall()
                now = int(time.time())
                if (now - last_kill_timestamp) < 3 * timestamp_max_diff:
                    logging.info("Rebooting, last kill too recent, timestamp_diff: " + str(
                        timestamp_diff) + ", current command:" + cur_command)
                    sockConnect()
                    os.system("shutdown -r now")
                    time.sleep(20)
                else:
                    last_kill_timestamp = now

                kill_count += 1
                logging.info("timestamp_max_diff kill(#" + str(kill_count) + "), timestamp_diff:" + str(
                    timestamp_diff) + " command '" + cur_command + "'")
                # Reboot if we reach the max number of kills allowed
                if kill_count >= max_kill:
                    logging.info("Rebooting, max_kill reached, current command:" + cur_command)
                    sockConnect()
                    os.system("shutdown -r now")
                    time.sleep(20)
                else:
                    cur_command = select_command()  # select properly the current command to be executed
                    execCommand(cur_command)  # start the command

            time.sleep(1)
    except KeyboardInterrupt:  # Ctrl+c
        print("\n\tKeyboardInterrupt detected, exiting gracefully!( at least trying :) )")
        killall()
        sys.exit(1)


if __name__ == '__main__':
    # timestampFile = ""
    var_dir = ""
    timestamp = 0
    timestamp_max_diff = 0
    commands = []
    last_kill_timestamp = 0
    timestamp_signal = 0
    timestamp_diff = 0
    main()

"""
# Log messages adding timestamp before the message
def logMsg(msg):
    global logFile
    now = datetime.now()
    date_str = str(now.ctime()) + ": " + str(msg)
    with open(logFile, 'a') as fp:
        fp.write(date_str + '\n')
    print(date_str)


# Update the timestamp file with machine current timestamp
def updateTimestamp():
    global timestampFile
    command = "echo " + str(int(time.time())) + " > " + timestampFile
    # retcode = os.system(command)
    global timestamp_signal
    timestamp_signal = int(time.time())
"""
