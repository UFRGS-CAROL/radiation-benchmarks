#!/usr/bin/python3
import logging
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

from server_parameters import SERVER_IP, SOCKET_PORT

# Time in seconds to wait for the timestamp update
TIMESTAMP_MAX_DIFF_DEFAULT = 30

# Max number of kills allowed
MAX_KILL = 5

# How long each command will execute, time window in seconds
TIME_WINDOW_COMMANDS = 3600

# Config file
CONF_FILE = '/etc/radiation-benchmarks.conf'

# logger name
DEFAULT_LOGGER_NAME = os.path.basename(__file__).upper()


def sock_connect():
    """
    Connect to server and close connection, kind of ping
    :return:
    """
    logger = logging.getLogger(DEFAULT_LOGGER_NAME)
    try:
        # create an INET, STREAMing socket
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # python 3 requires a TIMEOUT
        s.settimeout(2)
        # Now, connect with IP (or hostname) and PORT
        s.connect((SERVER_IP, SOCKET_PORT))
        s.close()
    except socket.error:
        logger.debug("could not connect to remote server, socket error")


def create_logger(log_path, logger_name):
    formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                  datefmt='%d-%m-%y %H:%M:%S')
    # create logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    # file handler
    fh = logging.FileHandler(f"{log_path}", mode='a')
    fh.setFormatter(formatter)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    # console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def update_timestamp_file(timestamp_file):
    """
    Update the timestamp file with machine current timestamp
    :return:
    """
    command = "echo " + str(int(time.time())) + " > " + timestamp_file
    return_code = os.system(command)
    timestamp_signal = int(time.time())
    return return_code, timestamp_signal


def clean_command_exec_logs(var_dir):
    """
    Remove files with start timestamp of commands executing
    :return:
    """
    os.system("rm -f " + var_dir + "command_execstart_*")


def check_command_list_changes(var_dir, commands):
    """
    Return True if the commandFile changed from the
    last time it was executed. If the file was never executed returns False
    :return:
    """
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


def get_command(index, commands):
    # timestamp_max_diff = 0
    try:
        # if commands[index]["tmDiff"]:
        timestamp_max_diff = commands[index]["tmDiff"]
    except (KeyError, IndexError, ValueError, TypeError):
        timestamp_max_diff = TIMESTAMP_MAX_DIFF_DEFAULT
    return commands[index]["exec"], timestamp_max_diff


def select_command(var_dir, commands, timestamp):
    """
    Select the correct command to be executed from the commands
    :return:
    """
    logger = logging.getLogger(DEFAULT_LOGGER_NAME)

    if check_command_list_changes(var_dir=var_dir, commands=commands):
        clean_command_exec_logs(var_dir=var_dir)

    # Get the index of last existent file
    i = 0
    while os.path.isfile(var_dir + "command_execstart_" + str(i)):
        i += 1
    i -= 1

    # If there is no file, create the first file with current timestamp
    # and return the first command of commands list
    if i == -1:
        os.system("echo " + str(int(time.time())) + " > " + var_dir + "command_execstart_0")
        return get_command(index=0, commands=commands)

    # Check if last command executed is still in the defined time window for each command
    # and return it

    # Read the timestamp file
    try:
        fp = open(var_dir + "command_execstart_" + str(i), 'r')
        timestamp = int(float(fp.readline().strip()))
        fp.close()
    except ValueError as eDetail:
        logger.info("Rebooting, command execstart timestamp read error: " + str(eDetail))
        sock_connect()
        os.system("rm -f " + var_dir + "command_execstart_" + str(i))
        os.system("shutdown -r now")
        time.sleep(20)
    # fp = open(varDir+"command_execstart_"+str(i),'r')
    # timestamp = int(float(fp.readline().strip()))
    # fp.close()

    now = int(time.time())
    if (now - timestamp) < TIME_WINDOW_COMMANDS:
        return get_command(index=i, commands=commands)

    i += 1
    # If all commands executed their time window, start all over again
    if i >= len(commands):
        clean_command_exec_logs(var_dir=var_dir)
        os.system("echo " + str(int(time.time())) + " > " + var_dir + "command_execstart_0")
        return get_command(index=0, commands=commands)

    # Finally, select the next command not executed so far
    os.system("echo " + str(int(time.time())) + " > " + var_dir + "command_execstart_" + str(i))
    return get_command(index=i, commands=commands)


def execCommand(command):
    """
    Execute command
    :param command: cmd to execute
    :return:
    """
    logger = logging.getLogger(DEFAULT_LOGGER_NAME)
    try:
        # TODO: Check why this update is here
        # updateTimestamp(timestamp_file=)
        if re.match(r".*&\s*$", command):
            # print "command should be ok"
            return os.system(command)
        else:
            # print "command not ok, inserting &"
            return os.system(command + " &")
    except OSError as detail:
        logger.exception("Error launching command '" + command + "'; error detail: " + str(detail))
        return None


def killall(commands):
    logger = logging.getLogger(DEFAULT_LOGGER_NAME)
    sys_return = 0
    try:
        for cmd in commands:
            sys_return += os.system(cmd["killcmd"])
        if sys_return != 0:
            raise ValueError
    except (KeyError, ValueError, TypeError):
        logger.debug("Could not issue the kill command for each entry, config file error!")


# When SIGUSR1 or SIGUSR2 is received update timestamp
def receive_signal(signum, stack):
    global timestampSignal
    timestampSignal = int(time.time())


def read_commands(file_list):
    logger = logging.getLogger(DEFAULT_LOGGER_NAME)
    commands = list()
    if os.path.isfile(file_list):
        with open(file_list, "r") as fp:
            for f in fp:
                f = f.strip()
                if f.startswith("#") or f.startswith("%"):
                    continue
                if os.path.isfile(f):
                    with open(f, "r") as fjson:
                        data = json.load(fjson)
                    commands.extend(data)
                else:
                    logger.error(f"ERROR: File with commands not found - {f} - continuing with other files")
    return commands


def main():
    ################################################
    # KillTest Main Execution
    ################################################
    # call the routine "receive_sginal" when SIGUSR1 is received
    signal.signal(signal.SIGUSR1, receive_signal)
    # call the routine "receive_sginal" when SIGUSR2 is received
    signal.signal(signal.SIGUSR2, receive_signal)

    if not os.path.isfile(CONF_FILE):
        raise FileNotFoundError(f"System configuration file not found!({CONF_FILE})")

    try:
        config = configparser.RawConfigParser()
        config.read(CONF_FILE)

        install_dir = config.get('DEFAULT', 'installdir') + "/"
        var_dir = config.get('DEFAULT', 'vardir') + "/"
        log_dir = config.get('DEFAULT', 'logdir') + "/"
        tmp_dir = config.get('DEFAULT', 'tmpdir') + "/"

        if not os.path.isdir(log_dir):
            os.mkdir(log_dir, 0o777)
            os.chmod(log_dir, 0o777)

    except IOError as e:
        raise IOError("System configuration setup error: " + str(e))
        # sys.exit(1)

    log_file = log_dir + "killtest.log"
    timestamp_file = var_dir + "timestamp.txt"

    # create the logger
    logger = create_logger(log_path=log_file, logger_name=DEFAULT_LOGGER_NAME)

    if len(sys.argv) != 2:
        logger.debug(f"Usage: {sys.argv[0]} <file with absolute paths of json files>")
        sys.exit(1)

    commands = read_commands(sys.argv[1])

    if len(commands) < 1:
        raise ValueError("ERROR: No commands read, there is nothing to execute")
        # sys.exit(1)

    timestamp_max_diff = TIMESTAMP_MAX_DIFF_DEFAULT
    # Start last kill timestamp with an old enough timestamp
    last_kill_timestamp = int(time.time()) - 50 * timestamp_max_diff
    timestamp_signal = int(time.time())

    try:
        kill_count = 0  # Counts how many kills were executed throughout execution
        cur_command, timestamp_max_diff = select_command(var_dir=var_dir, commands=commands, timestamp=timestamp_signal)
        execCommand(cur_command)
        while True:
            sock_connect()

            # Get the current timestamp
            now = int(time.time())
            # timestampDiff = now - timestamp
            timestamp_diff = now - timestamp_signal
            # If timestamp was not update properly
            if timestamp_diff > timestamp_max_diff:
                # Check if last kill was in the last 60 seconds and reboot
                killall(commands=commands)
                now = int(time.time())
                if (now - last_kill_timestamp) < 3 * timestamp_max_diff:
                    logger.info(
                        f"Rebooting, last kill too recent, timestampDiff: {timestamp_diff}, "
                        f"current command:{cur_command}")
                    sock_connect()
                    os.system("shutdown -r now")
                    time.sleep(20)
                else:
                    last_kill_timestamp = now

                kill_count += 1
                logger.info("timestampMaxDiff kill(#" + str(kill_count) + "), timestampDiff:" + str(
                    timestamp_diff) + " command '" + cur_command + "'")
                # Reboot if we reach the max number of kills allowed
                if kill_count >= MAX_KILL:
                    logger.info("Rebooting, maxKill reached, current command:" + cur_command)
                    sock_connect()
                    os.system("shutdown -r now")
                    time.sleep(20)
                else:
                    # select properly the current command to be executed
                    cur_command = select_command(var_dir=var_dir, commands=commands, timestamp=timestamp_signal)
                    execCommand(cur_command)  # start the command

            time.sleep(1)
    except KeyboardInterrupt:  # Ctrl+c
        logger.debug("\n\tKeyboardInterrupt detected, exiting gracefully!( at least trying :) )")
        killall(commands=commands)
        # 130 is the key for CTRL + c
        exit(130)


if __name__ == '__main__':
    timestampSignal = 0
    main()
