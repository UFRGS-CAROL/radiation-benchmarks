#!/usr/bin/python3
import configparser
import json
import logging
import os
import os.path
import re
import signal
import socket
import sys
import time

# PORT the socket will listen to
from typing import List, Dict

SOCKET_PORT = 8080
# IP of the remote socket server (hardware watchdog)
SERVER_IP = "25.89.165.225"
# Time in seconds to wait for the timestamp update
TIMESTAMP_MAX_DIFF_DEFAULT = 30
# Max number of kills allowed
MAX_KILL = 5
# How long each command will execute, time window in seconds
TIME_WINDOW_COMMANDS = 3600
# Config file
CONF_FILE = '/etc/radiation-benchmarks.conf'
"""
Parameters related to the script directly
"""
# logger name
DEFAULT_LOGGER_NAME = os.path.basename(__file__).upper().replace(".PY", "")
# Default file to store the Queue
DEFAULT_QUEUE_FILE = "commandExecutionStart"
# Kill test log
KILL_TEST_LOG = "killtest.log"

# Exit success status
EXIT_SUCCESS_STATUS = 0


def execute_system_command(command):
    """ Execute a command in the system
    :param command:    """
    sys_return = os.system(command)  # if "shutdown" not in command else 1
    # TODO: improve the checking of the system commands' outputs
    return sys_return


def execute_benchmark(command):
    """ Execute command
    :param command: cmd to execute with or without &
    :return:
    """
    logger = logging.getLogger(DEFAULT_LOGGER_NAME)
    try:
        cmd = command
        # Check if the end of the string is the & character
        if re.match(r".*&\s*$", command) is None:
            cmd += " &"
        if execute_system_command(cmd) != EXIT_SUCCESS_STATUS:
            logger.debug(f"Clean benchmark command could not be executed - {cmd}")
    except OSError as detail:
        logger.error(f"Error launching command '{command}'; error detail: {detail}")


def create_logger(log_path):
    """ Create the logger for this section
    :param log_path: is the path for the output logging file    """
    formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s - %(filename)s:%(lineno)d',
                                  datefmt='%d-%m-%y %H:%M:%S')
    # create logger
    logger = logging.getLogger(DEFAULT_LOGGER_NAME)
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


def sock_connect():
    """ Connect to server and close connection, kind of ping    """
    try:
        # create an INET, STREAMing socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            # python 3 requires a TIMEOUT
            s.settimeout(2)
            # Now, connect with IP (or hostname) and PORT
            s.connect((SERVER_IP, SOCKET_PORT))
    except socket.error:
        logging.getLogger(DEFAULT_LOGGER_NAME).debug("could not connect to remote server, socket error")


def select_command(var_dir, commands):
    """ Select the correct command to be executed from the commands
    :param var_dir:
    :param commands:
    """
    logger = logging.getLogger(DEFAULT_LOGGER_NAME)
    raise NotImplementedError("Finish the list selection first")
    return None, 0


def killall(commands):
    """  Killall the benchmarks based on commands list
    :param commands:
    :return:
    """
    logger = logging.getLogger(DEFAULT_LOGGER_NAME)
    try:
        for cmd in commands:
            kill_cmd = cmd["killcmd"]
            if execute_system_command(kill_cmd) != EXIT_SUCCESS_STATUS:
                logger.debug(f"Could not issue {kill_cmd}")
    except (KeyError, ValueError, TypeError):
        logger.debug("Could not issue the kill command for each entry, config file error!")


# noinspection PyUnusedLocal
def receive_signal(signum, stack):
    """ When SIGUSR1 or SIGUSR2 is received update timestamp
    :param signum:
    :param stack:
    :return:
    """
    global timestamp_signal
    timestamp_signal = int(time.time())


def read_commands(file_list) -> List[Dict]:
    """  Read the commands from the file
    :param file_list:
    :return:
    """
    commands = list()
    if os.path.isfile(file_list):
        with open(file_list, "r") as fp:
            for f in fp:
                f = f.strip()
                if f.startswith("#") or f.startswith("%"):
                    continue
                if os.path.isfile(f):
                    with open(f, "r") as json_file:
                        data = json.load(json_file)
                    commands.extend(data)
                else:
                    logging.getLogger(DEFAULT_LOGGER_NAME).error(
                        f"ERROR: File with commands not found - {f} - continuing with other files")
    return commands


def main():
    """ KillTest Main Execution
    :return:
    """
    global timestamp_signal

    # call the routine "receive_signal" when SIGUSR1 is received
    signal.signal(signal.SIGUSR1, receive_signal)
    # call the routine "receive_signal" when SIGUSR2 is received
    signal.signal(signal.SIGUSR2, receive_signal)

    if not os.path.isfile(CONF_FILE):
        raise FileNotFoundError("System configuration file not found!(" + CONF_FILE + ")")

    try:
        config = configparser.RawConfigParser()
        config.read(CONF_FILE)

        var_dir = config.get('DEFAULT', 'vardir') + "/"
        log_dir = config.get('DEFAULT', 'logdir') + "/"

        if not os.path.isdir(log_dir):
            os.mkdir(log_dir, 0o777)
            os.chmod(log_dir, 0o777)

    except IOError as e:
        raise IOError("System configuration setup error: " + str(e))

    # Get logger
    log_file = log_dir + "killtest.log"
    logger = create_logger(log_path=log_file)

    if len(sys.argv) != 2:
        logger.debug(f"Usage: {sys.argv[0]} <file with absolute paths of json files>")
        sys.exit(1)

    commands = read_commands(sys.argv[1])

    if len(commands) < 1:
        raise ValueError("ERROR: No commands read, there is nothing to execute")

    timestamp_max_diff = TIMESTAMP_MAX_DIFF_DEFAULT
    # Start last kill timestamp with an old enough timestamp
    last_kill_timestamp = int(time.time()) - 50 * timestamp_max_diff
    timestamp_signal = int(time.time())

    try:
        # Counts how many kills were executed throughout execution
        kill_count = 0
        cur_command, timestamp_max_diff = select_command(var_dir=var_dir, commands=commands)
        execute_benchmark(cur_command)
        while True:
            sock_connect()

            # Get the current timestamp
            now = int(time.time())
            # timestamp_diff = now - timestamp
            timestamp_diff = now - timestamp_signal
            # If timestamp was not update properly
            if timestamp_diff > timestamp_max_diff:
                # kill this benchmark
                killall(commands=commands)
                # Check if last kill was in the last 60 seconds and reboot
                now = int(time.time())
                if (now - last_kill_timestamp) < 3 * timestamp_max_diff:
                    logger.info(f"Rebooting, last kill too recent, timestamp_diff: {timestamp_diff}, "
                                f"current command:{cur_command}")
                    sock_connect()
                    if execute_system_command("shutdown -r now") != EXIT_SUCCESS_STATUS:
                        logger.debug("shutdown -r not successful")
                    time.sleep(20)
                else:
                    last_kill_timestamp = now

                kill_count += 1
                logger.info(
                    f"timestamp_max_diff kill(#{kill_count}), timestamp_diff:{timestamp_diff} command '{cur_command}'")
                # Reboot if we reach the max number of kills allowed
                if kill_count >= MAX_KILL:
                    logger.info(f"Rebooting, maxKill reached, current command:{cur_command}")
                    sock_connect()
                    if execute_system_command("shutdown -r now") != EXIT_SUCCESS_STATUS:
                        logger.debug("shutdown -r not successful")
                    time.sleep(20)
                else:
                    # select properly the current command to be executed
                    cur_command, timestamp_max_diff = select_command(var_dir=var_dir, commands=commands)
                    execute_benchmark(cur_command)  # start the command

            time.sleep(1)
    except KeyboardInterrupt:  # Ctrl+c
        print("\n\tKeyboardInterrupt detected, exiting gracefully!( at least trying :) )")
        killall(commands=commands)
        # 130 is the key for CTRL + c
        exit(130)


if __name__ == '__main__':
    timestamp_signal = 0
    main()
