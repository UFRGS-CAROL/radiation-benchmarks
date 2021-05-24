#!/usr/bin/python3
import collections
import logging
import socket
import time
import os
import os.path
import configparser
import sys
import re
import signal
import json

# PORT the socket will listen to
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

# logger name
DEFAULT_LOGGER_NAME = os.path.basename(__file__).upper().replace(".PY", "")

# Default amount of benchmarks for each json config file
# 24h  * week of experiments
MAX_BENCHMARK_QUEUE = 24 * 7

# Default file to store the Queue
DEFAULT_QUEUE_FILE = "commandExecutionStart"

# Kill test log
KILL_TEST_LOG = "killtest.log"


def execute_system_command(command):
    logger = logging.getLogger(DEFAULT_LOGGER_NAME)
    sys_return = os.system(command)
    if sys_return != 0:
        logger.debug(f"Could not execute command {command}")


class Benchmark:
    def __init__(self, kill_command, execution_command, timestamp_max_diff=TIMESTAMP_MAX_DIFF_DEFAULT):
        # Mandatory execute command from json
        self.__execution_command = execution_command
        # Mandatory kill command from json
        self.__kill_command = kill_command
        # Attribute to keep the timestamp diff
        self.__timestamp_diff = 0
        # Counts how many kills were executed throughout execution
        self.__kill_counter = 0
        # Start last kill timestamp with an old enough timestamp
        self.__last_kill_timestamp = int(time.time()) - 50 * timestamp_max_diff
        # Max diff allowed
        self.__timestamp_max_diff = timestamp_max_diff
        # Max kill counter
        self.__max_kill_counter = MAX_KILL
        # Last timestamp registered
        self.__last_timestamp_registered = int(time.time())

    # Following PEP8 all the attribute must be private
    @property
    def execution_command(self): return self.__execution_command

    @property
    def kill_counter(self): return self.__kill_counter

    @property
    def timestamp_diff(self): return self.__timestamp_diff

    def is_inside_benchmark_time_window(self):
        # Get the current timestamp
        now = int(time.time())
        return (now - self.__last_timestamp_registered) < self.__timestamp_max_diff

    def is_timestamp_outdated(self, timestamp):
        # Get the current timestamp
        now = int(time.time())
        self.__timestamp_diff = now - timestamp
        return self.__timestamp_diff > self.__timestamp_max_diff

    def was_the_last_kill_too_recent(self):
        now = int(time.time())
        return (now - self.__last_kill_timestamp) < 3 * self.__timestamp_max_diff

    def reach_max_kill(self):
        return self.__kill_counter >= self.__max_kill_counter

    def execute_command(self):
        """ Treat and execute this benchmark
        :return: None
        """
        command = self.execution_command
        if not re.match(r".*&\s*$", command):
            # "command not ok, inserting &"
            command += " &"
        execute_system_command(command)

    def kill_this_benchmark(self):
        """ Kill this benchmark
        :return: None
        """
        self.__kill_counter += 1
        execute_system_command(self.__kill_command)
        self.__last_kill_timestamp = int(time.time())


def sock_connect():
    """
    Connect to server and close connection, kind of ping
    :return:
    """
    logger = logging.getLogger(DEFAULT_LOGGER_NAME)
    try:
        # create an INET, STREAMing socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
            # python 3 requires a TIMEOUT
            client_socket.settimeout(2)
            # Now, connect with IP (or hostname) and PORT
            client_socket.connect((SERVER_IP, SOCKET_PORT))
    except socket.error:
        logger.debug("could not connect to remote server, socket error")


def create_logger(log_path):
    """
    Create the logger for this section
    :param log_path: is the path for the output logging file
    :return:
    """
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


def select_command(benchmarks_command_file, benchmarks_queue: collections.deque):
    """
    Select the correct command to be executed from the commands
    :return:
    """
    logger = logging.getLogger(DEFAULT_LOGGER_NAME)
    # TODO: correct the command selection
    current_benchmark = benchmarks_queue.pop()
    benchmarks_queue.appendleft(current_benchmark)


    return current_benchmark


def read_commands(file_list):
    """
    This function reads from the json config file
    then populate a deque with each benchmark
    :param file_list:
    :return:
    """
    logger = logging.getLogger(DEFAULT_LOGGER_NAME)
    # now it is a Circular Queue of benchmarks
    benchmarks_queue = collections.deque(maxlen=MAX_BENCHMARK_QUEUE)
    benchmarks_counter = 0
    if os.path.isfile(file_list):
        with open(file_list, "r") as fp:
            for line in fp:
                line = line.strip()
                if not line.startswith("#") and not line.startswith("%") and os.path.isfile(line):
                    with open(line, "r") as fjson:
                        benchmarks = json.load(fjson)
                    for benchmark in benchmarks:
                        benchmarks_counter += 1
                        timestamp_max_diff = TIMESTAMP_MAX_DIFF_DEFAULT
                        if "tmDiff" in benchmark:
                            timestamp_max_diff = benchmark["tmDiff"]
                        benchmarks_queue.appendleft(
                            Benchmark(kill_command=benchmark["killcmd"], execution_command=benchmark["exec"],
                                      timestamp_max_diff=timestamp_max_diff))
                else:
                    logger.error(f"ERROR: File with commands not found - {line} - continuing with other files")
    # It must have more than 1 benchmark
    assert benchmarks_counter >= 1, "ERROR: No commands read, there is nothing to execute"
    return benchmarks_queue


def receive_signal(signum, stack):
    """
    When SIGUSR1 or SIGUSR2 is received update timestamp
    :param signum:
    :param stack:
    :return:
    """
    global timestamp_signal
    timestamp_signal = int(time.time())


def main():
    """
    KillTest Main Execution
    :return: None
    """
    global timestamp_signal
    # call the routine "receive_sginal" when SIGUSR1 is received
    signal.signal(signal.SIGUSR1, receive_signal)
    # call the routine "receive_sginal" when SIGUSR2 is received
    signal.signal(signal.SIGUSR2, receive_signal)

    if not os.path.isfile(CONF_FILE):
        raise FileNotFoundError(f"System configuration file not found!({CONF_FILE})")

    try:
        config = configparser.RawConfigParser()
        config.read(CONF_FILE)
        var_dir = config.get('DEFAULT', 'vardir') + "/"
        log_dir = config.get('DEFAULT', 'logdir') + "/"

        if not os.path.isdir(log_dir):
            os.mkdir(log_dir, 0o777)
            os.chmod(log_dir, 0o777)
    except IOError as e:
        raise IOError(f"System configuration setup error: {e}")

    log_file = f"{log_dir}/{KILL_TEST_LOG}"

    # create the logger
    logger = create_logger(log_path=log_file)

    if len(sys.argv) != 2:
        logger.debug(f"Usage: {sys.argv[0]} <file with absolute paths of json files>")
        sys.exit(1)

    benchmarks_command_file = f"{var_dir}/{DEFAULT_QUEUE_FILE}"
    commands = read_commands(file_list=sys.argv[1])

    try:
        current_benchmark = select_command(benchmarks_command_file=benchmarks_command_file, benchmarks_queue=commands)
        current_benchmark.execute_command()
        while True:
            sock_connect()
            # If timestamp was not update properly
            if current_benchmark.is_timestamp_outdated(timestamp=timestamp_signal):
                # Kill this benchmark running
                current_benchmark.kill_this_benchmark()
                # Check if last kill was in the last 60 seconds and reboot
                if current_benchmark.was_the_last_kill_too_recent():
                    logger.info(
                        f"Rebooting, last kill too recent, timestampDiff: {current_benchmark.timestamp_diff}, "
                        f"current command:{current_benchmark.execution_command}")
                    sock_connect()
                    execute_system_command("shutdown -r now")
                    time.sleep(20)

                logger.info(f"timestampMaxDiff kill(#{current_benchmark.kill_counter}),"
                            f" timestampDiff:{current_benchmark.timestamp_diff} {current_benchmark.execution_command}")
                # Reboot if we reach the max number of kills allowed
                if current_benchmark.reach_max_kill():
                    logger.info(f"Rebooting, maxKill reached, current command:{current_benchmark.execution_command}")
                    sock_connect()
                    execute_system_command("shutdown -r now")
                    time.sleep(20)
                else:
                    # select properly the current command to be executed
                    current_benchmark = select_command(benchmarks_command_file=benchmarks_command_file,
                                                       benchmarks_queue=commands)
                    # start the benchmark
                    current_benchmark.execute_command()

            time.sleep(1)
    except KeyboardInterrupt:  # Ctrl+c
        logger.debug("\n\tKeyboardInterrupt detected, exiting gracefully!( at least trying :) )")
        for benchmark in commands:
            benchmark.kill_this_benchmark()
        # 130 is the key for CTRL + c
        exit(130)


if __name__ == '__main__':
    timestamp_signal = 0
    main()
