#!/usr/bin/python3
import os
import socket
import time
import logging
import queue

from server_parameters import *
from server_package.Machine import Machine
# from server_package.RebootMachine import RebootMachine
from server_package.LoggerFormatter import ColoredLogger
from server_package.CopyLogs import CopyLogs


def start_copying():
    server_logs_path = "logs/"
    if os.path.exists(server_logs_path) is False:
        os.mkdir(server_logs_path)

    formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                  datefmt='%d-%m-%y %H:%M:%S')
    # create logger with 'spam_application'
    fh = logging.FileHandler(f"{server_logs_path}/copy.log", mode='a')
    fh.setFormatter(formatter)
    logger = logging.getLogger(COPY_LOGGER_NAME)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    copy_obj = CopyLogs(machines=MACHINES,
                        sleep_copy_interval=COPY_LOG_INTERVAL,
                        destination_folder=server_logs_path,
                        logger_name=None,
                        to_copy_folder="/var/radiation-benchmarks/log/")
    copy_obj.start()

    return copy_obj


def generate_machine_hash(messages_queue):
    """
    Generate the objects for the devices
    :return:
    """
    machines_hash = dict()
    for mac in MACHINES:
        if mac["enabled"]:
            mac_obj = Machine(
                ip=mac["ip"],
                diff_reboot=mac["diff_reboot"],
                hostname=mac["hostname"],
                power_switch_ip=mac["power_switch_ip"],
                power_switch_port=mac["power_switch_port"],
                power_switch_model=mac["power_switch_model"],
                messages_queue=messages_queue,
                sleep_time=MACHINE_CHECK_SLEEP_TIME,
                logger_name=LOGGER_NAME,
                boot_problem_max_delta=BOOT_PROBLEM_MAX_DELTA,
                reboot_sleep_time=REBOOTING_SLEEP
                # RebootMachine=RebootMachine
            )

            machines_hash[mac["ip"]] = mac_obj
            mac_obj.start()
    return machines_hash


def logging_setup():
    """
    Logging setup
    :return: logger
    """
    # create logger with 'spam_application'
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(LOG_FILE, mode='a')
    fh.setLevel(logging.INFO)
    # create formatter and add it to the handlers
    formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                  datefmt='%d-%m-%y %H:%M:%S')

    # add the handlers to the logger
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # create console handler with a higher log level for console
    console = ColoredLogger(LOGGER_NAME)
    # noinspection PyTypeChecker
    logger.addHandler(console)
    return logger


def main():
    """
    Main function
    :return: None
    """
    # log format
    logger = logging_setup()

    # copy obj and logging
    copy_obj = start_copying()

    # Queue to print the messages in a good way
    messages_queue = queue.Queue()

    # safe check for socket
    client_socket = None

    # attach signal handler for CTRL + C
    try:
        # Start the server socket
        # Create an INET, STREAMing socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
            # Bind the socket to a public host, and a well-known port
            server_socket.bind((SERVER_IP, SOCKET_PORT))

            # Initialize a list that contains all Machines
            machines_hash = generate_machine_hash(messages_queue)

            logger.info(f"\tServer bind to: {SERVER_IP}")
            # Become a server socket
            # TODO: find the correct value for backlog parameter
            server_socket.listen(15)

            while True:
                # Accept connections from outside
                client_socket, address_list = server_socket.accept()
                address = address_list[0]

                # Set new timestamp
                timestamp = time.time()
                machines_hash[address].set_timestamp(timestamp=timestamp)

                # Close the connection
                client_socket.close()
                logger.debug(f"\tConnection from {address} machine {machines_hash[address].get_hostname()}")
    except KeyboardInterrupt:
        # Stop mac objects
        for mac_obj in machines_hash.values():
            mac_obj.join()

        # Close client socket
        if client_socket:
            client_socket.close()

        # Stop copy thread
        copy_obj.join()

        logger.error("KeyboardInterrupt detected, exiting gracefully!( at least trying :) )")
        exit(130)


if __name__ == '__main__':
    main()
