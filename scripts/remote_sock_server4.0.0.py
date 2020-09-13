#!/usr/bin/python3

import socket
import sys
import time
import logging
import queue

from server_package.server_parameters import SERVER_IP, SOCKET_PORT, MACHINES, LOG_FILE, LOGGER_NAME
from server_package.common import Codes
from server_package.Machine import Machine
from server_package.LoggerFormatter import ColoredLogger


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
                messages_queue=messages_queue
            )

            machines_hash[mac["ip"]] = mac_obj
            mac_obj.start()
    return machines_hash


def logging_setup():
    """
    Logging setup
    :return: logger
    """
    # set up logging to file - see previous section for more details
    # logging.basicConfig(
    #     level=logging.DEBUG,
    #     format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
    #     datefmt='%m-%d %H:%M',
    #     filename=LOG_FILE,
    #     filemode='w'
    # )

    # create logger with 'spam_application'
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(LOG_FILE)
    fh.setLevel(logging.INFO)
    # create formatter and add it to the handlers
    formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                  datefmt='%d-%m-%y %H:%M:%S')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # create console handler with a higher log level for console
    # ch = logging.StreamHandler()
    console = ColoredLogger(LOGGER_NAME)

    # add the handlers to the logger
    logger.addHandler(console)
    return logger


def main():
    """
    Main function
    :return: None
    """
    # log format
    logger = logging_setup()

    # Queue to print the messages in a good way
    messages_queue = queue.Queue()

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
        for mac_obj in machines_hash.values():
            mac_obj.join()
        logger.error("\tKeyboardInterrupt detected, exiting gracefully!( at least trying :) )")
        exit(Codes.CTRL_C)


if __name__ == '__main__':
    main()
