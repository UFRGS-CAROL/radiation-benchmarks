import logging
import socket
import threading
import time
import os
import paramiko
import scp


class CopyLogs(threading.Thread):

    def __init__(self, hostname, ip, password, username, sleep_copy_interval, destination_folder, to_copy_folder):
        """
        Create a CopyLogs obj that keeps copying from the machines list
        :param hostname: the hostname
        :param ip: IP
        :param password: password
        :param username: username
        :param sleep_copy_interval: interval of each copy
        :param destination_folder: local folder to store the logs
        :param to_copy_folder: where the logs will be located at the client device
        """

        super(CopyLogs, self).__init__()
        self.__hostname = hostname
        self.__ip = ip
        self.__password = password
        self.__username = username
        self.__destination_folder = destination_folder
        self.__sleep_copy_interval = sleep_copy_interval
        self.__to_copy_folder = to_copy_folder
        self.__stop_event = threading.Event()

        # now the output log file is the combination of the board + _copy.log
        log_messages_file = f"{self.__destination_folder}/{self.__hostname}_copy.log"

        # Start the logger
        if os.path.exists(self.__destination_folder) is False:
            os.mkdir(self.__destination_folder)

        formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                      datefmt='%d-%m-%y %H:%M:%S')
        # create logger with 'spam_application'
        fh = logging.FileHandler(log_messages_file, mode='a')
        fh.setFormatter(formatter)
        board_logger_name = f"{__name__}.{self.__hostname}"
        self.__logger = logging.getLogger(board_logger_name)
        self.__logger.setLevel(logging.DEBUG)
        self.__logger.addHandler(fh)

    def run(self):
        # Create the path if it doesnt exist
        destination = f"{self.__destination_folder}/{self.__hostname}"
        try:
            if os.path.exists(destination) is False:
                os.mkdir(destination)
        except FileExistsError:
            self.__logger.exception(f"Failed on creating {destination} path")
        self.__logger.debug(f"Running copy thread for {self.__hostname}")

        # Start thread
        while not self.__stop_event.isSet():
            # Try until a successful boot
            was_copy_successful = False
            while was_copy_successful is False:
                self.__logger.info(f"Attempt to copy from IP {self.__ip} hostname {self.__hostname}")
                was_copy_successful = self.__copy_from_address(ip=self.__ip, destination_folder=destination,
                                                               username=self.__username,
                                                               password=self.__password)
                # Avoid to flood the network
                self.__stop_event.wait(1)

            # Wait for the main time
            self.__stop_event.wait(self.__sleep_copy_interval)

    def __copy_from_address(self, ip, destination_folder, username, password):
        """
        Copy from an address
        :param ip:
        :param destination_folder:
        :return: if it was successful or not
        """
        try:
            ssh = paramiko.SSHClient()
            ssh.load_system_host_keys()
            ssh.connect(ip, username=username, password=password, allow_agent=False, look_for_keys=False)

            ssh_client = scp.SCPClient(ssh.get_transport())

            # GET the logs from machine
            ssh_client.get(self.__to_copy_folder, destination_folder, recursive=True)
            ssh_client.close()
            ssh.close()
            self.__logger.info(f"Copy was successful from IP {ip} username {username}")
            return True
        except (paramiko.BadHostKeyException, paramiko.AuthenticationException,
                paramiko.SSHException, socket.error, paramiko.ChannelException,
                scp.SCPException, paramiko.ssh_exception.NoValidConnectionsError) as ex:
            self.__logger.error(f"Could not download logs from {ip}. Exception {ex}")
            return False

    def join(self, *args, **kwargs):
        """
        Stop the copy thread
        :return:
        """
        self.__stop_event.set()
        super(CopyLogs, self).join(*args, **kwargs)


if __name__ == '__main__':
    # FOR DEBUG USE ONLY
    import sys

    sys.path.append("..")
    from server_parameters import MACHINES

    # copy_logs = CopyLogs(hostname=MACHINES,
    #                      sleep_copy_interval=10,
    #                      destination_folder="/tmp",
    #                      to_copy_folder="/var/radiation-benchmarks/log/", log_messages_file="unit_test_CopyLogs.log")
    # copy_logs.start()
    # time.sleep(10)
    # copy_logs.join()
