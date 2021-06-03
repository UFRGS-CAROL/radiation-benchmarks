import logging
import socket
import threading
import time
import os
import paramiko
import scp


class CopyLogs(threading.Thread):

    def __init__(self, machines, sleep_copy_interval, destination_folder, to_copy_folder, log_messages_file="copy.log"):
        """
        Create a CopyLogs obj that keeps copying from the machines list
        :param machines: list of machines
        :param sleep_copy_interval: interval of each copy
        :param destination_folder: local folder to store the logs
        :param to_copy_folder: where the logs will be located at the client device
        :param log_messages_file: output file for the info messages
        """

        super(CopyLogs, self).__init__()
        self.__machines_list = machines
        self.__destination_folder = destination_folder
        self.__sleep_copy_interval = sleep_copy_interval
        self.__to_copy_folder = to_copy_folder
        self.__stop_event = threading.Event()

        # Start the logger

        if os.path.exists(self.__destination_folder) is False:
            os.mkdir(self.__destination_folder)

        formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                      datefmt='%d-%m-%y %H:%M:%S')
        # create logger with 'spam_application'
        fh = logging.FileHandler(f"{self.__destination_folder}/{log_messages_file}", mode='a')
        fh.setFormatter(formatter)
        self.__logger = logging.getLogger(__name__)
        self.__logger.setLevel(logging.DEBUG)
        self.__logger.addHandler(fh)

    def run(self):
        # Create the path if it doesnt exist
        for mac in self.__machines_list:
            hostname = mac["hostname"]
            is_active_device = mac["enabled"]
            destination = f"{self.__destination_folder}/{hostname}"

            try:
                if is_active_device and os.path.exists(destination) is False:
                    os.mkdir(destination)
            except FileExistsError:
                self.__logger.exception(f"Failed on creating {destination} path")
        self.__logger.debug("Running copy thread")

        # Start thread
        while not self.__stop_event.isSet():
            for mac in self.__machines_list:
                ip = mac["ip"]
                hostname = mac['hostname']
                is_active_device = mac["enabled"]
                password = mac["password"]
                username = mac["username"]
                destination = f"{self.__destination_folder}/{hostname}"

                if is_active_device:
                    self.__logger.info(f"Attempt to copy from IP {ip} hostname {hostname}")
                    self.__copy_from_address(ip=ip, destination_folder=destination, username=username,
                                             password=password)

            self.__stop_event.wait(self.__sleep_copy_interval)  # instead of sleeping

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

        except (paramiko.BadHostKeyException, paramiko.AuthenticationException,
                paramiko.SSHException, socket.error, paramiko.ChannelException,
                scp.SCPException, paramiko.ssh_exception) as ex:
            self.__logger.error(f"Could not download logs from {ip}. Exception {ex}")

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

    copy_logs = CopyLogs(machines=MACHINES,
                         sleep_copy_interval=10,
                         destination_folder="/tmp",
                         to_copy_folder="/var/radiation-benchmarks/log/", log_messages_file="unit_test_CopyLogs.log")
    copy_logs.start()
    time.sleep(10)
    copy_logs.join()
