import logging
import socket
import threading
import time
import os
import paramiko
import scp


class CopyLogs(threading.Thread):

    def __init__(self, machines, sleep_copy_interval, destination_folder, logger_name, to_copy_folder):
        super(CopyLogs, self).__init__()
        self.__machines_list = machines
        self.__destination_folder = destination_folder
        self.__sleep_copy_interval = sleep_copy_interval
        self.__logger = logging.getLogger(logger_name)
        self.__to_copy_folder = to_copy_folder
        self.__stop_event = threading.Event()

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
                    self.__copy_from_address(ip=ip, destination_folder=destination, username=username,
                                             password=password)

            self.__stop_event.wait(self.__sleep_copy_interval)  # instead of sleeping

    def __copy_from_address(self, ip, destination_folder, username, password):
        """
        Copy from an addres
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
            return True
        except (paramiko.BadHostKeyException, paramiko.AuthenticationException,
                paramiko.SSHException, socket.error):
            self.__logger.debug(f"Could not download logs from {ip}")
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

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M',
        filename="unit_test_CopyLogs.log",
        filemode='w'
    )
    copy_logs = CopyLogs(machines=MACHINES,
                         sleep_copy_interval=10,
                         destination_folder="/tmp",
                         logger_name="COPY_LOGS_LOG",
                         to_copy_folder="/var/radiation-benchmarks/log/")
    copy_logs.start()
    time.sleep(10)
    copy_logs.stop_copying()
    copy_logs.join()
