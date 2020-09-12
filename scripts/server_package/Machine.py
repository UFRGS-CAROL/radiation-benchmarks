import threading
import time
from datetime import datetime
import logging

from .server_parameters import SLEEP_TIME


class Machine(threading.Thread):
    __time_min_reboot_threshold = 3
    __time_max_reboot_threshold = 10

    def __init__(self, *args, **kwargs):
        """
        Initialize a new thread that represents a setup machine
        :param args: None
        :param ip
        :param diff_reboot
        :param hostname
        :param power_switch_ip
        :param power_switch_port
        :param power_switch_model
        """
        self.__ip = kwargs.pop("ip")
        self.__diff_reboot = kwargs.pop("diff_reboot")
        self.__hostname = kwargs.pop("hostname")
        self.__switch_ip = kwargs.pop("power_switch_ip")
        self.__switch_port = kwargs.pop("power_switch_port")
        self.__switch_model = kwargs.pop("power_switch_model")

        super(Machine, self).__init__(*args, **kwargs)
        self.__timestamp = time.time()
        self.__machine_status = True

    def run(self):
        # Check if machine is working fine
        now = datetime.now()
        then = datetime.fromtimestamp(self.__timestamp)
        seconds = (now - then).total_seconds()
        lower_threshold = self.__time_min_reboot_threshold * self.__diff_reboot
        upper_threshold = self.__time_max_reboot_threshold * self.__diff_reboot
        # If machine is not working fine reboot it
        while self.__machine_status:

            # If machine is not working fine reboot it
            if self.__diff_reboot < seconds < lower_threshold:
                # reboot = datetime.fromtimestamp(rebooting[address])
                # if (now - reboot).total_seconds() > IPtoDiffReboot[address]:
                #     rebooting[address] = time.time()
                #         print("Rebooting IP " + address + " (" + IPtoNames[address] + ")")
                #     # Reboot machine in another thread
                #     RebootMachine(address).start()
                logging.info(f"Rebooting IP {self.__ip} ({self.__hostname})")

            # If machine did not reboot, log this and set it to not check again
            elif lower_threshold < seconds < upper_threshold:
                # print("Boot Problem IP " + address + " (" + IPtoNames[address] + ")")
                logging.error(f"Boot Problem IP  {self.__ip} ({self.__hostname})")

            # IPActiveTest[address] = False
            elif seconds > upper_threshold:
                pass
                # if address in IPtoNames:
                #     print("Rebooting IP " + address + " (" + IPtoNames[address] + ")")
            logging.info(f"Rebooting IP {self.__ip} ({self.__hostname})")
            time.sleep(SLEEP_TIME)
        # RebootMachine(address).start()

    def reboot(self):
        """
        reboot the device based on RebootMachine class
        :return: None
        """

    def set_timestamp(self, timestamp):
        """
        Set the timestamp for the connection machine
        :param timestamp: current timestamp for this board
        :return: None
        """
        self.__timestamp = timestamp

    def join(self, *args, **kwargs):
        """
        Set if thread should stops or not
        :return:
        """
        self.__machine_status = False
        super(Machine, self).join(*args, **kwargs)
