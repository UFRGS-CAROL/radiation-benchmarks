import threading
import time
import logging

from .RebootMachine import RebootMachine
from .server_parameters import BOOT_PROBLEM_MAX_DELTA, SLEEP_TIME, LOGGER_NAME


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
        self.__queue = kwargs.pop("messages_queue")
        super(Machine, self).__init__(*args, **kwargs)
        self.__timestamp = time.time()
        self.__logger = logging.getLogger(LOGGER_NAME)

        # Stops the thread when false
        self.__is_machine_active = True

    def run(self):
        # lower and upper threshold for reboot interval
        lower_threshold = self.__time_min_reboot_threshold * self.__diff_reboot
        upper_threshold = self.__time_max_reboot_threshold * self.__diff_reboot

        # Last reboot timestamp
        last_reboot_timestamp = 0

        # boot problem disable
        boot_problem_disable = False
        while self.__is_machine_active:
            # Check if machine is working fine
            now = time.time()
            last_conn_delta = now - self.__timestamp
            if not boot_problem_disable:
                # print(last_conn_delta)
                # If machine is not working fine reboot it
                if self.__diff_reboot < last_conn_delta < lower_threshold:
                    reboot_delta = now - last_reboot_timestamp
                    # If the reboot delta is bigger than the allowed reboot
                    if reboot_delta > self.__diff_reboot:
                        reboot_status, last_reboot_timestamp = self.__reboot_this_machine()
                        self.__log(reboot_status)

                # If machine did not reboot, log this and set it to not check again
                elif lower_threshold < last_conn_delta < upper_threshold:
                    message_string = f"Boot Problem IP  {self.__ip} ({self.__hostname})"
                    # print(message_string)
                    self.__logger.error(message_string)
                    boot_problem_disable = True
                # IPActiveTest[address] = False
                elif last_conn_delta > upper_threshold:
                    reboot_status, last_reboot_timestamp = self.__reboot_this_machine()
                    self.__log(reboot_status)
            else:
                msg = f"IP {self.__ip} waiting due boot problem f{BOOT_PROBLEM_MAX_DELTA}s"
                self.__logger.info(msg)
                # print(msg)
                time.sleep(BOOT_PROBLEM_MAX_DELTA)

            # sleep before re-check again
            time.sleep(SLEEP_TIME)

    def __log(self, reboot_status):
        message = {
            "msg": "Rebooted", "ip": self.__ip, "status": reboot_status
        }
        # TODO: finish enqueue process
        # self.__queue.put(message)
        reboot_msg = f"Rebooting IP {self.__ip} ({self.__hostname}) status {reboot_status}"
        self.__logger.info(reboot_msg)

    def __reboot_this_machine(self):
        """
        reboot the device based on RebootMachine class
        :return reboot_status
        :return: last_last_reboot_timestamp
        when the last reboot was performed
        """
        last_reboot_timestamp = time.time()
        # Reboot machine in another thread
        reboot_thread = RebootMachine(machine_address=self.__ip,
                                      switch_model=self.__switch_model,
                                      switch_port=self.__switch_port,
                                      switch_ip=self.__switch_ip)
        reboot_thread.start()
        reboot_thread.join()
        reboot_status = reboot_thread.get_reboot_status()

        return reboot_status, last_reboot_timestamp

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
        self.__is_machine_active = False
        super(Machine, self).join(*args, **kwargs)


"""
# Debug 
from queue import Queue
from .server_parameters import LOG_FILE
print("CREATING THE MACHINE")
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
    datefmt='%m-%d %H:%M',
    filename=LOG_FILE,
    filemode='w'
)
machine = Machine(
    ip="127.0.0.1",
    diff_reboot=1,
    hostname="test",
    power_switch_ip="127.0.0.1",
    power_switch_port=1,
    power_switch_model="lindy",
    messages_queue=Queue()
)

print("EXECUTING THE MACHINE")
machine.start()
machine.set_timestamp(999999)

print("SLEEPING THE MACHINE")
time.sleep(300)

print("JOINING THE MACHINE")
machine.join()

print("RAGE AGAINST THE MACHINE")
"""

