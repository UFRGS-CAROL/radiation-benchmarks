import threading
import time
import logging

from .RebootMachine import RebootMachine


class Machine(threading.Thread):
    __time_min_reboot_threshold = 3
    __time_max_reboot_threshold = 10

    def __init__(self, *args, **kwargs):
        """
        Initialize a new thread that represents a setup machine
        :param args: None
        :param ip:
        :param diff_reboot:
        :param hostname:
        :param power_switch_ip:
        :param power_switch_port:
        :param power_switch_model:
        :param messages_queue:
        :param sleep_time:
        :param logger_name:
        :param boot_problem_max_delta:
        """
        self.__ip = kwargs.pop("ip")
        self.__diff_reboot = kwargs.pop("diff_reboot")
        self.__hostname = kwargs.pop("hostname")
        self.__switch_ip = kwargs.pop("power_switch_ip")
        self.__switch_port = kwargs.pop("power_switch_port")
        self.__switch_model = kwargs.pop("power_switch_model")
        self.__queue = kwargs.pop("messages_queue")
        self.__sleep_time = kwargs.pop("sleep_time")
        self.__logger_name = kwargs.pop("logger_name")
        self.__boot_problem_max_delta = kwargs.pop("boot_problem_max_delta")
        self.__timestamp = time.time()
        self.__logger = logging.getLogger(self.__logger_name)
        self.__reboot_sleep_time = kwargs.pop("reboot_sleep_time")
        # self.__RebootMachine = kwargs.pop("RebootMachine")
        self.__stop_event = threading.Event()

        super(Machine, self).__init__(*args, **kwargs)

        # Stops the thread when false
        # self.__is_machine_active = True

    def run(self):
        # lower and upper threshold for reboot interval
        lower_threshold = self.__time_min_reboot_threshold * self.__diff_reboot
        upper_threshold = self.__time_max_reboot_threshold * self.__diff_reboot

        # Last reboot timestamp
        last_reboot_timestamp = 0

        # boot problem disable
        boot_problem_disable = False
        while not self.__stop_event.isSet():
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
                    message_string = f"\tBoot Problem IP  {self.__ip} ({self.__hostname})"
                    # print(message_string)
                    self.__logger.error(message_string)
                    boot_problem_disable = True
                # IPActiveTest[address] = False
                elif last_conn_delta > upper_threshold:
                    reboot_status, last_reboot_timestamp = self.__reboot_this_machine()
                    self.__log(reboot_status)
            else:
                msg = f"\tIP {self.__ip} waiting due boot problem f{self.__boot_problem_max_delta}s"
                self.__logger.info(msg)
                # print(msg)
                # time.sleep(self.__boot_problem_max_delta)
                self.__stop_event.wait(self.__boot_problem_max_delta)  # instead of sleeping

                boot_problem_disable = False

            # sleep before re-check again
            # time.sleep(self.__sleep_time)
            self.__stop_event.wait(self.__sleep_time)

    def __log(self, reboot_status):
        message = {
            "msg": "Rebooted", "ip": self.__ip, "status": reboot_status
        }
        # TODO: finish enqueue process
        # self.__queue.put(message)
        reboot_msg = f"\tRebooting IP {self.__ip} ({self.__hostname}) status {reboot_status}"
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
                                      switch_ip=self.__switch_ip,
                                      rebooting_sleep=self.__reboot_sleep_time,
                                      logger_name=self.__logger_name)
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
        # self.__is_machine_active = False
        self.__stop_event.set()
        super(Machine, self).join(*args, **kwargs)

    def get_hostname(self):
        """
        Return hostname
        :return: hostname str
        """
        return self.__hostname


if __name__ == '__main__':
    # FOR DEBUG ONLY
    from queue import Queue
    # from RebootMachine import RebootMachine

    print("CREATING THE MACHINE")
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M',
        filename="unit_test_log_Machine.log",
        filemode='w'
    )
    machine = Machine(
        ip="127.0.0.1",
        diff_reboot=1,
        hostname="test",
        power_switch_ip="127.0.0.1",
        power_switch_port=1,
        power_switch_model="lindy",
        messages_queue=Queue(),
        sleep_time=5,
        logger_name="MACHINE_LOG",
        boot_problem_max_delta=300,
        reboot_sleep_time=2,
    )

    print("EXECUTING THE MACHINE")
    machine.start()
    machine.set_timestamp(999999)

    print("SLEEPING THE MACHINE")
    time.sleep(20)

    print("JOINING THE MACHINE")
    machine.join()

    print("RAGE AGAINST THE MACHINE")
