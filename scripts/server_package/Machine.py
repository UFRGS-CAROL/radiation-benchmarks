import threading
import time
import logging

from server_package.RebootMachine import RebootMachine
from server_package.ErrorCodes import ErrorCodes


class Machine(threading.Thread):
    """
    Machine Thread
    do not change the machine constants unless you
    really know what you are doing, most of the constants
    describes the behavior of HARD reboot execution
    """
    __TIME_MIN_REBOOT_THRESHOLD = 3
    __TIME_MAX_REBOOT_THRESHOLD = 10

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
        self.__reboot_sleep_time = kwargs.pop("reboot_sleep_time")
        self.__timestamp = time.time()
        self.__logger = logging.getLogger(self.__logger_name)
        self.__stop_event = threading.Event()
        self.__reboot_status = ErrorCodes.SUCCESS

        super(Machine, self).__init__(*args, **kwargs)

    def run(self):
        """
        Run execution of thread
        :return:
        """
        # lower and upper threshold for reboot interval
        lower_threshold = self.__TIME_MIN_REBOOT_THRESHOLD * self.__diff_reboot
        upper_threshold = self.__TIME_MAX_REBOOT_THRESHOLD * self.__diff_reboot
        # mandatory: It must start the machine on
        self.__turn_machine_on()
        # Last reboot timestamp. It makes sense set it to now,
        # since the first thing performed is the machine on
        last_reboot_timestamp = time.time()
        # boot problem disable
        boot_problem_disable = False

        while not self.__stop_event.isSet():
            # Check if machine is working fine
            now = time.time()
            last_conn_delta = now - self.__timestamp
            if boot_problem_disable is False:
                # print(last_conn_delta)
                # If machine is not working fine reboot it
                if self.__diff_reboot < last_conn_delta < lower_threshold:
                    # If the reboot delta is bigger than the allowed reboot
                    if (now - last_reboot_timestamp) > self.__diff_reboot:
                        last_reboot_timestamp = self.__reboot_this_machine()
                        self.__log(ErrorCodes.REBOOTING)
                # If machine did not reboot, log this and set it to not check again
                elif lower_threshold < last_conn_delta < upper_threshold:
                    self.__log(ErrorCodes.WAITING_FOR_POSSIBLE_BOOT)
                # Sanity checks
                elif last_conn_delta > upper_threshold:
                    self.__log(ErrorCodes.BOOT_PROBLEM)
                    # Disable only when upper threshold is reached
                    boot_problem_disable = True
            else:
                self.__log(ErrorCodes.WAITING_BOOT_PROBLEM)
                # instead of sleeping
                self.__stop_event.wait(self.__boot_problem_max_delta)
                boot_problem_disable = False
            # sleep before re-check again
            self.__stop_event.wait(self.__sleep_time)

    def __log(self, kind):
        """
        Log some behavior
        :param kind:
        :return:
        """
        reboot_msg = ""
        logger_function = self.__logger.info
        if kind == ErrorCodes.REBOOTING:
            if self.__reboot_status == ErrorCodes.SUCCESS:
                reboot_msg = f"Rebooted IP:{self.__ip} HOSTNAME:{self.__hostname} STATUS:{self.__reboot_status}"
            else:
                reboot_msg = f"Reboot failed for IP:{self.__ip} "
                reboot_msg += f" HOSTNAME:{self.__hostname} STATUS:{self.__reboot_status}"
                logger_function = self.__logger.error
        elif kind == ErrorCodes.WAITING_BOOT_PROBLEM:
            reboot_msg = f"Waiting {self.__boot_problem_max_delta}s due boot problem IP:{self.__ip} "
            reboot_msg += f"HOSTNAME:{self.__hostname}"
        elif kind == ErrorCodes.WAITING_FOR_POSSIBLE_BOOT:
            reboot_msg = f"Waiting for a possible boot in the future from IP:{self.__ip} HOSTNAME:{self.__hostname}"
            logger_function = self.__logger.debug
        elif kind == ErrorCodes.BOOT_PROBLEM:
            reboot_msg = f"Boot Problem IP:{self.__ip} HOSTNAME:{self.__hostname}. "
            reboot_msg += f"The thread will wait for a connection for {self.__boot_problem_max_delta}s"
            logger_function = self.__logger.error
        elif kind == ErrorCodes.MAX_SEQ_REBOOT_REACHED:
            reboot_msg = f"Maximum number of reboots allowed reached for IP:{self.__ip} HOSTNAME:{self.__hostname}"
            logger_function = self.__logger.error
        elif kind == ErrorCodes.TURN_ON:
            reboot_msg = f"Turning ON IP:{self.__ip} HOSTNAME:{self.__hostname} STATUS:{self.__reboot_status}"

        logger_function(reboot_msg)
        # TODO: finish enqueue process
        # message = {"msg": msg, "ip": self.__ip, "status": self.__reboot_status, "kind": kind}
        # self.__queue.put(message)

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
        self.__reboot_status = reboot_thread.reboot_status

        return last_reboot_timestamp

    def __turn_machine_on(self):
        """
        Turn on the machine
        :return:
        """
        reboot_thread = RebootMachine(machine_address=self.__ip,
                                      switch_model=self.__switch_model,
                                      switch_port=self.__switch_port,
                                      switch_ip=self.__switch_ip,
                                      rebooting_sleep=self.__reboot_sleep_time,
                                      logger_name=self.__logger_name)
        self.__log(ErrorCodes.TURN_ON)
        reboot_thread.on()
        self.__reboot_status = reboot_thread.reboot_status

    def update_machine_timestamp(self, timestamp):
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

    @property
    def hostname(self):
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
        datefmt='%d-%m-%y %H:%M:%S',
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
        boot_problem_max_delta=10,
        reboot_sleep_time=2,
    )

    print("EXECUTING THE MACHINE")
    machine.update_machine_timestamp(time.time())

    machine.start()

    sleep_time = 100
    print(f"SLEEPING THE MACHINE FOR {sleep_time}s")
    time.sleep(sleep_time)

    print("JOINING THE MACHINE")
    machine.join()

    print("RAGE AGAINST THE MACHINE")
