#!/usr/bin/python3

from queue import Queue
import time
import logging

from server_parameters import *
from server_package.Machine import Machine
from server_package.CopyLogs import CopyLogs


def test_machine():
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


def test_copy_logs():
    copy_thread_list = list()
    for mac in MACHINES:
        if mac["enabled"]:
            copy_obj = CopyLogs(
                ip=mac["ip"],
                password=mac["password"],
                username=mac["username"],
                hostname=mac["hostname"],
                sleep_copy_interval=COPY_LOG_INTERVAL,
                to_copy_folder=DEFAULT_RADIATION_LOGS_PATH,
                destination_folder=SERVER_LOG_PATH
            )

            copy_obj.start()
            copy_thread_list.append(copy_obj)
            copy_obj.start()
    time.sleep(10)
    for cp in copy_thread_list:
        cp.join()


def test_reboo_machine():
    raise NotImplemented("Implement it first")


if __name__ == '__main__':
    # FOR DEBUG USE ONLY
    # test_machine()
    test_copy_logs()
