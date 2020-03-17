#!/usr/bin/python3

from datetime import datetime, timedelta
import subprocess
import os
from time import sleep
import sys
sys.path.insert(0, os.path.abspath("."))

from remote_sock_parameters import IPmachines, IPtoNames


DEFAULT_RADIATION_PATH = "/var/radiation-benchmarks/log"
LOGS_DIR = "logs"
TAR_COMMAND = "tar czf {} {}"
TAR_DIR_FINAL = "/home/fernando/Dropbox/ChipIR202002"

TIMEDELTA_DROPBOX = timedelta(hours=8)
TIMEDELTA_DOWNLOAD = timedelta(minutes=20)
MAX_SCP_TRIES = 10


def tar_files_from_device(device_folder, machine_name):
    now = datetime.now()
    dt_string = now.strftime("%H_%M_%S_%Y_%m_%d")
    final_tar = "{}_{}.tar.gz".format(dt_string, machine_name)
    result = os.system(TAR_COMMAND.format(final_tar, device_folder))
    dropbox_dir = "{}/{}".format(TAR_DIR_FINAL, final_tar)
    os.rename(final_tar, dropbox_dir)

    if result != 0:
        raise ValueError("Could not tar {}".format(final_tar))


def scp_all_files_from_device(device_ip, device_folder):
    scp_string = "carol@{}:{}/*.log".format(device_ip, DEFAULT_RADIATION_PATH)
    output_scp = "{}/".format(device_folder)
    print(scp_string)
    for i in range(1, MAX_SCP_TRIES + 1):
        p = subprocess.Popen(["scp", scp_string, output_scp], stdout=subprocess.DEVNULL)
        os.waitpid(p.pid, 0)

        if p.stderr is None:
            break
        print("Trying for the {}th time".format(i))
        print(p.stderr)
        sleep(10)


def main():
    if not os.path.exists(LOGS_DIR):
        os.mkdir(LOGS_DIR)
    last_copy = datetime.now()
    for device_ip in IPmachines:
        machine_name = IPtoNames[device_ip]
        device_folder = "{}/{}".format(LOGS_DIR, machine_name)
        print("First copying from board {}".format(machine_name))
        if not os.path.exists(device_folder):
            os.mkdir(device_folder)
        scp_all_files_from_device(device_ip=device_ip, device_folder=device_folder)

    while True:
        tdelta = (datetime.now() - last_copy)

        if tdelta > TIMEDELTA_DOWNLOAD:
            for device_ip in IPmachines:
                machine_name = IPtoNames[device_ip]
                device_folder = "{}/{}".format(LOGS_DIR, machine_name)

                print("Copying from board {}".format(machine_name))
                scp_all_files_from_device(device_ip=device_ip, device_folder=device_folder)

                if tdelta > TIMEDELTA_DROPBOX:
                    print("Uploading to dropbox from board {}".format(machine_name))
                    tar_files_from_device(device_folder=device_folder, machine_name=machine_name)

            last_copy = datetime.now()


if __name__ == '__main__':
    main()
