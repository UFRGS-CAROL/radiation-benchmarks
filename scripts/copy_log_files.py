#!/usr/bin/python3

from datetime import datetime
import subprocess
import os

DEFAULT_RADIATION_PATH = "/var/radiation-benchmarks/log"
LOGS_DIR = "logs"
TAR_COMMAND = "tar czf {} {}"
TAR_DIR_FINAL = "/home/fernando/Dropbox/ChipIR202002"

# Set the machines IP to check, comment the ones we are not checking
IPmachines = [
    "192.168.1.11",  # k201
    "192.168.1.21",  # k401
    # "192.168.1.15", #CarolTitanV1
    # "192.168.1.16", #CarolTeslaV1001
]

# Set the machine names for each IP
IPtoNames = {
    "192.168.1.11": "carolk201",
    "192.168.1.21": "carolk401",
    # "192.168.1.6": "CarolXeon1",
    # "192.168.1.7": "CarolXeon2",
}


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
    p = subprocess.Popen(["scp", scp_string, output_scp])
    sts = os.waitpid(p.pid, 0)


def main():
    if not os.path.exists(LOGS_DIR):
        os.mkdir(LOGS_DIR)

    for device_ip in IPmachines:
        machine_name = IPtoNames[device_ip]
        device_folder = "{}/{}".format(LOGS_DIR, machine_name)
        if not os.path.exists(device_folder):
            os.mkdir(device_folder)

        scp_all_files_from_device(device_ip=device_ip, device_folder=device_folder)
        tar_files_from_device(device_folder=device_folder, machine_name=machine_name)


if __name__ == '__main__':
    main()
