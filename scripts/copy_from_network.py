#!/usr/bin/python

import os
import re

from paramiko import SSHClient, AutoAddPolicy, util
from scp import SCPClient
from time import sleep, asctime, localtime
import curses
import glob
import sys

sys.path.append("/home/fernando/radiation-benchmarks-parsers/JTX2InstParser/")

import JTX2InstParser

TIME_TO_SLEEP = 10

IPmachines = [
    # "192.168.1.21",  # tegrax2
    "192.168.1.10",  # k20
    # "192.168.1.15", #CarolTitanV1
    # "192.168.1.16", #CarolTeslaV1001
]

# Set the machine names for each IP

# Set the machine names for each IP
IPtoNames = {
    #"192.168.1.21": "caroltegrax21",
    "192.168.1.10": "carolk201",
    # "192.168.1.6": "CarolXeon1",
    # "192.168.1.7": "CarolXeon2",
}

file_list = []


def plot_graph():
    global file_list

    while True:
        if len(file_list):
            latest_file = max(file_list, key=os.path.getatime)
            latest_csv = latest_file
            if ".log" in latest_file:
                latest_csv = latest_file.replace(".log", "JTX2INST.csv")

            temperature, power, timestamp_start, timestamp_end = JTX2InstParser.parseFile(latest_csv)
            for p in power:
                print("Average power of {} : {}".format(p.replace("VDD_", "").replace("_POWER", ""), power[p]))

            for t in temperature:
                print("Average temperature of {} : {}".format(t.replace("_TEMPERATURE", ""), temperature[t]))
        sleep(1)


def get_power_temperature():
    data_ret = ""
    if len(file_list):
        new_file_list = []
        for i in file_list:
            if ".csv" in i:
                new_file_list.append(i)
        # latest_file = max(new_file_list, key=os.path.getatime)
        latest_csv = max(new_file_list)

        temperature, power, timestamp_start, timestamp_end = JTX2InstParser.parseFile(latest_csv)
        for p in power:
            # print("Average power of {} : {}".format(p.replace("VDD_", "").replace("_POWER", ""), power[p]))
            data_ret += "Average power of {} : {}\n".format(p.replace("VDD_", "").replace("_POWER", ""), power[p])

        for t in temperature:
            # print("Average temperature of {} : {}".format(t.replace("_TEMPERATURE", ""), temperature[t]))
            data_ret += "Average temperature of {} : {}\n".format(t.replace("_TEMPERATURE", ""), temperature[t])

        data_ret += "Power/Temp start timestamp: {}\nPower/Temp end timestamp {}\n".format(timestamp_start.strftime(
            "%c"), timestamp_end.strftime("%c"))
    return data_ret


def get_frequencies(frequency_log_path):
    m_gpu = None
    m_cpu = None
    with open(frequency_log_path, "r") as fq_log_file:
        lines = fq_log_file.readlines()

        # cpu0: Online=1 Governor=userspace MinFreq=1267200 MaxFreq=1267200 CurrentFreq=1267200 IdleStates: C1=1 c7=1
        # GPU MinFreq=1134750000 MaxFreq=1134750000 CurrentFreq=1134750000

        for l in lines:
            m = re.match(".*GPU.*CurrentFreq=(\d+).*", l)
            if m:
                m_gpu = m
            m = re.match(".*cpu(\d+): Online=1 Governor=userspace.*CurrentFreq=(\d+).*", l)
            if m:
                m_cpu = m
    result = "CPU frequency (Ghz) {:2.2f}\nGPU frequency (Ghz) {:2.2f}\n"
    if m_cpu and m_gpu:
        result = result.format(float(m_cpu.group(2)) / 1e6, float(m_gpu.group(1)) / 1e9)
    else:
        result = result.format(-1, -1)
    return result


def create_ssh_client():
    ssh_clients = {}
    for ip in IPmachines:
        ssh = SSHClient()
        ssh.load_system_host_keys()
        ssh.connect(ip, username="carol", password="qwerty0", allow_agent=False, look_for_keys=False)

        scp = SCPClient(ssh.get_transport())

        ssh_clients[ip] = {"ssh": ssh, "scp": scp}

    return ssh_clients


def print_information(path, connection_status):
    ret_str = str(asctime(localtime())) + "\n"
    info_count = {"SDC": 0, "ABORT": 0, "CUDA Framework error": 0, "END": 0}

    global file_list
    file_list = glob.glob(pathname=path + "log/*")

    log_len = 0
    for file in file_list:
        if ".log" in file:
            log_len += 1
            with open(file) as fp:
                data = fp.readlines()

                for dt_l in data:
                    for inf_ct in info_count:
                        if inf_ct in dt_l:
                            info_count[inf_ct] += 1

    ret_str += "LOG NUMBER: {}\n".format(log_len)
    for inf_ct in info_count:
        ret_str += "{}: {}\n".format(inf_ct, info_count[inf_ct])

    # crashes + abort
    ret_str += "TOTAL CRASHES: {}\n".format((log_len - info_count["END"]) +
                                            info_count["ABORT"] + info_count["CUDA Framework error"])

    ret_str += "CONNECTION STATUS: {}\n".format("Good" if connection_status else "Broken")

    try:
        ret_str += get_power_temperature()
    except:
        # ret_str += "CANNOT SHOW POWER AND TEMPERATURE YET. KEEP TRYING.\n"
        pass
    try:
        ret_str += get_frequencies(frequency_log_path=path + "frequency.log")
    except:
        # ret_str += "CANNOT SHOW FREQUENCIES YET. KEEP TRYING\n"
        pass

    return ret_str


def scp_jetson(full_path):
    try:

        ssh_clients = create_ssh_client()
        general_var_dir = "/var/radiation-benchmarks/"

        for dir_ip in IPtoNames:
            # GET the logs from machine
            scp = ssh_clients[dir_ip]["scp"]
            ssh = ssh_clients[dir_ip]["ssh"]

            scp.get(general_var_dir, full_path, recursive=True)
            scp.close()
            ssh.close()
        return True
    except:
        return False


def copy_log_folder(window):
    script_path = "/home/fernando/radiation-benchmarks/scripts/logs"
    if not os.path.exists(script_path):
        os.mkdir(script_path)

    for dir_ip in IPtoNames:
        full_path = "{}/{}".format(script_path, IPtoNames[dir_ip])
        if not os.path.exists(full_path):
            os.mkdir(full_path)

    util.log_to_file(script_path + "/ssh_file.log")

    while True:
        for dir_ip in IPtoNames:
            full_path = "{}/{}/".format(script_path, IPtoNames[dir_ip])
            connected = scp_jetson(full_path)

            # Pre parser the information inside the logs
            to_window = print_information(full_path + "radiation-benchmarks/", connected)
            window.addstr(1, 1, to_window)
            window.refresh()

            # print(to_window)
            # sys.stdout.write(to_window)
            # sys.stdout.flush()

            sleep(4)


if __name__ == "__main__":
    curses.wrapper(copy_log_folder)
