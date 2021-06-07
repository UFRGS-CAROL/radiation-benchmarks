#!/usr/bin/python3

import configparser
import sys
import os
import re
from pathlib import Path

yes = {'yes', 'y', 'ye', ''}
no = {'no', 'n'}


def check_path(path):
    print("The install directory is '" + path + "', is that correct [Y/n]: ", end="")

    choice = input().lower()
    if choice in yes:
        return True
    elif choice in no:
        return False
    else:
        sys.stdout.write("Please respond with 'yes' or 'no'")
        return False


def install_path():
    path = os.getcwd()
    path = re.sub(r'/scripts', '', path)
    while not check_path(path):
        print("Please, enter the install path (radiation-benchmarks directory):", end="")
        path = input()
    return path


def remove_sudo():
    """
    Remove sudo password requesting
    :return:
    """
    print("[CAUTION] Remove sudo password for all users [Y/n] (default yes):", end="")
    choice = input().lower()
    sudo_str = "	ALL	ALL = (ALL) NOPASSWD: ALL\n"
    pattern = r".*ALL.*ALL.*=.*(ALL).*NOPASSWD:.*ALL.*"
    if choice in yes:
        sudoers_path = "/etc/sudoers"
        contains_line = False
        with open(sudoers_path, "r") as sudoers_file:
            for line in sudoers_file.readlines():
                if re.match(pattern, line):
                    contains_line = True
                    break
        if contains_line is False:
            with open(sudoers_path, "a") as sudoers_file:
                sudoers_file.write(sudo_str)
        print("sudo password request removed, remove last line to add it again")


def replace(old_file_path, new_file_path, pattern, subst):
    with open(new_file_path, 'w') as new_file, open(old_file_path) as old_file:
        for line in old_file:
            new_file.write(line.replace(pattern, subst))


def place_rc_local(install_path__):
    print("[CAUTION] Do you wish to create an /etc/rc.local [Y/n] (default yes):", end="")
    choice = input().lower()
    etc_path = "/etc/rc.local"
    home = str(Path.home())
    at_boot_path = f"{home}/atBoot.sh"
    at_boot_example_path = f"{install_path__}/scripts/atBoot.sh.example"
    file_content = ["#!/bin/bash\n\n", f"sudo {home}/atBoot.sh &\n\n", "exit 0\n"]

    if choice in yes:
        with open(etc_path, "w+") as etc_fp:
            etc_fp.writelines(file_content)

        replace(at_boot_example_path, at_boot_path, "/home/carol", home)
        print(f"{etc_path} file created, fill {at_boot_path} with the desirable script")
        os.chmod(at_boot_path, 0o777)
        os.chmod(etc_path, 0o777)


var_dir = "/var/radiation-benchmarks"
conf_file = "/etc/radiation-benchmarks.conf"

log_dir = var_dir + "/log"
mic_log = "/micNfs/carol/logs"

# Each update demands addition here
kill_test_version = "-2.0"
signal_cmd = f"killall -q -USR1 killtestSignal{kill_test_version}.py;"
signal_cmd += f" killall -q -USR1 test_killtest_commands_json{kill_test_version}.py; killall -q -USR1 python3"

config = configparser.ConfigParser()

install_path_ = install_path()
config.set("DEFAULT", "installdir", install_path_)
config.set("DEFAULT", "vardir", var_dir)
config.set("DEFAULT", "logdir", log_dir)
config.set("DEFAULT", "tmpdir", "/tmp")
config.set("DEFAULT", "miclogdir", mic_log)
config.set("DEFAULT", "signalcmd", signal_cmd)
try:
    if not os.path.isdir(var_dir):
        os.mkdir(var_dir, 0o777)
    os.chmod(var_dir, 0o777)
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir, 0o777)
    os.chmod(log_dir, 0o777)
    with open(conf_file, 'w') as configfile:
        config.write(configfile)

    remove_sudo()
    place_rc_local(install_path_)

except IOError:
    print("I/O Error, please make sure to run as root (sudo)")
    sys.exit(1)

print("var directory created (" + var_dir + ")")
print("log directory created (" + log_dir + ")")
print("Config file written (" + conf_file + ")")
