#!/usr/bin/python3

import configparser
import sys
import os
import re


def check_path(path):
    print("The install directory is '" + path + "', is that correct [Y/n]: ")
    yes = {'yes', 'y', 'ye', ''}
    no = {'no', 'n'}

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
        print("Please, enter the install path (radiation-benchmarks directory): ")
        path = input()
    return path


var_dir = "/var/radiation-benchmarks"
conf_file = "/etc/radiation-benchmarks.conf"

log_dir = var_dir + "/log"
mic_log = "/micNfs/carol/logs"

signal_cmd = "killall -q -USR1 killtestSignal.py;"
signal_cmd += " killall -q -USR1 test_killtest_commands_json.py; killall -q -USR1 python"

config = configparser.ConfigParser()

config.set("DEFAULT", "installdir", install_path())
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
except IOError:
    print("I/O Error, please make sure to run as root (sudo)")
    sys.exit(1)

print("var directory created (" + var_dir + ")")
print("log directory created (" + log_dir + ")")
print("Config file written (" + conf_file + ")")
