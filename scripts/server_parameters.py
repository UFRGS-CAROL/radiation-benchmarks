"""
All parameters necessary to a radiation test
no more magic numbers
"""

# PORT the socket will listen to
SOCKET_PORT = 8080
# IP of the remote socket server (hardware watchdog)
SERVER_IP = "192.168.1.5"
# SERVER_IP = "25.91.229.61"

# Time between machine checks
MACHINE_CHECK_SLEEP_TIME = 5

# time of a hard reboot
REBOOTING_SLEEP = 10

# log in whatever path you are executing this script
LOG_FILE = "server.log"
# Logger obj name
LOGGER_NAME = 'SOCK_SERVER_4.0.0'

# Copy logs interval
# time in seconds between the connections
COPY_LOG_INTERVAL = 600

"""
Boot problem disable delta
Machine class will wait this time
to reboot again after a boot problem
e.g. 300s (5min)
"""
BOOT_PROBLEM_MAX_DELTA = 300

"""
Set the machines IP to check
the ones which enabled parameter is false are not checked
:param ip the IP address of the machine
:param enabled True if the server must use this machine, False otherwise
:param reboot_diff delta to wait before reboot again
:param hostname the hostname of the machine, 
       that must contains the device in the name
:param power_switch_ip IP address of the power switch
:param power_switch_port outlet of the power switch
:param power_switch_model brand of the power switch
:param username
:param password for connection
"""
MACHINES = [
    {
        "ip": "192.168.1.11",
        "enabled": True,
        "diff_reboot": 100,
        "hostname": "carolk201",
        "power_switch_ip": "192.168.1.100",
        "power_switch_port": 1,
        "power_switch_model": "lindy",
        "username": "carol",
        "password": "R@di@tion"
    },
    {
        "ip": "192.168.1.14",
        "enabled": False,
        "diff_reboot": 100,
        "hostname": "carolk202",
        "power_switch_ip": "192.168.1.100",
        "power_switch_port": 2,
        "power_switch_model": "lindy",
        "username": "carol",
        "password": "R@di@tion"
    },
    {
        "ip": "192.168.1.15",
        "enabled": False,
        "diff_reboot": 100,
        "hostname": "caroltitanv1",
        "power_switch_ip": "192.168.1.100",
        "power_switch_port": 3,
        "power_switch_model": "lindy",
        "username": "carol",
        "password": "R@di@tion"
    },
]
