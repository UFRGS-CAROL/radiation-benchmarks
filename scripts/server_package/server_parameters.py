"""
All parameters necessary to a radiation test
no more magic numbers
"""

socketPort = 8080  # PORT the socket will listen to
sleepTime = 5  # Time between checks

serverIP = "192.168.1.5"  # IP of the remote socket server (hardware watchdog)

# time of a hard reboot
rebootingSleep = 10

# Set the machines IP to check, comment the ones we are not checking
IPmachines = [
    "192.168.1.11",  # k201
    "192.168.1.14",  # v1001
    "192.168.1.15",  # k202
    "192.168.1.51",  # apu1
]

# Set the machine names for each IP
IPtoDiffReboot = {
    "192.168.1.11": 100,  # k201
    "192.168.1.14": 100,  # v1001
    "192.168.1.15": 100,  # k202
    "192.168.1.51": 100,  # apu1
}

# Set the machine names for each IP
IPtoNames = {
    "192.168.1.11": "carolk201",
    "192.168.1.14": "carolv1001",
    "192.168.1.15": "carolk202",
    "192.168.1.51": "carolapu1",
}

# Set the switch IP that a machine IP is connected
IPtoSwitchIP = {
    "192.168.1.11": "192.168.1.100",  # carolk201
    "192.168.1.14": "192.168.1.100",  # carolv1001
    "192.168.1.15": "192.168.1.100",  # carolk202
    "192.168.1.51": "192.168.1.100",  # carolapu1
}

# Set the switch Port that a machine IP is connected
IPtoSwitchPort = {
    "192.168.1.11": 1,  # carolk201
    "192.168.1.14": 2,  # carolv1001
    "192.168.1.15": 3,  # carolk202
    "192.168.1.51": 4,  # carolapu1
}

SwitchIPtoModel = {
    "192.168.1.100": "lindy",
    # "192.168.1.101": "default",
    # "192.168.1.102": "default",
    # "192.168.1.103": "lindy",
    # "192.168.1.104": "default",
}

# log in whatever path you are executing this script
logFile = "server.log"
