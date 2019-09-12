

socketPort = 8080  # PORT the socket will listen to
sleepTime = 5  # Time between checks

serverIP = "143.54.10.211" # "192.168.1.5"  # IP of the remote socket server (hardware watchdog)

# Set the machines IP to check, comment the ones we are not checking
IPmachines = [
    # "192.168.1.21",  # tegrax2
    "143.54.10.255",  # k20
    # "192.168.1.15", #CarolTitanV1
    # "192.168.1.16", #CarolTeslaV1001
]

# Set the machine names for each IP
IPtoDiffReboot = {
    "143.54.10.255": 60,  # caroltegrax21
    # "192.168.1.10": 50,  # k20
    # "192.168.1.6": 120,  # CarolXeon1
    # "192.168.1.7": 200,  # CarolXeon2
}

# Set the machine names for each IP
IPtoNames = {
    "143.54.10.255": "caroltegrax21",
    # "192.168.1.10": "carolk201",
    # "192.168.1.6": "CarolXeon1",
    # "192.168.1.7": "CarolXeon2",
}

# Set the switch IP that a machine IP is connected
IPtoSwitchIP = {
    "192.168.1.21": "192.168.1.100",  # caroltegrax21
    "192.168.1.10": "192.168.1.100",  # carolk201
    "143.54.10.255": "192.168.1.100",
    # "192.168.1.6": "192.168.1.102",  # CarolXeon1
    # "192.168.1.7": "192.168.1.104",  # CarolXeon2
}

# Set the switch Port that a machine IP is connected
IPtoSwitchPort = {
    "192.168.1.21": 1,  # caroltegrax2
    "192.168.1.10": 1,  # carolk201
    "143.54.10.255": 1,
    # "192.168.1.6": 1,  # CarolXeon1
    # "192.168.1.7": 4,  # CarolXeon2
}

SwitchIPtoModel = {
    "192.168.1.100": "default",
    "192.168.1.101": "default",
    # "192.168.1.102": "default",
    # "192.168.1.103": "lindy",
    # "192.168.1.104": "default",
}

# log in whatever path you are executing this script
logFile = "server.log"