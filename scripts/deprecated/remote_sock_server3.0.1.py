#!/usr/bin/python3
import threading
import socket
import time
import os
from datetime import datetime

import requests
import json
from .remote_sock_parameters import *

# This val will be multiplied by IPtoDiffReboot value
IP_TO_DIFF_REBOOT_THRESHOLD = 3


class Codes:
    SUCCESS_CODE = 0
    ERROR_CODE = 1


# Log messages adding timestamp before the message
def log_msg(msg):
    now = datetime.now()
    with open(logFile, 'a') as fp:
        fp.write(now.ctime() + ": " + str(msg) + "\n")


def lindy_switch(port_number, status, switch_ip):
    # led = replace_str_index("000000000000000000000000", port_number - 1, "1")
    to_change = "000000000000000000000000"
    led = f"{to_change[:(port_number - 1)]}1{to_change[port_number:]}"

    if status == "On":
        url = f'http://{switch_ip}/ons.cgi?led={led}'
    else:
        url = f'http://{switch_ip}/offs.cgi?led={led}'
    payload = {
        "led": led,
    }
    headers = {
        "Host": switch_ip,
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.12; rv:56.0) Gecko/20100101 Firefox/56.0",
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate",
        "Referer": "http://" + switch_ip + "/outlet.htm",
        "Authorization": "Basic c25tcDoxMjM0",
        "Connection": "keep-alive",
        "Content-Length": "0",
    }

    try:
        result_request = requests.post(url, data=json.dumps(payload), headers=headers)
        result_request.raise_for_status()
        return Codes.SUCCESS_CODE
    except requests.RequestException:
        log_msg(
            f"Could not change Lindy IP switch status, port_number:{port_number} status:{status} switch_ip:{switch_ip}"
        )
        return Codes.ERROR_CODE


class Switch:
    def __init__(self, ip, portCount):
        self.ip = ip
        self.portCount = portCount
        self.portList = []
        for i in range(0, self.portCount):
            self.portList.append(
                'pw%1dName=&P6%1d=%%s&P6%1d_TS=&P6%1d_TC=&' %
                (i + 1, i, i, i)
            )

    def cmd(self, port, c):
        assert (port <= self.portCount)

        cmd = 'curl --data \"'

        # the port list is indexed from 0, so fix it
        port = port - 1

        for i in range(0, self.portCount):
            if i == port:
                cmd += self.portList[i] % c
            else:
                cmd += self.portList[i]

        cmd += '&Apply=Apply\" '
        cmd += f'http://%s/tgi/iocontrol.tgi {self.ip}'
        cmd += '-o /dev/null 2>/dev/null'
        return os.system(cmd)

    def on(self, port):
        return self.cmd(port, 'On')

    def off(self, port):
        return self.cmd(port, 'Off')


def set_ip_switch(portNumber, status, switchIP):
    if status in ['on', 'On', 'ON']:
        cmd = 'On'
    elif status in ['off', 'Off', 'OFF']:
        cmd = 'Off'
    else:
        return Codes.ERROR_CODE

    if SwitchIPtoModel[switchIP] == "default":
        s = Switch(switchIP, 4)
        # print(cmd, switchIP)
        return s.cmd(int(portNumber), cmd)
    elif SwitchIPtoModel[switchIP] == "lindy":
        return lindy_switch(int(portNumber), cmd, switchIP)


class RebootMachine(threading.Thread):
    def __init__(self, address):
        threading.Thread.__init__(self)
        self.address = address

    def run(self):
        port = IPtoSwitchPort[self.address]
        switch_ip = IPtoSwitchIP[self.address]
        print(f"\tRebooting machine: {self.address}, switch IP: {switch_ip}, switch port: {port}")
        ip_switch_result = set_ip_switch(port, "Off", switch_ip)
        if ip_switch_result != Codes.SUCCESS_CODE:
            print(f"\tReboot OFF failed for {self.address}, switch IP: {switch_ip}, switch port: {port}")

        time.sleep(rebootingSleep)
        ip_switch_result = set_ip_switch(port, "On", switch_ip)
        if ip_switch_result != Codes.SUCCESS_CODE:
            print(f"\tReboot ON failed for {self.address}, switch IP: {switch_ip}, switch port: {port}")


def start_socket():
    # Bind the socket to a public host, and a well-known port
    server_socket.bind((serverIP, socketPort))
    print("\tServer bind to: ", serverIP)
    # Become a server socket
    server_socket.listen(15)

    while True:
        # Accept connections from outside
        (clientSocket, address_list) = server_socket.accept()
        now = datetime.now()
        address = address_list[0]
        if address in IPtoNames:
            print(f"Connection from {address} ({IPtoNames[address]}) {now}")
        else:
            print(f"connection from {address} {now}")

        if address in IPmachines:
            ip_last_conn[address] = time.time()  # Set new timestamp
            # If machine was set to not check again, now it's alive so start to check again
            ip_active_test[address] = True
        clientSocket.close()


################################################
# Routines to check machine status
################################################
def check_machines():
    global ip_last_conn, ip_active_test, rebooting

    for address, timestamp in ip_last_conn.copy().items():
        # If machine had a boot problem, stop rebooting it
        if ip_active_test[address]:
            # Check if machine is working fine
            now = datetime.now()
            then = datetime.fromtimestamp(timestamp)
            seconds = (now - then).total_seconds()
            # If machine is not working fine reboot it
            if IPtoDiffReboot[address] < seconds < IP_TO_DIFF_REBOOT_THRESHOLD * IPtoDiffReboot[address]:
                reboot = datetime.fromtimestamp(rebooting[address])
                if (now - reboot).total_seconds() > IPtoDiffReboot[address]:
                    rebooting[address] = time.time()
                    if address in IPtoNames:
                        print("Rebooting IP " + address + " (" + IPtoNames[address] + ")")
                        log_msg("Rebooting IP " + address + " (" + IPtoNames[address] + ")")
                    else:
                        print("Rebooting IP " + address)
                        log_msg("Rebooting IP " + address)
                    # Reboot machine in another thread
                    RebootMachine(address).start()
            # If machine did not reboot, log this and set it to not check again
            elif IP_TO_DIFF_REBOOT_THRESHOLD * IPtoDiffReboot[address] < seconds < IP_TO_DIFF_REBOOT_THRESHOLD * IPtoDiffReboot[address]:
                if address in IPtoNames:
                    print("Boot Problem IP " + address + " (" + IPtoNames[address] + ")")
                    log_msg("Boot Problem IP " + address + " (" + IPtoNames[address] + ")")
                else:
                    print("Boot Problem IP " + address)
                    log_msg("Boot Problem IP " + address)
                ip_active_test[address] = False
            elif seconds > 10 * IPtoDiffReboot[address]:
                if address in IPtoNames:
                    print("Rebooting IP " + address + " (" + IPtoNames[address] + ")")
                    log_msg("Rebooting IP " + address + " (" + IPtoNames[address] + ")")
                else:
                    print("Rebooting IP " + address)
                    log_msg("Rebooting IP " + address)
                RebootMachine(address).start()


class HandleMachines(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        print("\tStarting thread to check machine connections")
        while True:
            check_machines()
            time.sleep(sleepTime)


def main():
    """
    Main function
    :return: None
    """
    global ip_last_conn, ip_active_test, rebooting, server_socket

    try:
        # Set the initial timestamp for all IPs
        for ip in IPmachines:
            rebooting[ip] = time.time()
            ip_last_conn[ip] = time.time()  # Current timestamp
            ip_active_test[ip] = True
            port = IPtoSwitchPort[ip]
            switch_ip = IPtoSwitchIP[ip]
            set_ip_switch(port, "On", switch_ip)

        handle = HandleMachines()
        handle.setDaemon(True)
        handle.start()
        start_socket()
    except KeyboardInterrupt:
        print("\n\tKeyboardInterrupt detected, exiting gracefully!( at least trying :) )")
        server_socket.close()


if __name__ == '__main__':
    ip_last_conn = dict()
    ip_active_test = dict()
    rebooting = dict()

    ################################################
    # Socket server
    ################################################
    # Create an INET, STREAMing socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    main()
