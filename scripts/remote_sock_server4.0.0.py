#!/usr/bin/python3

import threading
import socket
import time
import os
from datetime import datetime
import sys
import requests
import json

import remote_sock_parameters as par


# Log messages adding timestamp before the message
def logMsg(msg):
    now = datetime.now()
    # fp = open(logFile, 'a')
    # print >> fp, now.ctime() + ": " + str(msg)
    # fp.close()

    with open(par.logFile, 'a') as fp:
        fp.write(now.ctime() + ": " + str(msg) + "\n")


################################################
# Routines to perform power cycle user IP SWITCH
################################################
def replace_str_index(text, index=0, replacement=''):
    return '%s%s%s' % (text[:index], replacement, text[index + 1:])


def lindySwitch(portNumber, status, switchIP):
    led = replace_str_index("000000000000000000000000", portNumber - 1, "1")

    if status == "On":
        url = 'http://' + switchIP + '/ons.cgi?led=' + led
    else:
        url = 'http://' + switchIP + '/offs.cgi?led=' + led
    payload = {
        "led": led,
    }
    headers = {
        "Host": switchIP,
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.12; rv:56.0) Gecko/20100101 Firefox/56.0",
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate",
        "Referer": "http://" + switchIP + "/outlet.htm",
        "Authorization": "Basic c25tcDoxMjM0",
        "Connection": "keep-alive",
        "Content-Length": "0",
    }

    try:
        return requests.post(url, data=json.dumps(payload), headers=headers)
    except:
        logMsg("Could not change Lindy IP switch status, portNumber: " + str(
            portNumber) + ", status" + status + ", switchIP:" + switchIP)
        return 1


class Switch():
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
            if i == (port):
                cmd += self.portList[i] % c
            else:
                cmd += self.portList[i]

        cmd += '&Apply=Apply\" '
        cmd += 'http://%s/tgi/iocontrol.tgi ' % (self.ip)
        cmd += '-o /dev/null 2>/dev/null'
        return os.system(cmd)

    def on(self, port):
        return self.cmd(port, 'On')

    def off(self, port):
        return self.cmd(port, 'Off')


def setIPSwitch(portNumber, status, switchIP):
    if status == 'on' or status == 'On' or status == 'ON':
        cmd = 'On'
    elif status == 'off' or status == 'Off' or status == 'OFF':
        cmd = 'Off'
    else:
        return 1
    if par.SwitchIPtoModel[switchIP] == "default":
        s = Switch(switchIP, 4)
        return s.cmd(int(portNumber), cmd)
    elif par.SwitchIPtoModel[switchIP] == "lindy":
        return lindySwitch(int(portNumber), cmd, switchIP)


class RebootMachine(threading.Thread):
    def __init__(self, address):
        threading.Thread.__init__(self)
        self.address = address

    def run(self):
        port = par.IPtoSwitchPort[self.address]
        switchIP = par.IPtoSwitchIP[self.address]

        print("\tRebooting machine: " + self.address + ", switch IP: " + str(switchIP) + ", switch port: " + str(port))
        if setIPSwitch(port, "Off", switchIP) != 0:
            raise ValueError("setIPSwitch not working, maybe curl is not instaled")

        time.sleep(10)
        if setIPSwitch(port, "On", switchIP) != 0:
            raise ValueError("setIPSwitch not working, maybe curl is not instaled")


################################################
# Socket server
################################################
# Create an INET, STREAMing socket
serverSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)


def startSocket():
    # Bind the socket to a public host, and a well-known port
    serverSocket.bind((par.serverIP, par.socketPort))
    print("\tServer bind to: ", par.serverIP)
    # Become a server socket
    serverSocket.listen(15)

    while True:
        # Accept connections from outside
        (clientSocket, address) = serverSocket.accept()
        now = datetime.now()
        if address[0] in par.IPtoNames:
            print("Connection from " + address[0] + " (" + par.IPtoNames[address[0]] + ") " + str(now))
        else:
            print("connection from " + str(address[0]) + " " + str(now))

        if address[0] in par.IPmachines:
            IPLastConn[address[0]] = time.time()  # Set new timestamp
            # If machine was set to not check again, now it's alive so start to check again
            IPActiveTest[address[0]] = True
        clientSocket.close()


################################################
# Routines to check machine status
################################################
def checkMachines():
    for address, timestamp in IPLastConn.copy().items():
        # If machine had a boot problem, stop rebooting it
        if not IPActiveTest[address]:
            continue

        # Check if machine is working fine
        now = datetime.now()
        then = datetime.fromtimestamp(timestamp)
        seconds = (now - then).total_seconds()
        # If machine is not working fine reboot it
        if par.IPtoDiffReboot[address] < seconds < 3 * par.IPtoDiffReboot[address]:
            reboot = datetime.fromtimestamp(rebooting[address])
            if (now - reboot).total_seconds() > par.IPtoDiffReboot[address]:
                rebooting[address] = time.time()
                if address in par.IPtoNames:
                    print("Rebooting IP " + address + " (" + par.IPtoNames[address] + ")")
                    logMsg("Rebooting IP " + address + " (" + par.IPtoNames[address] + ")")
                else:
                    print("Rebooting IP " + address)
                    logMsg("Rebooting IP " + address)
                # Reboot machine in another thread
                RebootMachine(address).start()
        # If machine did not reboot, log this and set it to not check again
        elif 3 * par.IPtoDiffReboot[address] < seconds < 10 * par.IPtoDiffReboot[address]:
            if address in par.IPtoNames:
                print("Boot Problem IP " + address + " (" + par.IPtoNames[address] + ")")
                logMsg("Boot Problem IP " + address + " (" + par.IPtoNames[address] + ")")
            else:
                print("Boot Problem IP " + address)
                logMsg("Boot Problem IP " + address)
            IPActiveTest[address] = False
        elif seconds > 10 * par.IPtoDiffReboot[address]:
            if address in par.IPtoNames:
                print("Rebooting IP " + address + " (" + par.IPtoNames[address] + ")")
                logMsg("Rebooting IP " + address + " (" + par.IPtoNames[address] + ")")
            else:
                print("Rebooting IP " + address)
                logMsg("Rebooting IP " + address)
            RebootMachine(address).start()


class handleMachines(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        print("\tStarting thread to check machine connections")
        while 1:
            checkMachines()
            time.sleep(par.sleepTime)


################################################
# Main Execution
################################################

def main():
    global IPLastConn, IPActiveTest, rebooting
    try:
        # Set the initial timestamp for all IPs
        IPLastConn = dict()
        IPActiveTest = dict()
        rebooting = dict()
        for ip in par.IPmachines:
            rebooting[ip] = time.time()
            IPLastConn[ip] = time.time()  # Current timestamp
            IPActiveTest[ip] = True
            port = par.IPtoSwitchPort[ip]
            switchIP = par.IPtoSwitchIP[ip]
            setIPSwitch(port, "On", switchIP)

        handle = handleMachines()
        handle.setDaemon(True)
        handle.start()
        startSocket()
    except KeyboardInterrupt:
        print("\n\tKeyboardInterrupt detected, exiting gracefully!( at least trying :) )")
        serverSocket.close()
        sys.exit(1)


if __name__ == "__main__":
    main()
