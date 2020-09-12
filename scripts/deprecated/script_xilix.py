#!/usr/bin/python3

import json
import requests

from time import sleep
from datetime import datetime, timedelta

TIMEDELTA = timedelta(minutes=90)


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
        print("Could not change Lindy IP switch status, portNumber: " + str(
            portNumber) + ", status" + status + ", switchIP:" + switchIP)
        return 1


def main():
    portNumber = 1
    onStatus = "On"
    offStatus = "Off"

    switchIP = "192.168.1.66"
    lastTime = datetime.now()
    while True:
        diffTime = datetime.now() - lastTime
        if diffTime >= TIMEDELTA:
            print("Rebooting xilinx port {}".format(portNumber))
            lindySwitch(portNumber=portNumber, status=offStatus, switchIP=switchIP)
            sleep(1)
            lindySwitch(portNumber=portNumber, status=onStatus, switchIP=switchIP)
            lastTime = datetime.now()
        sleep(1)


if __name__ == "__main__":
    main()
