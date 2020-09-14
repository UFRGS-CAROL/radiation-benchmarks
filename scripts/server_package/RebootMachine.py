import threading
import time
import requests
import json
import logging

from .server_parameters import REBOOTING_SLEEP, LOGGER_NAME
from .common import Codes, execute_command


class RebootMachine(threading.Thread):
    __ON = "ON"
    __OFF = "OFF"

    def __init__(self, machine_address, switch_model, switch_port, switch_ip):
        super(RebootMachine, self).__init__()
        self.__address = machine_address
        self.__switch_port = switch_port
        self.__switch_ip = switch_ip
        self.__reboot_status = Codes.SUCCESS
        self.__switch_model = switch_model
        self.__logger = logging.getLogger(LOGGER_NAME)

    def run(self):
        # TODO: Remove FAKE when rebooting lines are uncommented
        self.__logger.info(f"\tRebooting machine: {self.__address}, switch IP: {self.__switch_ip},"
                           f" switch switch_port: {self.__switch_port}")
        self.__select_command_on_switch(self.__OFF)
        time.sleep(REBOOTING_SLEEP)
        self.__select_command_on_switch(self.__ON)

    def __select_command_on_switch(self, status):
        if self.__switch_model == "default":
            self.__common_switch_command(status)
        elif self.__switch_model == "lindy":
            self.__lindy_switch(status)
        else:
            raise ValueError("Incorrect switch switch_model")

    def __lindy_switch(self, status):
        to_change = "000000000000000000000000"
        led = f"{to_change[:(self.__switch_port - 1)]}1{to_change[self.__switch_port:]}"

        if status == self.__ON:
            url = f'http://{self.__switch_ip}/ons.cgi?led={led}'
        else:
            url = f'http://{self.__switch_ip}/offs.cgi?led={led}'
        payload = {
            "led": led,
        }
        headers = {
            "Host": self.__switch_ip,
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.12; rv:56.0) Gecko/20100101 Firefox/56.0",
            "Accept": "*/*",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "Referer": f"http://{self.__switch_ip}/outlet.htm",
            "Authorization": "Basic c25tcDoxMjM0",
            "Connection": "keep-alive",
            "Content-Length": "0",
        }

        # print(url)
        # print(headers)
        try:
            requests_status = requests.post(url, data=json.dumps(payload), headers=headers)
            requests_status.raise_for_status()
            self.__reboot_status = Codes.SUCCESS
        except requests.RequestException:
            self.__logger.exception(f"\tCould not change Lindy IP switch status, portNumber: {self.__switch_port} "
                                    f" status:{status} switchIP: {self.__switch_ip}")
            self.__reboot_status = Codes.ERROR

    def __common_switch_command(self, status):
        port_default_cmd = 'pw%1dName=&P6%1d=%%s&P6%1d_TS=&P6%1d_TC=&' % (
            self.__switch_port, self.__switch_port - 1, self.__switch_port - 1, self.__switch_port - 1)

        cmd = 'curl --data \"'
        cmd += port_default_cmd % ("On" if status == self.__ON else "Off")
        cmd += '&Apply=Apply\" '
        cmd += f'http://%s/tgi/iocontrol.tgi {self.__switch_ip}'
        cmd += '-o /dev/null '
        self.__reboot_status = execute_command(cmd)

    def get_reboot_status(self):
        return self.__reboot_status


if __name__ == '__main__':
    # FOR DEBUG ONLY
    from .server_parameters import LOG_FILE

    print("CREATING THE RebootMachine")
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M',
        filename=LOG_FILE,
        filemode='w'
    )
    reboot = RebootMachine(machine_address="192.168.0.4", switch_model="lindy", switch_port=2,
                           switch_ip="192.168.1.101")
    reboot.start()
    print(f"Reboot status {reboot.get_reboot_status()}")
    reboot.join()
