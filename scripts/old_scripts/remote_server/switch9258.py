#!/usr/bin/python

import os
import sys

class Switch():
    def __init__(self, ip, portCount):
        self.ip = ip
        self.portCount = portCount
        self.portList = []
        for i in range(0, self.portCount):
            self.portList.append(
                        'pw%1dName=&P6%1d=%%s&P6%1d_TS=&P6%1d_TC=&' %
                        (i+1, i, i, i)
                    )

    def cmd(self, port, c):
        assert(port <= self.portCount)

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

if __name__ == '__main__':
    usage = 'Usage: %s <port number> <status> <ip>\nWhere status is in [on, off, On, Off]'

    if len(sys.argv) < 4:
        print usage
        sys.exit(1)

    s = Switch(sys.argv[3], 4)

    if sys.argv[2] == 'on' or sys.argv[2] == 'On':
        cmd = 'On'
    elif sys.argv[2] == 'off' or sys.argv[2] == 'Off':
        cmd = 'Off'
    else:
        print "Unknown command: " + sys.argv[2]
        print usage
        sys.exit(1)

    sys.exit(s.cmd(int(sys.argv[1]), cmd))
