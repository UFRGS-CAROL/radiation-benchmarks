import re

from ParsersClasses import Parser


class HotspotParser(Parser):
    # Return [posX, posY, read, expected] -> [int, int, float, float]
    # Return [posX, posY, expected*2, read] if read is NaN
    # Returns None if it is not possible to parse
    def parseErr(self, errString):
        # print "Passou"
        try:
            m = re.match(".*ERR.*.*r\:([0-9e\+\-\.nan]+).*e\:([0-9e\+\-\.]+).*\[(\d+),(\d+)\]", errString)
            # OCL -> ERR r:293.943054 e:293.943024 [165,154]
            if m:
                posX = int(m.group(3))
                posY = int(m.group(4))
                read = float(m.group(1))
                expected = float(m.group(2))
                if re.match(".*nan.*", read):
                    return [posX, posY, expected * 2, expected]
                else:
                    read = float(read)
                    return [posX, posY, read, expected]

            m = re.match(".*ERR.*\[(\d+)..(\d+)\].*r\: ([0-9e\+\-\.nan]+).*e\: ([0-9e\+\-\.]+)", errString)
            # CUDA -> ERR stream: 0, p: [0, 0], r: 3.0815771484375000e+02, e: 0.0000000000000000e+00
            if m:
                print m.group(1) + " " + m.group(2) + " " + m.group(3) + " " + m.group(4) + " "
                posX = int(m.group(1))
                posY = int(m.group(2))
                read = m.group(3)
                expected = float(m.group(4))
                if re.match(".*nan.*", read):
                    return [posX, posY, expected * 2, expected]
                else:
                    read = float(read)
                    return [posX, posY, read, expected]
            return None
        except Exception as e:
            return None


    def getLogHeader(self, header):
        self.size = None
        m = re.match(".*size\:(\d+).*", header)
        if m:
            try:
                self.size = int(m.group(1))
            except:
                self.size = None