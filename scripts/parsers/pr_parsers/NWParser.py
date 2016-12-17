import Parser
import re



class NWParser(Parser):
    # Return [posX, posY, read, expected] -> [int, int, float, float]
    # Returns None if it is not possible to parse
    def parseErr(self, errString):
        try:
            # ERR  p: [1, 467], r: -4654, e: 21, error: 467
            m = re.match(".*ERR.*\[(\d+)..(\d+)\].*r\: ([0-9\+\-]+).*e\: ([0-9\+\-]+).*", errString)
            # print errString
            # print m.group(1) , m.group(2) , m.group(3) , m.group(4)
            if m:
                posX = int(m.group(1))
                posY = int(m.group(2))
                read = int(m.group(3))
                expected = int(m.group(4))
                # print m.group(1) , m.group(2) , m.group(3) , m.group(4)
                return [posX, posY, read, expected]
            else:
                return None
        except ValueError:
            return None


    def getLogHeader(self, header):

        self.size = None
        m = re.match(".*size\:(\d+).*", header)
        if m:
            try:
                self.size = int(m.group(1))
            except:
                self.size = None


    def getLogHeader(self, header):
        self.size = None
        m = re.match(".*size\:(\d+).*", header)
        if m:
            try:
                self.size = int(m.group(1))
            except:
                self.size = None
