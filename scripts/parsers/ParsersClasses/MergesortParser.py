import re

from Parser import Parser


class MergesortParser(Parser):
    # Return [posX, posY, read, expected] -> [int, int, float, float]
    # Returns None if it is not possible to parse
    def parseErrMethod(self, errString):
        try:
            # ERR stream: 0, p: [0, 0], r: 3.0815771484375000e+02, e: 0.0000000000000000e+00
            m = re.match(".*ERR.*\[(\d+)..(\d+)\].*r\: ([0-9e\+\-\.]+).*e\: ([0-9e\+\-\.]+)", errString)
            if m:
                posX = int(m.group(1))
                posY = int(m.group(2))
                read = float(m.group(3))
                expected = float(m.group(4))
                return [posX, posY, read, expected]
            else:
                return None
        except ValueError:
            return None




    def setSize(self, header):
        self._size = None
        m = re.match(".*size\:(\d+).*", header)
        if m:
            try:
                self._size = int(m.group(1))
            except:
                self._size = None
        self._size = str(self._size)



    def buildImageMethod(self):
        return False

    def getBenchmark(self):
        return self._benchmark