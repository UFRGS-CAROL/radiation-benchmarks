import re

from Parser import Parser


class NWParser(Parser):
    # Return [posX, posY, read, expected] -> [int, int, float, float]
    # Returns None if it is not possible to parse
    def parseErrMethod(self, errString):
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


    def setSize(self, header):
        self._max_rows = None
        self._max_cols = None
        self._penalty = None
        # for nw

        m = re.match(".*max_rows\:(\d+).*max_cols\:(\d+).*penalty\:(\d+).*", header)
        if m:

            try:
                self._max_rows = int(m.group(1))
                self._max_cols = int(m.group(2))
                self._penalty = int(m.group(3))
            except:
                self._max_rows = None
                self._max_cols = None
                self._penalty = None
        else:  # for old logs
            m = re.match(".*size\:(\d+).*(\d+).*", header)
            if m:
                self._max_rows = int(m.group(1))
                self._max_cols = int(m.group(2))
                self._penalty = None
        self._size = str(self._max_cols) + str(self._max_rows)



    def buildImageMethod(self):
        return False

    def getBenchmark(self):
        return self._benchmark