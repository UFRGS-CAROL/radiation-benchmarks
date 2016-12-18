import re

from ParsersClasses import Parser


class ACCLParser(Parser):
    # Return [posX, posY, read, expected, comp_or_spans] -> [int, int, float, float, string]
    # Returns None if it is not possible to parse
    def parseErr(self, errString):
        try:
            # ERR stream: 0, p: [0, 0], r: 3.0815771484375000e+02, e: 0.0000000000000000e+00
            # ERR t: [components], p: [256][83], r: 90370, e: 80131
            ##ERR t: [components], p: [770][148], r: 80130, e: -1
            m = re.match(".*ERR.*p\: \[(\d+)\]\[(\d+)\].*r\: ([(\d+)]+).*e\: ([(\d+\-)]+)", errString)
            if m:
                posX = int(m.group(1))
                posY = int(m.group(2))
                read = int(m.group(3))
                expected = int(m.group(4))
                # print [posX, posY, read, expected]
                return [posX, posY, read, expected]
            else:
                m = re.match(".*ERR.*p\: \[(\d+)\].*r\: ([(\d+\-)]+).*e\: ([(\d+\-)]+)", errString)
                if m:
                    pos = int(m.group(1))
                    # posY = int(m.group(2))
                    read = int(m.group(2))
                    expected = int(m.group(3))
                    i = 0
                    j = 0
                    done = False

                    if 'spans' in errString:
                        while (i < 512):
                            j = 0
                            while (j < (4096 * 2)):
                                if ((i * j) >= pos):
                                    # print (i * j) , pos , errString
                                    done = True
                                    break
                                j += 1
                            if done:
                                break
                            i += 1

                    if 'components' in errString:
                        while (i < 512):
                            j = 0
                            while (j < 4096):
                                if ((i * j) >= pos):
                                    # print (i * j) , pos , errString
                                    done = True
                                    break
                                j += 1

                            if done:
                                break
                            i += 1

                    posX = i
                    posY = j

                    return [posX, posY, read, expected]
                else:
                    return None
        except ValueError:
            return None


    def setLogHeader(self, header):
        self.header = header
        # for accl
        m = re.match(".*frames[\:-](\d+).*", self.header)
        self.frames = None
        self.framesPerStream = None
        if m:
            try:
                self.frames = int(m.group(1))
                m = re.match(".*framesPerStream[\:-](\d+).*", self.header)
                self.framesPerStream = int(m.group(1))
            except:
                self.frames = None
                self.framesPerStrem = None
        self.max_rows = None
        self.max_cols = None
        self.penalty = None