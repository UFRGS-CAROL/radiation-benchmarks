import re


from Parser import Parser

class ACCLParser(Parser):

    def __init__(self, **kwargs):
        super(Parser, self).__init__(kwargs)


    __frames = None
    __framesPerStream = None
    __maxRows = None
    __maxCols = None
    __penalty = None


    def getBenchmark(self):
        return self._benchmark

    # Return [posX, posY, read, expected, comp_or_spans] -> [int, int, float, float, string]
    # Returns None if it is not possible to parse
    def parseErrMethod(self, errString):
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


    def setSize(self, header):
        # for accl
        m = re.match(".*frames[\:-](\d+).*", header)
        self.__frames = None
        self.__framesPerStream = None
        if m:
            try:
                self.__frames = int(m.group(1))
                m = re.match(".*framesPerStream[\:-](\d+).*", header)
                self.__framesPerStream = int(m.group(1))
            except:
                self.__frames = None
                self.__framesPerStrem = None
        # return self.__frames
        self._size = "frames_" + str(self.__frames) + "_framesPerStream_" + str(self.__framesPerStream)

    def buildImageMethod(self):
        return False