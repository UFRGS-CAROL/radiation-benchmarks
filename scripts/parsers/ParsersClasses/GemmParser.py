import re

from ParsersClasses import Parser


class GemmParser(Parser):
    # Return [posX, posY, read, expected] -> [int, int, float, float]
    # Returns None if it is not possible to parse
    def __parseErr(self, errString):
        try:
            # ERR stream: 0, p: [0, 0], r: 3.0815771484375000e+02, e: 0.0000000000000000e+00
            if 'nan' in errString:
                m = re.match(".*ERR.*\[(\d+)..(\d+)\].*r\:.*nan.*e\: ([0-9e\+\-\.]+)", errString)
                if m:
                    posX = int(m.group(1))
                    posY = int(m.group(2))
                    read = float('nan')
                    expected = float(m.group(3))
                    return [posX, posY, read, expected]
                else:
                    return None
            else:
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


    """
        build image, based on object parameters
        #currObj.buildImage(errorsParsed, size,
        #                            currObj.dirName + '/' + currObj.header + '/' + currObj.logFileNameNoExt + '_' + str(imageIndex))
    """
    def __buildImage(self, imgIndex):
        raise NotImplementedError()


    def getLogHeader(self, header):
        self.size = None
        m = re.match(".*size\:(\d+).*", header)
        if m:
            try:
                self.size = int(m.group(1))
            except:
                self.size = None