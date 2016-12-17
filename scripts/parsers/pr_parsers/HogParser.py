import PrecisionAndRecall as pr
import re
import Parser

class HogParser(Parser):
    def __init__(self, **kwargs):
        self.prThreshold = float(kwargs.pop("threshold"))
        self.precisionAndRecall = pr.PrecisionAndRecall(self.prThreshold)

    def parseErr(self, errString):
        # ERR Image: set08_V009_1237.jpg
        # ERR 101,50,176,76,226,177
        #
        ret = {}
        try:
            if 'image' in errString:
                error = re.match(".*Image: (\S+).*")
                if error:
                    ret = re.group(1)
            else:
                error = re.match(".* (\S+),(\S+),(\S+),(\S+),(\S+),(\S+).*")

                if error:
                    # r.height, r.width, r.x,	r.y, r.br().x, r.br().y
                    ret["r_height"] = error.group(1)
                    ret["r_width"] = error.group(2)
                    ret["r_x"] = error.group(3)
                    ret["r_y"] = error.group(4)
                    ret["r_br_x"] = error.group(5)
                    ret["r_br_y"] = error.group(6)
        except:
            print "Error on HogParser.parseErrHog"
            raise

        return (ret if len(ret) > 0 else None)

    def relativeErrorParser(self, errList):
        return [None, None]

    def header(self):
        return ""




    def getLogHeader(self, header):
        self.size = None
        m = re.match(".*size\:(\d+).*", header)
        if m:
            try:
                self.size = int(m.group(1))
            except:
                self.size = None
